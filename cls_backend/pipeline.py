"""RAG+CAG instant backend — the millisecond retrieval system.

This is System 2 of the four-layer architecture, kept free of Streamlit so the UI layer
(`app.py`) can call it without the presentation and the retrieval concerns bleeding into
each other:

    1. UI/UX            — app.py (Streamlit)
    2. RAG+CAG backend  — THIS module (Retrieval Encoder + CAG + Evidence Store -> instant text)
    3. Carrier cleanup  — cls_backend/dllm.py (sparse, guarded)
    4. Wiring/contract  — `instant_answer(...)` is the contract the UI consumes

The backend does no LLM work by default: it encodes, checks the CAG cache, searches the
Evidence Store, and assembles a grounded extractive answer. All functions are pure /
dependency-injected (collection, cache, encoder are passed in) so they are testable without
app state.

When `debate_enabled=True`, an optional second self-debate loop uses the configured
inference carrier to audit retrieved chunks for relevance before answer generation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Iterable

from cls_backend.spectrum import classify_query
from cls_config import DEFAULT_DLLM_API_KEY, DEFAULT_DLLM_API_URL, DEFAULT_DLLM_MODEL

EMBED_DIM = 768


class OllamaEmbedder:
    """Retrieval Encoder: Ollama-based embedder using nomic-embed-text."""

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10.0) as response:
                data = json.loads(response.read().decode("utf-8"))
                embedding = data.get("embedding")
                if isinstance(embedding, list) and len(embedding) == EMBED_DIM:
                    return embedding
        except Exception:
            pass
        # Fallback to zero vector to avoid crashing the pipeline; callers should surface
        # an offline/embedder warning in the UI when all hits have zero similarity.
        return [0.0] * EMBED_DIM


def collection_count(collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0


def corpus_signature(collection) -> str:
    """Signature of the Evidence Store so cached evidence is invalidated when the corpus
    changes. Hash of the sorted distinct source hashes (prototype scale)."""
    try:
        existing = collection.get(include=["metadatas"])
    except Exception:
        return "empty"
    hashes = sorted(
        {(meta or {}).get("source_hash", "") for meta in existing.get("metadatas", [])}
    )
    if not hashes:
        return "empty"
    return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()[:16]


_STOP_WORDS = {
    "about", "after", "before", "could", "from", "have", "into", "manual", "procedure",
    "should", "tell", "that", "their", "there", "these", "this", "what", "when", "where",
    "which", "with", "would",
}


def lexical_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-/\.]*", text.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    }


_CHUNK_HEADER = re.compile(
    r"^Source:\s*(?P<source>.*?)\nSection:\s*(?P<section>.*?)\nPage:\s*(?P<page>.*?)\n\n",
    re.S,
)
_ADJACENT_SPLIT = re.compile(r"\n\n--- adjacent segment ---\n\n")


def _segment_metadata(segment: str, fallback: dict) -> tuple[str, dict]:
    """Return a chunk segment body plus metadata from its embedded chunk header."""
    meta = dict(fallback or {})
    match = _CHUNK_HEADER.match(segment)
    if not match:
        return segment, meta

    source = match.group("source").strip()
    section = match.group("section").strip()
    page = match.group("page").strip()
    if source:
        meta["source"] = source
    if section:
        meta["section"] = section
    if page:
        try:
            meta["page"] = int(page)
        except ValueError:
            meta["page"] = page
    return segment[match.end():], meta


def iter_document_segments(row: dict) -> Iterable[tuple[str, dict]]:
    """Yield each chunk body with its own metadata, preserving adjacent-page citations."""
    document = row.get("document", "")
    fallback = row.get("metadata", {}) or {}
    for segment in _ADJACENT_SPLIT.split(document):
        body, metadata = _segment_metadata(segment, fallback)
        if body.strip():
            yield body, metadata


def extract_sentences(text: str) -> list[str]:
    body = _CHUNK_HEADER.sub("", text, count=1)
    normalized = re.sub(r"\s+", " ", body)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalized) if s.strip()]


def build_extractive_answer(query: str, rows: list[dict], max_sentences: int = 5) -> list[str]:
    terms = lexical_terms(query)
    candidates: list[tuple[int, float, str, dict]] = []
    for row in rows[:5]:
        for segment, metadata in iter_document_segments(row):
            for sentence in extract_sentences(segment):
                overlap = len(terms & lexical_terms(sentence))
                if overlap:
                    candidates.append((overlap, row["score"], sentence, metadata))

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    chosen: list[str] = []
    seen: set[str] = set()
    for _, _, sentence, metadata in candidates:
        clean = sentence.strip()
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        source = metadata.get("source", "source")
        page = metadata.get("page", "?")
        chosen.append(f"{clean} [Source: {source}, page {page}]")
        if len(chosen) >= max_sentences:
            break
    return chosen


# Deterministic, instant text cleanup — DocuSearch-style "clean parsed text", no LLM.
_HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\s+([a-z]{2,})")  # "undula- tor" -> "undulator"
_CITE_SUFFIX = re.compile(r"\s*(\[Source:[^\]]*\])\s*$")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")


def clean_sentence(sentence: str) -> str:
    """Repair common PDF-extraction artifacts deterministically, preserving the citation."""
    match = _CITE_SUFFIX.search(sentence)
    citation = match.group(1) if match else ""
    body = _CITE_SUFFIX.sub("", sentence).strip()
    body = _HYPHEN_BREAK.sub(r"\1\2", body)
    body = _SPACE_BEFORE_PUNCT.sub(r"\1", body)
    body = re.sub(r"\s{2,}", " ", body).strip()
    return f"{body} {citation}".strip() if citation else body


def clean_sentences(sentences: list[str]) -> list[str]:
    """Clean each sentence and drop case-insensitive duplicates, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for sentence in sentences:
        cleaned = clean_sentence(sentence)
        key = _CITE_SUFFIX.sub("", cleaned).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(cleaned)
    return out


def merge_adjacent_segments(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    by_source: dict[str, dict[int, dict]] = {}
    for row in rows:
        meta = row["metadata"]
        by_source.setdefault(meta.get("source_hash", ""), {})[int(meta.get("chunk_index", -1))] = row

    merged: list[dict] = []
    used: set[tuple[str, int]] = set()
    for row in rows[:4]:
        meta = row["metadata"]
        source_hash = meta.get("source_hash", "")
        index = int(meta.get("chunk_index", -1))
        if (source_hash, index) in used:
            continue
        neighbors = []
        for candidate_index in range(index - 1, index + 2):
            neighbor = by_source.get(source_hash, {}).get(candidate_index)
            if neighbor:
                neighbors.append(neighbor)
                used.add((source_hash, candidate_index))
        merged.append(
            {
                "metadata": meta,
                "score": max(n["score"] for n in neighbors),
                "document": "\n\n--- adjacent segment ---\n\n".join(n["document"] for n in neighbors),
                "parts": len(neighbors),
            }
        )
    return merged


def retrieve(
    collection,
    embedder,
    query: str,
    n_results: int = 8,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Vector search against the Evidence Store. Returns scored rows (empty if unindexed)."""
    if collection_count(collection) == 0:
        return []
    query_params: dict[str, Any] = {
        "query_embeddings": embedder.embed([query]),
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if metadata_filter:
        query_params["where"] = metadata_filter
    result = collection.query(**query_params)
    rows: list[dict] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        rows.append({"document": document, "metadata": metadata or {}, "distance": distance, "score": score})
    return rows


def _dllm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if DEFAULT_DLLM_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_DLLM_API_KEY}"
    return headers


def self_debate_refine(query: str, rows: list[dict], model: str = DEFAULT_DLLM_MODEL) -> list[dict]:
    """Second relevance-audit loop: uses the configured inference carrier to prune
    irrelevant chunks before synthesis.
    """
    if not rows:
        return []
    if not DEFAULT_DLLM_API_URL:
        return rows

    context_str = ""
    for index, row in enumerate(rows):
        document = row.get("document", "")[:500]
        context_str += f"ID {index}: {document}\n\n"

    prompt = (
        "You are a retrieval auditor for a scientific document system. Keep only chunks "
        "that directly support answering the query; discard off-topic or low-signal chunks.\n"
        f"Query: {query}\n\n"
        f"Chunks:\n{context_str}\n"
        "Respond with a JSON list of the IDs that are HIGHLY RELEVANT and should be kept. "
        "Example: [0, 2, 5]. Exclude anything that does not directly help answer the query. "
        "Output ONLY the JSON list."
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_DLLM_API_URL}/chat/completions",
        data=payload,
        headers=_dllm_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        keep_ids = json.loads(content)
        if isinstance(keep_ids, list):
            refined = [rows[i] for i in keep_ids if isinstance(i, int) and 0 <= i < len(rows)]
            return refined if refined else rows[:1]
    except Exception:
        pass
    return rows


def instant_answer(
    query: str,
    *,
    collection,
    cache,
    embedder,
    top_k: int = 8,
    cache_enabled: bool = True,
    metadata_filter: dict | None = None,
    debate_enabled: bool = False,
) -> dict:
    """The contract the UI consumes: query -> grounded instant answer + provenance.

    Checks the CAG Layer first (evidence reuse), falls back to an Evidence Store search,
    then assembles the extractive answer. No LLM is involved unless debate_enabled=True.
    """
    category = classify_query(query)
    sig = corpus_signature(collection)

    hit = cache.lookup(query, sig) if cache_enabled else None
    if hit:
        rows, from_cache, similarity = hit["rows"], True, hit["similarity"]
    else:
        rows = retrieve(collection, embedder, query, n_results=top_k, metadata_filter=metadata_filter)
        from_cache, similarity = False, None
        if debate_enabled and rows:
            rows = self_debate_refine(query, rows)
        if cache_enabled and rows:
            cache.store(query, rows, sig, category=category, top_k=top_k)

    merged = merge_adjacent_segments(rows)
    # Instant, deterministic clean parse — no LLM augmentation.
    answer = clean_sentences(build_extractive_answer(query, merged or rows)) if rows else []
    return {
        "rows": rows,
        "merged": merged,
        "answer": answer,
        "category": category,
        "from_cache": from_cache,
        "similarity": similarity,
    }
