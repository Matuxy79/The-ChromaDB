"""RAG+CAG instant backend — the millisecond retrieval system.

This is System 2 of the four-layer architecture, kept free of Streamlit so the UI layer
(`app.py`) can call it without the presentation and the retrieval concerns bleeding into
each other:

    1. UI/UX            — app.py (Streamlit)
    2. RAG+CAG backend  — THIS module (Retrieval Encoder + CAG + Evidence Store -> instant text)
    3. Carrier cleanup  — examples/cls_dllm.py (sparse, guarded)
    4. Wiring/contract  — `instant_answer(...)` is the contract the UI consumes

The backend does no LLM work: it encodes, checks the CAG cache, searches the Evidence Store,
and assembles a grounded extractive answer. All functions are pure / dependency-injected
(collection, cache, encoder are passed in) so they are testable without app state.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable

from cls_backend.spectrum import classify_query

EMBED_DIM = 512


class HashEmbedder:
    """Retrieval Encoder: small deterministic embedder for offline manual retrieval."""

    def __init__(self, dimensions: int = EMBED_DIM) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-/\.]*", text.lower())
        features: Counter[str] = Counter(tokens)
        features.update(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))

        vector = [0.0] * self.dimensions
        for feature, count in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


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


def retrieve(collection, embedder, query: str, n_results: int = 8) -> list[dict]:
    """Vector search against the Evidence Store. Returns scored rows (empty if unindexed)."""
    if collection_count(collection) == 0:
        return []
    result = collection.query(
        query_embeddings=embedder.embed([query]),
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    rows: list[dict] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        rows.append({"document": document, "metadata": metadata or {}, "distance": distance, "score": score})
    return rows


def instant_answer(
    query: str,
    *,
    collection,
    cache,
    embedder,
    top_k: int = 8,
    cache_enabled: bool = True,
) -> dict:
    """The contract the UI consumes: query -> grounded instant answer + provenance.

    Checks the CAG Layer first (evidence reuse), falls back to an Evidence Store search,
    then assembles the extractive answer. No LLM is involved.
    """
    category = classify_query(query)
    sig = corpus_signature(collection)

    hit = cache.lookup(query, sig) if cache_enabled else None
    if hit:
        rows, from_cache, similarity = hit["rows"], True, hit["similarity"]
    else:
        rows = retrieve(collection, embedder, query, n_results=top_k)
        from_cache, similarity = False, None
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
