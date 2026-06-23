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
from threading import RLock
from typing import Any, Iterable

from cls_backend.spectrum import classify_query
from cls_config import (
    DEFAULT_DLLM_API_KEY,
    DEFAULT_DLLM_API_URL,
    DEFAULT_DLLM_MODEL,
    RETRIEVAL_ONLY,
)

EMBED_DIM = 384  # all-MiniLM-L6-v2
MIN_EXTRACTIVE_SCORE = 0.38
MAX_ANSWER_ROWS = 16
VECTOR_STORE_RECOVERY_MESSAGE = (
    "The vector index could not be read. It may be from an older embedding model, "
    "or a previous run may have left a partial HNSW segment on disk. Reset the "
    "Chroma index and re-index your documents."
)

_LEXICAL_INDEX_LOCK = RLock()
_LEXICAL_INDEX_CACHE: dict[tuple[int, int], list[dict[str, Any]]] = {}
_LEXICAL_VOCAB_CACHE: dict[tuple[int, int], frozenset[str]] = {}

# Partial-match reach for keyword retrieval. A query fragment of at least this length
# matches any indexed term it is a substring of (and vice versa), so "sapi" finds
# "sapiens", "chro" finds "chromosome", and arbitrary fragment combinations still
# retrieve. Shorter fragments fall back to exact match so they don't pull in almost
# everything as the corpus grows.
PARTIAL_MATCH_MIN_LEN = 3


class EmbeddingUnavailableError(RuntimeError):
    """Raised when the configured retrieval encoder cannot produce an embedding."""


class VectorStoreUnavailableError(RuntimeError):
    """Raised when Chroma's persisted vector/index files cannot be read safely."""


def _vector_store_error(exc: Exception) -> VectorStoreUnavailableError:
    return VectorStoreUnavailableError(VECTOR_STORE_RECOVERY_MESSAGE)


class SentenceTransformerEmbedder:
    """Semantic retrieval encoder: sentence-transformers, runs fully offline on CPU.

    Downloads all-MiniLM-L6-v2 (~80 MB) on first use and caches in
    ~/.cache/huggingface/hub/. After that, load is local and instant.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingUnavailableError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        vecs = self._load().encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]


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
    terms: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-/\.]*", text.lower()):
        if len(token) <= 2 or token in _STOP_WORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        terms.add(token)
    return terms


def clear_lexical_index_cache() -> None:
    """Drop the in-memory keyword snapshot after corpus writes or reset."""
    with _LEXICAL_INDEX_LOCK:
        _LEXICAL_INDEX_CACHE.clear()
        _LEXICAL_VOCAB_CACHE.clear()


def _term_matches(query_term: str, target_term: str) -> bool:
    """Exact match, or — for query fragments of at least PARTIAL_MATCH_MIN_LEN — the
    query fragment is contained in the indexed term ("sapi" -> "sapiens").

    The match is one-directional on purpose: we expand the *typed* fragment outward to
    longer indexed terms, but we never treat a short indexed word as a match for a long
    typed word. Otherwise a precise query like "metabolism" would dredge up every chunk
    containing "meta", "bol", "ism", etc."""
    if query_term == target_term:
        return True
    if len(query_term) < PARTIAL_MATCH_MIN_LEN:
        return False
    return query_term in target_term


def expand_query_terms(query_terms: set[str], vocabulary: frozenset[str]) -> dict[str, set[str]]:
    """Map each query fragment to the indexed vocabulary terms it matches (incl. partials).

    The vocabulary is scanned once per query rather than per chunk, so partial matching
    stays cheap even as the Evidence Store grows: the per-chunk step below is then a plain
    set intersection against the union of matched terms.
    """
    expanded: dict[str, set[str]] = {}
    for term in query_terms:
        if len(term) < PARTIAL_MATCH_MIN_LEN:
            expanded[term] = {term} if term in vocabulary else set()
        else:
            expanded[term] = {vocab_term for vocab_term in vocabulary if term in vocab_term}
    return expanded


def partial_overlap(query_terms: set[str], target_terms: set[str]) -> int:
    """Count target terms matching any query fragment (exact or partial). Used at the
    sentence level, where the target set is small enough for a direct scan."""
    return sum(
        1 for target in target_terms
        if any(_term_matches(term, target) for term in query_terms)
    )


def _metadata_matches(metadata: dict, metadata_filter: dict | None) -> bool:
    """Small in-memory subset of Chroma where matching used by this project."""
    if not metadata_filter:
        return True
    for key, expected in metadata_filter.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
            unsupported = set(expected) - {"$eq", "$in"}
            if unsupported:
                return False
        elif actual != expected:
            return False
    return True


def warm_lexical_index(collection) -> int:
    """Build/reuse the full in-memory keyword snapshot and return its row count."""
    return len(_lexical_snapshot(collection, collection_count(collection)))


def _lexical_snapshot(collection, count: int | None = None) -> list[dict[str, Any]]:
    """Fetch documents once, cache token sets in RAM, then serve keyword queries in ms."""
    if count is None:
        count = collection_count(collection)
    if count == 0:
        return []

    key = (id(collection), count)
    with _LEXICAL_INDEX_LOCK:
        cached = _LEXICAL_INDEX_CACHE.get(key)
        if cached is not None:
            return cached

    try:
        result = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        raise _vector_store_error(exc) from exc

    documents = result.get("documents", []) or []
    metadatas = result.get("metadatas", []) or []
    snapshot = [
        {
            "document": document or "",
            "metadata": metadata or {},
            "terms": lexical_terms(document or ""),
        }
        for document, metadata in zip(documents, metadatas)
    ]
    with _LEXICAL_INDEX_LOCK:
        stale_keys = [cache_key for cache_key in _LEXICAL_INDEX_CACHE if cache_key[0] == id(collection)]
        for cache_key in stale_keys:
            _LEXICAL_INDEX_CACHE.pop(cache_key, None)
        _LEXICAL_INDEX_CACHE[key] = snapshot
    return snapshot


def _lexical_vocabulary(collection, count: int | None = None) -> frozenset[str]:
    """Union of every indexed term — the keyword vocabulary used to expand partial
    queries. Built once from the snapshot and cached alongside it (same count key, so it
    is rebuilt automatically whenever the corpus changes size)."""
    if count is None:
        count = collection_count(collection)
    if count == 0:
        return frozenset()

    key = (id(collection), count)
    with _LEXICAL_INDEX_LOCK:
        cached = _LEXICAL_VOCAB_CACHE.get(key)
        if cached is not None:
            return cached

    vocabulary: set[str] = set()
    for item in _lexical_snapshot(collection, count):
        vocabulary.update(item["terms"])
    frozen = frozenset(vocabulary)

    with _LEXICAL_INDEX_LOCK:
        stale_keys = [cache_key for cache_key in _LEXICAL_VOCAB_CACHE if cache_key[0] == id(collection)]
        for cache_key in stale_keys:
            _LEXICAL_VOCAB_CACHE.pop(cache_key, None)
        _LEXICAL_VOCAB_CACHE[key] = frozen
    return frozen


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


def _sentence_candidates(query: str, rows: list[dict]) -> list[tuple[int, float, str, dict]]:
    terms = lexical_terms(query)
    candidates: list[tuple[int, float, str, dict]] = []
    for row in rows[:MAX_ANSWER_ROWS]:
        for segment, metadata in iter_document_segments(row):
            for sentence in extract_sentences(segment):
                overlap = partial_overlap(terms, lexical_terms(sentence))
                candidates.append((overlap, row["score"], sentence, metadata))
    return candidates


def _format_extractive_sentences(candidates: list[tuple[float, float, str, dict]], max_sentences: int) -> list[str]:
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


def build_extractive_answer(query: str, rows: list[dict], max_sentences: int = 5) -> list[str]:
    candidates = [
        candidate for candidate in _sentence_candidates(query, rows)
        if candidate[0] > 0
    ]

    if not candidates:
        return []

    candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    return _format_extractive_sentences(candidates, max_sentences)


def build_semantic_extractive_answer(
    query: str,
    rows: list[dict],
    embedder,
    max_sentences: int = 5,
) -> list[str]:
    """Rank candidate sentences with the same local encoder used for retrieval."""
    candidates = [
        candidate for candidate in _sentence_candidates(query, rows)
        if len(candidate[2]) >= 30
    ]
    if not candidates:
        return []

    # Keep sentence reranking bounded so the no-LLM answer lane stays responsive.
    candidates = candidates[:160]
    try:
        vectors = embedder.embed([query] + [sentence for _, _, sentence, _ in candidates])
    except EmbeddingUnavailableError:
        return build_extractive_answer(query, rows, max_sentences=max_sentences)

    query_vec = vectors[0]
    ranked: list[tuple[float, float, str, dict]] = []
    for candidate, sentence_vec in zip(candidates, vectors[1:]):
        overlap, row_score, sentence, metadata = candidate
        semantic = sum(a * b for a, b in zip(query_vec, sentence_vec))
        score = semantic + (0.04 * overlap) + (0.08 * row_score)
        ranked.append((score, row_score, sentence, metadata))

    ranked.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    return _format_extractive_sentences(ranked, max_sentences)


# Deterministic, instant text cleanup — DocuSearch-style "clean parsed text", no LLM.
_HYPHEN_BREAK = re.compile(r"([A-Za-z]{2,})-\s+([a-z]{2,})")  # "synchro- tron" -> "synchrotron"
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


# ------------------------------------------------------------------------------- #
# Cross-reference expansion — post-retrieval, pre-LLM enrichment.
#
# Problem: a retrieved chunk may say "see Section 4.2" or "as shown in Figure 3".
# The embedding for that sentence is about the *topic*, not the content of the
# reference target — so the referenced content is never in the top-k. The LLM
# gets a dangling pointer it cannot resolve.
#
# Fix: after the top-k rows are assembled, (1) fetch the ±1 adjacent chunks from
# the same source (catches "see above/below" and gives the LLM wider context),
# and (2) regex-extract explicit cross-refs and do a secondary lexical lookup
# restricted to the same document to pull in the referenced content.
#
# Expansion rows are tagged {"expanded": True, "score": 0.0} so they are
# invisible to extractive-bullet scoring (MIN_EXTRACTIVE_SCORE gate) but travel
# with `rows` into the LLM context window.
# ------------------------------------------------------------------------------- #

_XREF_RE = re.compile(
    r"\b(?:"
    r"(?:Section|Sec\.?|§)\s*\d+(?:\.\d+)*"
    r"|(?:Figure|Fig\.?)\s*\d+(?:\.\d+)*"
    r"|(?:Table)\s*\d+(?:\.\d+)*"
    r"|(?:Appendix|App\.?)\s*[A-Z]"
    r"|(?:Chapter|Ch\.?)\s*\d+"
    r")",
    re.IGNORECASE,
)


def extract_cross_refs(text: str) -> list[str]:
    """Return unique cross-reference strings found in *text* (Section X.Y, Figure N…)."""
    seen: set[str] = set()
    refs: list[str] = []
    for match in _XREF_RE.finditer(text):
        ref = re.sub(r"\s+", " ", match.group(0)).strip()
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return refs


def expand_cross_refs(
    collection,
    rows: list[dict],
    embedder,
    *,
    max_ref_expansions: int = 6,
) -> list[dict]:
    """Enrich retrieved rows with adjacent chunks and resolved cross-references.

    Step 1 — adjacent chunks: for each of the top-8 rows fetch chunk_index ±1
    from the same source (cheap Chroma ``get`` by ID).

    Step 2 — explicit ref resolution: regex-extract cross-refs (Section X.Y,
    Figure N…) from each row's text and do a secondary *lexical* lookup inside
    the same document to pull the referenced content.

    Returns the original rows list with expansion rows appended.  Expansion rows
    carry ``{"expanded": True, "score": 0.0}`` so they are excluded from
    extractive-bullet scoring but included in the LLM context passed downstream.
    """
    if not rows:
        return rows

    # Track already-present chunk IDs so we don't duplicate.
    def _cid(meta: dict) -> str:
        return f"{meta.get('source_hash', '')}:{meta.get('chunk_index', -999999):05d}"

    existing: set[str] = {_cid(row.get("metadata", {})) for row in rows}
    expansion: list[dict] = []

    # ------------------------------------------------------------------ #
    # Step 1: adjacent chunk fetch (±1 neighbor within same source)       #
    # ------------------------------------------------------------------ #
    adj_ids: list[str] = []
    for row in rows[:8]:
        meta = row.get("metadata", {})
        source_hash = meta.get("source_hash", "")
        chunk_index = meta.get("chunk_index")
        if not source_hash or chunk_index is None:
            continue
        for offset in (-1, 1):
            cid = f"{source_hash}:{chunk_index + offset:05d}"
            if cid not in existing:
                adj_ids.append(cid)
                existing.add(cid)  # reserve so duplicates from two rows don't double-fetch

    if adj_ids:
        try:
            fetched = collection.get(ids=adj_ids, include=["documents", "metadatas"])
            for doc, meta in zip(
                fetched.get("documents") or [],
                fetched.get("metadatas") or [],
            ):
                if doc:
                    expansion.append({
                        "document": doc,
                        "metadata": meta or {},
                        "score": 0.0,
                        "distance": 1.0,
                        "expanded": True,
                        "expand_reason": "adjacent",
                    })
        except Exception:
            pass  # non-existent IDs (first/last chunk) silently skipped

    # ------------------------------------------------------------------ #
    # Step 2: explicit cross-ref resolution (lexical, same-source gate)   #
    # ------------------------------------------------------------------ #
    ref_count = 0
    for row in rows[:6]:
        if ref_count >= max_ref_expansions:
            break
        meta = row.get("metadata", {})
        source_hash = meta.get("source_hash", "")
        refs = extract_cross_refs(row.get("document", ""))
        for ref in refs[:3]:
            if ref_count >= max_ref_expansions:
                break
            mf = {"source_hash": source_hash} if source_hash else None
            try:
                ref_rows = lexical_retrieve(collection, ref, n_results=2, metadata_filter=mf)
            except Exception:
                continue
            for rrow in ref_rows:
                cid = _cid(rrow.get("metadata", {}))
                if cid not in existing:
                    existing.add(cid)
                    rrow = dict(rrow)  # don't mutate the original
                    rrow["expanded"] = True
                    rrow["expand_reason"] = f"xref:{ref}"
                    rrow["score"] = 0.0  # invisible to extractive scoring
                    expansion.append(rrow)
                    ref_count += 1

    return rows + expansion


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


def _row_key(row: dict) -> str:
    meta = row.get("metadata", {})
    return f"{meta.get('source_hash', '')}:{meta.get('chunk_index', '')}"


def _merge_hybrid(semantic: list[dict], lexical: list[dict], top_k: int) -> list[dict]:
    """Blend semantic search with keyword hits so exact user terms can rescue recall."""
    seen: dict[str, dict] = {}
    for row in semantic:
        key = _row_key(row)
        seen[key] = dict(row)
        seen[key]["score"] = row["score"]
        seen[key]["semantic_score"] = row["score"]
        seen[key]["lexical_score"] = 0.0

    for row in lexical:
        key = _row_key(row)
        lexical_score = row["score"]
        if key in seen:
            semantic_score = seen[key].get("semantic_score", seen[key]["score"])
            blended = min(1.0, (0.72 * semantic_score) + (0.28 * lexical_score) + 0.08)
            seen[key]["score"] = max(seen[key]["score"], blended)
            seen[key]["lexical_score"] = lexical_score
        else:
            supplement = dict(row)
            supplement["semantic_score"] = 0.0
            supplement["lexical_score"] = lexical_score
            supplement["score"] = min(0.92, 0.85 * lexical_score)
            seen[key] = supplement

    ranked = sorted(seen.values(), key=lambda r: (r["score"], r.get("lexical_score", 0.0)), reverse=True)
    return ranked[:top_k]


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
    try:
        result = collection.query(**query_params)
    except Exception as exc:
        # A half-written / version-mismatched HNSW segment ("Nothing found on disk") or any
        # other store-level read failure must not crash the app. Surface an actionable error
        # the UI already renders cleanly; the fix is always Reset Chroma index + re-index.
        raise _vector_store_error(exc) from exc
    rows: list[dict] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        rows.append({"document": document, "metadata": metadata or {}, "distance": distance, "score": score})
    return rows


def lexical_retrieve(
    collection,
    query: str,
    n_results: int = 8,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Deterministic keyword retrieval backed by a cached in-memory lexical snapshot.

    Query fragments are expanded against the indexed vocabulary first, so partial words
    ("sapi" -> "sapiens") and arbitrary fragment combinations still retrieve evidence.
    """
    count = collection_count(collection)
    if count == 0:
        return []
    query_terms = lexical_terms(query)
    if not query_terms:
        return []

    vocabulary = _lexical_vocabulary(collection, count)
    expanded = expand_query_terms(query_terms, vocabulary)
    matchable: set[str] = set().union(*expanded.values()) if expanded else set()
    if not matchable:
        return []

    ranked: list[tuple[int, float, dict]] = []
    for item in _lexical_snapshot(collection, count):
        metadata = item["metadata"]
        if not _metadata_matches(metadata, metadata_filter):
            continue
        document_terms = item["terms"]
        matched = document_terms & matchable
        if not matched:
            continue
        covered = sum(1 for hits in expanded.values() if document_terms & hits)
        coverage = covered / len(query_terms)
        specificity = len(matched) / max(1, len(document_terms) ** 0.5)
        score = max(0.0, min(1.0, coverage + min(0.2, specificity / 4)))
        row = {
            "document": item["document"],
            "metadata": metadata or {},
            "distance": 1.0 - score,
            "score": score,
        }
        ranked.append((len(matched), score, row))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [row for _, _, row in ranked[:n_results]]


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
    if RETRIEVAL_ONLY:
        return rows
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
    top_k: int = 16,
    cache_enabled: bool = True,
    metadata_filter: dict | None = None,
    debate_enabled: bool = False,
    keyword_only: bool = False,
) -> dict:
    """The contract the UI consumes: query -> grounded instant answer + provenance.

    Checks the CAG Layer first (evidence reuse), falls back to an Evidence Store search,
    then assembles the extractive answer. No LLM is involved unless debate_enabled=True.
    """
    category = classify_query(query)
    sig = "keyword_only" if keyword_only else corpus_signature(collection)

    retrieval_mode = "keyword_only" if keyword_only else "hybrid"
    try:
        hit = cache.lookup(query, sig) if cache_enabled and not keyword_only else None
    except EmbeddingUnavailableError:
        hit = None
        retrieval_mode = "lexical_fallback"
    if hit:
        rows, from_cache, similarity = hit["rows"], True, hit["similarity"]
    else:
        if retrieval_mode in {"keyword_only", "lexical_fallback"}:
            rows = lexical_retrieve(collection, query, n_results=top_k, metadata_filter=metadata_filter)
        else:
            try:
                semantic_rows = retrieve(
                    collection, embedder, query,
                    n_results=top_k, metadata_filter=metadata_filter,
                )
                # Always run lexical in parallel as a keyword safety net; merge results.
                lexical_rows = lexical_retrieve(
                    collection, query,
                    n_results=max(4, top_k // 2), metadata_filter=metadata_filter,
                )
                rows = _merge_hybrid(semantic_rows, lexical_rows, top_k)
            except EmbeddingUnavailableError:
                retrieval_mode = "lexical_fallback"
                rows = lexical_retrieve(collection, query, n_results=top_k, metadata_filter=metadata_filter)
        from_cache, similarity = False, None
        if debate_enabled and rows:
            rows = self_debate_refine(query, rows)
        if cache_enabled and rows and retrieval_mode == "hybrid":
            try:
                cache.store(query, rows, sig, category=category, top_k=top_k)
            except EmbeddingUnavailableError:
                pass

    # Reference expansion: pull adjacent chunks and resolve explicit cross-refs
    # (Section X.Y, Figure N…) so the LLM context window contains the referenced
    # content. Expansion rows are invisible to extractive-bullet scoring.
    if rows and retrieval_mode == "hybrid":
        try:
            rows = expand_cross_refs(collection, rows, embedder)
        except Exception:
            pass  # expansion is best-effort; never block the main answer

    merged = merge_adjacent_segments(rows)
    if retrieval_mode == "hybrid":
        answer_rows = [
            row for row in rows
            if row.get("semantic_score", row.get("score", 0.0)) >= MIN_EXTRACTIVE_SCORE
        ]
    else:
        answer_rows = [row for row in rows if row.get("score", 0.0) >= MIN_EXTRACTIVE_SCORE]
    # Instant, deterministic clean parse — no LLM augmentation.
    if answer_rows and retrieval_mode == "hybrid":
        answer = clean_sentences(build_semantic_extractive_answer(query, answer_rows, embedder))
    else:
        answer = clean_sentences(build_extractive_answer(query, answer_rows)) if answer_rows else []
    return {
        "rows": rows,
        "merged": merged,
        "answer": answer,
        "category": category,
        "from_cache": from_cache,
        "similarity": similarity,
        "retrieval_mode": retrieval_mode,
    }
