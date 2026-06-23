from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Generator
import urllib.error
import urllib.request
from urllib.parse import urlparse

import chromadb

from cls_backend.readers import load_document
from cls_config import (
    CACHE_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    COLLECTION_NAME,
    DEFAULT_DLLM_API_KEY,
    DEFAULT_DLLM_API_URL,
    DEFAULT_DLLM_MODEL,
    KEYWORD_ONLY_RETRIEVAL,
    RETRIEVAL_ONLY,
    DEFAULT_PARROT_MODEL,
    DEFAULT_PARROT_URL,
)
from cls_backend.cag_cache import SemanticEvidenceCache
from cls_backend.dllm import (
    ANSWER_SYSTEM,
    ASSIST_SYSTEM,
    PARROT_SYSTEM,
    answer_user,
    assist_user,
    parrot_grounded,
    parrot_user,
)
from cls_backend.pipeline import (
    EMBED_DIM,
    VECTOR_STORE_RECOVERY_MESSAGE,
    SentenceTransformerEmbedder,
    clear_lexical_index_cache,
    collection_count,
    expand_cross_refs,
    extract_cross_refs,
    instant_answer,
    retrieve,
    warm_lexical_index,
)


# Embed + write this many chunks at a time during ingest. Bounds peak memory so a single
# large document can't OOM the process on small-RAM hosts.
INGEST_BATCH_SIZE = 256


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict
    chunk_id: str


_RESOURCE_LOCK = RLock()
_embedder: SentenceTransformerEmbedder | None = None
_chroma_client: Any | None = None
_collection: chromadb.Collection | None = None
_cache_collection: chromadb.Collection | None = None
_cache: SemanticEvidenceCache | None = None


def get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    with _RESOURCE_LOCK:
        if _embedder is None:
            _embedder = SentenceTransformerEmbedder()
        return _embedder


def get_chroma_client() -> Any:
    global _chroma_client
    with _RESOURCE_LOCK:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _chroma_client


def get_collection() -> chromadb.Collection:
    global _collection
    with _RESOURCE_LOCK:
        if _collection is None:
            _collection = get_chroma_client().get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={
                    "description": "CLS synchrotron research document retrieval",
                    "embedding": f"all-MiniLM-L6-v2_{EMBED_DIM}d",
                    "hnsw:space": "cosine",
                },
            )
        return _collection


def get_cache_collection() -> chromadb.Collection:
    global _cache_collection
    with _RESOURCE_LOCK:
        if _cache_collection is None:
            _cache_collection = get_chroma_client().get_or_create_collection(
                name=CACHE_COLLECTION_NAME,
                metadata={
                    "description": "CLS CAG semantic evidence cache",
                    "embedding": f"all-MiniLM-L6-v2_{EMBED_DIM}d",
                    "hnsw:space": "cosine",
                },
            )
        return _cache_collection


def get_cache() -> SemanticEvidenceCache:
    global _cache
    with _RESOURCE_LOCK:
        if _cache is None:
            _cache = SemanticEvidenceCache(get_cache_collection(), get_embedder())
        return _cache


def file_signature(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()[:16]


def uploaded_signature(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]





def detect_section(text: str, fallback: str) -> str:
    for line in text.splitlines()[:12]:
        clean = re.sub(r"\s+", " ", line).strip()
        if re.match(r"^\d+(\.\d+)*\s+[A-ZA-Za-z]", clean):
            return clean[:120]
        if 8 <= len(clean) <= 90 and clean[:1].isupper() and not clean.endswith("."):
            return clean[:120]
    return fallback


def split_page_text(
    text: str,
    page_number: int,
    source_name: str,
    source_hash: str,
    starting_index: int,
    extra_metadata: dict | None = None,
) -> list[Chunk]:
    section = detect_section(text, f"Page {page_number}")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[Chunk] = []
    buffer = ""
    chunk_index = starting_index

    def flush() -> None:
        nonlocal buffer, chunk_index
        clean = re.sub(r"\n{3,}", "\n\n", buffer).strip()
        if not clean:
            return
        context_text = f"Source: {source_name}\nSection: {section}\nPage: {page_number}\n\n{clean}"
        chunk_id = f"{source_hash}:{chunk_index:05d}"
        meta = {
            "source": source_name,
            "source_hash": source_hash,
            "page": page_number,
            "section": section,
            "chunk_index": chunk_index,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        chunks.append(
            Chunk(
                text=context_text,
                metadata=meta,
                chunk_id=chunk_id,
            )
        )
        overlap = clean[-CHUNK_OVERLAP_CHARS:] if len(clean) > CHUNK_OVERLAP_CHARS else clean
        buffer = overlap
        chunk_index += 1

    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
        if len(candidate) > CHUNK_TARGET_CHARS and buffer:
            flush()
            buffer = paragraph
        else:
            buffer = candidate

        while len(buffer) > CHUNK_TARGET_CHARS * 1.4:
            cut = buffer.rfind(". ", 0, CHUNK_TARGET_CHARS)
            if cut < CHUNK_TARGET_CHARS // 2:
                cut = CHUNK_TARGET_CHARS
            head, buffer = buffer[: cut + 1], buffer[cut + 1 :]
            saved_buffer = buffer
            buffer = head
            flush()
            buffer = f"{head[-CHUNK_OVERLAP_CHARS:]}\n\n{saved_buffer}".strip()

    flush()
    return chunks


def build_chunks(
    path: Path,
    source_hash: str,
    extra_metadata: dict | None = None,
    source_name: str | None = None,
) -> list[Chunk]:
    pages = load_document(path)
    chunks: list[Chunk] = []
    next_index = 0
    for page_number, text in pages:
        page_chunks = split_page_text(
            text=text,
            page_number=page_number,
            source_name=source_name or path.name,
            source_hash=source_hash,
            starting_index=next_index,
            extra_metadata=extra_metadata,
        )
        chunks.extend(page_chunks)
        next_index += len(page_chunks)
    return chunks


def source_is_indexed(collection: chromadb.Collection, source_hash: str) -> bool:
    try:
        result = collection.get(where={"source_hash": source_hash}, limit=1)
    except Exception as exc:
        raise RuntimeError(VECTOR_STORE_RECOVERY_MESSAGE) from exc
    return bool(result.get("ids"))


def ingest_path(
    path: Path,
    source_hash: str,
    force: bool = False,
    extra_metadata: dict | None = None,
    source_name: str | None = None,
    embedder: Any | None = None,
) -> tuple[int, str]:
    collection = get_collection()
    existing_ids: list[str] = []
    if source_is_indexed(collection, source_hash):
        if not force:
            return 0, "already indexed"
        try:
            existing = collection.get(where={"source_hash": source_hash})
        except Exception as exc:
            raise RuntimeError(VECTOR_STORE_RECOVERY_MESSAGE) from exc
        existing_ids = existing.get("ids") or []

    chunks = build_chunks(
        path,
        source_hash,
        extra_metadata=extra_metadata,
        source_name=source_name,
    )
    if not chunks:
        return 0, "no readable text found"

    active_embedder = embedder or get_embedder()
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    write = collection.upsert if existing_ids else collection.add
    # Embed and write in batches so a single large file never materialises all of its
    # chunks + vectors in memory at once (that pattern OOM-kills on small-RAM hosts).
    try:
        for start in range(0, len(chunks), INGEST_BATCH_SIZE):
            batch = chunks[start : start + INGEST_BATCH_SIZE]
            write(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[chunk.metadata for chunk in batch],
                embeddings=active_embedder.embed(chunk.text for chunk in batch),
            )
    except Exception as exc:
        raise RuntimeError(VECTOR_STORE_RECOVERY_MESSAGE) from exc
    stale_ids = list(set(existing_ids) - set(chunk_ids))
    if stale_ids:
        try:
            collection.delete(ids=stale_ids)
        except Exception as exc:
            raise RuntimeError(VECTOR_STORE_RECOVERY_MESSAGE) from exc
    clear_lexical_index_cache()
    return len(chunks), "indexed"


def reset_collection() -> None:
    """Factory-reset the vector store: drop *every* collection, not just the current one, so
    stale/old-version collections (e.g. earlier 768d/512d builds) can't pile up or shadow the
    active store. The next get_collection() recreates a clean, empty v2 collection to re-index.
    """
    global _collection, _cache_collection, _cache
    with _RESOURCE_LOCK:
        client = get_chroma_client()
        for collection in client.list_collections():
            try:
                client.delete_collection(collection.name)
            except Exception:
                pass
        _collection = None
        _cache_collection = None
        _cache = None
        clear_lexical_index_cache()


def warm_keyword_index() -> int:
    """Preload the keyword snapshot so first user search avoids a full Chroma document fetch."""
    return warm_lexical_index(get_collection())


def evidence_breakdown(collection: chromadb.Collection | None = None) -> list[dict]:
    try:
        existing = (collection or get_collection()).get(include=["metadatas"])
    except Exception:
        return []
    by_source: dict[str, dict[str, Any]] = {}
    for meta in existing.get("metadatas", []) or []:
        meta = meta or {}
        source = meta.get("source", "unknown")
        entry = by_source.setdefault(source, {"source": source, "chunks": 0, "pages": set()})
        entry["chunks"] += 1
        page = meta.get("page")
        if isinstance(page, int):
            entry["pages"].add(page)
    rows: list[dict] = []
    for entry in by_source.values():
        pages = entry["pages"]
        rows.append(
            {
                "source": entry["source"],
                "chunks": entry["chunks"],
                "page_span": (min(pages), max(pages)) if pages else None,
            }
        )
    rows.sort(key=lambda r: r["chunks"], reverse=True)
    return rows


EVAL_CASES = [
    {
        "question": "What are the main CLS control room and floor coordinator contacts?",
        "keywords": ["control room", "floor coordinator", "3570", "3639"],
    },
    {
        "question": "What emergency numbers are listed for fire or medical help?",
        "keywords": ["911", "security", "emergency"],
    },
    {
        "question": "How do I request hutch access or beamline support?",
        "keywords": ["hutch", "access", "support", "procedure"],
    },
    {
        "question": "Where does the documentation describe sample mounting procedures?",
        "keywords": ["sample", "mount", "procedure"],
    },
]


def evaluate_retrieval() -> list[dict]:
    evaluations: list[dict] = []
    for case in EVAL_CASES:
        rows = retrieve(get_collection(), get_embedder(), case["question"], n_results=6)
        combined = " ".join(row["document"].lower() for row in rows[:3])
        hits = [keyword for keyword in case["keywords"] if keyword.lower() in combined]
        keyword_score = len(hits) / len(case["keywords"])
        top_score = rows[0]["score"] if rows else 0.0
        evaluations.append(
            {
                "question": case["question"],
                "keyword_score": keyword_score,
                "top_relevance": top_score,
                "hits": ", ".join(hits) if hits else "none",
                "status": "pass" if keyword_score >= 0.5 else "review",
            }
        )
    return evaluations


def ask_manual(
    query: str,
    *,
    top_k: int = 16,
    cache_enabled: bool = True,
    min_similarity: float = 0.80,
    metadata_filter: dict | None = None,
    debate_enabled: bool = False,
    keyword_only: bool | None = None,
) -> dict:
    if keyword_only is None:
        keyword_only = KEYWORD_ONLY_RETRIEVAL
    cache = get_cache()
    cache.distance_max = 1.0 - min_similarity
    return instant_answer(
        query,
        collection=get_collection(),
        cache=cache,
        embedder=get_embedder(),
        top_k=top_k,
        cache_enabled=cache_enabled,
        metadata_filter=metadata_filter,
        debate_enabled=debate_enabled,
        keyword_only=keyword_only,
    )


def answer_text(sentences: list[str]) -> str:
    if not sentences:
        return "No grounded sentence-level answer was found. Review the returned retrieval evidence."
    return "\n".join(f"- {sentence}" for sentence in sentences)


def dllm_online(timeout: float = 1.0) -> bool:
    return bool(dllm_status(timeout=timeout)["online"])


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _is_local_endpoint(url: str) -> bool:
    """Local carriers (Ollama) need no API key; remote carriers (OpenRouter, Groq) do."""
    return (urlparse(url).hostname or "").lower() in _LOCAL_HOSTS


def _carrier_name(url: str) -> str:
    """Friendly label for the configured generative carrier, derived from its host."""
    host = (urlparse(url).hostname or "").lower()
    if not host:
        return "Inference carrier"
    if "openrouter" in host:
        return "OpenRouter"
    if "groq" in host:
        return "Groq"
    if host in _LOCAL_HOSTS:
        return "Local (Ollama)"
    return host


def dllm_status(timeout: float = 1.0) -> dict:
    configured = bool(DEFAULT_DLLM_API_URL)
    local = configured and _is_local_endpoint(DEFAULT_DLLM_API_URL)
    has_key = bool(DEFAULT_DLLM_API_KEY)
    carrier = _carrier_name(DEFAULT_DLLM_API_URL)
    online = False

    if RETRIEVAL_ONLY:
        detail = (
            "Retrieval-only mode is active (CLS_RETRIEVAL_ONLY=1): LLM synthesis, "
            "cleanup, self-debate, parrot phrasing, and proxy calls are disabled."
        )
    elif not configured:
        detail = "Set CLS_DLLM_API_URL to enable the gpt-oss-120b answer."
    elif not local and not has_key:
        # Remote carrier wired in, but no key yet — the on-by-default toggle stays disabled
        # until the key is present so searches don't 401 on every call.
        detail = (
            f"Carrier set to {carrier} but no API key found. Paste your key into cls.env "
            "(CLS_DLLM_API_KEY), then relaunch."
        )
    else:
        try:
            request = urllib.request.Request(
                f"{DEFAULT_DLLM_API_URL}/models",
                headers=_dllm_headers(),
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                online = 200 <= response.status < 300
                detail = f"{carrier} carrier reachable." if online else f"{carrier} returned HTTP {response.status}."
        except urllib.error.HTTPError as exc:
            detail = f"{carrier} returned HTTP {exc.code}."
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            detail = f"{carrier} unreachable: {exc}"
    return {
        "provider": "api",
        "carrier": carrier,
        "base_url": DEFAULT_DLLM_API_URL,
        "model": DEFAULT_DLLM_MODEL,
        "configured": configured,
        "has_key": has_key,
        "local": local,
        "online": online,
        "disabled": RETRIEVAL_ONLY,
        "retrieval_only": RETRIEVAL_ONLY,
        "detail": detail,
    }


def _dllm_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if DEFAULT_DLLM_API_KEY:
        headers["Authorization"] = f"Bearer {DEFAULT_DLLM_API_KEY}"
    return headers


def call_dllm_api(
    messages: list[dict[str, str]],
    *,
    system: str | None = None,
    model: str = DEFAULT_DLLM_MODEL,
    timeout: float = 60.0,
) -> str:
    if RETRIEVAL_ONLY:
        raise RuntimeError("Retrieval-only mode is active; LLM generation is disabled.")
    if not DEFAULT_DLLM_API_URL:
        raise RuntimeError("Inference carrier is not configured. Set CLS_DLLM_API_URL.")
    if system:
        messages = [{"role": "system", "content": system}] + messages
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_DLLM_API_URL}/chat/completions",
        data=payload,
        headers=_dllm_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Inference carrier call failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Inference carrier returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise RuntimeError("Inference carrier returned no text content.")


def generate_answer(
    query: str,
    rows: list[dict],
    *,
    model: str = DEFAULT_DLLM_MODEL,
    timeout: float = 60.0,
    grounded: bool = True,
) -> str:
    """Synthesize a short natural-language answer from the retrieved context.

    Opt-in generative RAG path: unlike `call_dllm_api` correction, this reads the question +
    retrieved passages and writes a direct answer. Uses the same OpenAI-compatible endpoint, so
    it works against Ollama, OpenRouter, Groq, etc.

    `grounded=True` (default) is the strict path: answer only from the context, and refuse with
    "Not found in the indexed documents." when it does not cover the question. `grounded=False`
    is the Hybrid path: the context is optional supporting material and the carrier may also
    answer from its own general knowledge, so it can respond even with no retrieved rows.
    """
    if RETRIEVAL_ONLY:
        raise RuntimeError("Retrieval-only mode is active; LLM generation is disabled.")
    if grounded and not rows:
        return ""
    system = ANSWER_SYSTEM if grounded else ASSIST_SYSTEM
    user = answer_user(query, rows) if grounded else assist_user(query, rows)
    return call_dllm_api(
        [{"role": "user", "content": user}],
        system=system,
        model=model,
        timeout=timeout,
    ).strip()


def stream_generate_answer(
    query: str,
    rows: list[dict],
    *,
    model: str = DEFAULT_DLLM_MODEL,
    timeout: float = 90.0,
    grounded: bool = True,
) -> Generator[str, None, None]:
    """Stream a grounded answer token-by-token via SSE (OpenAI chat/completions with stream=True).

    Yields individual text tokens as they arrive so the caller can pass this directly to
    ``st.write_stream()`` for live rendering without waiting for the full response.
    """
    if RETRIEVAL_ONLY:
        raise RuntimeError("Retrieval-only mode is active; LLM generation is disabled.")
    if grounded and not rows:
        return
    if not DEFAULT_DLLM_API_URL:
        raise RuntimeError("Inference carrier is not configured. Set CLS_DLLM_API_URL.")

    system = ANSWER_SYSTEM if grounded else ASSIST_SYSTEM
    user = answer_user(query, rows) if grounded else assist_user(query, rows)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_DLLM_API_URL}/chat/completions",
        data=payload,
        headers=_dllm_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    token = (
                        data.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if token:
                        yield token
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Streaming call failed: {exc}") from exc


def parrot_status(timeout: float = 1.0) -> dict:
    """Is the small local parrot model reachable? (Ollama OpenAI-compatible endpoint.)"""
    if RETRIEVAL_ONLY:
        return {
            "online": False,
            "model": DEFAULT_PARROT_MODEL,
            "base_url": DEFAULT_PARROT_URL,
            "disabled": True,
            "detail": "Retrieval-only mode is active; parrot phrasing is disabled.",
        }
    online = False
    try:
        request = urllib.request.Request(f"{DEFAULT_PARROT_URL}/models", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            online = 200 <= response.status < 300
    except Exception:
        online = False
    return {"online": online, "model": DEFAULT_PARROT_MODEL, "base_url": DEFAULT_PARROT_URL}


def parrot_answer(sentences: list[str], *, timeout: float = 30.0) -> str | None:
    """Rephrase grounded extractive sentences into one natural-language paragraph using the
    small local model. Returns None if the model is unavailable or if the output drifts from
    the evidence (invents a number) — the caller then falls back to the extractive bullets.
    """
    if RETRIEVAL_ONLY or not sentences:
        return None
    payload = json.dumps(
        {
            "model": DEFAULT_PARROT_MODEL,
            "messages": [
                {"role": "system", "content": PARROT_SYSTEM},
                {"role": "user", "content": parrot_user(sentences)},
            ],
            "stream": False,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_PARROT_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        prose = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return None
    if not prose or not parrot_grounded(prose, sentences):
        return None
    return prose


def parrot_stream(sentences: list[str], *, timeout: float = 60.0):
    """Yield natural-language tokens that rephrase the grounded sentences, streamed from the
    small local model. The grounded extractive answer stays the source of truth; this stream
    is the fallible 'human' layer on top. Yields nothing if the model is unavailable.
    """
    if RETRIEVAL_ONLY or not sentences:
        return
    payload = json.dumps(
        {
            "model": DEFAULT_PARROT_MODEL,
            "messages": [
                {"role": "system", "content": PARROT_SYSTEM},
                {"role": "user", "content": parrot_user(sentences)},
            ],
            "stream": True,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{DEFAULT_PARROT_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                token = delta.get("content")
                if token:
                    yield token
    except Exception:
        return


def service_status() -> dict:
    dllm = dllm_status()
    return {
        "indexed_chunks": collection_count(get_collection()),
        "cached_queries": get_cache().count(),
        "documents": evidence_breakdown(get_collection()),
        "dllm": dllm,
        "retrieval_only": RETRIEVAL_ONLY,
        "keyword_only": KEYWORD_ONLY_RETRIEVAL,
    }
