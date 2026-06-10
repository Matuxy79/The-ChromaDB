from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any
import urllib.error
import urllib.request

import chromadb
import fitz

from cls_config import (
    CACHE_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    COLLECTION_NAME,
    DEFAULT_DLLM_API_KEY,
    DEFAULT_DLLM_API_URL,
    DEFAULT_DLLM_MODEL,
)
from examples.cls_cag_cache import SemanticEvidenceCache
from examples.cls_dllm import ANSWER_SYSTEM, answer_user
from examples.cls_pipeline import EMBED_DIM, HashEmbedder, collection_count, instant_answer, retrieve


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict
    chunk_id: str


_RESOURCE_LOCK = RLock()
_embedder: HashEmbedder | None = None
_chroma_client: Any | None = None
_collection: chromadb.Collection | None = None
_cache_collection: chromadb.Collection | None = None
_cache: SemanticEvidenceCache | None = None


def get_embedder() -> HashEmbedder:
    global _embedder
    with _RESOURCE_LOCK:
        if _embedder is None:
            _embedder = HashEmbedder()
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
                    "description": "CLS IVU beamline manual extractive retrieval prototype",
                    "embedding": f"local_hash_{EMBED_DIM}d",
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
                    "embedding": f"local_hash_{EMBED_DIM}d",
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


def load_pdf(path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append((page_index, text))
    return pages


def load_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(1, text)] if text else []


def load_document(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".txt", ".md"}:
        return load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use PDF, TXT, or MD.")


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
        chunks.append(
            Chunk(
                text=context_text,
                metadata={
                    "source": source_name,
                    "source_hash": source_hash,
                    "page": page_number,
                    "section": section,
                    "chunk_index": chunk_index,
                },
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


def build_chunks(path: Path, source_hash: str) -> list[Chunk]:
    pages = load_document(path)
    chunks: list[Chunk] = []
    next_index = 0
    for page_number, text in pages:
        page_chunks = split_page_text(
            text=text,
            page_number=page_number,
            source_name=path.name,
            source_hash=source_hash,
            starting_index=next_index,
        )
        chunks.extend(page_chunks)
        next_index += len(page_chunks)
    return chunks


def source_is_indexed(collection: chromadb.Collection, source_hash: str) -> bool:
    result = collection.get(where={"source_hash": source_hash}, limit=1)
    return bool(result.get("ids"))


def ingest_path(path: Path, source_hash: str, force: bool = False) -> tuple[int, str]:
    collection = get_collection()
    if source_is_indexed(collection, source_hash):
        if not force:
            return 0, "already indexed"
        existing = collection.get(where={"source_hash": source_hash})
        if existing.get("ids"):
            collection.delete(ids=existing["ids"])

    chunks = build_chunks(path, source_hash)
    if not chunks:
        return 0, "no readable text found"

    embeddings = get_embedder().embed(chunk.text for chunk in chunks)
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )
    return len(chunks), "indexed"


def reset_collection() -> None:
    global _collection
    with _RESOURCE_LOCK:
        client = get_chroma_client()
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
    try:
        get_cache().clear()
    except Exception:
        pass


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
        "question": "Who are the IVU beamline contacts and phone numbers?",
        "keywords": ["beatriz", "narayan", "al", "3868", "3648", "3530"],
    },
    {
        "question": "What are the emergency contact numbers for fire or ambulance?",
        "keywords": ["911", "security", "306-966-5555"],
    },
    {
        "question": "What beamline phone number is listed for the Undulator beamline?",
        "keywords": ["undulator", "soe-3", "3832"],
    },
    {
        "question": "Where does the manual describe the in-vacuum undulator?",
        "keywords": ["in-vacuum", "undulator"],
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
    top_k: int = 8,
    cache_enabled: bool = True,
    min_similarity: float = 0.97,
) -> dict:
    cache = get_cache()
    cache.distance_max = 1.0 - min_similarity
    return instant_answer(
        query,
        collection=get_collection(),
        cache=cache,
        embedder=get_embedder(),
        top_k=top_k,
        cache_enabled=cache_enabled,
    )


def answer_text(sentences: list[str]) -> str:
    if not sentences:
        return "No grounded sentence-level answer was found. Review the returned source passages."
    return "\n".join(f"- {sentence}" for sentence in sentences)


def dllm_online(timeout: float = 1.0) -> bool:
    return bool(dllm_status(timeout=timeout)["online"])


def dllm_status(timeout: float = 1.0) -> dict:
    configured = bool(DEFAULT_DLLM_API_URL)
    online = False
    detail = "Set CLS_DLLM_API_URL to enable dLLM API correction."
    if configured:
        try:
            request = urllib.request.Request(
                f"{DEFAULT_DLLM_API_URL}/models",
                headers=_dllm_headers(),
                method="GET",
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                online = 200 <= response.status < 300
                detail = "dLLM API reachable." if online else f"dLLM API returned HTTP {response.status}."
        except urllib.error.HTTPError as exc:
            detail = f"dLLM API returned HTTP {exc.code}."
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            detail = f"dLLM API unreachable: {exc}"
    return {
        "provider": "api",
        "base_url": DEFAULT_DLLM_API_URL,
        "model": DEFAULT_DLLM_MODEL,
        "configured": configured,
        "online": online,
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
    if not DEFAULT_DLLM_API_URL:
        raise RuntimeError("dLLM API is not configured. Set CLS_DLLM_API_URL.")
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
        raise RuntimeError(f"dLLM API call failed: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("dLLM API returned no choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise RuntimeError("dLLM API returned no text content.")


def generate_answer(
    query: str,
    rows: list[dict],
    *,
    model: str = DEFAULT_DLLM_MODEL,
    timeout: float = 60.0,
) -> str:
    """Synthesize a short natural-language answer grounded in the retrieved context.

    Opt-in generative RAG path: unlike `call_dllm_api` correction, this reads the question +
    retrieved passages and writes a direct answer — but only from the provided context. Uses
    the same OpenAI-compatible endpoint, so it works against Ollama, OpenRouter, Groq, etc.
    """
    if not rows:
        return ""
    return call_dllm_api(
        [{"role": "user", "content": answer_user(query, rows)}],
        system=ANSWER_SYSTEM,
        model=model,
        timeout=timeout,
    ).strip()


def service_status() -> dict:
    dllm = dllm_status()
    return {
        "indexed_chunks": collection_count(get_collection()),
        "cached_queries": get_cache().count(),
        "documents": evidence_breakdown(get_collection()),
        "dllm": dllm,
    }
