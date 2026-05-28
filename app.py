from __future__ import annotations

import hashlib
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import fitz
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "Training for perfect in ui graded"
DEFAULT_MANUAL = MANUAL_DIR / "IVU beamline manual - Apr 10 2026.pdf"
CHROMA_DIR = APP_ROOT / "chroma_store"
COLLECTION_NAME = "cls_ivu_manual_hash_v1"
EMBED_DIM = 512
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180


st.set_page_config(
    page_title="CLS IVU Manual Query Prototype",
    page_icon="🔬",
    layout="wide",
)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict
    chunk_id: str


class HashEmbedder:
    """Small deterministic embedding model for offline technical-manual retrieval."""

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


@st.cache_resource(show_spinner=False)
def get_embedder() -> HashEmbedder:
    return HashEmbedder()


@st.cache_resource(show_spinner=False)
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@st.cache_resource(show_spinner=False)
def get_collection() -> chromadb.Collection:
    return get_chroma_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "CLS IVU beamline manual extractive retrieval prototype",
            "embedding": f"local_hash_{EMBED_DIM}d",
            "hnsw:space": "cosine",
        },
    )


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


def collection_count(collection: chromadb.Collection) -> int:
    try:
        return collection.count()
    except Exception:
        return 0


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
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection.clear()


def query_manual(query: str, n_results: int = 8) -> list[dict]:
    collection = get_collection()
    if collection_count(collection) == 0:
        return []

    result = collection.query(
        query_embeddings=get_embedder().embed([query]),
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


def lexical_terms(text: str) -> set[str]:
    stop_words = {
        "about",
        "after",
        "before",
        "could",
        "from",
        "have",
        "into",
        "manual",
        "procedure",
        "should",
        "tell",
        "that",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-/\.]*", text.lower())
        if len(token) > 2 and token not in stop_words
    }


def extract_sentences(text: str) -> list[str]:
    body = re.sub(r"^Source:.*?\nSection:.*?\nPage:.*?\n\n", "", text, flags=re.S)
    normalized = re.sub(r"\s+", " ", body)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def build_extractive_answer(query: str, rows: list[dict], max_sentences: int = 5) -> list[str]:
    terms = lexical_terms(query)
    candidates: list[tuple[int, float, str, dict]] = []
    for row in rows[:5]:
        for sentence in extract_sentences(row["document"]):
            sentence_terms = lexical_terms(sentence)
            overlap = len(terms & sentence_terms)
            if overlap:
                candidates.append((overlap, row["score"], sentence, row["metadata"]))

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
        key = (source_hash, index)
        if key in used:
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
                "score": max(neighbor["score"] for neighbor in neighbors),
                "document": "\n\n--- adjacent segment ---\n\n".join(neighbor["document"] for neighbor in neighbors),
                "parts": len(neighbors),
            }
        )
    return merged


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
        rows = query_manual(case["question"], n_results=6)
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


st.title("🔬 CLS IVU Beamline Manual Query")
st.caption(
    "Local-first ChromaDB retrieval prototype for offline scientific manual search, extraction, and scoring."
)

with st.sidebar:
    st.header("One-click corpus")
    st.write("Default manual:")
    st.code(str(DEFAULT_MANUAL.relative_to(APP_ROOT)) if DEFAULT_MANUAL.exists() else "Missing IVU PDF")
    force_reindex = st.checkbox("Force rebuild existing IVU index", value=False)
    if st.button("Index IVU manual", type="primary", use_container_width=True):
        if DEFAULT_MANUAL.exists():
            with st.status("Indexing IVU beamline manual...", expanded=True) as status:
                signature = file_signature(DEFAULT_MANUAL)
                count, message = ingest_path(DEFAULT_MANUAL, signature, force=force_reindex)
                st.write(f"ChromaDB status: {message}")
                st.write(f"Chunks added: {count}")
                status.update(label="Index ready", state="complete", expanded=False)
        else:
            st.error("The default IVU manual PDF was not found.")

    st.header("Upload more docs")
    uploaded_files = st.file_uploader(
        "Add PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded files", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            for uploaded_file in uploaded_files:
                data = uploaded_file.getvalue()
                suffix = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(data)
                    temp_path = Path(handle.name)
                try:
                    count, message = ingest_path(temp_path, uploaded_signature(data), force=True)
                    st.success(f"{uploaded_file.name}: {message}, {count} chunks")
                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")
                finally:
                    temp_path.unlink(missing_ok=True)

    st.header("Database")
    st.metric("Stored chunks", collection_count(get_collection()))
    if st.button("Reset Chroma index", use_container_width=True):
        reset_collection()
        st.success("Chroma index reset. Re-index the manual to query again.")

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("📥 Ingestion pipeline")
    st.markdown(
        """
        1. **Semantic sectioning:** preserve source, page, and detected heading.
        2. **AutoContext-style prefix:** prepend source/page/section before embedding.
        3. **Chunking & embedding:** store deterministic local vectors in ChromaDB.
        """
    )

    st.subheader("🔎 Query & retrieval")
    query = st.text_area(
        "Scientist / operator prompt",
        placeholder="Example: What phone number is listed for the Undulator beamline?",
        height=100,
    )
    top_k = st.slider("Top-K chunks", 3, 12, 8)

    if st.button("Search IVU Manual", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            rows = query_manual(query, n_results=top_k)
            if not rows:
                st.error("No indexed chunks found. Click 'Index IVU manual' first.")
            else:
                merged = merge_adjacent_segments(rows)
                answer_sentences = build_extractive_answer(query, merged or rows)
                st.session_state["last_rows"] = rows
                st.session_state["last_merged"] = merged
                st.session_state["last_answer"] = answer_sentences

    if st.button("Run graded offline checks", use_container_width=True):
        if collection_count(get_collection()) == 0:
            st.error("Index the IVU manual before running graded checks.")
        else:
            st.session_state["eval_rows"] = evaluate_retrieval()

with right:
    st.subheader("📊 Retrieval score")
    rows = st.session_state.get("last_rows", [])
    answer = st.session_state.get("last_answer", [])
    if rows:
        top_score = rows[0]["score"]
        confidence = "High" if top_score >= 0.55 else "Medium" if top_score >= 0.35 else "Needs review"
        st.metric("Top relevance", f"{top_score:.2f}", confidence)
        st.progress(min(top_score, 1.0))

        st.markdown("### Extractive answer draft")
        if answer:
            for sentence in answer:
                st.write(f"- {sentence}")
        else:
            st.warning("No strong sentence-level extraction found. Review the source passages below.")

        st.markdown("### Source passages")
        for index, row in enumerate(rows, start=1):
            meta = row["metadata"]
            label = (
                f"{index}. score {row['score']:.2f} — {meta.get('source', 'source')}, "
                f"page {meta.get('page', '?')}, {meta.get('section', 'section')}"
            )
            with st.expander(label, expanded=index <= 2):
                st.text(row["document"])
    else:
        st.info("Index the IVU manual, then run a search to see scored source passages.")

eval_rows = st.session_state.get("eval_rows", [])
if eval_rows:
    st.divider()
    st.subheader("🧪 Graded offline query checks")
    st.dataframe(eval_rows, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Why ChromaDB: persistent local vectors, tiny single-node setup, Python-native API, no server required, "
    "and good fit for data-sovereign CLS manual retrieval. No dsrag package is imported."
)
