from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import chromadb
import fitz
import streamlit as st

from ragandcag.llm import OllamaAPI
from examples.cls_cag_cache import SemanticEvidenceCache
from examples.cls_dllm import (
    needs_correction,
    parse_bullets,
    stream_correction,
    validate_correction,
)
from examples.cls_pipeline import (
    EMBED_DIM,
    HashEmbedder,
    collection_count,
    instant_answer,
    retrieve,
)
from examples.cls_spectrum import (
    SUGGESTED_PROBLEMS,
    category_meta,
    classify_query,
    decorate,
    glow_css,
)


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "Training for perfect in ui graded"
DEFAULT_MANUAL = MANUAL_DIR / "IVU beamline manual - Apr 10 2026.pdf"
CHROMA_DIR = APP_ROOT / "chroma_store"
COLLECTION_NAME = "cls_ivu_manual_hash_v1"           # Evidence Store
CACHE_COLLECTION_NAME = "cls_cag_evidence_cache_v1"  # CAG Layer
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

APP_VERSION = "v0.6"
# Architecture: ONE active model — the embedding Retrieval Encoder. The answer is instant
# clean parsed text from the RAG/CAG dual layer (DocuSearch-style); the LLM does no text
# augmentation by default. A single optional LLM stays wired in but OFF unless toggled on.
DLLM_MODEL = "llama3.2:3b"


st.set_page_config(
    page_title=f"CLS IVU Manual Query · {APP_VERSION}",
    page_icon="🔬",
    layout="wide",
)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict
    chunk_id: str


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


@st.cache_resource(show_spinner=False)
def get_cache_collection() -> chromadb.Collection:
    """CAG Layer: a separate collection of prior (query -> retrieved evidence) results."""
    return get_chroma_client().get_or_create_collection(
        name=CACHE_COLLECTION_NAME,
        metadata={
            "description": "CLS CAG semantic evidence cache",
            "embedding": f"local_hash_{EMBED_DIM}d",
            "hnsw:space": "cosine",
        },
    )


@st.cache_resource(show_spinner=False)
def get_cache() -> SemanticEvidenceCache:
    return SemanticEvidenceCache(get_cache_collection(), get_embedder())


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
    client = get_chroma_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_collection.clear()
    # The corpus changed, so any cached evidence is now stale.
    try:
        get_cache().clear()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Optional dLLM. ONE LLM, wired in but OFF by default — it never augments the
# displayed text unless the user toggles it on. The instant answer is pure clean
# parsed text from the RAG/CAG backend (examples/cls_pipeline.py).
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_dllm() -> OllamaAPI:
    return OllamaAPI(DLLM_MODEL)


def ollama_online() -> bool:
    try:
        return get_dllm().is_available(timeout=1.0)
    except Exception:
        return False


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


st.markdown(
    """
    <style>
      :root {
        --bg: #faf6f2;
        --panel: #ffffff;
        --ink: #241a1d;
        --muted: #6d5b56;
        --faint: #927d77;
        --line: #ecddd4;
        --rose: #ff4d6d;
        --orange: #ff8a3d;
        --amber: #ffc24b;
        --green: #6fd58a;
        --blue: #6aa9ff;
      }

      html, body, #root {
        background: var(--bg) !important;
        min-height: 100vh !important;
      }
      header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
      }
      .stAppDeployButton { display: none !important; }

      .stApp {
        background:
          radial-gradient(1100px 360px at 0% -8%, rgba(255,138,61,0.10) 0%, rgba(255,138,61,0) 60%),
          var(--bg);
        color: var(--ink);
        min-height: 100vh;
      }
      div[data-testid="stAppViewContainer"],
      section[data-testid="stMain"] {
        background: transparent !important;
        min-height: 100vh;
      }
      .block-container {
        max-width: 1280px;
        padding: 1.35rem 2rem 3rem;
      }
      .stApp h1, .stApp h2, .stApp h3, .stApp h4,
      .stApp p, .stApp li, .stApp label, .stMarkdown,
      .stCaptionContainer, div[data-testid="stText"] {
        color: var(--ink) !important;
      }
      .stCaptionContainer, .stApp small {
        color: var(--muted) !important;
      }
      .stApp h2, .stApp h3 {
        letter-spacing: 0;
        font-weight: 760;
      }
      .cls-spectrum-rule {
        height: 5px; border-radius: 999px; margin: 0.15rem 0 0;
        background: linear-gradient(90deg,var(--rose),var(--orange),var(--amber),var(--green),var(--blue),#b478ff);
      }
      .cls-hero {
        border-radius: 14px; padding: 1.15rem 1.35rem 1rem; margin-bottom: 1.25rem;
        background: var(--panel);
        border: 1px solid var(--line);
        box-shadow: 0 8px 26px rgba(120,80,60,0.08);
      }
      .cls-hero h1 {
        margin: 0;
        font-size: clamp(1.45rem, 2vw, 1.85rem);
        color: var(--ink) !important;
      }
      .cls-hero p  {
        margin: 0.3rem 0 0.7rem;
        color: var(--muted) !important;
        font-size: 0.95rem;
      }
      .cls-badge {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.22rem 0.66rem; border-radius: 999px; font-size: 0.82rem; font-weight: 700;
        color: #3a2a2e; background: color-mix(in srgb, var(--hue) 16%, #ffffff);
        border: 1px solid color-mix(in srgb, var(--hue) 55%, #ffffff);
      }
      .cls-answer {
        border-radius: 12px; padding: 1.05rem 1.2rem; margin-top: 0.4rem;
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 5px solid var(--hue,#ff8a3d);
        box-shadow: var(--glow, 0 6px 22px rgba(120,80,60,0.10));
        color: var(--ink); line-height: 1.6;
      }
      .cls-answer code { color: #b8541a; background: #fbeee4; padding: 0 0.3em; border-radius: 5px; }
      .cls-q { color: var(--muted); font-style: italic; font-size: 0.92rem; }
      div[data-testid="stMetricValue"] { color: var(--ink) !important; }
      div[data-testid="stMetricDelta"] { color: var(--muted) !important; }
      .tok { border-radius: 6px; padding: 0 0.28em; font-weight: 600; }
      .tok-phone  { color: #a8480f; background: #fcebdd; box-shadow: inset 0 0 0 1px rgba(244,120,31,0.40); }
      .tok-acr    { color: #b3203f; background: #fbe2e7; box-shadow: inset 0 0 0 1px rgba(224,57,92,0.35); }
      .tok-safety { color: #c01838; text-decoration: underline; text-decoration-color: #ff4d6d; text-underline-offset: 3px; font-weight: 700; }
      .tok-cite   { color: #8a7a74; font-weight: 500; font-size: 0.9em; }
      /* DocuSearch-style query-term hit highlight. */
      .tok-hit    { color: #1f6b38; background: #d6f3df; box-shadow: inset 0 0 0 1px rgba(47,158,87,0.45); }
      mark.tok-hit { color: #1f6b38; }
      .cls-snippet {
        white-space: pre-wrap; word-break: break-word;
        color: #4a3a3e; font-size: 0.9rem; line-height: 1.5;
      }

      section[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--line);
      }
      section[data-testid="stSidebar"] * {
        color: var(--ink) !important;
      }
      section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #b3203f !important;
      }
      section[data-testid="stSidebar"] code,
      section[data-testid="stSidebar"] pre {
        background: #f6efe9 !important;
        color: #3a2a2e !important;
        border: 1px solid var(--line);
        white-space: pre-wrap !important;
        word-break: break-word !important;
      }

      .stButton > button,
      button[data-testid="stBaseButton-secondary"] {
        border-radius: 10px !important;
        border: 1px solid var(--line) !important;
        background: var(--panel) !important;
        color: var(--ink) !important;
        box-shadow: 0 2px 8px rgba(120,80,60,0.06);
        min-height: 2.35rem;
        transition: border-color 120ms ease, transform 120ms ease, background 120ms ease;
      }
      .stButton > button:hover,
      button[data-testid="stBaseButton-secondary"]:hover {
        border-color: var(--orange) !important;
        background: #fff6ef !important;
        transform: translateY(-1px);
      }
      button[data-testid="stBaseButton-primary"] {
        border: 0 !important;
        background: linear-gradient(90deg, #ff4d6d 0%, #ff7a45 55%, #ffa516 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 20px rgba(255,122,69,0.30);
      }
      button[data-testid="stBaseButton-primary"]:hover {
        filter: brightness(1.04);
        transform: translateY(-1px);
      }

      div[data-testid="stTextArea"] textarea,
      div[data-testid="stTextInput"] input {
        background: #ffffff !important;
        color: var(--ink) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
      }
      div[data-testid="stTextArea"] textarea::placeholder {
        color: #a8938c !important;
      }
      div[data-testid="stSlider"] [role="slider"] {
        background: var(--orange) !important;
        border-color: var(--orange) !important;
      }
      .stProgress > div > div > div { background: #efe4dc !important; }
      .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--rose), var(--orange), var(--amber)) !important;
      }
      div[data-testid="stAlert"] {
        background: #fff6ef !important;
        color: var(--ink) !important;
        border: 1px solid var(--line);
        border-radius: 10px;
      }
      div[data-testid="stFileUploader"] section {
        background: #faf4ef !important;
        border-color: var(--line) !important;
      }
      div[data-testid="stExpander"] {
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        background: var(--panel) !important;
      }
      div[data-testid="stExpander"] * {
        color: var(--ink) !important;
      }
      code, pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

online = ollama_online()
st.markdown(
    f"""
    <div class="cls-hero">
      <h1>🔬 CLS IVU Beamline Manual Query <span style="opacity:0.55;font-size:1rem;">{APP_VERSION}</span></h1>
      <p>Cited manual answers for IVU operators and beamline staff.</p>
      <div class="cls-spectrum-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
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

    st.header("Evidence Store")
    st.metric("Indexed chunks", collection_count(get_collection()))
    if st.button("Reset Chroma index", use_container_width=True):
        reset_collection()
        st.success("Evidence Store reset (CAG cache cleared too). Re-index to query again.")

    st.header("♻ CAG Layer")
    st.caption("Semantic cache of prior question → retrieved evidence.")
    cag_enabled = st.toggle("Reuse cached evidence", value=True)
    min_similarity = st.slider("Min similarity for a cache hit", 0.80, 1.00, 0.97, 0.01)
    get_cache().distance_max = 1.0 - min_similarity
    st.metric("Cached queries", get_cache().count())
    if st.button("Clear answer cache", use_container_width=True):
        get_cache().clear()
        st.success("CAG cache cleared.")

    st.header("✎ dLLM (optional)")
    st.caption("Off by default — the answer is instant clean parsed text with no LLM. Turn on to let "
               "a single downstream model correct extraction artifacts (it never invents facts).")
    dllm_enabled = st.toggle(
        "Enable downstream correction",
        value=False,
        disabled=not online,
        help="Needs Ollama. When on, streams a guarded correction in place; clean answers stay instant.",
    )
    if not online:
        st.caption("Ollama offline — answers are instant parsed text only.")

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("📥 Pipeline")
    st.markdown(
        f"""
        1. **Retrieval Encoder** (HashEmbedder) turns text into vectors — the one active model.
        2. **CAG Layer** reuses a prior query's evidence when a similar question reappears.
        3. **Evidence Store** (ChromaDB) returns the closest cited passages on a miss.
        4. **Clean parse** repairs extraction artifacts deterministically — instant, no LLM.
        """
    )
    with st.expander("ⓘ How it works"):
        st.markdown(
            f"""
            Like DocuSearch, the answer is **instant**: clean parsed text straight from the
            RAG/CAG dual layer, with query terms highlighted. The only model in this path is the
            **embedding Retrieval Encoder**; artifact cleanup (hyphenation breaks, spacing,
            duplicates) is **deterministic code**, not an LLM.

            A single optional **dLLM** ({DLLM_MODEL}) stays wired in but **off by default** — it
            never augments the displayed text unless you toggle it on. When on, it streams a
            *correction* in place and is rejected unless every number and `[Source: …]` citation
            survives verbatim.

            `question → Retrieval Encoder → CAG / Evidence Store → clean parse → instant answer`
            """
        )

    st.subheader("🔎 Ask a problem")
    st.caption("Pick a starting point, or type your own.")

    # Problem-asking chips: clicking one pre-fills the query (seeds the text_area key
    # before the widget is created on this run, so no extra rerun is needed).
    for chip_row in (SUGGESTED_PROBLEMS[:3], SUGGESTED_PROBLEMS[3:]):
        cols = st.columns(len(chip_row))
        for col, (chip_cat, chip_text) in zip(cols, chip_row):
            glyph = category_meta(chip_cat)["glyph"]
            if col.button(f"{glyph} {chip_text}", key=f"chip_{chip_text}", use_container_width=True):
                st.session_state["query_text"] = chip_text

    query = st.text_area(
        "Scientist / operator prompt",
        placeholder="Example: What phone number is listed for the Undulator beamline?",
        height=100,
        key="query_text",
    )

    # Live category badge — reclassifies on each run (chip click / submit), not per keystroke.
    live_cat = classify_query(query)
    live_meta = category_meta(live_cat)
    st.markdown(
        f'<span class="cls-badge" style="--hue:{live_meta["hue"]}">'
        f'{live_meta["glyph"]} {live_meta["label"]}</span>',
        unsafe_allow_html=True,
    )

    top_k = st.slider("Top-K chunks", 3, 12, 8)

    if st.button("Search IVU Manual", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            # System 2: the RAG+CAG backend returns the instant clean parsed answer.
            result = instant_answer(
                query,
                collection=get_collection(),
                cache=get_cache(),
                embedder=get_embedder(),
                top_k=top_k,
                cache_enabled=cag_enabled,
            )
            if not result["rows"]:
                st.error("No indexed chunks found. Click 'Index IVU manual' first.")
            else:
                st.session_state["last_query"] = query.strip()
                st.session_state["last_category"] = result["category"]
                st.session_state["last_rows"] = result["rows"]
                st.session_state["last_answer"] = result["answer"]
                st.session_state["last_from_cache"] = result["from_cache"]
                st.session_state["last_similarity"] = result["similarity"]
                # dLLM is off by default and never touches text unless toggled on. If on,
                # arm it only when the instant text shows artifacts worth correcting.
                activate, reason = needs_correction(result["answer"])
                st.session_state["last_dllm"] = {
                    "status": "pending" if (activate and result["answer"]) else "dormant",
                    "reason": reason,
                    "text": None,
                }

    if st.button("Run graded offline checks", use_container_width=True):
        if collection_count(get_collection()) == 0:
            st.error("Index the IVU manual before running graded checks.")
        else:
            st.session_state["eval_rows"] = evaluate_retrieval()

with right:
    rows = st.session_state.get("last_rows", [])
    answer = st.session_state.get("last_answer", [])
    if rows:
        top_score = rows[0]["score"]

        # Spectral framing: question type picks the hue (the trustworthy signal); the
        # hash embedder's absolute cosine score isn't a calibrated relevance, so it is
        # not shown — it only sets the answer card's glow intensity.
        category = st.session_state.get("last_category", "general")
        meta = category_meta(category)
        hue = meta["hue"]
        glow = glow_css(hue, top_score)
        card_open = f'<div class="cls-answer" style="--hue:{hue};--glow:{glow}">'

        # CAG provenance: did this evidence come from the cache or a fresh search?
        from_cache = st.session_state.get("last_from_cache", False)
        similarity = st.session_state.get("last_similarity")
        if from_cache:
            sim_txt = f"{similarity:.2f}" if similarity is not None else "—"
            cag_badge = (
                '<span class="cls-badge" style="--hue:#6fd58a">'
                f'♻ CAG hit · similarity {sim_txt}</span>'
            )
        else:
            cag_badge = '<span class="cls-q">↯ fresh retrieval</span>'

        st.markdown("### ✨ Answer")
        st.markdown(
            f'Question type · <span class="cls-badge" style="--hue:{hue}">'
            f'{meta["glyph"]} {meta["label"]}</span> &nbsp; {cag_badge}',
            unsafe_allow_html=True,
        )

        stored_query = st.session_state.get("last_query", "")

        def card(sentences: list[str]) -> str:
            body = "<br>".join(f"• {decorate(s, category, stored_query)}" for s in sentences)
            return f"{card_open}{body}</div>"

        if answer:
            dllm = st.session_state.get("last_dllm", {"status": "dormant"})
            placeholder = st.empty()

            if dllm.get("status") == "corrected" and dllm.get("text"):
                # Settled from a prior run — render directly, no flash, no model call.
                placeholder.markdown(card(dllm["text"]), unsafe_allow_html=True)
                st.caption(f"✎ dLLM corrected — {dllm.get('reason')}.")
            elif dllm.get("status") == "pending" and online and dllm_enabled:
                # Instant text appears first; the dLLM then streams its correction *in place*
                # into the same card. Re-checked for grounding at the end; reverts on drift.
                placeholder.markdown(card(answer), unsafe_allow_html=True)
                buffer = ""
                try:
                    for chunk in stream_correction(get_dllm(), answer):
                        buffer += chunk
                        placeholder.markdown(
                            f"{card_open}{decorate(buffer, category, stored_query)}</div>",
                            unsafe_allow_html=True,
                        )
                    corrected = parse_bullets(buffer)
                    if corrected and validate_correction(" ".join(corrected), answer):
                        placeholder.markdown(card(corrected), unsafe_allow_html=True)
                        st.session_state["last_dllm"] = {
                            "status": "corrected", "reason": dllm.get("reason"), "text": corrected
                        }
                        st.caption(f"✎ dLLM corrected — {dllm.get('reason')}.")
                    else:
                        placeholder.markdown(card(answer), unsafe_allow_html=True)
                        st.session_state["last_dllm"] = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                        st.caption("⚡ instant — dLLM drifted, kept the grounded extraction.")
                except Exception:
                    placeholder.markdown(card(answer), unsafe_allow_html=True)
                    st.session_state["last_dllm"] = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                    st.caption("⚡ instant — dLLM unavailable, kept the grounded extraction.")
            else:
                placeholder.markdown(card(answer), unsafe_allow_html=True)
                st.caption("⚡ instant — clean parsed text from RAG/CAG, no LLM.")
        else:
            st.warning("No strong sentence-level extraction found. Review the source passages below.")

        st.markdown("### Source passages")
        hit_terms = stored_query
        for index, row in enumerate(rows, start=1):
            meta = row["metadata"]
            label = (
                f"{index}. {meta.get('source', 'source')}, "
                f"page {meta.get('page', '?')} — {meta.get('section', 'section')}"
            )
            with st.expander(label, expanded=index <= 2):
                st.markdown(
                    f'<div class="cls-snippet">{decorate(row["document"], category, hit_terms)}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Index the IVU manual, then run a search to see scored source passages.")

eval_rows = st.session_state.get("eval_rows", [])
if eval_rows:
    st.divider()
    st.subheader("🧪 Graded offline query checks")
    st.dataframe(eval_rows, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"{APP_VERSION} · DocuSearch-style: the answer is **instant clean parsed text** from the RAG/CAG "
    "dual layer — **Retrieval Encoder** (HashEmbedder, the one active model) → **CAG Layer** / "
    "**Evidence Store** (ChromaDB) → deterministic clean-parse + highlighting. **No LLM augments the "
    f"text by default**; a single optional **dLLM** ({DLLM_MODEL}) stays wired in but OFF unless toggled, "
    "and even then only corrects artifacts (never invents facts). Fully offline-capable; no dsrag import."
)
