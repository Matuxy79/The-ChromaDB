from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
import urllib.error
import urllib.request

import streamlit as st

from cls_config import (
    APP_ROOT,
    APP_VERSION,
    DEFAULT_API_URL,
    DEFAULT_DLLM_MODEL,
    DEFAULT_MANUAL,
)
from cls_service import (
    ask_manual,
    call_dllm_api,
    collection_count,
    dllm_status as service_dllm_status,
    evaluate_retrieval,
    evidence_breakdown,
    file_signature,
    generate_answer,
    get_cache,
    get_collection,
    ingest_path,
    reset_collection,
    uploaded_signature,
)
from cls_backend.dllm import (
    CORRECTION_SYSTEM,
    answer_numbers_grounded,
    correction_user,
    needs_correction,
    parse_bullets,
    validate_correction,
)
from cls_backend.spectrum import (
    SUGGESTED_PROBLEMS,
    category_meta,
    classify_query,
    decorate,
    glow_css,
)


# --------------------------------------------------------------------------- #
# Role tiers. The left sidebar delegates one of four paths; each role reshapes
# the screen by capability. Corpus lifecycle (indexing/reset) is admin-specific;
# lower tiers narrow down to a clean ask-and-read surface. This is a visual +
# permission separation only — it does not change how chunks are embedded.
# --------------------------------------------------------------------------- #
ROLES: dict[str, dict] = {
    "Admin": {
        "glyph": "🛡",
        "accent": "#ff4d6d",
        "tagline": "Full control — corpus lifecycle, tuning, and evaluation.",
        "caps": {"index", "upload", "reindex", "reset", "tune", "cag_tune", "dllm", "eval"},
    },
    "Scientist": {
        "glyph": "🔬",
        "accent": "#6aa9ff",
        "tagline": "Bring your own docs and query with full precision controls.",
        "caps": {"upload", "reindex", "tune", "cag_tune", "dllm", "eval"},
    },
    "Staff": {
        "glyph": "🛠",
        "accent": "#ffb347",
        "tagline": "Ask the manual and see what's indexed. Read-only corpus.",
        "caps": {"cag_toggle"},
    },
    "User": {
        "glyph": "👤",
        "accent": "#6fd58a",
        "tagline": "Just ask a problem and read the cited answer.",
        "caps": set(),
    },
}
ROLE_NAMES = list(ROLES)
# Architecture: ONE active *local* model — the embedding Retrieval Encoder. Retrieval is instant
# and primary (DocuSearch-style). The generative answer is carried by an external OpenAI-compatible
# LLM — default carrier OpenRouter · openai/gpt-oss-120b — wired in as a single on/off toggle that
# is ON by default once a carrier key is configured. It synthesizes strictly from the retrieved
# context, and the grounded extractive answer is always shown beneath it for audit.
DLLM_MODEL = DEFAULT_DLLM_MODEL


st.set_page_config(
    page_title=f"CLS IVU Manual Query · {APP_VERSION}",
    page_icon="🔬",
    layout="wide",
)

def render_evidence_store(rows: list[dict]) -> None:
    """Idle-state panel: visualize what is indexed, one bar per document."""
    st.markdown("### 📚 Evidence Store")
    if not rows:
        st.info("Index the IVU manual, then run a search to see scored retrieval evidence.")
        return
    total_chunks = sum(r["chunks"] for r in rows)
    st.caption(f"{len(rows)} document(s) · {total_chunks} indexed chunks — run a search to query them.")
    top = max(r["chunks"] for r in rows)
    blocks: list[str] = []
    for r in rows:
        pct = max(6, round(100 * r["chunks"] / top))
        span = r["page_span"]
        if span and span[0] != span[1]:
            meta = f"pages {span[0]}–{span[1]}"
        elif span:
            meta = f"page {span[0]}"
        else:
            meta = ""
        blocks.append(
            '<div class="cls-evrow">'
            '<div class="cls-evhead">'
            f'<span class="cls-evname">📄 {html.escape(str(r["source"]))}</span>'
            f'<span class="cls-evcount">{r["chunks"]} chunks</span></div>'
            f'<div class="cls-evtrack"><div class="cls-evfill" style="width:{pct}%"></div></div>'
            f'<div class="cls-evmeta">{meta}</div></div>'
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


API_URL = DEFAULT_API_URL.rstrip("/")
USE_API_BACKEND = os.getenv("CLS_USE_API", "0").strip().lower() in {"1", "true", "yes", "on"}


def post_json(path: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API bridge unavailable at {API_URL}: {exc}") from exc


def get_json(path: str, timeout: float = 2.0) -> dict:
    request = urllib.request.Request(
        f"{API_URL}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


def query_backend(query: str, top_k: int, cache_enabled: bool, min_similarity: float) -> dict:
    if USE_API_BACKEND:
        return post_json(
            "/v1/query",
            {
                "query": query,
                "top_k": top_k,
                "cache_enabled": cache_enabled,
                "min_similarity": min_similarity,
            },
        )
    return ask_manual(
        query,
        top_k=top_k,
        cache_enabled=cache_enabled,
        min_similarity=min_similarity,
    )


def dllm_api_status() -> dict:
    if not USE_API_BACKEND:
        return service_dllm_status()
    status = get_json("/v1/dllm/status")
    if status:
        return status
    return {
        "provider": "api",
        "base_url": API_URL,
        "model": DLLM_MODEL,
        "configured": False,
        "online": False,
        "detail": f"Start the API bridge at {API_URL}, or use embedded retrieval-only mode.",
    }


def correct_with_dllm_api(sentences: list[str]) -> list[str]:
    if not USE_API_BACKEND:
        return parse_bullets(
            call_dllm_api(
                [{"role": "user", "content": correction_user(sentences)}],
                system=CORRECTION_SYSTEM,
                model=DLLM_MODEL,
            )
        )
    response = post_json(
        "/v1/dllm/chat",
        {
            "model": DLLM_MODEL,
            "system": CORRECTION_SYSTEM,
            "messages": [{"role": "user", "content": correction_user(sentences)}],
        },
        timeout=60.0,
    )
    return parse_bullets(response.get("content", ""))


def display_model_name(model: str) -> str:
    return (model or "model").rsplit("/", 1)[-1]


def retrieval_evidence_summary(rows: list[dict]) -> tuple[str, bool]:
    """Compact HTML summary of the retrieval rows and their source documents."""
    sources: dict[str, dict] = {}
    for row in rows:
        meta = row.get("metadata", {}) or {}
        source = str(meta.get("source") or "source")
        entry = sources.setdefault(source, {"count": 0, "pages": []})
        entry["count"] += 1
        page = meta.get("page")
        if page not in (None, "", "?") and page not in entry["pages"]:
            entry["pages"].append(page)

    doc_count = len(sources)
    row_word = "row" if len(rows) == 1 else "rows"
    doc_word = "document" if doc_count == 1 else "documents"
    pieces: list[str] = []
    for source, entry in list(sources.items())[:4]:
        pages = entry["pages"]
        pages_text = ""
        if pages:
            shown = ", ".join(str(page) for page in pages[:4])
            if len(pages) > 4:
                shown += ", ..."
            pages_text = f" · pages {shown}"
        count_word = "row" if entry["count"] == 1 else "rows"
        pieces.append(
            f"<strong>{html.escape(source)}</strong> "
            f"({entry['count']} {count_word}{html.escape(pages_text)})"
        )
    if len(sources) > 4:
        pieces.append(f"{len(sources) - 4} more")

    summary = (
        f"{len(rows)} ranked evidence {row_word} from {doc_count} {doc_word}: "
        + "; ".join(pieces)
    )
    return summary, doc_count > 1


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
      /* Role tier separator — sidebar card + hero chip share the --role accent. */
      .cls-rolecard {
        border-radius: 12px; padding: 0.7rem 0.85rem; margin: 0.1rem 0 1rem;
        background: color-mix(in srgb, var(--role) 12%, #ffffff);
        border: 1px solid color-mix(in srgb, var(--role) 42%, #ffffff);
        border-left: 5px solid var(--role);
      }
      .cls-rolehead { font-weight: 800; font-size: 1.04rem; color: #2a1d20; }
      .cls-roletag { font-size: 0.8rem; color: var(--muted); margin-top: 0.18rem; line-height: 1.35; }
      .cls-rolelock { color: var(--faint); font-size: 0.82rem; margin: 0.1rem 0 0.4rem; }

      .cls-answer {
        border-radius: 12px; padding: 1.05rem 1.2rem; margin-top: 0.4rem;
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 5px solid var(--hue,#ff8a3d);
        box-shadow: var(--glow, 0 6px 22px rgba(120,80,60,0.10));
        color: var(--ink); line-height: 1.6;
      }
      .cls-statusline {
        display: flex; flex-wrap: wrap; align-items: center; gap: 0.45rem;
        margin: 0.1rem 0 0.65rem;
      }
      .cls-mode-note {
        color: var(--muted); font-size: 0.86rem; line-height: 1.45;
        margin: 0.25rem 0 0.65rem;
      }
      .cls-evidence-strip {
        border-radius: 10px; padding: 0.72rem 0.82rem; margin: 0.25rem 0 0.7rem;
        background: #f7fbff;
        border: 1px solid #d9e8f6;
        border-left: 5px solid var(--blue);
        line-height: 1.45;
      }
      .cls-evidence-title {
        font-weight: 800; color: var(--ink); margin-bottom: 0.15rem;
      }
      .cls-evidence-copy {
        color: var(--muted); font-size: 0.86rem;
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

      /* Evidence Store visualization — one bar per indexed document. */
      .cls-evrow { margin: 0.55rem 0 0.75rem; }
      .cls-evhead {
        display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem;
      }
      .cls-evname { font-weight: 650; color: var(--ink); font-size: 0.92rem; word-break: break-word; }
      .cls-evcount { color: var(--muted); font-size: 0.82rem; white-space: nowrap; }
      .cls-evtrack {
        height: 8px; border-radius: 999px; background: #efe4dc;
        margin: 0.32rem 0 0.18rem; overflow: hidden;
      }
      .cls-evfill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, var(--rose), var(--orange), var(--amber));
      }
      .cls-evmeta { color: var(--faint); font-size: 0.78rem; }

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

dllm_endpoint = dllm_api_status()
dllm_api_online = bool(dllm_endpoint.get("online"))

# Resolve the active role before the hero so the chip + accent reflect it on the
# same run. The sidebar radio (key="role") persists the choice across reruns.
active_role = st.session_state.get("role", ROLE_NAMES[0])
if active_role not in ROLES:
    active_role = ROLE_NAMES[0]
role_meta = ROLES[active_role]
role_accent = role_meta["accent"]
role_caps = role_meta["caps"]


def role_can(capability: str) -> bool:
    return capability in role_caps


# Per-run accent so the whole path is visually tinted (sidebar edge + chip).
st.markdown(
    f"<style>:root {{ --role: {role_accent}; }}"
    f"section[data-testid=\"stSidebar\"] {{ border-right: 3px solid {role_accent} !important; }}</style>",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="cls-hero">
      <h1>🔬 CLS IVU Beamline Manual Query <span style="opacity:0.55;font-size:1rem;">{APP_VERSION}</span>
        &nbsp;<span class="cls-badge" style="--hue:{role_accent}">{role_meta['glyph']} {active_role}</span></h1>
      <p>Cited manual answers for IVU operators and beamline staff.</p>
      <div class="cls-spectrum-rule"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

def _human_size(num_bytes: int) -> str:
    """Compact human-readable byte count for batch upload readouts."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _render_corpus_admin() -> None:
    """Admin-only path: index the canonical IVU manual."""
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


def _render_upload_section() -> None:
    """Admin + Scientist path: drag-and-drop a whole batch of docs and index them at once."""
    st.subheader("📥 Upload & index documents")
    st.caption(
        "Drag and drop a batch of files into the drop zone below — PDF, TXT, or MD. "
        "They join the Evidence Store; retrieval evidence stays separate from any carrier synthesis."
    )
    uploaded_files = st.file_uploader(
        "Drag and drop files here, or click to browse — multiple files supported",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        sizes = [(f.name, len(f.getvalue())) for f in uploaded_files]
        total_bytes = sum(size for _, size in sizes)
        st.caption(f"📎 {len(uploaded_files)} file(s) ready · {_human_size(total_bytes)} total.")
        with st.expander(f"Files queued for indexing ({len(sizes)})", expanded=False):
            st.markdown("\n".join(f"- {name} — {_human_size(size)}" for name, size in sizes))
    reindex_uploads = (
        st.checkbox(
            "Re-index files already in the store",
            value=False,
            help="Off: files already indexed are skipped (fast for big batches). "
                 "On: re-embed every file even if its content is unchanged.",
        )
        if role_can("reindex")
        else False
    )
    if st.button("Index uploaded files", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one file first.")
        else:
            total = len(uploaded_files)
            progress = st.progress(0.0, text=f"Indexing 0 / {total}…")
            tally = {"indexed": 0, "reindexed": 0, "skipped": 0, "empty": 0, "failed": 0}
            chunks_added = 0
            details: list[str] = []
            for position, uploaded_file in enumerate(uploaded_files, start=1):
                name = uploaded_file.name
                progress.progress((position - 1) / total, text=f"Indexing {position} / {total} · {name}")
                data = uploaded_file.getvalue()
                suffix = Path(name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                    handle.write(data)
                    temp_path = Path(handle.name)
                try:
                    count, message = ingest_path(
                        temp_path, uploaded_signature(data), force=reindex_uploads
                    )
                    if message == "already indexed":
                        tally["skipped"] += 1
                        details.append(f"⏭ {name} — already indexed")
                    elif message == "no readable text found":
                        tally["empty"] += 1
                        details.append(f"⚠ {name} — no readable text")
                    else:
                        tally["reindexed" if reindex_uploads else "indexed"] += 1
                        chunks_added += count
                        details.append(f"✓ {name} — {count} chunks")
                except Exception as exc:
                    tally["failed"] += 1
                    details.append(f"✗ {name} — {exc}")
                finally:
                    temp_path.unlink(missing_ok=True)
                progress.progress(position / total, text=f"Indexed {position} / {total}")
            progress.empty()

            labels = [
                ("indexed", "indexed"),
                ("reindexed", "re-indexed"),
                ("skipped", "skipped"),
                ("empty", "empty"),
                ("failed", "failed"),
            ]
            parts = [f"{tally[key]} {word}" for key, word in labels if tally[key]]
            summary = " · ".join(parts) + f" · {chunks_added} chunks added"
            (st.warning if tally["failed"] else st.success)(summary)
            with st.expander(f"Per-file detail ({total})", expanded=bool(tally["failed"])):
                st.markdown("\n".join(f"- {line}" for line in details))

# Defaults so every role path has these names defined even when its tier hides
# the control that would otherwise set them.
cag_enabled = True
min_similarity = 0.97
dllm_enabled = False
synth_enabled = False
answer_mode = "Grounded"

with st.sidebar:
    st.header("Active path")
    st.radio(
        "Role",
        ROLE_NAMES,
        key="role",
        format_func=lambda name: f"{ROLES[name]['glyph']} {name}",
        label_visibility="collapsed",
    )
    st.markdown(
        f'<div class="cls-rolecard" style="--role:{role_accent}">'
        f'<div class="cls-rolehead">{role_meta["glyph"]} {active_role}</div>'
        f'<div class="cls-roletag">{role_meta["tagline"]}</div></div>',
        unsafe_allow_html=True,
    )
    st.header("Frontend bridge")
    st.caption(f"Streamlit → {'FastAPI' if USE_API_BACKEND else 'embedded service'}")
    st.code(f"{API_URL}/v1/query\n{API_URL}/v1/chat/completions\n{API_URL}/v1/dllm/status", language="text")
    dllm_state = "online" if dllm_api_online else "offline"
    carrier_name = dllm_endpoint.get("carrier", "OpenRouter")
    st.caption(
        f"Inference carrier · {carrier_name} · {dllm_endpoint.get('model', DLLM_MODEL)} · {dllm_state}"
    )

    if role_can("index"):
        _render_corpus_admin()

    # Read-only corpus visibility for every operating tier; the bare User path
    # stays a pure ask-and-read surface, so it sees no store internals.
    if active_role != "User":
        st.header("Evidence Store")
        store_rows = evidence_breakdown(get_collection())
        st.metric("Indexed chunks", collection_count(get_collection()))
        if store_rows:
            st.caption(f"across {len(store_rows)} document(s)")
        if role_can("reset") and st.button("Reset Chroma index", use_container_width=True):
            reset_collection()
            st.success("Evidence Store reset (CAG cache cleared too). Re-index to query again.")

    if role_can("cag_tune"):
        st.header("♻ CAG Layer")
        st.caption("Semantic cache of prior question → retrieved evidence.")
        cag_enabled = st.toggle("Reuse cached evidence", value=True)
        min_similarity = st.slider("Min similarity for a cache hit", 0.80, 1.00, 0.97, 0.01)
        st.metric("Cached queries", get_cache().count())
        if st.button("Clear answer cache", use_container_width=True):
            get_cache().clear()
            st.success("CAG cache cleared.")
    elif role_can("cag_toggle"):
        st.header("♻ CAG Layer")
        st.caption("Reuse evidence from prior identical questions.")
        cag_enabled = st.toggle("Reuse cached evidence", value=True)
        st.metric("Cached queries", get_cache().count())

    get_cache().distance_max = 1.0 - min_similarity

    if role_can("dllm"):
        carrier_name = dllm_endpoint.get("carrier", "OpenRouter")
        model_name = display_model_name(DLLM_MODEL)
        st.header("💬 Inference carrier")
        st.caption(
            f"{carrier_name} · {DLLM_MODEL}. The carrier writes the optional synthesis only; "
            "document names, pages, and evidence rows always come from Chroma retrieval."
        )
        # Single on/off for carrier synthesis — the generative RAG path, ON by default once the
        # carrier is ready (key present + reachable).
        synth_enabled = st.toggle(
            f"Synthesize answer with {model_name}",
            value=dllm_api_online,
            disabled=not dllm_api_online,
            help="Reads the question + retrieval evidence and writes a direct answer. Off falls "
                 "back to deterministic extraction.",
        )
        # Carrier scope: Grounded (default) keeps the strict trust contract — answer only from the
        # indexed documents, else refuse. Hybrid lets the carrier also answer general / natural-
        # language questions from its own knowledge when the documents don't cover them.
        answer_mode = st.radio(
            "Answer mode",
            ["Grounded", "Hybrid"],
            horizontal=True,
            disabled=not (dllm_api_online and synth_enabled),
            help="Grounded · answers strictly from the indexed documents (refuses otherwise). "
                 "Hybrid · also answers general questions from the model's own knowledge.",
        )
        st.caption(
            "**Grounded** answers only from your indexed documents. **Hybrid** also answers "
            "natural-language / general questions from the model when the documents don't cover them."
        )
        # Secondary, optional: let the same carrier also repair PDF extraction artifacts in the
        # grounded bullets. Off by default; orthogonal to the generative answer above.
        dllm_enabled = st.checkbox(
            "Also clean extraction with carrier",
            value=False,
            disabled=not dllm_api_online,
            help="Applies a guarded in-place correction to the extractive bullets (never invents "
                 "facts). Independent of the generative answer toggle.",
        )
        if not dllm_api_online:
            st.caption(dllm_endpoint.get("detail", "Carrier offline — retrieval evidence and extraction still work."))

    if not role_can("index") and not role_can("upload"):
        st.markdown(
            '<div class="cls-rolelock">📚 Corpus is curated by an administrator.</div>',
            unsafe_allow_html=True,
        )

# Full-width drag-and-drop batch upload — roomy drop target for many files at once,
# shown above the ask/answer columns for upload-capable tiers (Admin / Scientist).
if role_can("upload"):
    _render_upload_section()
    st.divider()

left, right = st.columns([1.05, 1], gap="large")

with left:
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
        height=240,
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

    # Top-K is a precision knob — only the tuning tiers (Admin / Scientist) see it;
    # operating tiers (Staff / User) run on a sensible fixed default.
    top_k = st.slider("Top-K chunks", 3, 12, 8) if role_can("tune") else 8

    if st.button("Search IVU Manual", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            # System 2: the RAG+CAG backend returns the instant clean parsed answer.
            try:
                result = query_backend(query, top_k, cag_enabled, min_similarity)
            except RuntimeError as exc:
                st.error(str(exc))
                result = None
            if result is None:
                pass
            elif not result["rows"]:
                st.error("No indexed chunks found. Click 'Index IVU manual' first.")
            else:
                st.session_state["last_query"] = query.strip()
                st.session_state["last_category"] = result["category"]
                st.session_state["last_rows"] = result["rows"]
                st.session_state["last_answer"] = result["answer"]
                st.session_state["last_from_cache"] = result["from_cache"]
                st.session_state["last_similarity"] = result["similarity"]
                # Carrier cleanup is separate from synthesis. If enabled, arm it only when
                # the instant text shows artifacts worth correcting.
                activate, reason = needs_correction(result["answer"])
                st.session_state["last_dllm"] = {
                    "status": "pending" if (activate and result["answer"]) else "dormant",
                    "reason": reason,
                    "text": None,
                }
                # Generative answer (on by default when the carrier is ready): arm it for this
                # query when the toggle is on, so the right column synthesises once and caches
                # the text across reruns.
                st.session_state["last_synth"] = {
                    "status": "pending" if synth_enabled else "off",
                    "text": None,
                    "strict": answer_mode == "Grounded",
                }

    if role_can("eval") and st.button("Run graded offline checks", use_container_width=True):
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

        carrier_name = dllm_endpoint.get("carrier", "OpenRouter")
        model_name = display_model_name(DLLM_MODEL)
        # Mode this query was run under (armed at submit time), so the badge stays truthful across reruns.
        run_grounded = st.session_state.get("last_synth", {}).get("strict", True)
        mode_label = "grounded" if run_grounded else "hybrid"
        if dllm_api_online and synth_enabled:
            inference_badge = (
                '<span class="cls-badge" style="--hue:#6aa9ff">'
                f'💬 Synthesis on · {mode_label} · {html.escape(model_name)}</span>'
            )
        else:
            inference_badge = '<span class="cls-q">💬 synthesis off — extraction only</span>'

        st.markdown("### ✨ Answer")
        st.markdown(
            '<div class="cls-statusline">'
            f'<span>Question type · </span><span class="cls-badge" style="--hue:{hue}">'
            f'{meta["glyph"]} {meta["label"]}</span>'
            f'{cag_badge}'
            f'{inference_badge}'
            '</div>',
            unsafe_allow_html=True,
        )

        stored_query = st.session_state.get("last_query", "")

        def card(sentences: list[str]) -> str:
            body = "<br>".join(f"• {decorate(s, category, stored_query)}" for s in sentences)
            return f"{card_open}{body}</div>"

        # Generative RAG answer (on by default when the carrier is ready): synthesised from the
        # retrieved context and shown as the headline answer above the trusted extractive bullets.
        # Runs once per query, then caches
        # the text in session state so reruns (chip clicks, toggles) don't re-call the API.
        synth = st.session_state.get("last_synth", {"status": "off"})
        synth_display_enabled = bool(dllm_api_online and synth_enabled)
        if synth_display_enabled and synth.get("status") == "off":
            synth = {"status": "pending", "text": None, "strict": synth.get("strict", answer_mode == "Grounded")}
            st.session_state["last_synth"] = synth

        # Mode this query was armed under: drives grounding strictness and how much provenance we surface.
        run_strict = synth.get("strict", True)

        if synth.get("status") == "pending" and synth_display_enabled:
            with st.spinner("💬 Synthesizing answer…"):
                try:
                    synth_text = generate_answer(stored_query, rows, model=DLLM_MODEL, grounded=run_strict)
                    synth = (
                        {"status": "done", "text": synth_text, "strict": run_strict,
                         "grounded": answer_numbers_grounded(synth_text, rows)}
                        if synth_text
                        else {"status": "empty", "text": None, "strict": run_strict}
                    )
                except Exception as exc:
                    synth = {"status": "error", "text": str(exc), "strict": run_strict}
            st.session_state["last_synth"] = synth

        if synth_display_enabled and synth.get("status") == "done" and synth.get("text"):
            cite_hint = (
                '<span class="cls-q">citations [n] map to retrieval evidence rows below</span>'
                if run_strict else ""
            )
            st.markdown(
                '<div class="cls-statusline">'
                f'<span class="cls-badge" style="--hue:#6aa9ff">💬 Carrier synthesis · '
                f'{html.escape(carrier_name)} · {html.escape(model_name)}</span>'
                f'{cite_hint}'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown(synth["text"])
            if run_strict:
                if not synth.get("grounded", True):
                    st.caption("⚠ Contains numbers not found verbatim in the retrieved context — "
                               "verify against the retrieval evidence below.")
                st.caption(
                    f"💬 Generated by the {carrier_name} inference carrier from the retrieval evidence; "
                    "the deterministic extraction follows for audit."
                )
            else:
                st.caption(
                    f"💬 Hybrid answer from the {carrier_name} carrier — uses your indexed documents "
                    "when they're relevant, otherwise the model's own knowledge."
                )
        elif synth_display_enabled and synth.get("status") == "error":
            st.caption(f"💬 Carrier synthesis failed — showing deterministic extraction. ({synth.get('text')})")
        elif synth_display_enabled and synth.get("status") == "empty":
            st.caption("💬 Carrier returned no synthesis — showing deterministic extraction.")

        if answer:
            dllm = st.session_state.get("last_dllm", {"status": "dormant"})
            st.markdown(
                '<span class="cls-badge" style="--hue:#6fd58a">⚡ Deterministic extraction · RAG/CAG</span>',
                unsafe_allow_html=True,
            )
            placeholder = st.empty()

            if dllm.get("status") == "corrected" and dllm.get("text"):
                # Settled from a prior run — render directly, no flash, no model call.
                placeholder.markdown(card(dllm["text"]), unsafe_allow_html=True)
                st.caption(f"✎ Extraction cleaned by {carrier_name} — {dllm.get('reason')}.")
            elif dllm.get("status") == "pending" and dllm_api_online and dllm_enabled:
                # Instant text appears first; the carrier then corrects it in place.
                # Re-checked for grounding at the end; reverts on drift.
                placeholder.markdown(card(answer), unsafe_allow_html=True)
                try:
                    corrected = correct_with_dllm_api(answer)
                    if corrected and validate_correction(" ".join(corrected), answer):
                        placeholder.markdown(card(corrected), unsafe_allow_html=True)
                        st.session_state["last_dllm"] = {
                            "status": "corrected", "reason": dllm.get("reason"), "text": corrected
                        }
                        st.caption(f"✎ Extraction cleaned by {carrier_name} — {dllm.get('reason')}.")
                    else:
                        placeholder.markdown(card(answer), unsafe_allow_html=True)
                        st.session_state["last_dllm"] = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                        st.caption("⚡ Carrier cleanup drifted, so the deterministic extraction was kept.")
                except Exception:
                    placeholder.markdown(card(answer), unsafe_allow_html=True)
                    st.session_state["last_dllm"] = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                    st.caption("⚡ Carrier cleanup unavailable, so the deterministic extraction was kept.")
            else:
                placeholder.markdown(card(answer), unsafe_allow_html=True)
                st.caption(
                    "⚡ Deterministic extraction from retrieval evidence. No inference carrier modified this text."
                )
        else:
            synth_answered = (not run_strict) and synth.get("status") == "done" and bool(synth.get("text"))
            if synth_answered:
                st.caption(
                    "⚡ No verbatim passage matched — the Hybrid answer above came from the carrier's "
                    "own knowledge. Retrieval evidence below for reference."
                )
            else:
                st.warning("No strong sentence-level extraction found. Review the retrieval evidence below.")

        evidence_summary, mixed_evidence = retrieval_evidence_summary(rows)
        st.markdown("### Retrieval evidence")
        st.markdown(
            '<div class="cls-evidence-strip">'
            '<div class="cls-evidence-title">Chroma-ranked evidence rows, not model output</div>'
            f'<div class="cls-evidence-copy">{evidence_summary}</div>'
            f'<div class="cls-evidence-copy">Carrier citations like [1] refer to these row numbers; '
            f'extractive citations name the source document and page directly.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        if mixed_evidence:
            st.caption(
                "Mixed evidence set: retrieved rows came from multiple indexed documents. "
                "For broad or misspelled prompts, tighten the query or lower Top-K for a cleaner evidence set."
            )
        hit_terms = stored_query
        for index, row in enumerate(rows, start=1):
            meta = row["metadata"]
            label = (
                f"Evidence {index} · {meta.get('source', 'source')}, "
                f"page {meta.get('page', '?')} — {meta.get('section', 'section')}"
            )
            with st.expander(label, expanded=index <= 2):
                st.markdown(
                    f'<div class="cls-snippet">{decorate(row["document"], category, hit_terms)}</div>',
                    unsafe_allow_html=True,
                )
    else:
        render_evidence_store(evidence_breakdown(get_collection()))

eval_rows = st.session_state.get("eval_rows", [])
if eval_rows:
    st.divider()
    st.subheader("🧪 Graded offline query checks")
    st.dataframe(eval_rows, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    f"{APP_VERSION} · DocuSearch-style RAG/CAG — **Retrieval Encoder** (HashEmbedder, the one active "
    "*local* model) → **CAG Layer** / **Evidence Store** (ChromaDB) → deterministic clean-parse + "
    f"highlighting. The **{DLLM_MODEL}** answer (via the {dllm_endpoint.get('carrier', 'OpenRouter')} "
    "inference carrier) is **on by default** when a key is configured. In **Grounded** mode (default) "
    "it synthesizes strictly from retrieval evidence rows; switch **Answer mode** to **Hybrid** to also "
    "answer general / natural-language questions from the model's own knowledge. Source documents and "
    "pages always come from Chroma, not the carrier. Toggle synthesis off for deterministic extraction "
    "only; fully offline-capable with no carrier."
)
