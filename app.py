"""Streamlit dual-UI — the browser-facing entry point for the CLS RAG+CAG system.

Two UIs are served from this single file, selected on the landing page:

  Full App  — admin + user roles, corpus admin, file upload, precision controls,
               graded evaluation, and optional LLM synthesis.  The role tier
               (Explorer / Analyst / Engineer) only controls *which widgets are
               visible*; it is not a security boundary.

  Ask Lane  — a fast, minimal chat interface for non-technical users.  No LLM,
               no engineering telemetry — instant retrievaland cited bullets only.

Both UIs call the same backend via ``cls_service.py``.  All retrieval, ranking,
CAG cache look-up, and answer assembly happen there and in ``cls_backend/``; this
file is pure presentation.

Key environment switches (set in ``cls.env`` before launch):
    CLS_RETRIEVAL_ONLY=1  — disables the generative carrier (default ON).
    CLS_KEYWORD_ONLY=1    — disables semantic vector search; lexical only (default ON).

Start the app:  ./scripts/launch_cls.sh   (creates .venv, installs deps, opens browser)
"""

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
    DEFAULT_DOCUMENT_DOMAIN,
    DEFAULT_DLLM_MODEL,
    DEFAULT_DOCUMENTS_DIR,
    KEYWORD_ONLY_RETRIEVAL,
    RESEARCH_SCOPES,
    RETRIEVAL_ONLY,
    add_research_scope,
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
    stream_generate_answer,
    get_cache,
    get_collection,
    ingest_path,
    flush_cag_cache,
    reset_collection,
    store_status,
    uploaded_signature,
    warm_keyword_index,
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
    HUE_AMBER,
    HUE_GREEN,
    HUE_ORANGE,
    HUE_RED,
    category_meta,
    classify_query,
    decorate,
    glow_css,
)
from cls_backend.query_repair import repair_query


# --------------------------------------------------------------------------- #
# Role tiers. The left sidebar delegates one of two paths; each role reshapes
# the screen by capability. Corpus lifecycle (indexing/reset) is admin-specific;
# lower tiers narrow down to a clean ask-and-read surface. This is a visual +
# permission separation only — it does not change how chunks are embedded.
# --------------------------------------------------------------------------- #
ROLES: dict[str, dict] = {
    "Admin": {
        "glyph": "🛡",
        "accent": "#ff4d6d",
        "tagline": "Full control — corpus lifecycle, tuning, and evaluation.",
        "caps": {"index", "upload", "reindex", "reset", "tune", "cag_tune", "dllm", "eval", "scopes"},
    },
    "User": {
        "glyph": "👤",
        "accent": "#6fd58a",
        "tagline": "Ask a question and read the cited answer.",
        "caps": set(),
    },
}
ROLE_NAMES = list(ROLES)


def _role_has(role_name: str, capability: str) -> bool:
    """Capability check by role name — usable before the active role is resolved
    (e.g. inside the role-agnostic Ask Lane, which still reads st.session_state)."""
    return capability in ROLES.get(role_name, {}).get("caps", set())
# Architecture: retrieval is primary. During the temporary speed-first phase,
# CLS_RETRIEVAL_ONLY disables every LLM augmentation path and CLS_KEYWORD_ONLY skips
# semantic query embedding for deterministic keyword retrieval.
DLLM_MODEL = DEFAULT_DLLM_MODEL

st.set_page_config(
    page_title=f"CLS RAG+CAG Prototype · {APP_VERSION}",
    page_icon="🔬",
    layout="wide",
)



def render_evidence_store(rows: list[dict]) -> None:
    """Idle-state panel: visualize what is indexed, one bar per document."""
    st.markdown("### 📚 Evidence Store")
    if not rows:
        st.info("Index research documents, then run a search to see scored retrieval evidence.")
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


# RESEARCH_SCOPES now lives in cls_config (shared across frontends); imported below.


def query_backend(query: str, top_k: int, cache_enabled: bool, min_similarity: float, metadata_filter: dict | None = None, debate_enabled: bool = False) -> dict:
    search_query = repair_query(query)["search"]
    if USE_API_BACKEND:
        return post_json(
            "/v1/query",
            {
                "query": search_query,
                "top_k": top_k,
                "cache_enabled": cache_enabled,
                "min_similarity": min_similarity,
                "metadata_filter": metadata_filter,
                "debate_enabled": debate_enabled,
                "keyword_only": KEYWORD_ONLY_RETRIEVAL,
            },
        )
    return ask_manual(
        search_query,
        top_k=top_k,
        cache_enabled=cache_enabled,
        min_similarity=min_similarity,
        metadata_filter=metadata_filter,
        debate_enabled=debate_enabled,
        keyword_only=KEYWORD_ONLY_RETRIEVAL,
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
        "disabled": RETRIEVAL_ONLY,
        "retrieval_only": RETRIEVAL_ONLY,
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


def _answer_card_html(sentences: list[str], category: str, query: str, hue: str, glow: str) -> str:
    """Build the deterministic extraction answer card as a single HTML block."""
    card_open = f'<div class="cls-answer" style="--hue:{hue};--glow:{glow}">'
    body = "<br>".join(f"• {decorate(s, category, query)}" for s in sentences)
    return f"{card_open}{body}</div>"


def render_answer_component(
    stored_query: str,
    rows: list[dict],
    answer: list[str],
    *,
    dllm_endpoint: dict,
    dllm_api_online: bool,
    dllm_enabled: bool,
    synth_enabled: bool,
    answer_mode: str,
) -> None:
    """Single one-go answer component for the scientist / operator prompt."""
    top_score = rows[0]["score"]
    category = st.session_state.get("last_category", "general")
    meta = category_meta(category)
    hue = meta["hue"]
    glow = glow_css(hue, top_score)
    carrier_name = dllm_endpoint.get("carrier", "OpenRouter")

    st.markdown("### ✨ Answer")

    synth = st.session_state.get("last_synth", {"status": "off"})
    synth_display_enabled = bool(dllm_api_online and synth_enabled)
    if synth_display_enabled and synth.get("status") == "off":
        synth = {"status": "pending", "text": None, "strict": synth.get("strict", answer_mode == "Grounded")}
        st.session_state["last_synth"] = synth

    run_strict = synth.get("strict", True)
    if synth.get("status") == "pending" and synth_display_enabled:
        with st.spinner("💬 Answering…"):
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
        with st.container(border=True):
            st.markdown(synth["text"])
        if run_strict and not synth.get("grounded", True):
            st.caption("⚠ Contains numbers not found verbatim in the retrieved context — "
                       "verify against the retrieval evidence below.")
    elif synth_display_enabled and synth.get("status") == "error":
        st.caption(f"💬 Couldn't generate an answer — showing the extracted passages instead. ({synth.get('text')})")
    elif synth_display_enabled and synth.get("status") == "empty":
        st.caption("💬 The model returned no answer — showing the extracted passages instead.")

    dllm = st.session_state.get("last_dllm", {"status": "dormant"})
    extraction_sentences = answer
    extraction_caption = (
        "⚡ Deterministic extraction from retrieval evidence. No inference carrier modified this text."
    )
    if answer and dllm.get("status") == "pending" and dllm_api_online and dllm_enabled:
        with st.spinner("✎ Cleaning extraction…"):
            try:
                corrected = correct_with_dllm_api(answer)
                if corrected and validate_correction(" ".join(corrected), answer):
                    extraction_sentences = corrected
                    dllm = {"status": "corrected", "reason": dllm.get("reason"), "text": corrected}
                    extraction_caption = f"✎ Extraction cleaned by {carrier_name} — {dllm.get('reason')}."
                else:
                    dllm = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                    extraction_caption = "⚡ Carrier cleanup drifted, so the deterministic extraction was kept."
            except Exception:
                dllm = {"status": "reverted", "reason": dllm.get("reason"), "text": None}
                extraction_caption = "⚡ Carrier cleanup unavailable, so the deterministic extraction was kept."
        st.session_state["last_dllm"] = dllm
    elif answer and dllm.get("status") == "corrected" and dllm.get("text"):
        extraction_sentences = dllm["text"]
        extraction_caption = f"✎ Extraction cleaned by {carrier_name} — {dllm.get('reason')}."

    answer_placeholder = st.empty()
    if extraction_sentences:
        answer_placeholder.markdown(
            '<span class="cls-badge" style="--hue:#6fd58a">⚡ Deterministic extraction · RAG/CAG</span>'
            + _answer_card_html(extraction_sentences, category, stored_query, hue, glow)
            + f'<div class="cls-mode-note">{extraction_caption}</div>',
            unsafe_allow_html=True,
        )
    else:
        synth_answered = synth.get("status") == "done" and bool(synth.get("text"))
        if not synth_answered:
            answer_placeholder.markdown(
                '<div class="cls-mode-note">The indexed documents don\'t contain a direct passage for this — '
                'the retrieval evidence below shows the closest matches.</div>',
                unsafe_allow_html=True,
            )

    evidence_summary, mixed_evidence = retrieval_evidence_summary(rows)
    st.markdown("### Retrieval evidence")
    if st.session_state.get("last_retrieval_mode") == "lexical_fallback":
        st.warning(
            "Semantic retrieval is offline, so these rows were ranked with the deterministic "
            "text fallback. Start Ollama with `nomic-embed-text` to restore semantic ranking."
        )
    citation_note = (
        "Retrieval-only mode: row numbers are evidence labels; extractive citations name "
        "the source document and page directly."
        if RETRIEVAL_ONLY
        else "Carrier citations like [1] refer to these row numbers; extractive citations "
             "name the source document and page directly."
    )
    st.markdown(
        '<div class="cls-evidence-strip">'
        '<div class="cls-evidence-title">Chroma-ranked evidence rows, not model output</div>'
        f'<div class="cls-evidence-copy">{evidence_summary}</div>'
        f'<div class="cls-evidence-copy">{citation_note}</div>'
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


# Bright, llama.cui-inspired chat lane: clean light surface, source chips,
# native chat bubbles, bottom send-arrow input bar.
_BRIGHT_LANE_CSS = """
<style>
:root {
    --bg:     #faf9f7;
    --panel:  #ffffff;
    --ink:    #2a2730;
    --muted:  #6b6470;
    --faint:  #a89fa8;
    --line:   #e8e4de;
    --accent: #ff8a3d;
    --magenta:      #c2185b;
    --magenta-ink:  #ad1457;
    --magenta-line: #f3aac8;
    --magenta-soft: #fde7f0;
    --magenta-hot:  #fbd0e3;
}
html, body, #root { background: var(--bg) !important; }
.stApp,
div[data-testid="stAppViewContainer"],
section[data-testid="stMain"] {
    background:
        radial-gradient(1000px 320px at 50% -6%, rgba(255,138,61,0.06) 0%, rgba(255,138,61,0) 62%),
        var(--bg) !important;
    color: var(--ink) !important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp p,.stApp li,.stApp label,.stMarkdown,
div[data-testid="stText"] { color: var(--ink) !important; }
.stCaptionContainer,.stApp small { color: var(--muted) !important; }

/* Sidebar — light, slim orange edge */
section[data-testid="stSidebar"] {
    background: #f4f1ec !important;
    border-right: 3px solid var(--accent) !important;
}
section[data-testid="stSidebar"] * { color: var(--ink) !important; }

/* Research Scope multiselect — magenta lane. Force the BaseWeb control stack onto
   light surfaces with a magenta accent so the scope picker reads as its own lane. */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
    background: var(--panel) !important;
    border-color: var(--magenta-line) !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    border-radius: 12px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
    border-color: var(--magenta) !important;
    box-shadow: 0 0 0 2px rgba(194,24,91,0.18) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
    background: transparent !important;
    color: var(--ink) !important;
    caret-color: var(--magenta) !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] input::placeholder {
    color: var(--faint) !important;
}
/* Selected scope pills */
section[data-testid="stSidebar"] div[data-baseweb="tag"] {
    background: var(--magenta-soft) !important;
    color: var(--magenta-ink) !important;
    border: 1px solid var(--magenta-line) !important;
    border-radius: 999px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="tag"] span,
section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {
    color: var(--magenta) !important;
    fill: currentColor !important;
}

/* The options menu is portalled outside the sidebar DOM. Target several BaseWeb
   DOM shapes so the popup is a light magenta surface (this is the panel that was
   rendering dark), not just the inner listbox. */
div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul[role="listbox"],
ul[role="listbox"],
div[role="listbox"] {
    background: var(--panel) !important;
    border: 1px solid var(--magenta-line) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 26px rgba(173,20,87,0.18) !important;
}
li[role="option"],
div[role="option"] {
    background: transparent !important;
    color: var(--ink) !important;
}
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background: var(--magenta-soft) !important;
    color: var(--magenta) !important;
}
li[role="option"]:hover,
div[role="option"]:hover {
    background: var(--magenta-hot) !important;
    color: var(--magenta-ink) !important;
}

/* Hero header */
.lane-hero { text-align:center; padding: 1.6rem 1rem 0.4rem; }
.lane-hero h1 { font-size: clamp(1.4rem, 3vw, 2rem); margin: 0; color: var(--ink); }
.lane-hero p  { color: var(--muted); margin: 0.4rem 0 0; font-size: 0.95rem; }
.lane-hero .lane-rule {
    height: 3px; border-radius: 3px; margin: 0.9rem auto 0; max-width: 620px;
    background: linear-gradient(90deg,#ff4d6d,#ff8a3d,#ffc24b,#6fd58a,#6aa9ff);
}

/* Native chat bubbles -> clean cards */
div[data-testid="stChatMessage"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(120,80,60,0.06);
    margin-bottom: 0.7rem;
}
div[data-testid="stChatMessage"] * { color: var(--ink) !important; }

/* Source chips — llama.cui pill row */
.lane-chips { display:flex; flex-wrap:wrap; gap:6px; margin: 0 0 0.7rem; }
.lane-chip {
    font-size: 0.74rem; font-weight: 600; line-height: 1;
    padding: 5px 10px; border-radius: 999px;
    background: #f1ede7; border: 1px solid var(--line); color: var(--muted);
    white-space: nowrap;
}

/* Grounded evidence bullets — escaped text rendered at a normal, fixed size so
   stray markdown in the source (a leading "# ", pipes, asterisks) can never blow
   the type up into a heading. */
.lane-evidence { margin: 0.2rem 0 0.4rem 1.15rem; padding: 0; }
.lane-evidence li {
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    line-height: 1.5;
    margin: 0 0 0.4rem;
}

/* Bottom chat input — rounded send-arrow bar, light blue */
div[data-testid="stChatInput"] {
    background: #eaf4fb !important;
    border: 1px solid #bfe0f3 !important;
    border-radius: 14px !important;
    box-shadow: 0 6px 22px rgba(80,140,180,0.14);
}
div[data-testid="stChatInput"] > div {
    background: #eaf4fb !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"] [data-baseweb="textarea"],
div[data-testid="stChatInput"] [data-baseweb="base-input"] {
    background: #eaf4fb !important;
    border: 0 !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea {
    background: #eaf4fb !important;
    color: var(--ink) !important;
    caret-color: var(--ink) !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea::placeholder { color: #7fa8c4 !important; }
div[data-testid="stChatInputSubmitButton"],
div[data-testid="stChatInput"] button {
    background: #d6ecf7 !important;
    color: #2f7fb0 !important;
    border: 1px solid #bfe0f3 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
button[data-testid="stChatInputSubmitButton"] svg,
div[data-testid="stChatInput"] button svg {
    color: #2f7fb0 !important;
    fill: currentColor !important;
}
button[data-testid="stChatInputSubmitButton"]:disabled,
div[data-testid="stChatInput"] button:disabled {
    background: #eaf4fb !important;
    color: #9bb8cc !important;
    border-color: #bfe0f3 !important;
    opacity: 1 !important;
}
button[data-testid="stChatInputSubmitButton"]:disabled svg,
div[data-testid="stChatInput"] button:disabled svg {
    color: #9bb8cc !important;
}
div[data-testid="stChatInput"] button:hover {
    border-color: #2f7fb0 !important;
    background: #c8e4f4 !important;
}
/* Main scroll column — give it enough bottom padding so the last message can
   scroll fully above the fixed input bar, not behind it. */
section[data-testid="stMain"] .block-container,
div[data-testid="stMainBlockContainer"] {
    padding-bottom: 120px !important;
}

/* Fixed bottom bar — opaque on every layer. Do NOT override position/left/right:
   Streamlit already places stBottom correctly relative to the sidebar. Overriding
   left:0 makes the bar bleed behind the sidebar panel. Just lift z-index and set
   backgrounds so it masks the message list above it. */
div[data-testid="stBottom"] {
    background: var(--bg) !important;
    border-top: 1px solid var(--line) !important;
    z-index: 100 !important;
}
div[data-testid="stBottom"] > div,
div[data-testid="stBottomBlockContainer"] {
    background: var(--bg) !important;
}

/* Hide Streamlit's native sidebar collapse/expand toggle (the << >> chevron).
   This app uses the custom floating ← Back dock for navigation, so the native
   toggle is redundant and clutters the top-left corner near the input area. */
button[data-testid="stSidebarCollapsedControl"],
button[data-testid="stBaseButton-headerNoPadding"],
section[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

/* Buttons */
.stButton > button,
button[data-testid="stBaseButton-secondary"] {
    background: var(--panel) !important; color: var(--ink) !important;
    border: 1px solid var(--line) !important; border-radius: 10px !important;
}
.stButton > button:hover { border-color: var(--accent) !important; }
div[data-testid="stAlert"] { background: #fbf7f2 !important; border-color: var(--line) !important; }
.lane-empty { padding: 2.6rem 1rem; text-align:center; color: var(--faint); font-size: 0.95rem; }
</style>
"""


def _source_chips(rows: list[dict]) -> str:
    """Build the llama.cui-style pill row of distinct sources for a turn."""
    seen: set[tuple] = set()
    chips: list[str] = []
    for row in rows:
        meta = row.get("metadata", {})
        src = str(meta.get("source", "")).rsplit(".", 1)[0]
        page = meta.get("page")
        key = (src, page)
        if src and key not in seen:
            seen.add(key)
            label = html.escape(src) + (f" · p{page}" if page else "")
            chips.append(f'<span class="lane-chip">{label}</span>')
    return f'<div class="lane-chips">{"".join(chips[:6])}</div>' if chips else ""


_WEAK_SCORE_THRESHOLD = 0.45  # below this, retrieved rows are too weak to ground an answer


def _fallback_tier(result: dict) -> str:
    """Three-tier signal: 'grounded' | 'weak' | 'general'.

    grounded — extractive bullets found; RAG/CAG answer is self-sufficient.
    weak     — rows retrieved but score < threshold and no bullets; LLM augments with context.
    general  — nothing retrieved or score near-zero; LLM answers from its own knowledge.
    """
    if result.get("answer"):
        return "grounded"
    rows = result.get("rows", [])
    if not rows:
        return "general"
    top_score = rows[0].get("score", 0.0)
    return "weak" if top_score >= _WEAK_SCORE_THRESHOLD else "general"


def _evidence_bullets_html(sentences: list[str]) -> str:
    """Render grounded extractive sentences as an escaped HTML list.

    Raw PDF evidence can begin with markdown block syntax — e.g. a literal "# "
    section marker carried over from the source — which st.markdown parses as a
    heading *inside* the list item ("- # foo"), blowing the font up to H1 size.
    html.escape neutralises all markdown/HTML so the evidence renders verbatim at
    a normal size. Returns "" when nothing is left after stripping.
    """
    items = "".join(
        f"<li>{html.escape(clean)}</li>"
        for sentence in sentences
        if (clean := sentence.split(" [Source:")[0].strip())
    )
    return f'<ul class="lane-evidence">{items}</ul>' if items else ""


def _render_turn(message: dict, dllm_online: bool) -> None:
    """Render one stored chat turn: user question + assistant answer with chips."""
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(message["query"])
    with st.chat_message("assistant", avatar="🔬"):
        scope_results = message.get("scope_results")
        if scope_results:
            for scope, result in scope_results.items():
                _render_scope_summary(scope, result, dllm_online=dllm_online)
            return
        chips = _source_chips(message.get("rows", []))
        if chips:
            st.markdown(chips, unsafe_allow_html=True)
        answer = message.get("answer", [])
        if answer:
            # Pass 1 — grounded extractive bullets. Provenance lives in the chips above.
            # Escaped HTML, not markdown: raw evidence may start with "# " etc.
            st.markdown(_evidence_bullets_html(answer), unsafe_allow_html=True)
        # Pass 2 — stored LLM augmentation (present when DLLM was online and RAG couldn't fully answer).
        augmentation = message.get("augmentation")
        aug_label = message.get("augmentation_label", "💬 AI augmentation · grounded synthesis")
        if augmentation:
            st.divider()
            st.caption(aug_label)
            st.markdown(augmentation)
        elif not answer:
            st.markdown("_Nothing relevant found — try rephrasing._")


def _render_scope_summary(scope: str, result: dict | None, *, dllm_online: bool) -> None:
    """Compact per-beamline answer card for multi-scope Ask Lane queries."""
    with st.container(border=True):
        st.markdown(
            f'<strong>{html.escape(scope)}</strong>',
            unsafe_allow_html=True,
        )
        if not result or not result.get("rows"):
            st.caption("_Nothing relevant found — try rephrasing._")
            return
        chips = _source_chips(result["rows"])
        if chips:
            st.markdown(chips, unsafe_allow_html=True)
        if result.get("answer"):
            st.markdown(_evidence_bullets_html(result["answer"]), unsafe_allow_html=True)
        else:
            tier = _fallback_tier(result)
            if tier == "weak":
                st.caption("_Weak corpus match — no direct passage found._")
            else:
                st.caption("_Nothing relevant found — try rephrasing._")


def _render_ask_lane() -> None:
    """Chat lane over instant RAG/CAG extraction."""
    dllm_online = dllm_api_status().get("online", False)
    dllm_model = dllm_api_status().get("model", DLLM_MODEL) if dllm_online else None

    if RETRIEVAL_ONLY:
        pass2_label = "keyword retrieval only" if KEYWORD_ONLY_RETRIEVAL else "retrieval only"
    else:
        pass2_label = "RAG/CAG + AI augmentation" if dllm_online else "RAG/CAG · extractive only"
    st.markdown(
        _BRIGHT_LANE_CSS +
        f"""<div class="lane-hero">
          <h1>🔬 CLS Research Documents <span style="opacity:0.4;font-size:1rem">{APP_VERSION}</span></h1>
          <p>Ask a question — {pass2_label}.</p>
          <div class="lane-rule"></div>
        </div>""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if st.button("🔬 Home", use_container_width=True, key="ask_lane_home_btn"):
            st.session_state.pop("ui_mode", None)
            st.rerun()
        st.divider()
        st.header("Research Scope")
        st.caption("Filter retrieval to one or more CLS beamlines.")
        selected_scopes = st.multiselect(
            "Active scopes",
            list(RESEARCH_SCOPES.keys()),
            default=[],
            key="lane_scopes",
            label_visibility="collapsed",
        )
        if not selected_scopes:
            selected_scopes = ["All beamlines"]
        effective_scopes = selected_scopes
        if "All beamlines" in effective_scopes and len(effective_scopes) > 1:
            # All already covers every lane, so keep only the specific selections for comparison.
            effective_scopes = [s for s in effective_scopes if s != "All beamlines"]
            st.caption("ℹ All beamlines covers every lane; querying the selected scopes instead.")
        scope_filters = [(s, RESEARCH_SCOPES[s]) for s in effective_scopes]
        single_scope = len(scope_filters) == 1
        active_mfilter = scope_filters[0][1] if single_scope else None
        st.divider()
        if RETRIEVAL_ONLY:
            st.caption("💬 AI augmentation disabled — retrieval-only mode.")
        elif dllm_online:
            st.caption(f"💬 AI augmentation: {dllm_model}")
        else:
            st.caption("💬 AI augmentation offline — extractive only.")
        st.divider()
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state["lane_messages"] = []
            st.rerun()

    messages: list[dict] = st.session_state.setdefault("lane_messages", [])

    if not messages:
        st.markdown(
            '<div class="lane-empty">Your conversation will appear here — '
            'ask anything about the indexed documents.</div>',
            unsafe_allow_html=True,
        )
    else:
        for message in messages:
            _render_turn(message, dllm_online)

    prompt = st.chat_input("Ask a question")
    if prompt and prompt.strip():
        query = prompt.strip()

        # ------------------------------------------------------------------ #
        # Render the new user message inline (visible during processing).
        # ------------------------------------------------------------------ #
        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(query)

        if single_scope:
            result = None
            augmentation: str | None = None
            aug_label: str | None = None

            with st.chat_message("assistant", avatar="🔬"):
                # ---------------------------------------------------------------- #
                # Pass 1 — instant RAG/CAG: retrieval + extractive bullets.        #
                # ---------------------------------------------------------------- #
                try:
                    result = query_backend(query, 16, True, 0.80, metadata_filter=active_mfilter)
                except RuntimeError as exc:
                    st.error(str(exc))

                if result is not None:
                    tier = _fallback_tier(result)
                    chips = _source_chips(result["rows"])
                    if chips:
                        st.markdown(chips, unsafe_allow_html=True)

                    if result["answer"]:
                        st.markdown(_evidence_bullets_html(result["answer"]), unsafe_allow_html=True)

                    # ---------------------------------------------------------------- #
                    # Pass 2 — streaming LLM fallback, tier-gated.                    #
                    #   grounded → also augment with context (score was good)          #
                    #   weak     → augment with retrieved context, flag as partial     #
                    #   general  → answer from general knowledge, flag clearly         #
                    # ---------------------------------------------------------------- #
                    aug_label = None
                    if dllm_online:
                        if tier == "grounded":
                            aug_label = "💬 AI augmentation · grounded synthesis"
                            aug_grounded = True
                        elif tier == "weak":
                            st.caption("_Weak corpus match — augmenting with available context…_")
                            aug_label = "💬 AI augmentation · partial context"
                            aug_grounded = True
                        else:  # general
                            st.caption("_Not found in indexed documents — answering from general knowledge…_")
                            aug_label = "💬 General knowledge answer · not from corpus"
                            aug_grounded = False

                        st.divider()
                        st.caption(aug_label)
                        try:
                            augmentation = st.write_stream(
                                stream_generate_answer(
                                    query, result["rows"], model=DLLM_MODEL, grounded=aug_grounded
                                )
                            )
                        except Exception as exc:
                            st.caption(f"_Augmentation unavailable: {exc}_")
                            augmentation = None
                    elif tier != "grounded":
                        # DLLM offline and no extractive answer
                        st.markdown("_Nothing relevant found — try rephrasing._")

            if result is not None:
                turn: dict = {
                    "query": query,
                    "rows": result["rows"],
                    "answer": result["answer"],
                }
                if augmentation:
                    turn["augmentation"] = augmentation
                    if aug_label:
                        turn["augmentation_label"] = aug_label
                messages.append(turn)
                st.session_state["lane_rows"] = result["rows"]
                st.session_state["lane_from_cache"] = result["from_cache"]
                st.rerun()
        else:
            # ---------------------------------------------------------------- #
            # Multi-scope mode — run the same query against each selected lane. #
            # No LLM pass: comparison cards are for quick beamline triage.      #
            # ---------------------------------------------------------------- #
            with st.chat_message("assistant", avatar="🔬"):
                st.caption(f"Querying {len(scope_filters)} selected scopes…")
                scope_results: dict[str, dict] = {}
                any_rows = False
                for scope, mfilter in scope_filters:
                    try:
                        res = query_backend(query, 16, True, 0.80, metadata_filter=mfilter)
                    except RuntimeError as exc:
                        st.error(f"{scope}: {exc}")
                        res = None
                    scope_results[scope] = res
                    if res and res.get("rows"):
                        any_rows = True
                    _render_scope_summary(scope, res, dllm_online=dllm_online)

            if scope_results:
                turn = {
                    "query": query,
                    "scope_results": scope_results,
                }
                messages.append(turn)
                first_rows = next(
                    (r["rows"] for r in scope_results.values() if r and r.get("rows")),
                    [],
                )
                st.session_state["lane_rows"] = first_rows
                st.session_state["lane_from_cache"] = any(
                    r.get("from_cache") for r in scope_results.values() if r
                )
                st.rerun()


def _home_gate() -> None:
    """Landing page: pick Full App or Ask Lane. Skipped once a mode is chosen."""
    if st.session_state.get("ui_mode"):
        return

    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown(
            f"""<div class="cls-hero" style="text-align:center;padding:1.6rem 1.5rem 1.3rem">
              <h1 style="font-size:clamp(1.5rem,3vw,2rem)">🔬 CLS Synchrotron Research Query</h1>
              <p style="margin-bottom:0.2rem">Choose how you want to work today.</p>
              <div class="cls-spectrum-rule" style="margin-top:0.9rem"></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """<style>
            .cls-mode-card {
                border-radius:14px; padding:1.2rem 1.3rem; margin:0.6rem 0;
                background:var(--panel); border:1px solid var(--line);
                box-shadow:0 6px 20px rgba(120,80,60,0.08);
            }
            .cls-mode-card h4 { margin:0 0 0.3rem; font-size:1.1rem; color:var(--ink); }
            .cls-mode-card p  { margin:0; font-size:0.88rem; color:var(--muted); line-height:1.45; }
            </style>""",
            unsafe_allow_html=True,
        )

        left_card, right_card = st.columns(2, gap="medium")
        with left_card:
            st.markdown(
                '<div class="cls-mode-card">'
                '<h4>🛡 Full App</h4>'
                '<p>Admin &amp; User roles — corpus admin, upload, eval, and precision controls.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Full App", type="primary", use_container_width=True, key="btn_full"):
                st.session_state["ui_mode"] = "full"
                st.rerun()
        with right_card:
            st.markdown(
                '<div class="cls-mode-card">'
                '<h4>💬 Ask Lane</h4>'
                '<p>Clean ask-and-read surface — just type a question and get a cited answer.</p>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button("Open Ask Lane", use_container_width=True, key="btn_ask"):
                st.session_state["ui_mode"] = "ask_lane"
                st.rerun()

    st.stop()


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

            /* Floating back dock — minimal cross-mode control, independent of role/UI HUD state. */
            .cls-backdock {
                position: fixed; right: 1.1rem; bottom: 1.1rem; z-index: 9999;
                display: flex; align-items: center; gap: 0.45rem;
                padding: 0.3rem; border-radius: 999px;
                background: rgba(255,255,255,0.94);
                border: 1px solid var(--line);
                backdrop-filter: blur(10px);
                box-shadow: 0 10px 28px rgba(120,80,60,0.14);
                user-select: none;
            }
            .cls-backgrip {
                display: inline-flex; align-items: center; justify-content: center;
                width: 1.55rem; height: 1.55rem;
                border-radius: 999px;
                background: #f4f1ec;
                border: 1px solid var(--line);
                color: var(--faint);
                cursor: grab;
                font-size: 0.8rem; line-height: 1;
            }
            .cls-backgrip:active {
                cursor: grabbing;
                background: #efe8df;
            }
            .cls-backbtn {
                display: inline-flex; align-items: center; justify-content: center;
                min-height: 1.55rem; padding: 0 0.8rem;
                border-radius: 999px;
                background: var(--panel);
                border: 1px solid var(--line);
                color: var(--ink) !important;
                font-size: 0.86rem; font-weight: 600;
                text-decoration: none !important;
            }
            .cls-backbtn:hover {
                background: #fff8f2;
                border-color: var(--accent);
                color: var(--accent) !important;
            }
    </style>
    """,
    unsafe_allow_html=True,
)

dllm_endpoint = dllm_api_status()
dllm_api_online = bool(dllm_endpoint.get("online"))
if KEYWORD_ONLY_RETRIEVAL:
    try:
        warm_keyword_index()
    except RuntimeError:
        pass

_home_gate()

if st.session_state.get("ui_mode") == "ask_lane":
    _render_ask_lane()
    st.stop()

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
      <h1>🔬 CLS Synchrotron Research Query <span style="opacity:0.55;font-size:1rem;">{APP_VERSION}</span>
        &nbsp;<span class="cls-badge" style="--hue:{role_accent}">{role_meta['glyph']} {active_role}</span></h1>
      <p>Cited answers from indexed facility manuals and research documents.</p>
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


def _render_db_status_badge() -> None:
    """Colour-coded DB health indicator shown at the top of the Workspace expander.

    Three states:
      Green  — both collections healthy, no write lock held by any process.
      Amber  — an exclusive write lock is held (reset/flush in progress elsewhere).
      Red    — ChromaDB collections unreachable (corrupt store or mid-reset crash).

    Both the Full App and the Ask Lane (separate OS processes) write to the same
    on-disk ChromaDB files.  The amber state is the visual signal that a cross-
    process admin operation is running and queries may block momentarily.
    """
    status = store_status()
    if not status["healthy"]:
        colour  = HUE_RED
        icon    = "◉"
        label   = "DB Error"
        detail  = "ChromaDB collections unreachable — a full reset may be needed."
    elif status["write_locked"]:
        colour  = HUE_AMBER
        icon    = "◉"
        label   = "Write Locked"
        detail  = "Admin operation in progress (another process holds the write guard)."
    else:
        colour  = HUE_GREEN
        icon    = "◉"
        label   = "Healthy"
        detail  = (
            f"{status['evidence_count']} evidence chunks · "
            f"{status['cache_count']} cached queries"
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
        f'<span style="color:{colour};font-size:1.15em;line-height:1">{icon}</span>'
        f'<span style="font-weight:600;font-size:0.9em">{label}</span>'
        f'<span style="color:#888;font-size:0.82em">{detail}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_corpus_admin() -> None:
    """Admin-only path: index every supported document in the default directory.

    The default points at the local literature test corpus unless overridden by
    CLS_DEFAULT_DOCUMENTS_DIR.
    """
    from cls_backend.readers import is_supported

    st.header("One-click corpus")
    if DEFAULT_DOCUMENTS_DIR.exists():
        default_files = sorted(p for p in DEFAULT_DOCUMENTS_DIR.iterdir() if is_supported(p))
        try:
            corpus_label = str(DEFAULT_DOCUMENTS_DIR.relative_to(APP_ROOT))
        except ValueError:
            corpus_label = str(DEFAULT_DOCUMENTS_DIR)
        st.write(f"Default documents directory: `{corpus_label}`")
        st.caption(f"{len(default_files)} supported file(s) ready to index.")
        if DEFAULT_DOCUMENT_DOMAIN:
            st.caption(f"Default domain: `{DEFAULT_DOCUMENT_DOMAIN}`")
        with st.expander("Files in default corpus", expanded=False):
            st.markdown("\n".join(f"- {p.name}" for p in default_files) or "_none_")
    else:
        default_files = []
        st.error("The default documents directory was not found.")

    force_reindex = st.checkbox("Force rebuild existing index", value=False)
    if st.button("Index default documents", type="primary", use_container_width=True):
        if not default_files:
            st.warning("No supported documents found in the default directory.")
            return
        with st.status("Indexing default documents...", expanded=True) as status:
            total_chunks = 0
            extra_meta = {"domain": DEFAULT_DOCUMENT_DOMAIN} if DEFAULT_DOCUMENT_DOMAIN else None
            for path in default_files:
                signature = file_signature(path)
                count, message = ingest_path(path, signature, force=force_reindex, extra_metadata=extra_meta)
                st.write(f"{path.name}: {message} ({count} chunks)")
                total_chunks += count
            st.write(f"Total chunks added: {total_chunks}")
            status.update(label="Index ready", state="complete", expanded=False)


def _render_upload_section() -> None:
    """Admin path: drag-and-drop a whole batch of docs and index them at once."""
    st.subheader("📥 Upload & index documents")
    st.caption(
        "Drag and drop a batch of files into the drop zone below — PDF, TXT, MD, DOCX, "
        "HTML, CSV, TSV, and JSON are supported. They join the Evidence Store; retrieval "
        "evidence stays separate from any carrier synthesis."
    )
    uploaded_files = st.file_uploader(
        "Drag and drop files here, or click to browse — multiple files supported",
        type=["pdf", "txt", "md", "docx", "html", "htm", "csv", "tsv", "json"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        sizes = [(f.name, len(f.getvalue())) for f in uploaded_files]
        total_bytes = sum(size for _, size in sizes)
        st.caption(f"📎 {len(uploaded_files)} file(s) ready · {_human_size(total_bytes)} total.")
        with st.expander(f"Files queued for indexing ({len(sizes)})", expanded=False):
            st.markdown("\n".join(f"- {name} — {_human_size(size)}" for name, size in sizes))
    
    scope_tag = st.selectbox("Assign a beamline:", list(RESEARCH_SCOPES.keys()), index=0)
    extra_meta = RESEARCH_SCOPES[scope_tag] or {}

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
                        temp_path,
                        uploaded_signature(data),
                        force=reindex_uploads,
                        extra_metadata=extra_meta,
                        source_name=name,
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
min_similarity = 0.80
dllm_enabled = False
# Synthesis is unavailable in temporary retrieval-only mode.
synth_enabled = False if RETRIEVAL_ONLY else dllm_api_online
answer_mode = "Grounded"

with st.sidebar:
    if st.button("🔬 Home", use_container_width=True, key="full_app_home_btn"):
        st.session_state.pop("ui_mode", None)
        st.rerun()
    st.divider()
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

    st.header("Research Scope")
    st.caption("Filter retrieval to a specific CLS beamline.")
    selected_scope = st.selectbox("Active scope", list(RESEARCH_SCOPES.keys()), label_visibility="collapsed")
    active_mfilter = RESEARCH_SCOPES[selected_scope]
    if role_can("scopes"):
        scope_flash = st.session_state.pop("scope_flash", None)
        if scope_flash:
            st.success(scope_flash)
        with st.expander("➕ Add research scope", expanded=False):
            st.caption(
                "Admin — saved permanently and shown to everyone. "
                "Ingest docs under data/corpus/<slug>/ to populate it."
            )
            new_name = st.text_input(
                "Scope name",
                key="new_scope_name",
                placeholder="e.g. SXRMB — Soft X-ray Microcharacterization",
            )
            new_domain = st.text_input(
                "Domain slug (optional — derived from name if blank)",
                key="new_scope_domain",
                placeholder="e.g. sxrmb",
            )
            if st.button("Add scope", use_container_width=True, key="add_scope_btn"):
                try:
                    added_name, added_filter = add_research_scope(new_name, new_domain)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["scope_flash"] = (
                        f"Added \"{added_name}\" → corpus slug "
                        f"data/corpus/{added_filter['domain']}/"
                    )
                    st.rerun()

with st.expander("Search controls", expanded=False):
    st.caption("Tune retrieval and cache behavior without crowding the ask/search workspace.")
    if RETRIEVAL_ONLY:
        st.caption("Evidence refinement is disabled in retrieval-only mode; no carrier calls are made.")
        debate_enabled = False
        st.toggle("Enable evidence refinement", value=False, disabled=True)
    else:
        st.caption("Second-pass filtering for noisy retrieval sets before synthesis.")
        debate_enabled = st.toggle("Enable evidence refinement", value=False)

    if role_can("cag_tune"):
        cache_left, cache_right = st.columns([1.15, 0.85], gap="large")
        with cache_left:
            st.subheader("♻ CAG Layer")
            st.caption("Semantic cache of prior question → retrieved evidence.")
            cag_enabled = st.toggle("Reuse cached evidence", value=True)
            min_similarity = st.slider("Min similarity for a cache hit", 0.80, 1.00, 0.80, 0.01)
        with cache_right:
            st.metric("Cached queries", get_cache().count())
            if st.button("Clear answer cache", use_container_width=True):
                get_cache().clear()
                st.success("CAG cache cleared.")

get_cache().distance_max = 1.0 - min_similarity

if role_can("index") or active_role != "User" or role_can("dllm"):
    with st.expander("Workspace", expanded=False):
        _render_db_status_badge()
        if role_can("index"):
            _render_corpus_admin()
            st.divider()

        # Read-only corpus visibility for every operating tier; the bare User path
        # stays a pure ask-and-read surface, so it sees no store internals.
        if active_role != "User":
            st.subheader("Evidence Store")
            store_rows = evidence_breakdown(get_collection())
            st.metric("Indexed chunks", collection_count(get_collection()))
            if store_rows:
                st.caption(f"across {len(store_rows)} document(s)")
            col_reset, col_flush = st.columns(2)
            if role_can("reset") and col_reset.button("Reset Chroma index", use_container_width=True):
                reset_collection()
                st.success("Evidence Store reset (CAG cache cleared too). Re-index to query again.")
            if role_can("reset") and col_flush.button("Flush CAG cache", use_container_width=True):
                n = flush_cag_cache()
                st.success(f"CAG cache cleared ({n} entr{'y' if n == 1 else 'ies'} removed). Corpus untouched.")

        if role_can("dllm"):
            carrier_name = dllm_endpoint.get("carrier", "OpenRouter")
            model_name = display_model_name(DLLM_MODEL)
            st.divider()
            st.subheader("💬 Inference carrier")
            if RETRIEVAL_ONLY:
                st.caption(dllm_endpoint.get("detail", "Carrier offline — retrieval evidence and extraction still work."))
                st.caption("Set `CLS_RETRIEVAL_ONLY=0` before launch to re-enable generation.")
            else:
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
elif not role_can("index") and not role_can("upload"):
    st.caption("📚 Corpus is curated by an administrator.")

# Full-width drag-and-drop batch upload — roomy drop target for many files at once,
# shown above the ask/answer columns for the upload-capable tier (Admin).
if role_can("upload"):
    _render_upload_section()
    st.divider()

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("🔎 Search documents")
    if KEYWORD_ONLY_RETRIEVAL:
        st.caption("Temporary fast mode: deterministic keyword retrieval, no generation.")

    query = st.text_area(
        "Research query",
        placeholder="Example: What is the sample mounting procedure?",
        height=210,
        key="query_text",
    )

    # Live category badge — reclassifies on each run, not per keystroke.
    live_cat = classify_query(query)
    live_meta = category_meta(live_cat)
    st.markdown(
        f'<span class="cls-badge" style="--hue:{live_meta["hue"]}">'
        f'{live_meta["glyph"]} {live_meta["label"]}</span>',
        unsafe_allow_html=True,
    )

    # Top-K is a precision knob — only Admin sees it;
    # User runs on a sensible fixed default.
    top_k = st.slider("Top-K chunks", 3, 24, 16) if role_can("tune") else 16

    if st.button("Search Documents", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Enter a query first.")
        else:
            try:
                result = query_backend(query, top_k, cag_enabled, min_similarity, metadata_filter=active_mfilter, debate_enabled=debate_enabled)
            except RuntimeError as exc:
                st.error(str(exc))
                result = None
            if result is None:
                pass
            elif not result["rows"]:
                st.error("No indexed chunks found. Index a document first.")
            else:
                st.session_state["last_query"] = query.strip()
                st.session_state["last_category"] = result["category"]
                st.session_state["last_rows"] = result["rows"]
                st.session_state["last_answer"] = result["answer"]
                st.session_state["last_from_cache"] = result["from_cache"]
                st.session_state["last_similarity"] = result["similarity"]
                st.session_state["last_retrieval_mode"] = result.get("retrieval_mode", "semantic")
                # Carrier cleanup is separate from synthesis. If enabled, arm it only when
                # the instant text shows artifacts worth correcting.
                activate, reason = (
                    needs_correction(result["answer"])
                    if dllm_api_online and dllm_enabled and not RETRIEVAL_ONLY
                    else (False, None)
                )
                st.session_state["last_dllm"] = {
                    "status": "pending" if (activate and result["answer"]) else "dormant",
                    "reason": reason,
                    "text": None,
                }
                # Generative answer: only armed when retrieval-only mode is off and the
                # synthesis toggle is enabled.
                st.session_state["last_synth"] = {
                    "status": "pending" if synth_enabled else "off",
                    "text": None,
                    "strict": answer_mode == "Grounded",
                }

    if role_can("eval") and st.button("Run graded offline checks", use_container_width=True):
        if collection_count(get_collection()) == 0:
            st.error("Index documents before running graded checks.")
        else:
            st.session_state["eval_rows"] = evaluate_retrieval()

with right:
    rows = st.session_state.get("last_rows", [])
    answer = st.session_state.get("last_answer", [])
    if rows:
        render_answer_component(
            st.session_state.get("last_query", ""),
            rows,
            answer,
            dllm_endpoint=dllm_endpoint,
            dllm_api_online=dllm_api_online,
            dllm_enabled=dllm_enabled,
            synth_enabled=synth_enabled,
            answer_mode=answer_mode,
        )
    else:
        render_evidence_store(evidence_breakdown(get_collection()))

eval_rows = st.session_state.get("eval_rows", [])
if eval_rows:
    st.divider()
    st.subheader("🧪 Graded offline query checks")
    st.dataframe(eval_rows, use_container_width=True, hide_index=True)

st.divider()
mode_bits = [
    "retrieval-only" if RETRIEVAL_ONLY else f"{DLLM_MODEL} synthesis available",
    "keyword-only" if KEYWORD_ONLY_RETRIEVAL else "hybrid semantic+keyword",
]
st.caption(
    f"{APP_VERSION} · CLS Synchrotron Research Query — DocuSearch-style RAG/CAG. "
    "**Evidence Store** (ChromaDB) → deterministic clean-parse. "
    f"Mode: **{', '.join(mode_bits)}**. "
    "Source documents and pages always come from Chroma retrieval."
)
