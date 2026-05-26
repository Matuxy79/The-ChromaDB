import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from examples.cls_ingest import ingest_document
from examples.cls_kb_setup import build_cls_kb
from examples.cls_safety import EMERGENCY_CONTACTS, LOW_CONFIDENCE_DISTANCE
from examples.streamlit_cls_chat import build_flags, format_evidence_answer, retrieve_evidence


LANES = ["None", "purple", "green", "blue", "orange", "yellow"]
DOMAINS = ["beamline", "research", "outreach", "logistics", "education"]
STAGE_ORDER = ["extract", "chunk", "embed", "store"]
STAGE_LABEL = {
    "extract": "📄 Extract",
    "chunk":   "✂️ Chunk",
    "embed":   "🧠 Embed",
    "store":   "💾 Store",
}


st.set_page_config(page_title="CLS RAG+CAG Chat", layout="wide", page_icon="🔬")


CUSTOM_CSS = """
<style>
.stApp [data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.45rem;
}
.cls-pill {
    display: inline-block;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    vertical-align: middle;
    margin-left: 0.5rem;
}
.cls-pill-online  { background: #10331f; color: #57e08a; border: 1px solid #1f6a3b; }
.cls-pill-offline { background: #3a1414; color: #ff8b8b; border: 1px solid #7a2424; }
.cls-stage-row {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
    color: #c9c9c9;
}
.cls-summary-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    margin-top: 0.4rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.82rem;
}
.cls-safety-card {
    background: #3a1e08;
    border: 1px solid #b85d18;
    border-left: 4px solid #ff8b2a;
    border-radius: 10px;
    padding: 0.7rem 0.95rem;
    margin: 0.5rem 0 0.3rem;
    font-size: 0.88rem;
    color: #ffdcb6;
}
.cls-safety-card .cls-safety-head {
    font-weight: 700;
    font-size: 0.95rem;
    color: #ffb574;
    margin-bottom: 0.35rem;
}
.cls-safety-card table { width: 100%; border-collapse: collapse; font-family: ui-monospace, monospace; font-size: 0.8rem; }
.cls-safety-card th, .cls-safety-card td { text-align: left; padding: 0.15rem 0.5rem 0.15rem 0; }
.cls-safety-card th { color: #ffb574; border-bottom: 1px solid #6a3a16; }
.cls-warn-card {
    background: #2c2510;
    border: 1px solid #806118;
    border-left: 4px solid #e3b341;
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    margin: 0.4rem 0 0.2rem;
    font-size: 0.84rem;
    color: #f7e0a0;
}
.cls-info-card {
    background: #11253a;
    border: 1px solid #1f4a78;
    border-left: 4px solid #5aaaff;
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    margin: 0.4rem 0 0.2rem;
    font-size: 0.84rem;
    color: #cfe5ff;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def init_kb():
    return build_cls_kb()


kb = init_kb()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ollama_online" not in st.session_state:
    st.session_state.ollama_online = kb.embedding_model.is_available()


def render_offline_pill(online: bool) -> str:
    if online:
        return '<span class="cls-pill cls-pill-online">● Offline-only · Ollama reachable</span>'
    return '<span class="cls-pill cls-pill-offline">○ Ollama unreachable</span>'


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:04.1f}s"


class StageRenderer:
    """Per-file progress board rendered into a Streamlit container."""

    def __init__(self, container, filename: str):
        self.container = container
        self.start = time.perf_counter()
        with container:
            st.markdown(f"**{filename}**")
            self.timer_slot = st.empty()
            self.bar_slots: Dict[str, Any] = {}
            self.label_slots: Dict[str, Any] = {}
            for stage in STAGE_ORDER:
                self.label_slots[stage] = st.empty()
                self.bar_slots[stage] = st.progress(0, text=STAGE_LABEL[stage])
        self._tick_timer()

    def _tick_timer(self) -> None:
        elapsed = time.perf_counter() - self.start
        self.timer_slot.markdown(
            f"<span class='cls-stage-row'>⏱ {format_seconds(elapsed)} elapsed</span>",
            unsafe_allow_html=True,
        )

    def __call__(self, stage: str, current: int, total: int) -> None:
        if stage not in self.bar_slots:
            return
        pct = (current / total) if total else 1.0
        pct = max(0.0, min(pct, 1.0))
        self.bar_slots[stage].progress(pct, text=f"{STAGE_LABEL[stage]} — {current}/{total}")
        self._tick_timer()

    def finish(self, stats: Dict[str, Any]) -> None:
        elapsed = time.perf_counter() - self.start
        chunks = stats.get("chunks", 0)
        pages = stats.get("pages", 0)
        rate = chunks / elapsed if elapsed > 0 else 0.0
        per_stage = " · ".join(
            f"{STAGE_LABEL[s]} {format_seconds(stats.get(f'{s}_seconds', 0.0))}"
            for s in STAGE_ORDER
            if f"{s}_seconds" in stats
        )
        with self.container:
            st.markdown(
                "<div class='cls-summary-card'>"
                f"✅ <b>{stats.get('file', '')}</b><br>"
                f"{pages} page(s) · {chunks} chunk(s) · {format_seconds(elapsed)} total · "
                f"{rate:.1f} chunks/s<br>"
                f"{per_stage}"
                "</div>",
                unsafe_allow_html=True,
            )


def _emergency_contact_table() -> str:
    rows = "".join(
        f"<tr><td>{c['who']}</td><td><b>{c['number']}</b></td><td>{c['note']}</td></tr>"
        for c in EMERGENCY_CONTACTS
    )
    return (
        "<table><thead><tr><th>Contact</th><th>Number</th><th>Note</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_flags(flags: Optional[Dict[str, Any]], active_prism: Optional[str]) -> None:
    if not flags:
        return

    topic = flags.get("safety_topic")
    if topic:
        st.markdown(
            "<div class='cls-safety-card'>"
            f"<div class='cls-safety-head'>⚠ Safety topic detected — {topic.replace('_', ' ')}</div>"
            "For an active emergency call <b>911</b> or U-Sask Security "
            "<b>9-306-966-5555</b>. This tool only retrieves indexed evidence; "
            "confirm any procedure with beamline staff before acting.<br><br>"
            f"{_emergency_contact_table()}"
            "</div>",
            unsafe_allow_html=True,
        )

    if flags.get("low_confidence"):
        st.markdown(
            "<div class='cls-warn-card'>"
            "🟡 <b>Low-confidence retrieval.</b> The best matching chunk is past the "
            f"distance threshold ({LOW_CONFIDENCE_DISTANCE:.2f}). Treat the answer as a "
            "lead, not a citation, and check the source row in the retrieval trace."
            "</div>",
            unsafe_allow_html=True,
        )

    if flags.get("no_hits") and active_prism:
        st.markdown(
            "<div class='cls-info-card'>"
            f"ℹ No evidence found in the <b>{active_prism}</b> lane. "
            "Try setting the lane to <b>None</b> to search across all lanes."
            "</div>",
            unsafe_allow_html=True,
        )


def render_retrieval_trace(trace: Optional[Dict[str, Any]]) -> None:
    if not trace:
        return
    hits: List[Dict[str, Any]] = trace.get("hits", [])
    latency = trace.get("latency", 0.0)
    label = f"🔍 Retrieval trace — {len(hits)} hit(s) · {format_seconds(latency)}"
    with st.expander(label, expanded=False):
        if not hits:
            st.caption("No evidence retrieved for this query.")
            return
        rows = []
        for idx, hit in enumerate(hits, start=1):
            meta = hit.get("metadata", {}) or {}
            text = (hit.get("text") or "").strip().replace("\n", " ")
            rows.append({
                "#": idx,
                "source": meta.get("source_url", "local-doc"),
                "lane": meta.get("colour_code", "—"),
                "domain": meta.get("domain", "—"),
                "distance": round(hit.get("distance"), 4) if hit.get("distance") is not None else None,
                "preview": (text[:120] + "…") if len(text) > 120 else text,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Retrieval")
    active_prism = st.selectbox("Prism lane", LANES, index=0)
    lane_filter = None if active_prism == "None" else active_prism

    col_metric, col_recheck = st.columns([2, 1])
    with col_metric:
        st.metric("Indexed chunks", kb.count_chunks())
    with col_recheck:
        st.markdown("&nbsp;")
        if st.button("Re-check", use_container_width=True, help="Re-probe Ollama"):
            st.session_state.ollama_online = kb.embedding_model.is_available()
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Index maintenance", expanded=False):
        uploaded_files = st.file_uploader(
            "Add PDF/TXT documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
        )
        ingest_lane = st.selectbox(
            "Index as lane",
            LANES[1:],
            index=1,
            help=(
                "This is metadata stored with each chunk for fast filtering later. "
                "It does not call a chat LLM and does not meaningfully slow indexing."
            ),
        )
        domain = st.selectbox("Document domain", DOMAINS, index=0)

        if st.button("Index uploaded files", use_container_width=True):
            if not uploaded_files:
                st.warning("No files selected.")
            elif not st.session_state.ollama_online:
                st.error("Ollama is offline. Start it before indexing.")
            else:
                completed = 0
                with st.status("Indexing documents…", expanded=True) as status:
                    overall_start = time.perf_counter()
                    for uploaded_file in uploaded_files:
                        suffix = os.path.splitext(uploaded_file.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        metadata = {
                            "colour_code": ingest_lane,
                            "domain": domain,
                            "source_url": uploaded_file.name,
                            "trust_level": "official_cls",
                        }

                        file_container = st.container()
                        renderer = StageRenderer(file_container, uploaded_file.name)

                        try:
                            stats = ingest_document(
                                kb,
                                tmp_path,
                                metadata,
                                progress_callback=renderer,
                            )
                            if stats:
                                completed += 1
                                renderer.finish(stats)
                            else:
                                with file_container:
                                    st.warning(f"No extractable text in {uploaded_file.name}")
                        except Exception as exc:
                            with file_container:
                                st.error(f"Failed: {exc}")
                        finally:
                            os.remove(tmp_path)

                    elapsed_all = time.perf_counter() - overall_start
                    status.update(
                        label=f"Indexed {completed}/{len(uploaded_files)} · {format_seconds(elapsed_all)}",
                        state="complete",
                        expanded=True,
                    )
                st.rerun()


# ─── Header ────────────────────────────────────────────────────────────────
title_col, pill_col = st.columns([4, 2])
with title_col:
    st.title("🔬 CLS Scientist Chat")
with pill_col:
    st.markdown(
        f"<div style='text-align:right; padding-top:1.6rem;'>"
        f"{render_offline_pill(st.session_state.ollama_online)}"
        f"</div>",
        unsafe_allow_html=True,
    )
st.caption("Fast local semantic retrieval over the indexed CLS knowledge base. No chat LLM is used.")


# ─── Chat transcript ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_flags(message.get("flags"), message.get("lane"))
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_retrieval_trace(message.get("trace"))


# ─── Input ─────────────────────────────────────────────────────────────────
if st.session_state.ollama_online:
    prompt = st.chat_input("Ask a beamline, procedure, training, or policy question")
else:
    st.error(
        "Ollama is not reachable on 127.0.0.1:11434. Start it with `ollama serve`, "
        "then click **Re-check** in the sidebar."
    )
    prompt = None


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Retrieving evidence…", expanded=False)
        flags_slot = st.empty()
        response_container = st.empty()
        full_response = ""
        trace: Dict[str, Any] = {"hits": [], "latency": 0.0}
        flags: Dict[str, Any] = {}

        try:
            hits, latency = retrieve_evidence(prompt, kb, lane_filter)
            trace = {"hits": hits, "latency": latency}
            flags = build_flags(prompt, hits)
            with flags_slot.container():
                render_flags(flags, lane_filter)

            full_response = format_evidence_answer(hits)
            response_container.markdown(full_response)
            status.update(
                label=f"Evidence ready · {len(hits)} hit(s) in {format_seconds(latency)}",
                state="complete",
                expanded=False,
            )
        except Exception as exc:
            full_response = (
                "The local embedding service returned an error. "
                "Confirm Ollama is running and nomic-embed-text is installed."
            )
            status.update(label=f"Retrieval failed: {exc}", state="error", expanded=True)
            st.session_state.ollama_online = kb.embedding_model.is_available()

        response_container.markdown(full_response)
        render_retrieval_trace(trace)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "trace": trace,
        "flags": flags,
        "lane": lane_filter,
    })
