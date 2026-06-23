"""Chainlit Ask Lane — a streaming, layered RAG/CAG chat over the shared backend.

Run:  chainlit run chat_lane.py -w        (or ./scripts/launch_chat.sh)

Two-pass answer design:
  PASS 1 — instant grounded answer from the RAG/CAG engine, no LLM wait.
            Formatted markdown bullets with match badge (exact / close / loose / cached)
            and source chips attached.  This is always the source of truth and is visible
            immediately before the parrot starts.
  PASS 2 — natural-language prose streamed token-by-token from the small local parrot
            (qwen2.5:0.5b via Ollama).  The Pass 2 message is created on the first token
            so there is no empty-message flash.  Falls back silently to Pass 1 alone when
            the parrot is offline, and is flagged if prose drifts from the evidence.

The retrieval / CAG / parrot / embedder engine is reused unchanged from cls_service and
cls_backend — this file is only the Chainlit view.
"""

from __future__ import annotations

import asyncio
import threading

import chainlit as cl
from chainlit.input_widget import Select

from cls_config import APP_VERSION, KEYWORD_ONLY_RETRIEVAL, RESEARCH_SCOPES, RETRIEVAL_ONLY
from cls_backend.query_repair import repair_query
from cls_backend.dllm import numbers_grounded, relation_drift
from cls_service import ask_manual, parrot_stream


def _match_badge(result: dict) -> str:
    """Advertise the instant grounded layer's confidence."""
    if result.get("from_cache"):
        return "⚡ exact match · cached"
    rows = result.get("rows") or []
    score = rows[0]["score"] if rows else 0.0
    label = "exact match" if score >= 0.9 else "close match" if score >= 0.6 else "loose match"
    return f"● {label} · {score:.2f}"


def _grounded_bullets(answer: list[str]) -> str:
    return "\n".join(f"- {s.split(' [Source:')[0].strip()}" for s in answer)


def _sources(rows: list[dict]) -> list[str]:
    seen: set = set()
    items: list[str] = []
    for row in rows:
        meta = row.get("metadata", {}) or {}
        src = str(meta.get("source", "")).rsplit(".", 1)[0]
        page = meta.get("page")
        key = (src, page)
        if src and key not in seen:
            seen.add(key)
            items.append(f"{src}" + (f" · p{page}" if page else ""))
    return items


async def _astream_parrot(sentences: list[str]):
    """Bridge the blocking parrot generator to async so the event loop isn't blocked."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    sentinel = object()

    def worker():
        for token in parrot_stream(sentences):
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        token = await queue.get()
        if token is sentinel:
            break
        yield token


@cl.on_chat_start
async def start():
    scopes = list(RESEARCH_SCOPES.keys())
    await cl.ChatSettings(
        [Select(id="scope", label="Research scope", values=scopes, initial_index=0)]
    ).send()
    cl.user_session.set("scope", scopes[0])
    await cl.Message(
        content=(
            f"### 🔬 CLS Research Documents `{APP_VERSION}`\n"
            "Ask a question — instant cited answers from your indexed documents. "
            + (
                "Temporary retrieval-only mode is active; no phrased model answer will run."
                if RETRIEVAL_ONLY
                else "The grounded evidence is the source of truth; the phrased answer rides on top."
            )
        )
    ).send()


@cl.on_settings_update
async def update_settings(settings: dict):
    cl.user_session.set("scope", settings.get("scope"))


@cl.on_message
async def on_message(message: cl.Message):
    query = message.content.strip()
    if not query:
        return

    scope = cl.user_session.get("scope") or "All beamlines"
    mfilter = RESEARCH_SCOPES.get(scope)
    search = repair_query(query)["search"]

    # ── PASS 1: RAG/CAG engine — instant grounded answer, no LLM wait ────
    result = await cl.make_async(ask_manual)(
        search,
        top_k=16,
        metadata_filter=mfilter,
        keyword_only=KEYWORD_ONLY_RETRIEVAL,
    )
    rows   = result.get("rows") or []
    answer = result.get("answer") or []

    if not rows:
        await cl.Message(content="_Nothing relevant found — try rephrasing._").send()
        return

    badge   = _match_badge(result)
    sources = _sources(rows)
    p1 = cl.Message(content=f"**{badge}**\n\n{_grounded_bullets(answer)}")
    if sources:
        p1.elements = [
            cl.Text(
                name="Sources",
                content="\n".join(f"📄 {s}" for s in sources),
                display="inline",
            )
        ]
    await p1.send()  # user sees grounded answer immediately

    if RETRIEVAL_ONLY:
        return

    # ── PASS 2: Parrot — message created on first token, no empty flash ───
    p2: cl.Message | None = None
    streamed = ""
    async for token in _astream_parrot(answer):
        if p2 is None:
            p2 = cl.Message(content="")
            await p2.send()
        streamed += token
        await p2.stream_token(token)

    if p2 is None or not streamed.strip():
        return  # parrot offline — pass 1 stands alone

    # Wernicke check: numbers grounded AND no relation-drift near a fact.
    if not numbers_grounded(streamed, answer):
        await p2.stream_token(
            "\n\n> ⚠ _this phrasing introduced a number not in the source — "
            "trust the grounded evidence above._"
        )
    elif drift := relation_drift(streamed, answer):
        suspect = ", ".join(f"**{w}**" for w in drift[:4])
        await p2.stream_token(
            f"\n\n> ⚠ _phrasing may not match the source near: {suspect} — "
            "trust the grounded evidence above._"
        )

    await p2.update()
