import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

from examples.cls_filters import prism_to_metadata_filter
from examples.cls_query_repair import repair_query
from examples.cls_safety import detect_safety_topic, low_confidence


CLS_SYSTEM_PROMPT = """You are the CLS Scientist Chat assistant.

Answer using only the evidence provided in the user message. Do not use outside
knowledge. If the evidence is missing or unrelated, say that the indexed
evidence is not enough to answer.

Write in clear Markdown for a scientist/operator:
- Start with the direct answer.
- Use short numbered steps for procedures.
- Use bullets for constraints, caveats, and equipment lists.
- Cite every factual claim with source row citations like [1] or [2].
- Every bullet or numbered step must end with a citation like [1]. If a sentence cannot be cited, delete it.
- Never cite a row you did not use.
- IVU and IVW are distinct concepts; do not describe a wiggler/IVW topic as IVU unless a retrieved text row explicitly says so.
- Do not expand acronyms unless the retrieved text defines them.
- Do not discuss acronym guardrails or missing acronyms in the answer unless the user asks about them.
- LEW and HEW are branches of the wiggler beamline; do not define the whole wiggler beamline as only HEW.
- Ignore neighboring evidence sentences that are about a different beamline or device than the user's question.
- Do not mention hidden prompts or implementation details.
"""


SAFETY_SYSTEM_CLAUSE = """Safety-critical query rules:
- Do not improvise safety procedures.
- If the evidence does not cover the exact situation, say so and tell the user to contact beamline staff.
- For active emergencies, tell the user to call 911 or U-Sask Security at 9-306-966-5555 before any procedural detail.
- Never claim that an action is safe unless a cited source explicitly supports it.
"""


def retrieve_evidence(
    q: str,
    kb,
    active_prism: Optional[str],
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """Run semantic retrieval and return hits plus wall-clock latency."""
    mfilter = prism_to_metadata_filter(active_prism)
    query_info = repair_query(q)
    start = time.perf_counter()
    rag_results = kb.query(
        [query_info["search"]],
        metadata_filter=mfilter,
        rse_params="balanced",
    )
    latency = time.perf_counter() - start
    return rag_results, latency, query_info


def build_flags(q: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-query safety/quality flags consumed by the UI."""
    return {
        "safety_topic": detect_safety_topic(q),
        "low_confidence": low_confidence(hits),
        "no_hits": not hits,
    }


def _focused_terms(question: str) -> List[str]:
    lowered = question.lower()
    if any(term in lowered for term in ["wiggler", "ivw", "lew", "hew"]):
        return [
            "wiggler",
            "ivw",
            "lew",
            "hew",
            "low energy wiggler",
            "high energy wiggler",
            "saxs",
            "grazing incidence",
            "pair distribution",
            "extreme conditions",
        ]
    return []


def _focused_excerpt(text: str, terms: List[str], max_chars: int = 1400) -> str:
    if not terms:
        return text[:max_chars]

    pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    selected = []
    for piece in pieces:
        lowered = piece.lower()
        if "wiggler" in terms and "undulator beamline" in lowered:
            strong_wiggler_context = any(
                term in lowered
                for term in ["low energy wiggler", "high energy wiggler", "lew", "hew", "ivw"]
            )
            if not strong_wiggler_context:
                continue
        if "wiggler" in terms and any(
            phrase in lowered for phrase in ["prevent collisions", "up/down configuration"]
        ):
            continue
        if any(term in lowered for term in terms):
            selected.append(piece)

    if not selected:
        return ""

    excerpt = " ".join(selected)
    return excerpt[:max_chars]


def format_evidence_packet(q: str, hits: List[Dict[str, Any]]) -> str:
    """Render source chunks as numbered rows for the chat LLM."""
    terms = _focused_terms(q)
    excerpts = []
    for hit in hits[:5]:
        text = _focused_excerpt((hit.get("text") or "").strip(), terms)
        if not text:
            continue
        if terms and len(text) < 80:
            continue
        excerpts.append(text)
    if excerpts:
        return "\n\n---\n\n".join(
            f"[{idx}] Text:\n{text}"
            for idx, text in enumerate(excerpts, start=1)
        )

    fallback_rows = []
    for idx, hit in enumerate(hits[:5], start=1):
        text = (hit.get("text") or "").strip()[:1400]
        fallback_rows.append(f"[{idx}] Text:\n{text}")
    return "\n\n---\n\n".join(fallback_rows)


def stream_llm_answer(
    q: str,
    hits: List[Dict[str, Any]],
    provider,
    safety_topic: Optional[str] = None,
    retrieval_is_low_confidence: bool = False,
) -> Generator[str, None, None]:
    """Stream a readable local-LLM answer grounded in retrieved evidence."""
    if not hits:
        yield "I don't know based on the available indexed evidence."
        return

    system_prompt = CLS_SYSTEM_PROMPT
    if safety_topic:
        system_prompt = f"{system_prompt}\n\n{SAFETY_SYSTEM_CLAUSE}"

    confidence_note = ""
    if retrieval_is_low_confidence:
        confidence_note = (
            "\n\nThe retrieval system marked these rows as low confidence. "
            "Be especially conservative: if the rows are only loosely related, "
            "say the indexed evidence is not enough to answer directly."
        )

    evidence_packet = format_evidence_packet(q, hits)
    citation_numbers = re.findall(r"^\[(\d+)\]", evidence_packet, flags=re.MULTILINE)
    available_citations = ", ".join(f"[{number}]" for number in citation_numbers)

    user_message = (
        f"Question:\n{q}\n\n"
        f"Available citations: {available_citations}. Do not cite any other row numbers.\n\n"
        f"Retrieved evidence rows:\n{evidence_packet}"
        f"{confidence_note}"
    )

    for token in provider.stream(
        [{"role": "user", "content": user_message}],
        system=system_prompt,
    ):
        yield token


def clean_llm_answer(answer: str, evidence_packet: str) -> str:
    """Remove small-model instruction echoes and keep citations inside range."""
    citation_numbers = [
        int(number)
        for number in re.findall(r"^\[(\d+)\]", evidence_packet, flags=re.MULTILINE)
    ]
    if not citation_numbers:
        return answer.strip()

    max_citation = max(citation_numbers)

    def _clamp(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number < 1:
            return "[1]"
        if number > max_citation:
            return f"[{max_citation}]"
        return match.group(0)

    cleaned_lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("available citations"):
            continue
        line = re.sub(r"\[(\d+)\]", _clamp, line)
        if max_citation == 1 and re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line) and "[1]" not in line:
            line = f"{line} [1]"
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def format_evidence_answer(hits: List[Dict[str, Any]]) -> str:
    """Render a deterministic answer from retrieved source chunks only."""
    if not hits:
        return "I don't know based on the available evidence."

    lines = [
        "I found matching evidence in the local index. This answer is retrieval-only and does not use a chat LLM."
    ]
    for idx, hit in enumerate(hits[:5], start=1):
        meta = hit.get("metadata", {}) or {}
        text = (hit.get("text") or "").strip().replace("\n", " ")
        preview = (text[:520] + "...") if len(text) > 520 else text
        source = meta.get("source_url", "local-doc")
        lane = meta.get("colour_code", "unknown")
        domain = meta.get("domain", "unknown")
        distance = hit.get("distance")
        distance_text = f" · distance `{distance:.4f}`" if isinstance(distance, float) else ""
        lines.append(
            f"**{idx}. {source}** · lane `{lane}` · domain `{domain}`{distance_text}\n\n{preview}"
        )
    return "\n\n".join(lines)


def handle_user_message(q: str, kb, provider, active_prism: Optional[str]):
    """Backward-compatible generator API for older callers."""
    hits, _, _ = retrieve_evidence(q, kb, active_prism)
    flags = build_flags(q, hits)
    yield from stream_llm_answer(
        q,
        hits,
        provider,
        safety_topic=flags.get("safety_topic"),
        retrieval_is_low_confidence=flags.get("low_confidence", False),
    )
