import time
from typing import Any, Dict, List, Optional, Tuple

from examples.cls_filters import prism_to_metadata_filter
from examples.cls_safety import detect_safety_topic, low_confidence


def retrieve_evidence(
    q: str,
    kb,
    active_prism: Optional[str],
) -> Tuple[List[Dict[str, Any]], float]:
    """Run semantic retrieval and return hits plus wall-clock latency."""
    mfilter = prism_to_metadata_filter(active_prism)
    start = time.perf_counter()
    rag_results = kb.query(
        [q],
        metadata_filter=mfilter,
        rse_params="balanced",
    )
    latency = time.perf_counter() - start
    return rag_results, latency


def build_flags(q: str, hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-query safety/quality flags consumed by the UI."""
    return {
        "safety_topic": detect_safety_topic(q),
        "low_confidence": low_confidence(hits),
        "no_hits": not hits,
    }


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
    hits, _ = retrieve_evidence(q, kb, active_prism)
    yield format_evidence_answer(hits)
