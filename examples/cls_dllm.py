"""dLLM — a downstream, sparsely-activated correction LLM.

Inspiration: DocuSearch feels instant because it just retrieves and highlights — no
model in the hot path. So here the *instant* answer is the grounded extractive text,
shown with zero LLM latency. The dLLM ("downstream LLM") is deliberately lazy: it
**usually does not activate**. It only wakes up at the very end, downstream of retrieval,
when the instant text shows obvious extraction artifacts (PDF hyphenation breaks, truncated
fragments, table/OCR soup). When it does fire, it *corrects* text — it never rewrites,
summarises, or introduces facts, and every number and citation is preserved.

This module is pure-Python and side-effect-free:
- `needs_correction(sentences)` is the sparse activation gate (returns False for clean text).
- `CORRECTION_SYSTEM` / `correction_user(...)` build the constrained correction prompt; the
  app runs the actual one-shot Ollama call and re-checks groundedness.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple


# Strip the "[Source: ..., page N]" suffix the extractor appends before judging artifacts.
_CITE_SUFFIX = re.compile(r"\s*\[Source:[^\]]*\]\s*$")

# Artifact signals. Kept conservative so the gate stays *sparse* — clean prose passes.
_HYPHEN_BREAK = re.compile(r"[A-Za-z]{2,}-\s+[a-z]{2,}")  # "exam- ple", "undula- tor"
_LEFTOVER_HEADER = re.compile(r"\b(?:Source|Section|Page)\s*:", re.IGNORECASE)
_SINGLE_CHARS = re.compile(r"(?:\b[A-Za-z0-9]\b[\s.]*){4,}")  # "a 1 2 b" table/OCR soup


def _body(sentence: str) -> str:
    return _CITE_SUFFIX.sub("", sentence).strip()


def _non_alpha_ratio(text: str) -> float:
    stripped = text.replace(" ", "")
    if not stripped:
        return 0.0
    non_alpha = sum(1 for ch in stripped if not ch.isalnum())
    return non_alpha / len(stripped)


def needs_correction(sentences: Iterable[str]) -> Tuple[bool, Optional[str]]:
    """Sparse activation gate. Returns (activate, reason). False for clean extractions."""
    bodies = [_body(s) for s in sentences]
    bodies = [b for b in bodies if b]
    if not bodies:
        return False, None

    for body in bodies:
        if _HYPHEN_BREAK.search(body):
            return True, "joined hyphenation breaks"
        if _LEFTOVER_HEADER.search(body):
            return True, "stripped leftover header text"
        if _SINGLE_CHARS.search(body):
            return True, "tidied fragmented table text"
        if _non_alpha_ratio(body) > 0.35:
            return True, "cleaned symbol-heavy fragment"
        # Long sentence with no terminal punctuation = truncated mid-thought.
        if len(body) > 80 and body[-1] not in ".!?:)":
            return True, "completed a truncated fragment"

    # Near-duplicate sentences (extractor picked overlapping chunks).
    lowered = [b.lower() for b in bodies]
    if len(lowered) != len(set(lowered)):
        return True, "removed duplicated sentence"

    return False, None


CORRECTION_SYSTEM = (
    "You are a downstream text corrector for the Canadian Light Source IVU beamline manual. "
    "You receive already-extracted, factual sentences that contain minor artifacts from PDF "
    "extraction. Fix ONLY mechanical issues: rejoin hyphenation breaks, repair obvious spacing, "
    "drop leftover header fragments, and complete a sentence that was cut mid-word.\n"
    "Hard rules:\n"
    "1. Do NOT rewrite, summarise, reorder, or add information. Same facts, same order.\n"
    "2. Preserve every number and every '[Source: ..., page ...]' citation verbatim.\n"
    "3. Keep the bullet structure (one line per input bullet).\n"
    "4. Output only the corrected bullets, nothing else."
)


def correction_user(sentences: Iterable[str]) -> str:
    bullets = "\n".join(f"- {s}" for s in sentences)
    return f"Correct the mechanical artifacts in these bullets:\n{bullets}"


# --- Streaming correction + trust guards ------------------------------------------------ #
# The dLLM streams live into the UI; these helpers let the UI accept the result only if it
# stayed faithful. A weak model invents numbers and mangles citations, so both are checked.

def stream_correction(api, sentences: list[str]) -> Iterable[str]:
    """Yield the correction token-by-token. `api` is any object with `.stream(messages, system)`."""
    yield from api.stream(
        [{"role": "user", "content": correction_user(sentences)}],
        system=CORRECTION_SYSTEM,
    )


def parse_bullets(raw: str) -> list[str]:
    """Strip bullet markers and blank lines from streamed/one-shot model output."""
    lines = [re.sub(r"^[-*•]\s*", "", line).strip() for line in raw.splitlines()]
    return [line for line in lines if line]


def numbers_grounded(text: str, sentences: list[str]) -> bool:
    """Every multi-digit run the model emits must already exist in the source evidence."""
    evidence_digits = set(re.findall(r"\d+", " ".join(sentences)))
    return all(
        not (len(tok) >= 3 and tok not in evidence_digits)
        for tok in re.findall(r"\d+", text)
    )


def citations_preserved(text: str, sentences: list[str]) -> bool:
    """Every [Source: ...] citation must survive verbatim, with none invented."""
    source_cites = re.findall(r"\[Source:[^\]]*\]", " ".join(sentences))
    output_cites = re.findall(r"\[Source:[^\]]*\]", text)
    return all(cite in text for cite in source_cites) and len(output_cites) == len(source_cites)


def validate_correction(text: str, sentences: list[str]) -> bool:
    """The trust contract: a correction is only shown if it invented no numbers and kept
    every citation verbatim. Otherwise the instant grounded extraction stands."""
    return numbers_grounded(text, sentences) and citations_preserved(text, sentences)
