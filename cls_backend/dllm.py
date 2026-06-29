"""Carrier cleanup — downstream, sparsely activated extraction correction.

Inspiration: DocuSearch feels instant because it just retrieves and highlights — no
model in the hot path. So here the *instant* answer is the grounded extractive text,
shown with zero carrier latency. Carrier cleanup is deliberately lazy: it **usually does
not activate**. It only wakes up at the very end, downstream of retrieval, when the instant
text shows obvious extraction artifacts (PDF hyphenation breaks, truncated fragments,
table/OCR soup). When it does fire, it *corrects* text — it never rewrites, summarises, or
introduces facts, and every number and citation is preserved.

This module is pure-Python and side-effect-free:
- `needs_correction(sentences)` is the sparse activation gate (returns False for clean text).
- `CORRECTION_SYSTEM` / `correction_user(...)` build the constrained correction prompt; the
  app/API runs the actual one-shot carrier call and re-checks groundedness.
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
    "You are a downstream text corrector for Canadian Light Source research documents. "
    "You receive already-extracted, factual sentences that contain minor artifacts from PDF, "
    "web, or office-document extraction. Fix ONLY mechanical issues: rejoin hyphenation breaks, "
    "repair obvious spacing, drop leftover header fragments, and complete a sentence that was "
    "cut mid-word.\n"
    "Hard rules:\n"
    "1. Do NOT rewrite, summarise, reorder, or add information. Same facts, same order.\n"
    "2. Preserve every number and every '[Source: ..., page ...]' citation verbatim.\n"
    "3. Keep the bullet structure (one line per input bullet).\n"
    "4. Output only the corrected bullets, nothing else."
)


def correction_user(sentences: Iterable[str]) -> str:
    bullets = "\n".join(f"- {s}" for s in sentences)
    return f"Correct the mechanical artifacts in these bullets:\n{bullets}"


# --- Correction parsing + trust guards -------------------------------------------------- #
# The carrier cleanup result is accepted only if it stayed faithful. Weak models can invent
# numbers or mangle citations, so both are checked before the UI shows a correction.


def parse_bullets(raw: str) -> list[str]:
    """Strip bullet markers and blank lines from model output."""
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


# --- Generative RAG answer -------------------------------------------------------------- #
# A second LLM mode distinct from correction: instead of only repairing artifacts, it
# synthesises a natural-language answer to the user's question. Facts must come from the
# retrieved context passages (component-wise honest: answer what the context covers, say
# plainly what it doesn't); the extractive answer stays alongside it as the audit trail.

_CHUNK_HEADER = re.compile(r"^Source:.*?\nSection:.*?\nPage:.*?\n\n", re.S)


def answer_context(rows: list[dict], max_passages: int = 6, max_chars: int = 900) -> str:
    """Render retrieved rows as numbered, source-labelled context blocks for the prompt."""
    blocks: list[str] = []
    for position, row in enumerate(rows[:max_passages], start=1):
        meta = row.get("metadata", {}) or {}
        source = meta.get("source", "source")
        page = meta.get("page", "?")
        body = _CHUNK_HEADER.sub("", row.get("document", "")).strip()
        body = re.sub(r"\s+", " ", body)[:max_chars]
        blocks.append(f"[{position}] (Source: {source}, page {page})\n{body}")
    return "\n\n".join(blocks)


ANSWER_SYSTEM = (
    "You are a retrieval-grounded assistant for the Canadian Light Source. Answer the "
    "user's question naturally and directly, using the numbered context passages as your "
    "only source of facts about the facility.\n"
    "Hard rules:\n"
    "1. Factual claims (procedures, settings, contacts, specifications) must come from the "
    "context passages; cite the ones you use by their bracket numbers, e.g. [1], [2].\n"
    "2. Be honest component by component: answer the parts of the question the context does "
    "cover, and say plainly which parts it does not. Never refuse the whole question just "
    "because one part is missing.\n"
    "3. Greetings and purely conversational messages get a natural conversational reply — "
    "no citations needed.\n"
    "4. Never invent facts and never attribute outside knowledge to the documents.\n"
    "5. Preserve numbers, names, and identifiers exactly as written in the context.\n"
    "6. Lead with a direct one- or two-sentence answer to the question."
)


def answer_user(query: str, rows: list[dict]) -> str:
    return (
        f"Context passages:\n{answer_context(rows)}\n\n"
        f"Question: {query}\n\n"
        "Answer the question directly. Ground every factual claim in the context above with "
        "[n] citations, and say plainly if some part of the question isn't covered."
    )


# --- Hybrid "assist" answer (opt-in) ---------------------------------------------------- #
# A looser sibling of ANSWER_SYSTEM, selected only when the user switches the carrier to
# Hybrid mode. The retrieved passages become *optional* supporting material: the model
# leads with them and cites [n] when they cover the question, but is free to answer from its
# own general knowledge otherwise — it must never refuse with "Not found" and never dress
# outside knowledge up as a document citation.

ASSIST_SYSTEM = (
    "You are a helpful assistant for the Canadian Light Source. You may use both the numbered "
    "context passages provided and your own general knowledge.\n"
    "Hard rules:\n"
    "1. Prefer the context passages when they cover the question, and cite the ones you use by "
    "their bracket numbers, e.g. [1], [2].\n"
    "2. If the context does not cover the question, answer from your own general knowledge and "
    'just answer naturally. Never reply "Not found in the indexed documents."\n'
    "3. Never attribute outside knowledge to the documents and never invent bracket citations — "
    "only cite passages you actually drew from.\n"
    "4. Lead with a direct one- or two-sentence answer to the question.\n"
    "5. When you quote the context, preserve its numbers, names, and identifiers exactly."
)


def assist_user(query: str, rows: list[dict]) -> str:
    context = answer_context(rows)
    if context:
        return (
            f"Supporting passages (use if relevant, otherwise answer from your own knowledge):\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question directly, citing any passages you used."
        )
    return f"Question: {query}\n\nAnswer the question directly."


def answer_numbers_grounded(text: str, rows: list[dict]) -> bool:
    """Light guard for the generative answer: every multi-digit run it emits must already
    appear in the retrieved context (otherwise the UI flags it as unverified)."""
    context = " ".join(row.get("document", "") for row in rows)
    return numbers_grounded(text, [context])


# --- Ask Lane parrot -------------------------------------------------------------------- #
# A tiny local model (qwen2.5:0.5b via Ollama) that rephrases the already-grounded extractive
# sentences into one natural-language paragraph. Deliberately a parrot, not a reasoner: it
# adds nothing, infers nothing, and is held to the same numbers_grounded guard as the carrier.
# If the rephrase drifts (invents a number) the caller falls back to the extractive bullets.

_PARROT_CITE = re.compile(r"\s*\[Source:[^\]]*\]\s*")

PARROT_SYSTEM = (
    "You rewrite factual notes into one short, natural paragraph for a reader.\n"
    "You are a parrot, not a thinker: use ONLY the facts in the notes. Never add, infer, "
    "explain, or reason beyond what is written.\n"
    "Hard rules:\n"
    "1. Use only information present in the notes. Add nothing.\n"
    "2. Keep every number, name, unit, and identifier exactly as written.\n"
    "3. Do not write bracket citations or '[Source: ...]' tags.\n"
    "4. One to three plain sentences. Output only the paragraph, nothing else."
)


def parrot_user(sentences: Iterable[str]) -> str:
    notes = []
    for sentence in sentences:
        clean = _PARROT_CITE.sub(" ", sentence).strip()
        if clean:
            notes.append(f"- {clean}")
    body = "\n".join(notes)
    return f"Notes:\n{body}\n\nRewrite these notes as one short, natural paragraph."


def parrot_grounded(prose: str, sentences: list[str]) -> bool:
    """Parrot output is trusted only if it invented no multi-digit numbers."""
    return numbers_grounded(prose, sentences)


# --- Wernicke relation guard ------------------------------------------------------------ #
# The number guard catches invented digits but NOT relation-drift: the parrot can keep a real
# number while binding it to the wrong thing — "floor coordinator at ext. 3639" became
# "elevator number 3639". Both numbers are real, so numbers_grounded passes it.
#
# This deterministic check scrutinises the content words that *hug a protected value* (a
# multi-digit number / ext code) in the prose and flags any that are absent from the evidence
# vocabulary — the signature of a hallucinated relation. It is intentionally local and
# rule-based (trusting another model to judge meaning would inherit the same drift) and, like
# the safety detector, conservative: a false positive just surfaces a "verify" banner while
# the grounded base stays the source of truth; a false negative would let drift through.

_PROTECTED_VALUE = re.compile(r"\d{2,}")  # multi-digit numbers, ext codes, identifiers
# Alphanumeric runs with internal -/. only (so "access." -> "access", "ext." -> "ext").
_WORD = re.compile(r"[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)*")
_DRIFT_STOP = {
    # function words
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "at", "by", "is", "are",
    "was", "were", "be", "can", "via", "with", "this", "that", "it", "as", "from", "into",
    "ext", "no", "number",
    # generic relation/locative verbs the parrot swaps freely — not the entity-drift we hunt
    "located", "accessed", "reached", "called", "contacted", "found", "used", "set",
    "provided", "available", "listed", "given", "made", "obtained", "handled",
}


def _content_terms(text: str) -> set[str]:
    """Lower-cased content terms with light singularisation; mirrors the retrieval tokeniser."""
    terms: set[str] = set()
    for tok in _WORD.findall(text.lower()):
        if len(tok) <= 2 or tok in _DRIFT_STOP or tok.isdigit():
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = f"{tok[:-3]}y"
        elif len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        terms.add(tok)
    return terms


def relation_drift(prose: str, sentences: list[str], window: int = 3) -> list[str]:
    """Content words the prose binds next to a protected value but that never appear in the
    evidence — candidate relation-drift (the 'elevator 3639' case). Empty list == clean.

    Only words within `window` tokens of a number are scrutinised, so general rephrasing
    elsewhere in the sentence is tolerated; only vocabulary hugging a fact must be grounded.
    """
    evidence = " ".join(_CITE_SUFFIX.sub("", s) for s in sentences)
    evidence_vocab = _content_terms(evidence)
    tokens = _WORD.findall(prose)

    offending: list[str] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        if not _PROTECTED_VALUE.fullmatch(token):
            continue
        lo = max(0, index - window)
        hi = min(len(tokens), index + window + 1)
        for term in _content_terms(" ".join(tokens[lo:hi])):
            if term not in evidence_vocab and term not in seen:
                seen.add(term)
                offending.append(term)
    return offending


def parrot_trustworthy(prose: str, sentences: list[str]) -> bool:
    """Full parrot trust contract: no invented numbers AND no relation-drift near a fact."""
    return numbers_grounded(prose, sentences) and not relation_drift(prose, sentences)
