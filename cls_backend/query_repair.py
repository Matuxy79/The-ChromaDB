import re
from typing import Any, Dict, List, Tuple


# Strip common NL scaffolding that adds noise without adding retrieval signal.
# The semantic encoder handles natural language well; this catches verbose filler
# that dilutes the meaningful keywords in the embedding.
_FILLER_PREFIX = re.compile(
    r"^\s*(?:"
    r"i(?:'m| am| was) (?:wondering|trying to|looking to|not sure|having trouble|unsure)\s+(?:about\s+|if\s+)?"
    r"|(?:can|could|would) you (?:please\s*)?(?:tell me|explain|help me|show me|describe|give me)\s+(?:about\s+)?"
    r"|(?:please\s*)?(?:tell me|explain|help me|show me|describe)\s+(?:about\s+)?"
    r"|how\s+(?:do|does|can|would|should)\s+(?:i|you|one|we)\s+"
    r"|what\s+(?:is|are|was|were|'s)\s+(?:the\s+)?"
    r"|where\s+(?:is|are|can\s+i\s+find\s+)?"
    r"|who\s+(?:is|are|do\s+i\s+|should\s+i\s+)(?:contact\s+)?"
    r"|why\s+(?:is|are|does|do)\s+"
    r"|i\s+(?:need|want|would like)\s+(?:to\s+(?:know\s+(?:about\s+)?)?|information\s+(?:on\s+|about\s+)?)?"
    r")+",
    re.IGNORECASE,
)

_FILLER_WORDS = re.compile(
    r"\b(?:basically|literally|actually|essentially|kind of|sort of|like|um+|uh+|so|just|maybe|perhaps)\b",
    re.IGNORECASE,
)

_TRAILING_PUNCT = re.compile(r"[?!.]+$")
_WHITESPACE = re.compile(r"\s{2,}")


def _normalize_nl(query: str) -> str:
    """Strip conversational scaffolding so the embedding focuses on content words."""
    s = query.strip()
    s = _FILLER_PREFIX.sub("", s)
    s = _FILLER_WORDS.sub("", s)
    s = _TRAILING_PUNCT.sub("", s)
    s = _WHITESPACE.sub(" ", s).strip()
    # If normalization ate everything (e.g. pure filler input), fall back to original.
    return s if len(s) >= 3 else query.strip()


# ---------------------------------------------------------------------------
# Facility-specific typo corrections and query expansions.
# Add domain or acronym repairs here — the function structure stays the same;
# only the content is deployment-specific.
# ---------------------------------------------------------------------------
TYPO_REPLACEMENTS: List[Tuple[re.Pattern[str], str, str]] = []

QUERY_EXPANSIONS: List[Tuple[re.Pattern[str], str]] = []


def repair_query(query: str) -> Dict[str, Any]:
    """Return a search query improved for local retrieval, without using an LLM."""
    search_query = _normalize_nl(query)
    notes: List[str] = []

    if search_query != query.strip():
        notes.append("Stripped conversational scaffolding.")

    for pattern, replacement, note in TYPO_REPLACEMENTS:
        search_query, count = pattern.subn(replacement, search_query)
        if count:
            notes.append(note)

    lowered = search_query.lower()
    expansions = []
    for pattern, expansion in QUERY_EXPANSIONS:
        if pattern.search(search_query) and expansion.lower() not in lowered:
            expansions.append(expansion)

    if expansions:
        search_query = f"{search_query} {' '.join(expansions)}"
        notes.append("Expanded acronym context for semantic retrieval.")

    return {
        "original": query,
        "search": search_query,
        "changed": search_query != query,
        "notes": notes,
    }
