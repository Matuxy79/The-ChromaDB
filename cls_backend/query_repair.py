import re
from typing import Any, Dict, List, Tuple


TYPO_REPLACEMENTS: List[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bi[\s-]+v[\s-]+u\b", re.IGNORECASE), "IVU", "Normalized spaced IVU acronym."),
    (re.compile(r"\bi[\s-]+v[\s-]+w\b", re.IGNORECASE), "IVW", "Normalized spaced IVW acronym."),
    (re.compile(r"\bbxd\b", re.IGNORECASE), "BXDS", "Corrected BXD to BXDS."),
]

QUERY_EXPANSIONS: List[Tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\bivu\b", re.IGNORECASE),
        "IVU beamline BXDS in-vacuum undulator optical components overview",
    ),
    (
        re.compile(r"\bivw\b", re.IGNORECASE),
        "IVW wiggler beamline low energy wiggler LEW high energy wiggler HEW "
        "power diffraction SAXS grazing incidence diffraction pair distribution "
        "function extreme conditions",
    ),
    (
        re.compile(r"\bbxds\b", re.IGNORECASE),
        "BXDS beamline IVU IVW diffraction SAXS grazing incidence pair distribution function",
    ),
    (
        re.compile(r"\b(lew|hew)\b", re.IGNORECASE),
        "IVW wiggler beamline low energy wiggler high energy wiggler",
    ),
    (
        re.compile(r"\bwiggler\b", re.IGNORECASE),
        "IVW low energy wiggler LEW high energy wiggler HEW",
    ),
]


def repair_query(query: str) -> Dict[str, Any]:
    """Return a search query improved for local retrieval, without using an LLM."""
    search_query = query.strip()
    notes: List[str] = []

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
        notes.append("Expanded beamline acronym context for semantic retrieval.")

    return {
        "original": query,
        "search": search_query,
        "changed": search_query != query,
        "notes": notes,
    }
