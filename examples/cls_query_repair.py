import re
from typing import Any, Dict, List, Tuple


TYPO_REPLACEMENTS: List[Tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\bivw\b", re.IGNORECASE), "IVU", "Corrected IVW to IVU."),
    (re.compile(r"\bi\s*v\s*w\b", re.IGNORECASE), "IVU", "Corrected IVW to IVU."),
    (re.compile(r"\bbxd\b", re.IGNORECASE), "BXDS", "Corrected BXD to BXDS."),
]

QUERY_EXPANSIONS = {
    "ivu": "IVU beamline BXDS in-vacuum undulator optical components overview",
    "bxds": "BXDS IVU beamline in-vacuum undulator optical components overview",
}


def repair_query(query: str) -> Dict[str, Any]:
    """Return a search query improved for local retrieval, without using an LLM."""
    search_query = query.strip()
    notes: List[str] = []

    for pattern, replacement, note in TYPO_REPLACEMENTS:
        search_query, count = pattern.subn(replacement, search_query)
        if count:
            notes.append(note)

    lowered = search_query.lower()
    expansions = [
        expansion
        for trigger, expansion in QUERY_EXPANSIONS.items()
        if trigger in lowered and expansion.lower() not in lowered
    ]

    if expansions:
        search_query = f"{search_query} {' '.join(expansions)}"
        notes.append("Expanded beamline acronym context for semantic retrieval.")

    return {
        "original": query,
        "search": search_query,
        "changed": search_query != query,
        "notes": notes,
    }
