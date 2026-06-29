"""Spectral question classifier and presentation helpers.

Pure-Python, side-effect-free. The chat UI asks `classify_query` for every
prompt and uses the returned category to colour the answer card along the
visible-light spectrum, attach a symbolic glyph, and highlight semantic
entities in the retrieved text.

This is a *presentation* layer only — it never touches retrieval, embeddings,
or the facts in an answer.
"""

from __future__ import annotations

import html
import re


# ---------------------------------------------------------------------------
# Palette — the five hues of the visible-light spectrum, red → blue.
# These are the single source of truth for every colour in the UI: query cards,
# the DB status badge, HIT/MISS labels, CSS gradients.  Import these constants
# rather than repeating hex strings anywhere else in the codebase.
# ---------------------------------------------------------------------------
HUE_RED    = "#ff4d6d"   # danger · error · rose (leftmost, high energy)
HUE_ORANGE = "#ff8a3d"   # contacts · --accent
HUE_AMBER  = "#ffc24b"   # procedure · warning
HUE_GREEN  = "#6fd58a"   # specs · healthy · cache hit
HUE_BLUE   = "#6aa9ff"   # general · info (rightmost, low energy)

# Categories in visible-spectrum order.
# Contacts sits up front because it carries the strongest visual cue.
SPECTRUM: dict[str, dict] = {
    "contacts": {
        "hue": HUE_ORANGE,
        "glyph": "☎",
        "label": "Contacts",
        "triggers": [
            "phone", "number", "call", "contact", "who do i", "who should i",
            "reach", "extension", "ext.", "ext ", "email", "responsible",
        ],
    },
    "procedure": {
        "hue": HUE_AMBER,
        "glyph": "⚙",
        "label": "Procedure",
        "triggers": [
            "how do i", "how to", "procedure", "protocol", "method", "steps",
            "step ", "start up", "startup", "start-up", "warm up", "warm-up",
            "warmup", "shutdown", "shut down", "align", "operate", "operating",
            "set up", "configure", "preparation", "prepare",
        ],
    },
    "specs": {
        "hue": HUE_GREEN,
        "glyph": "📐",
        "label": "Specs",
        "triggers": [
            "energy", "range", "flux", "optics", "mirror", "mono", "monochromator",
            "resolution", "spec", "specification", "wavelength", "kev", "ev ",
            "beam size", "frequency", "concentration", "temperature", "pressure",
            "intensity", "diffraction", "absorption", "emission",
        ],
    },
    "general": {
        "hue": HUE_BLUE,
        "glyph": "◇",
        "label": "General",
        "triggers": [],  # fallback
    },
}

_SCAN_ORDER = ["contacts", "procedure", "specs"]


def classify_query(query: str) -> str:
    """Map a query to a spectral category key."""
    if not query or not query.strip():
        return "general"
    haystack = query.lower()
    for category in _SCAN_ORDER:
        if any(trigger in haystack for trigger in SPECTRUM[category]["triggers"]):
            return category
    return "general"


def category_meta(category: str) -> dict:
    """Hue / glyph / label for a category, falling back to general."""
    return SPECTRUM.get(category, SPECTRUM["general"])


def glow_css(hue: str, score: float) -> str:
    """Box-shadow whose blur and alpha scale with retrieval confidence."""
    s = max(0.0, min(1.0, float(score)))
    blur = int(8 + s * 18)
    alpha = round(0.15 + s * 0.35, 3)
    rgba = _hex_to_rgba(hue, alpha)
    return f"0 0 {blur}px {rgba}, 0 6px 20px rgba(120,80,60,0.12)"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Facility or domain-specific acronyms to highlight in answer text.
# Extend this list at deployment time for your specific beamline or domain vocabulary.
_ACRONYMS: list[str] = [
    "SAXS", "WAXS", "PDF", "XRD", "XRF", "XANES", "EXAFS",
]

_BASE_TOKEN = (
    r"(?P<cite>\[Source:[^\]]*\])"
    r"|(?P<phone>(?:ext\.?\s*)?\b\d{3,4}(?:[-\s]?\d{3,4})*\b)"
    + (r"|(?P<acr>\b(?:" + "|".join(map(re.escape, _ACRONYMS)) + r")\b)" if _ACRONYMS else "")
)


def _wrap(match: "re.Match[str]") -> str:
    kind = match.lastgroup
    return f'<span class="tok tok-{kind}">{match.group()}</span>'


_HIT_STOP = {
    "the", "and", "for", "are", "what", "which", "who", "how", "where", "when",
    "with", "from", "into", "does", "did", "can", "should", "would", "about",
    "this", "that", "these", "those", "you", "your",
}


def _hit_terms(query: str) -> list[str]:
    """Distinct content terms from the query, longest-first."""
    terms = {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (query or "").lower())
        if tok not in _HIT_STOP
    }
    return sorted(terms, key=len, reverse=True)


def decorate(text: str, category: str, query: str = "") -> str:
    """HTML-escape text then wrap semantic entities in styled spans."""
    safe = html.escape(text)
    parts = _BASE_TOKEN
    terms = _hit_terms(query)
    if terms:
        parts += r"|(?P<hit>\b(?:" + "|".join(map(re.escape, terms)) + r")\b)"
    return re.compile(parts, re.IGNORECASE).sub(_wrap, safe)
