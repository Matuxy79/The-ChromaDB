"""CLS spectral question classifier and presentation helpers.

Pure-Python, side-effect-free. The chat UI asks `classify_query` for every prompt
and uses the returned category to colour the answer along the visible-light
spectrum (rose/orange anchor the most common operations questions), attach a
symbolic glyph, and highlight semantic entities in the text.

This is a *presentation* layer only — it never touches retrieval, embeddings, or
the facts in an answer. Colour metadata can also be used by the research-scope
filters; the safety category reuses the audited keyword set in `cls_backend/safety.py`
rather than duplicating it.
"""

from __future__ import annotations

import html
import re
from typing import List, Tuple

from cls_backend.safety import SAFETY_TOPICS as _SAFETY_WORDS, detect_safety_topic


# Categories in visible-spectrum (wavelength) order. Rose/orange sit up front
# because safety and contact questions are the most common — and the ones where
# a clear, warm visual cue matters most.
SPECTRUM: dict[str, dict] = {
    "safety": {
        "hue": "#ff4d6d",  # rose-red
        "glyph": "⚠",
        "label": "Safety",
        "triggers": [],  # resolved via detect_safety_topic, not a flat list
    },
    "contacts": {
        "hue": "#ff8a3d",  # orange (preferred anchor)
        "glyph": "☎",
        "label": "Contacts",
        "triggers": [
            "phone", "number", "call", "contact", "who do i", "who should i",
            "reach", "extension", "ext.", "ext ", "email", "responsible",
        ],
    },
    "procedure": {
        "hue": "#ffc24b",  # amber
        "glyph": "⚙",
        "label": "Procedure",
        "triggers": [
            "how do i", "how to", "procedure", "steps", "step ", "start up",
            "startup", "start-up", "warm up", "warm-up", "warmup", "shutdown",
            "shut down", "align", "operate", "operating", "set up", "configure",
        ],
    },
    "specs": {
        "hue": "#6fd58a",  # green
        "glyph": "📐",
        "label": "Specs / Optics",
        "triggers": [
            "energy", "range", "flux", "optics", "mirror", "mono", "monochromator",
            "undulator", "wiggler", "gap", "resolution", "spec", "specification",
            "wavelength", "kev", "ev ", "beam size", "harmonic",
        ],
    },
    "general": {
        "hue": "#6aa9ff",  # blue
        "glyph": "◇",
        "label": "General",
        "triggers": [],  # fallback
    },
}

# Order to scan when no safety hit: most specific intent first, general last.
_SCAN_ORDER = ["contacts", "procedure", "specs"]


def classify_query(query: str) -> str:
    """Map a query to a spectral category key. Safety always wins."""
    if not query or not query.strip():
        return "general"
    if detect_safety_topic(query):
        return "safety"
    haystack = query.lower()
    for category in _SCAN_ORDER:
        if any(trigger in haystack for trigger in SPECTRUM[category]["triggers"]):
            return category
    return "general"


def category_meta(category: str) -> dict:
    """Hue / glyph / label for a category, falling back to general."""
    return SPECTRUM.get(category, SPECTRUM["general"])


def glow_css(hue: str, score: float) -> str:
    """A box-shadow whose blur and alpha scale with retrieval confidence.

    Bright, tight glow when the top hit is strong; dim and soft when it's a
    stretch. `score` is the 0..1 relevance already computed by the app.
    """
    s = max(0.0, min(1.0, float(score)))
    blur = int(8 + s * 18)          # 8px .. 26px
    alpha = round(0.15 + s * 0.35, 3)  # 0.15 .. 0.50
    rgba = _hex_to_rgba(hue, alpha)
    # Coloured halo scales with confidence, over a soft neutral shadow that reads
    # well on a light background.
    return f"0 0 {blur}px {rgba}, 0 6px 20px rgba(120,80,60,0.12)"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# Semantic entities to highlight in answer text.
_ACRONYMS = ["IVU", "IVW", "BXDS", "SOE-3", "LEW", "HEW", "SAXS", "PDF"]
_SAFETY_VOCAB = sorted(
    {word for words in _SAFETY_WORDS.values() for word in words}, key=len, reverse=True
)

# Entity patterns, applied in a single left-to-right pass so a match can never be
# wrapped inside another (e.g. an acronym sitting inside a [Source: ...] citation).
# Order matters: citations first (they swallow any acronyms/numbers they contain),
# then phones, then acronyms.
#   citations : [Source: IVU manual.pdf, page 4]
#   phones    : 911, ext. 3832, 306-241-1999, 9-306-966-5555
#   acronyms  : IVU, IVW, SOE-3 ...
_BASE_TOKEN = (
    r"(?P<cite>\[Source:[^\]]*\])"
    r"|(?P<phone>(?:ext\.?\s*)?\b\d{3,4}(?:[-\s]?\d{3,4})*\b)"
    r"|(?P<acr>\b(?:" + "|".join(map(re.escape, _ACRONYMS)) + r")\b)"
)
_SAFETY_ALT = r"|(?P<safety>(?<!\w)(?:" + "|".join(map(re.escape, _SAFETY_VOCAB)) + r")(?!\w))"

_TOKEN_RE = re.compile(_BASE_TOKEN, re.IGNORECASE)
_TOKEN_RE_SAFETY = re.compile(_BASE_TOKEN + _SAFETY_ALT, re.IGNORECASE)


def _wrap(match: "re.Match[str]") -> str:
    kind = match.lastgroup
    return f'<span class="tok tok-{kind}">{match.group()}</span>'


# Tiny stoplist so DocuSearch-style query highlighting doesn't light up filler words.
_HIT_STOP = {
    "the", "and", "for", "are", "what", "which", "who", "how", "where", "when",
    "with", "from", "into", "does", "did", "can", "should", "would", "about",
    "this", "that", "these", "those", "you", "your", "ivu", "ivw",
}


def _hit_terms(query: str) -> list[str]:
    """Distinct content terms from the query, longest-first (so overlaps prefer longer)."""
    terms = {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}", (query or "").lower())
        if tok not in _HIT_STOP
    }
    return sorted(terms, key=len, reverse=True)


def decorate(text: str, category: str, query: str = "") -> str:
    """HTML-escape `text`, then wrap semantic entities (and query hits) in styled spans.

    Output is inline HTML safe to drop into `st.markdown(..., unsafe_allow_html=True)`.
    A single left-to-right pass guarantees no match is wrapped inside another. Precedence:
    citation > phone > acronym > query-hit > safety. Safety words are only underlined when
    the question itself was safety-flagged. `query` adds DocuSearch-style green term hits.
    """
    safe = html.escape(text)
    parts = _BASE_TOKEN
    terms = _hit_terms(query)
    if terms:
        parts += r"|(?P<hit>\b(?:" + "|".join(map(re.escape, terms)) + r")\b)"
    if category == "safety":
        parts += _SAFETY_ALT
    return re.compile(parts, re.IGNORECASE).sub(_wrap, safe)


# Problem-asking chips: (category, text). Clicking one pre-fills the query.
# Spread across the spectrum so the colour system is visible at a glance.
SUGGESTED_PROBLEMS: List[Tuple[str, str]] = [
    ("safety", "radiation interlock and search and secure procedure"),
    ("contacts", "Who do I call for a vacuum failure?"),
    ("procedure", "IVU warm-up procedure steps"),
    ("specs", "IVU undulator energy range and optics"),
    ("contacts", "IVU beamline contacts and phone numbers"),
    # Conceptual / natural-language starter — shines in Hybrid mode, where the carrier may
    # answer from its own knowledge when the indexed manual doesn't explain the physics.
    ("specs", "Explain how an undulator produces X-rays"),
]
