"""CLS-specific safety flagging.

Pure-Python, side-effect-free. The chat layer asks `detect_safety_topic` for
every user query; if anything matches, the UI surfaces an emergency-contact
card and reminds the scientist to confirm procedures with beamline staff.

All thresholds and keyword sets are kept here so safety policy lives in one
file and can be audited without touching the chat or retrieval code.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Topic → trigger words. Lower-case match against the query.
# Keep lists short and conservative; false positives just show an extra
# banner, false negatives let an unsafe answer through.
SAFETY_TOPICS: Dict[str, List[str]] = {
    "beryllium": [
        "beryllium", "be window", "be dome", "cbd",
    ],
    "radiation": [
        "radiation", "shutter open", "ssh", "psh", "compton",
        "exposure", "interlock", "search and secure",
    ],
    "laser": [
        "laser", "class 2 laser", "class 3 laser", "laser safety",
    ],
    "cryogenics": [
        "cryostat", "cryocooler", "liquid helium", "helium leak",
        "warm-up", "warm up procedure", "venting the chamber",
    ],
    "high_voltage": [
        "high voltage", "1050 v", "hv", "nhq", "nmh 203m",
        "electrocut", "shock",
    ],
    "vacuum": [
        "vacuum failure", "vent the", "venting", "vac leak", "rupture",
    ],
    "fire_medical": [
        "fire", "smoke", "ambulance", "injury", "injured", "bleed",
        "burn", "burnt", "unconscious", "evacuate", "evacuation",
        "emergency",
    ],
    "mechanical": [
        "collision", "crushed", "pinch", "trapped", "fell into",
    ],
}


# Distance threshold above which retrieval is considered low-confidence.
# Chroma cosine distance >0.55 usually means the top hit is a stretch for this
# prototype's retrieval traces.
LOW_CONFIDENCE_DISTANCE = 0.55


# Authoritative contacts from the IVU beamline manual (Apr 2026).
# Update this dict — not the UI — when phone numbers change.
EMERGENCY_CONTACTS: List[Dict[str, str]] = [
    {"who": "Fire / Ambulance",        "number": "911",            "note": "Serious emergency"},
    {"who": "U-Sask Security",         "number": "9-306-966-5555", "note": "From a CLS phone"},
    {"who": "CLS Control Room",        "number": "ext. 3570",      "note": "Operations / beam"},
    {"who": "Floor Coordinator",       "number": "ext. 3639",      "note": "Hutch access / floor"},
    {"who": "Health & Safety (HSE)",   "number": "ext. 3663",      "note": "Reportable incidents"},
    {"who": "Beamline (SOE-3 / IVU)",  "number": "ext. 3832",      "note": "In-hutch phone"},
    {"who": "Beatriz Moreno",          "number": "306-241-1999",   "note": "Beamline responsible"},
    {"who": "Narayan Appathurai",      "number": "306-514-1384",   "note": "Beamline scientist"},
    {"who": "Al Rahemtulla",           "number": "519-993-9137",   "note": "Beamline scientist"},
]


def detect_safety_topic(query: str) -> Optional[str]:
    """Return the safety topic key if any trigger word appears in the query."""
    if not query:
        return None
    haystack = query.lower()
    for topic, triggers in SAFETY_TOPICS.items():
        for word in triggers:
            if word in haystack:
                return topic
    return None


def low_confidence(hits: List[dict]) -> bool:
    """True if every retrieved hit is past the confidence threshold."""
    if not hits:
        return False
    distances = [h.get("distance") for h in hits if h.get("distance") is not None]
    if not distances:
        return False
    return min(distances) >= LOW_CONFIDENCE_DISTANCE
