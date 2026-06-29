"""Central configuration — every tuneable constant and environment variable in one place.

All runtime behaviour is controlled through environment variables so the same
codebase can run in fast prototype mode (defaults) or with a live LLM carrier,
without touching source code.  Set values in ``cls.env`` (gitignored) before
launching; the launch scripts source that file automatically.

Key environment variables
-------------------------
CLS_RETRIEVAL_ONLY      1 = disable the generative carrier entirely (default: 1)
CLS_KEYWORD_ONLY        1 = skip vector search, use lexical term overlap only (default: 1)
CLS_DLLM_API_KEY        API key for the generative carrier (OpenRouter, OpenAI, etc.)
CLS_DLLM_API_URL        Base URL for the carrier (default: https://openrouter.ai/api/v1)
CLS_DLLM_MODEL          Model name passed to the carrier (default: openai/gpt-oss-120b)
CLS_PARROT_URL          Ollama / llama.cpp base URL for the Ask Lane parrot (default: localhost:11434)
CLS_API_URL             Internal FastAPI bridge URL (default: http://127.0.0.1:8010)

Beamline scopes (RESEARCH_SCOPES)
----------------------------------
Each entry maps a human-readable beamline name to a ChromaDB metadata filter dict.
Passing ``None`` as the filter returns the entire corpus (All beamlines).
The dict keys (e.g. ``"bioxas_imaging"``) also serve as folder names under
``data/corpus/`` for batch ingest via ``scripts/ingest_corpus.py``.

To add a new beamline: append one entry to RESEARCH_SCOPES and create the
corresponding sub-folder under ``data/corpus/``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "data" / "training_corpus"
DEFAULT_DOCUMENTS_DIR = Path(
    os.getenv("CLS_DEFAULT_DOCUMENTS_DIR", str(MANUAL_DIR / "test_books"))
)
DEFAULT_DOCUMENT_DOMAIN = os.getenv("CLS_DEFAULT_DOCUMENT_DOMAIN", "")
CHROMA_DIR = APP_ROOT / "chroma_store"

COLLECTION_NAME = "cls_v2_evidence"            # Evidence Store (384d, MiniLM)
CACHE_COLLECTION_NAME = "cls_v2_cag_cache"   # CAG Layer (384d, MiniLM)
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

APP_VERSION = "v1.2"

# Research scopes — one lane per CLS beamline. Shared by every frontend (Streamlit,
# Chainlit). None bypasses the Chroma metadata filter; any other value is passed as
# metadata_filter={"domain": ...}. The domain slugs double as the folder names under
# data/corpus/ that scripts/ingest_corpus.py reads (it derives its valid-domain set
# from this map), so a lane only returns evidence once docs are ingested under its slug.
RESEARCH_SCOPES = {
    "All beamlines":                                None,
    "BioXAS-Imaging":                               {"domain": "bioxas_imaging"},
    "BioXAS-Spectroscopy":                          {"domain": "bioxas_spectroscopy"},
    "BMIT — Biomedical Imaging & Therapy":          {"domain": "bmit"},
    "BXDS — Brockhouse Diffraction & Scattering":   {"domain": "bxds"},
    "CLS@APS":                                      {"domain": "cls_aps"},
    "CMCF — Macromolecular Crystallography":        {"domain": "cmcf"},
    "EIML — Electron Imaging & Microanalysis":      {"domain": "eiml"},
    "Far-IR — Far Infrared":                        {"domain": "far_ir"},
    "HXMA — Hard X-ray Micro-Analysis":             {"domain": "hxma"},
    "IDEAS":                                        {"domain": "ideas"},
    "Mid-IR — Mid Infrared Spectromicroscopy":      {"domain": "mid_ir"},
    "QMSC — Quantum Materials Spectroscopy":        {"domain": "qmsc"},
    "REIXS — Resonant In/Elastic X-ray Scattering": {"domain": "reixs"},
    "SGM — Spherical Grating Monochromator":        {"domain": "sgm"},
    "SM — Soft X-ray Spectromicroscopy":            {"domain": "sm"},
    "SXRMB — Soft X-ray Microcharacterization":     {"domain": "sxrmb"},
    "SyLMAND — Micro & Nano Devices":               {"domain": "sylmand"},
    "VESPERS":                                      {"domain": "vespers"},
    "VLS-PGM — Variable Line Spacing PGM":          {"domain": "vls_pgm"},
}

# Admin-added scopes are layered on top of the built-ins above and persisted here
# so they survive restarts and are shared by every session/frontend. The built-in
# names stay the source of truth in code; only runtime additions go to disk.
CUSTOM_SCOPES_PATH = APP_ROOT / "data" / "custom_scopes.json"
_BUILTIN_SCOPE_NAMES = frozenset(RESEARCH_SCOPES)


def _slugify_domain(value: str) -> str:
    """Normalise free text into a corpus folder slug (lowercase, a-z0-9 and _)."""
    return re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")


def _load_custom_scopes() -> dict[str, dict | None]:
    """Read admin-added scopes from disk; tolerate a missing or corrupt file."""
    if not CUSTOM_SCOPES_PATH.exists():
        return {}
    try:
        raw = json.loads(CUSTOM_SCOPES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        name.strip(): mfilter
        for name, mfilter in raw.items()
        if isinstance(name, str) and name.strip()
    }


def _persist_custom_scopes() -> None:
    """Write every non-built-in scope back to CUSTOM_SCOPES_PATH as JSON."""
    custom = {
        name: mfilter
        for name, mfilter in RESEARCH_SCOPES.items()
        if name not in _BUILTIN_SCOPE_NAMES
    }
    CUSTOM_SCOPES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_SCOPES_PATH.write_text(
        json.dumps(custom, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_research_scope(name: str, domain: str = "") -> tuple[str, dict]:
    """Register a new beamline lane and persist it so it survives restarts.

    Mutates RESEARCH_SCOPES in place so any frontend holding a reference to the
    dict picks the new lane up on its next rerun, then saves the custom subset to
    disk. ``domain`` defaults to a slug derived from ``name``. The slug doubles as
    the folder name under ``data/corpus/`` that ingest reads, so a new lane stays
    empty until documents are ingested under that slug.

    Returns ``(display_name, metadata_filter)``. Raises ``ValueError`` on empty or
    duplicate input.
    """
    name = (name or "").strip()
    slug = _slugify_domain(domain or name)
    if not name:
        raise ValueError("Scope name cannot be empty.")
    if not slug:
        raise ValueError("Domain slug must contain at least one letter or digit.")
    if name in RESEARCH_SCOPES:
        raise ValueError(f"A scope named “{name}” already exists.")
    existing_slugs = {
        f["domain"] for f in RESEARCH_SCOPES.values()
        if isinstance(f, dict) and "domain" in f
    }
    if slug in existing_slugs:
        raise ValueError(f"The domain “{slug}” is already used by another scope.")
    mfilter = {"domain": slug}
    RESEARCH_SCOPES[name] = mfilter
    _persist_custom_scopes()
    return name, mfilter


# Layer any previously saved admin scopes on top of the built-ins, in place, so
# existing references to RESEARCH_SCOPES stay valid.
for _scope_name, _scope_filter in _load_custom_scopes().items():
    RESEARCH_SCOPES.setdefault(_scope_name, _scope_filter)

DEFAULT_API_URL = os.getenv("CLS_API_URL", "http://127.0.0.1:8010")

# Temporary speed-first mode: keep inference on deterministic retrieval only.
# Set CLS_RETRIEVAL_ONLY=0 to restore carrier synthesis/cleanup/proxy calls.
RETRIEVAL_ONLY = _env_flag("CLS_RETRIEVAL_ONLY", default=True)

# Temporary keyword-first mode: skip semantic embedding/CAG cache lookup on queries and
# rank directly by lexical term overlap. Set CLS_KEYWORD_ONLY=0 to restore hybrid
# semantic+lexical retrieval.
KEYWORD_ONLY_RETRIEVAL = _env_flag("CLS_KEYWORD_ONLY", default=True)

# Default generative carrier: OpenRouter + openai/gpt-oss-120b (one key, 100+ models).
# Plug-and-play — paste CLS_DLLM_API_KEY into cls.env (gitignored) and launch. Point
# CLS_DLLM_API_URL at http://localhost:11434/v1 to switch the carrier to a local Ollama
# model instead (no key needed).
DEFAULT_DLLM_API_URL = os.getenv("CLS_DLLM_API_URL", "https://openrouter.ai/api/v1").rstrip("/")
DEFAULT_DLLM_API_KEY = os.getenv("CLS_DLLM_API_KEY", "")
DEFAULT_DLLM_MODEL = os.getenv("CLS_DLLM_MODEL", "openai/gpt-oss-120b")

# Ask Lane parrot: a small local model (Ollama, llama.cpp underneath) that rephrases the
# grounded extractive evidence into natural language. No reasoning — pure faithful mimicry,
# guarded by the same number-grounding check the carrier uses. Independent of the carrier
# above so the Ask Lane stays local/offline even when the carrier points at OpenRouter.
DEFAULT_PARROT_URL = os.getenv("CLS_PARROT_URL", "http://localhost:11434/v1").rstrip("/")
DEFAULT_PARROT_MODEL = os.getenv("CLS_PARROT_MODEL", "qwen2.5:0.5b")
