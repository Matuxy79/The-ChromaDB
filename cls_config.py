from __future__ import annotations

import os
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
