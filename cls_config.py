from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "Training for perfect in ui graded"
DEFAULT_MANUAL = MANUAL_DIR / "IVU beamline manual - Apr 10 2026.pdf"
CHROMA_DIR = APP_ROOT / "chroma_store"

COLLECTION_NAME = "cls_ivu_manual_hash_v1"            # Evidence Store
CACHE_COLLECTION_NAME = "cls_cag_evidence_cache_v1"   # CAG Layer
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

APP_VERSION = "v0.9"

DEFAULT_API_URL = os.getenv("CLS_API_URL", "http://127.0.0.1:8010")
DEFAULT_DLLM_API_URL = os.getenv("CLS_DLLM_API_URL", "").rstrip("/")
DEFAULT_DLLM_API_KEY = os.getenv("CLS_DLLM_API_KEY", "")
DEFAULT_DLLM_MODEL = os.getenv("CLS_DLLM_MODEL", "gpt-oss-120b")
