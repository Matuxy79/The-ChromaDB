from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "Training for perfect in ui graded"
DEFAULT_MANUAL = MANUAL_DIR / "IVU beamline manual - Apr 10 2026.pdf"
CHROMA_DIR = APP_ROOT / "chroma_store"

COLLECTION_NAME = "cls_v1_dsrag_evidence"            # Evidence Store (768d)
CACHE_COLLECTION_NAME = "cls_v1_dsrag_cag_cache"   # CAG Layer (768d)
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

APP_VERSION = "v1.1"

DEFAULT_API_URL = os.getenv("CLS_API_URL", "http://127.0.0.1:8010")

# Default generative carrier: OpenRouter + openai/gpt-oss-120b (one key, 100+ models).
# Plug-and-play — paste CLS_DLLM_API_KEY into cls.env (gitignored) and launch. Point
# CLS_DLLM_API_URL at http://localhost:11434/v1 to switch the carrier to a local Ollama
# model instead (no key needed).
DEFAULT_DLLM_API_URL = os.getenv("CLS_DLLM_API_URL", "https://openrouter.ai/api/v1").rstrip("/")
DEFAULT_DLLM_API_KEY = os.getenv("CLS_DLLM_API_KEY", "")
DEFAULT_DLLM_MODEL = os.getenv("CLS_DLLM_MODEL", "openai/gpt-oss-120b")
