from __future__ import annotations

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
MANUAL_DIR = APP_ROOT / "Training for perfect in ui graded"
DEFAULT_DOCUMENTS_DIR = Path(
    os.getenv("CLS_DEFAULT_DOCUMENTS_DIR", str(MANUAL_DIR / "test_books"))
)
DEFAULT_DOCUMENT_DOMAIN = os.getenv("CLS_DEFAULT_DOCUMENT_DOMAIN", "literature")
CHROMA_DIR = APP_ROOT / "chroma_store"

COLLECTION_NAME = "cls_v2_dsrag_evidence"            # Evidence Store (384d, MiniLM)
CACHE_COLLECTION_NAME = "cls_v2_dsrag_cag_cache"   # CAG Layer (384d, MiniLM)
CHUNK_TARGET_CHARS = 1100
CHUNK_OVERLAP_CHARS = 180

APP_VERSION = "v1.2"

# Research scopes — shared by every frontend (Streamlit, Chainlit). None bypasses the
# Chroma metadata filter; any other value is passed as metadata_filter={"domain": ...}.
RESEARCH_SCOPES = {
    "All disciplines":  None,
    "Chemistry":        {"domain": "chemistry"},
    "Computer Science": {"domain": "computer_science"},
    "Biology":          {"domain": "biology"},
    "Physics":          {"domain": "physics"},
    "Mathematics":      {"domain": "mathematics"},
    "Literature":       {"domain": "literature"},   # student general-query lane
}

DEFAULT_API_URL = os.getenv("CLS_API_URL", "http://127.0.0.1:8010")

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
