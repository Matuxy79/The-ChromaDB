# CLS Synchrotron Research Query — RAG+CAG Prototype (1.5v)

A Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, FastAPI, and ChromaDB. Retrieval is gated by CLS beamline metadata so each beamline lane can be queried independently or together.

Temporary fast mode is enabled by default: generation is disabled and queries use deterministic keyword retrieval for millisecond lookups on the prototype corpus. Set `CLS_RETRIEVAL_ONLY=0` and `CLS_KEYWORD_ONLY=0` to restore hybrid semantic retrieval plus optional carrier synthesis.

## Beamline Scopes

The current app is beamline-scoped, not discipline-scoped. Uploaded documents are tagged from the **Assign a beamline** selector, and the same shared scope map drives both the Full App and Ask Lane.

Choose **All beamlines** to bypass the metadata filter and search the full indexed store. The named scopes currently cover BioXAS-Imaging, BioXAS-Spectroscopy, BMIT, BXDS, CLS@APS, CMCF, EIML, Far-IR, HXMA, IDEAS, Mid-IR, QMSC, REIXS, SGM, SM, SXRMB, SyLMAND, VESPERS, and VLS-PGM.

## Architecture

DocuSearch-inspired: retrieval is instant and primary. The grounded extractive answer from the RAG/CAG layer is always shown first. Optional generative carrier synthesis (default: OpenRouter · `openai/gpt-oss-120b`) is currently blocked by retrieval-only mode.

| # | Component | File |
| --- | --- | --- |
| 1 | Streamlit dual-UI (Full App + Ask Lane) | `app.py` |
| 2 | FastAPI shared bridge | `api.py` |
| 3 | RAG+CAG instant backend | `cls_service.py`, `cls_backend/pipeline.py` |
| 4 | Generative carrier | `cls_backend/dllm.py`, `/v1/dllm/*` |

## Two UIs

The landing page offers two entry points:

- **Full App** — Admin / User roles, corpus admin, upload, precision controls, graded eval, optional LLM synthesis.
- **Ask Lane** — bright llama.cui-style chat interface for non-technical users: chat input → cited answer with source chips, conversation history, no LLM, no engineering telemetry. Instant retrieval only, for speed.

Both UIs share the same underlying retrieval backend.

## Quick Launch

```bash
./scripts/launch_cls.sh
```

Creates `.venv` if needed, installs packages, and opens the UI at `http://localhost:8501`. Does not start Ollama or pull any LLM.

Fast mode defaults:

```bash
export CLS_RETRIEVAL_ONLY=1
export CLS_KEYWORD_ONLY=1
```

### Carrier (optional, Full App synthesis)

Carrier calls are disabled while `CLS_RETRIEVAL_ONLY=1`. To test generation again, set `CLS_RETRIEVAL_ONLY=0` and `CLS_KEYWORD_ONLY=0` before launch.

The carrier is any OpenAI-compatible `/v1/chat/completions` endpoint. Pick one — no code change:

```bash
# Cloud (OpenRouter)
export CLS_DLLM_API_KEY="sk-or-..."
# export CLS_DLLM_API_URL="https://openrouter.ai/api/v1"   # default
# export CLS_DLLM_MODEL="openai/gpt-oss-120b"              # default

# Local llama.cpp (offline, no key) — run: llama-server -m model.gguf --port 8080
# export CLS_DLLM_API_URL="http://localhost:8080/v1"
# unset CLS_DLLM_API_KEY

# Local Ollama
# export CLS_DLLM_API_URL="http://localhost:11434/v1"
```

Without a carrier the app is fully offline: semantic retrieval + instant cited extraction. The **Ask Lane never uses the carrier** — it is retrieval-only by design.

## API + Dual Frontend

```bash
./scripts/launch_api.sh
CLS_USE_API=1 CLS_API_URL=http://127.0.0.1:8010 ./scripts/launch_cls.sh
```

Key endpoints:

```text
GET  /health
POST /v1/query
POST /v1/chat/completions   # cls-rag-cag-v1.0; CLS_DLLM_MODEL only when retrieval-only is off
GET  /v1/dllm/status
POST /v1/dllm/chat
```

Example:

```bash
curl http://127.0.0.1:8010/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the sample mounting procedure?","top_k":16}'
```

## Indexing Documents

- **Workspace -> Corpus admin**: one-click index of the local literature test corpus (`data/training_corpus/test_books` by default; override with `CLS_DEFAULT_DOCUMENTS_DIR`).
- **Main page upload panel**: drag-and-drop batch upload of PDF, TXT, MD, DOCX, HTML, CSV, TSV, and JSON with beamline tagging.
- **`ingest_daemon.py`**: optional batch indexer for folder-watch experiments.

> **Upgrading from v1.1 to 1.5** The encoder changed to 384d MiniLM, so collections were renamed `cls_v2_*`. Open **Workspace**, hit **Reset Chroma index**, and re-index once.

## Prototype HUD

A fixed floating overlay (bottom-right) shows live session telemetry for dev use:

| Field | Meaning |
| --- | --- |
| `ui` | Current surface (Full App / Ask Lane) |
| `turn` | Query count this session |
| `score` | Top Chroma similarity score |
| `cache` | HIT (green) / MISS (orange) |

## Guide Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Inference carrier](docs/DLLM.md)
- [User guide](docs/USER_GUIDE.md)
