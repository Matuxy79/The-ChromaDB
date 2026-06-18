# CLS Synchrotron Research Query — RAG+CAG Prototype (v1.2)

A multi-domain Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, FastAPI, and ChromaDB. Supports six research disciplines indexed as separate metadata-gated scopes.

Retrieval is semantic (sentence-transformers `all-MiniLM-L6-v2`, fully offline on CPU) with a lexical keyword scan running in parallel as a safety net, plus natural-language query repair so messy human phrasing maps to the right vector.

## Research Domains

| Scope | `domain` tag | Who uses it |
| --- | --- | --- |
| All disciplines | `None` | cross-domain queries |
| Chemistry | `chemistry` | chemical analysis, XRF, crystallography |
| Computer Science | `computer_science` | control software, data pipelines |
| Biology | `biology` | protein crystallography, imaging |
| Physics | `physics` | condensed matter, diffraction, optics |
| Mathematics | `mathematics` | data analysis, modelling |
| Literature | `literature` | students, general queries |

Tag each document at upload time using the **Assign a research domain** selector. The sidebar scope filter gates Chroma retrieval per domain; **All disciplines** bypasses the filter.

## Architecture

DocuSearch-inspired: retrieval is instant and primary. The grounded extractive answer from the RAG/CAG layer is always shown first. An optional generative carrier (default: OpenRouter · `openai/gpt-oss-120b`) synthesizes a direct answer from the same evidence rows.

| # | Component | File |
| --- | --- | --- |
| 1 | Streamlit dual-UI (Full App + Ask Lane) | `app.py` |
| 2 | FastAPI shared bridge | `api.py` |
| 3 | RAG+CAG instant backend | `cls_service.py`, `cls_backend/pipeline.py` |
| 4 | Generative carrier | `cls_backend/dllm.py`, `/v1/dllm/*` |

## Two UIs

The landing page offers two entry points:

- **Full App** — all roles (Admin / Scientist / Staff / User), corpus admin, upload, precision controls, graded eval, optional LLM synthesis.
- **Ask Lane** — bright llama.cui-style chat interface for non-technical users: chat input → cited answer with source chips, conversation history, no LLM, no engineering telemetry. Instant retrieval only, for speed.

Both UIs share the same underlying retrieval backend.

## Quick Launch

```bash
./scripts/launch_cls.sh
```

Creates `.venv` if needed, installs packages, and opens the UI at `http://localhost:8501`. Does not start Ollama or pull any LLM. On first run the embedder (`all-MiniLM-L6-v2`, ~80 MB) downloads once to `~/.cache/huggingface/`; after that it runs fully offline.

### Carrier (optional, Full App synthesis)

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
POST /v1/chat/completions   # models: cls-rag-cag-v1.0 | CLS_DLLM_MODEL
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

- **Admin sidebar**: one-click index of the local literature test corpus (`data/training_corpus/test_books` by default; override with `CLS_DEFAULT_DOCUMENTS_DIR`).
- **Admin / Scientist sidebar**: drag-and-drop batch upload of PDF, TXT, MD, DOCX, HTML, CSV, TSV, and JSON with domain tagging.
- **`ingest_daemon.py`**: optional batch indexer for folder-watch experiments.

> **Upgrading from v1.1?** The encoder changed to 384d MiniLM, so collections were renamed `cls_v2_*`. Hit **Reset Chroma index** in the admin sidebar and re-index once.

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
- [Safety flags](docs/SAFETY.md)
- [User guide](docs/USER_GUIDE.md)
