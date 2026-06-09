# CLS RAG+CAG Prototype

A domain-specific Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, FastAPI, and ChromaDB.

## Architecture

This prototype separates indexing, shared retrieval, and frontend layers.

Like DocuSearch, the default answer is **instant clean parsed text** from the RAG/CAG dual layer. No LLM runs in the hot answer path. The systems are kept separate so the UI, retrieval backend, API bridge, and optional dLLM correction do not tangle:

| # | Component | Name | Where |
| --- | --- | --- | --- |
| 1 | Streamlit + OpenAI-compatible clients | **Dual frontend layers** | `app.py`, `/v1/chat/completions` |
| 2 | FastAPI | **Shared API bridge** | `api.py` |
| 3 | `HashEmbedder` + ChromaDB + CAG | **RAG+CAG instant backend** | `cls_service.py`, `examples/cls_pipeline.py` |
| 4 | `gpt-oss-120b` external connection | **dLLM API endpoint** (optional, off by default) | `/v1/dllm/*`, `cls_service.py`, `examples/cls_dllm.py` |

Within the backend, the **Retrieval Encoder** (`HashEmbedder`) encodes the query, the **CAG Layer** reuses prior evidence for near-identical questions, the **Evidence Store** (ChromaDB) is searched on a miss, and deterministic cleanup repairs extraction artifacts.

> The base app does not download or start local language models. The optional dLLM path calls an OpenAI-compatible API endpoint only when configured and toggled on.

## Prerequisites

The prototype does **not** require a heavyweight model download. The instant RAG/CAG path uses the deterministic `HashEmbedder` and works without `gpt-oss-120b`.

Optional dLLM correction is API-only:

```bash
export CLS_DLLM_API_URL="https://your-dllm-endpoint.example/v1"
export CLS_DLLM_API_KEY="..."
export CLS_DLLM_MODEL="gpt-oss-120b"
```

`CLS_DLLM_API_URL` should point to the external dLLM provider/runtime, not this app's own `http://127.0.0.1:8010/v1` API bridge.

## Quick Launch

For scientist chat, run:

```bash
./launch_cls.sh
```

The launcher creates `.venv` if needed, installs Python packages, and opens the UI on `http://localhost:8501`. It does not start Ollama, pull models, or check local model tags.

On Linux desktops, you can also double-click `CLS_RAG_CAG.desktop`. If the desktop asks whether to trust or execute the file, choose the execute/trust option.

## API + Dual Frontend Launch

Start the shared API:

```bash
./launch_api.sh
```

Then start Streamlit through the API bridge:

```bash
CLS_USE_API=1 CLS_API_URL=http://127.0.0.1:8010 ./launch_cls.sh
```

For an OpenAI-compatible frontend, add a connection with base URL:

```text
http://127.0.0.1:8010/v1
```

Useful endpoints:

```text
GET  /health
GET  /v1/models
POST /v1/query
POST /v1/chat/completions
GET  /v1/dllm/status
POST /v1/dllm/chat
```

`/v1/chat/completions` exposes two models:

- `cls-rag-cag-v0.9`: local RAG/CAG extractive answer.
- `gpt-oss-120b` or `CLS_DLLM_MODEL`: proxy to the configured external dLLM API.

Example direct query:

```bash
curl http://127.0.0.1:8010/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who are the IVU beamline contacts and phone numbers?","top_k":8}'
```

## Index Documents

Use the Streamlit sidebar for the normal prototype path:

- **Admin:** index the canonical IVU manual with one click.
- **Admin / Scientist:** upload PDF, TXT, or MD files and index them into the same ChromaDB Evidence Store.

The older `launch_indexer.sh` inbox daemon remains for batch experiments, but it now uses the same no-download hash encoder and no local chat model.

## What You'll See

- A **Frontend bridge** sidebar panel showing whether Streamlit is using the embedded service or the FastAPI bridge.
- A **dLLM API** status for the configured external model; if it is offline or unconfigured, the instant RAG/CAG answer still works.
- Per-file progress while uploaded files are indexed.
- A silent deterministic query repair step for common beamline acronym spacing/typos.
- An instant clean parsed answer by default, with source passages below it for audit.
- An optional **dLLM API correction** toggle for extraction artifacts, backed only by the configured API endpoint.

## Model Roles

- **Indexing and search use `HashEmbedder`** so the IVU manual path works offline without an embedding-model download.
- **Chat answers are extractive by default.** `gpt-oss-120b` is contacted only through the dLLM API for optional correction or direct dLLM chat.
- **Query repair is deterministic.** It expands known CLS beamline acronyms before retrieval; no hidden language model is called.
- **Legacy Ollama adapters are explicit-only.** They have no default model in v0.9 and are not installed by the main requirements file.

## Guide Docs

- [Scientist guide](docs/USER_GUIDE.md)
- [Architecture review](docs/ARCHITECTURE.md)
- [dLLM API correction](docs/DLLM.md)
- [CLS safety flags](docs/SAFETY.md)

## Project Structure

- `app.py`: Scientist-facing Streamlit application.
- `api.py`: Shared FastAPI layer for Streamlit bridge mode and OpenAI-compatible clients.
- `cls_service.py`: Shared ChromaDB, CAG, indexing, and dLLM API runtime wiring.
- `cls_config.py`: Version, paths, and API endpoint configuration.
- `examples/`: CLS-specific retrieval, cleanup, correction, and legacy batch helpers.
- `ragandcag/`: Legacy KnowledgeBase and vector DB abstractions.
- `ingest_daemon.py`: Optional batch document indexing process.
- `launch_cls.sh`: Chat UI launcher.
- `launch_api.sh`: API launcher.
- `launch_indexer.sh`: Optional batch-indexing launcher.
