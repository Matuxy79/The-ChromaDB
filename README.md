# CLS RAG+CAG Prototype

A domain-specific Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, FastAPI, and ChromaDB.

## Architecture

This prototype separates indexing, shared retrieval, and frontend layers.

Like DocuSearch, retrieval is instant and primary: the grounded **instant clean parsed text** from the RAG/CAG dual layer is always shown. On top of it, a generative answer is carried by an external OpenAI-compatible LLM — the **default carrier is OpenRouter · `openai/gpt-oss-120b`** — wired in as a single on/off toggle that is **ON by default once a carrier key is configured**. The systems are kept separate so the UI, retrieval backend, API bridge, and the LLM carrier do not tangle:

| # | Component | Name | Where |
| --- | --- | --- | --- |
| 1 | Streamlit + OpenAI-compatible clients | **Dual frontend layers** | `app.py`, `/v1/chat/completions` |
| 2 | FastAPI | **Shared API bridge** | `api.py` |
| 3 | `HashEmbedder` + ChromaDB + CAG | **RAG+CAG instant backend** | `cls_service.py`, `examples/cls_pipeline.py` |
| 4 | OpenRouter · `openai/gpt-oss-120b` (default) | **Generative carrier** (single on/off, on by default when keyed) | `/v1/dllm/*`, `cls_service.py`, `examples/cls_dllm.py` |

Within the backend, the **Retrieval Encoder** (`HashEmbedder`) encodes the query, the **CAG Layer** reuses prior evidence for near-identical questions, the **Evidence Store** (ChromaDB) is searched on a miss, and deterministic cleanup repairs extraction artifacts.

> The base app does not download or start local language models — the only local model is the `HashEmbedder`. The generative answer calls the configured OpenAI-compatible carrier (default OpenRouter); with no key (or the toggle off) the app falls back to instant extractive text and stays fully offline-capable.

## Prerequisites

The prototype does **not** require a heavyweight model download. The instant RAG/CAG path uses the deterministic `HashEmbedder` and works without any carrier.

The generative answer is API-only and OpenAI-compatible. The **default carrier is OpenRouter · `openai/gpt-oss-120b`** — the URL and model are already baked in, so plug-and-play is just your key. The easiest way is a gitignored `cls.env` that `launch_cls.sh` auto-sources:

```bash
cp cls.env.example cls.env   # then paste your OpenRouter key on the CLS_DLLM_API_KEY line
./launch_cls.sh
```

Only `CLS_DLLM_API_KEY` is required for the default carrier. To switch carriers, override the URL/model:

```bash
export CLS_DLLM_API_KEY="sk-or-..."                      # required for OpenRouter; omit for Ollama
export CLS_DLLM_API_URL="https://openrouter.ai/api/v1"   # default; or http://localhost:11434/v1 for Ollama
export CLS_DLLM_MODEL="openai/gpt-oss-120b"              # default; or e.g. llama3.2 on Ollama
```

`CLS_DLLM_API_URL` should point to the external provider/runtime, not this app's own `http://127.0.0.1:8010/v1` API bridge.

### Answer with gpt-oss-120b (generative RAG, on by default)

Once a carrier key is configured, the generative answer is **on by default** for **Admin** and **Scientist** — the sidebar's *💬 gpt-oss-120b* section exposes a single on/off toggle to turn it off. Each search synthesizes a short natural-language answer **strictly from the retrieved passages** (e.g. "great expectations author" → "Charles Dickens"), cites the passage numbers, and is flagged if it emits any number not found in the context. The grounded extractive answer is still shown directly below it for verification. With no key (or the toggle off), the answer stays instant extractive text only. A secondary checkbox can additionally let the same carrier repair PDF extraction artifacts in the grounded bullets.

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

- `cls-rag-cag-v1.0`: local RAG/CAG extractive answer.
- `openai/gpt-oss-120b` or `CLS_DLLM_MODEL`: proxy to the configured external carrier.

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
- A **carrier** status for the configured external model (default OpenRouter · `openai/gpt-oss-120b`); if it is offline or keyless, the instant RAG/CAG answer still works.
- Per-file progress while uploaded files are indexed.
- A silent deterministic query repair step for common beamline acronym spacing/typos.
- A generative **gpt-oss-120b** answer on by default (Admin/Scientist) with a single on/off toggle, plus the grounded extraction and retrieval evidence rows below it for audit.
- A secondary checkbox to additionally **correct extraction artifacts** in the grounded bullets, backed only by the configured carrier.

## Model Roles

- **Indexing and search use `HashEmbedder`** so the IVU manual path works offline without an embedding-model download.
- **The grounded extractive answer is always shown.** On top of it, the generative `openai/gpt-oss-120b` answer (via the configured carrier, default OpenRouter) is on by default once a key is set, synthesized strictly from retrieved context.
- **Query repair is deterministic.** It expands known CLS beamline acronyms before retrieval; no hidden language model is called.
- **Legacy Ollama adapters are explicit-only.** They have no default model in v1.0 and are not installed by the main requirements file (Ollama can still be used as a local carrier via `CLS_DLLM_API_URL`).

## Guide Docs

- [Scientist guide](docs/USER_GUIDE.md)
- [Architecture review](docs/ARCHITECTURE.md)
- [Inference carrier cleanup](docs/DLLM.md)
- [CLS safety flags](docs/SAFETY.md)

## Project Structure

- `app.py`: Scientist-facing Streamlit application.
- `api.py`: Shared FastAPI layer for Streamlit bridge mode and OpenAI-compatible clients.
- `cls_service.py`: Shared ChromaDB, CAG, indexing, and inference carrier runtime wiring.
- `cls_config.py`: Version, paths, and API endpoint configuration.
- `examples/`: CLS-specific retrieval, cleanup, correction, and legacy batch helpers.
- `ragandcag/`: Legacy KnowledgeBase and vector DB abstractions.
- `ingest_daemon.py`: Optional batch document indexing process.
- `launch_cls.sh`: Chat UI launcher.
- `launch_api.sh`: API launcher.
- `launch_indexer.sh`: Optional batch-indexing launcher.
