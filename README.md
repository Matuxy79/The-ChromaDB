# CLS RAG+CAG Prototype

A domain-specific Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, ChromaDB, and Ollama.

## Architecture

This prototype separates indexing from scientist-facing retrieval:
- **Offline Indexing Daemon:** A one-time or background maintenance process extracts text from documents (PDF, TXT), generates embeddings using `nomic-embed-text`, and stores them in local ChromaDB with Prism lane metadata.
- **Fast Evidence Chat UI:** The Streamlit app is optimized for scientists asking questions conversationally. It performs Prism-filtered vector search and returns source-backed evidence snippets without a chat LLM.

The intended operating model is: index documents once, then keep the chatbot open for fast repeated question-answering.

## Prerequisites

1. Install [Ollama](https://ollama.com/).
2. Pull the required embedding model:
   ```bash
   ollama pull nomic-embed-text
   ```

## Quick Launch

For scientist chat, run:

```bash
./launch_cls.sh
```

The launcher creates `.venv` if needed, installs Python packages, starts Ollama if it is installed but not running, checks the required local models, and opens the fast inference UI.

On Linux desktops, you can also double-click `CLS_RAG_CAG.desktop`. If the desktop asks whether to trust or execute the file, choose the execute/trust option.

## Index Documents

For the normal one-time indexing pass, place PDFs/TXTs in `docs/inbox`, then run:

```bash
./launch_indexer.sh --lane green --domain beamline
```

Processed files move to `docs/processed`; failed files move to `docs/failed`.

For a lightweight daemon that keeps watching the inbox:

```bash
./launch_indexer.sh --watch --interval 10 --lane green --domain beamline
```

Optional per-file metadata can be added with a sidecar JSON file such as `manual.pdf.metadata.json`:

```json
{
  "colour_code": "green",
  "domain": "beamline",
  "source_url": "IVU beamline manual"
}
```

## What you'll see

- An **offline-only pill** at the top right: green when Ollama is reachable on `127.0.0.1:11434`, red otherwise. The chat input is disabled while it's red.
- **Per-file progress bars** during indexing — `Extract → Chunk → Embed → Store` — with a running mm:ss timer and a final `chunks/s` rate. PDF extraction now runs page-parallel; embedding runs in batches of 16 so the bar moves every couple of seconds instead of waiting for the whole document.
- A collapsible **retrieval trace** under every assistant answer showing the top-k hits, their lane, distance, and a preview, plus the retrieval latency.

## Model Roles

- **Document lane is not an LLM feature.** It is metadata stored on every chunk as `colour_code`, then used by ChromaDB as a fast filter at query time.
- **Indexing needs the embedding model** (`nomic-embed-text`) so chunks can be searched semantically. This is the slow part on CPU.
- **No chat LLM is used.** The app returns deterministic source snippets from ChromaDB instead of generated prose.

## Guide Docs

- [Scientist guide](docs/USER_GUIDE.md)
- [Architecture review](docs/ARCHITECTURE.md)
- [CLS safety flags](docs/SAFETY.md)

## Project Structure

- `ragandcag/`: Core library for Knowledge Base, Vector DB adapters, and embedding wrappers.
- `examples/`: CLS-specific configurations, ingestion logic, and filters.
- `app.py`: Scientist-facing chat application.
- `ingest_daemon.py`: One-time/background document indexing process.
- `launch_cls.sh`: Chat UI launcher.
- `launch_indexer.sh`: Indexing launcher.

## Metadata & Prism Lanes

The system supports metadata filtering based on CLS Prism lanes:
- **Purple:** Research
- **Green:** Beamline
- **Blue:** Outreach
- **Orange:** Logistics
- **Yellow:** Education

These are applied at query time via ChromaDB's native filtering to ensure lane-correct results.
