# CLS RAG+CAG Prototype

A domain-specific Retrieval-Augmented and Cache-Augmented Generation (RAG+CAG) system for the Canadian Light Source (CLS), built with Streamlit, ChromaDB, and Ollama.

## Architecture

This prototype separates indexing from scientist-facing retrieval.

Like DocuSearch, the answer is **instant clean parsed text** from the RAG/CAG dual layer — no LLM
in the hot path. The four systems are kept as separate modules so they don't tangle:

| # | Component | Name | Where |
| --- | --- | --- | --- |
| 1 | Streamlit | **UI/UX** | `app.py` |
| 2 | `HashEmbedder` + ChromaDB + CAG | **RAG+CAG instant backend** | `examples/cls_pipeline.py` (`instant_answer`) |
| 3 | `llama3.2:3b` | **dLLM** (downstream LLM, *optional, off by default*) | `examples/cls_dllm.py` |
| 4 | Cache collection | **CAG Layer** | `examples/cls_cag_cache.py` |

Within the backend: the **Retrieval Encoder** (`HashEmbedder` — the one active model) encodes the
query, the **CAG Layer** reuses prior evidence on a similar query, the **Evidence Store**
(ChromaDB) is searched on a miss, and a **deterministic clean parse** repairs extraction artifacts.

> The embedding model is the dense retrieval encoder; the answer is clean parsed text from the
> RAG/CAG layer with no LLM augmentation by default. An optional downstream LLM (the dLLM) can be
> toggled on to correct artifacts, but it never invents facts.

- **Offline Indexing Daemon:** extracts text from documents (PDF, TXT), encodes chunks, and stores them in the local Evidence Store with Prism lane metadata.
- **Scientist Chat UI:** encodes the question, checks the CAG Layer, searches the Evidence Store on a miss, and shows the clean parsed answer **instantly** (DocuSearch-style, highlighted terms). No LLM augments the text unless the dLLM toggle is turned on; even then it only corrects artifacts (guarded so numbers + citations survive verbatim). See [docs/DLLM.md](docs/DLLM.md).

The intended operating model is: index documents once, then keep the app open for fast repeated question-answering. Repeated/near-identical questions are served from the CAG Layer. With the deterministic `HashEmbedder` this matches near-identical queries; a semantic encoder (`nomic-embed-text`) would generalise to paraphrases.

## Prerequisites

1. Install [Ollama](https://ollama.com/).
2. Pull the required local models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.2:1b
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
- A silent **query repair** step for common beamline acronym spacing/typos while preserving distinct acronyms like `IVU` and `IVW`.
- A local **LLM summary** answer mode by default, plus an **Evidence rows** mode for fast debugging and retrieval timing.
- A collapsible **retrieval trace** under every assistant answer showing the top-k hits, their lane, distance, and a preview, plus the retrieval latency.

## Model Roles

- **Document lane is not an LLM feature.** It is metadata stored on every chunk as `colour_code`, then used by ChromaDB as a fast filter at query time.
- **Indexing needs the embedding model** (`nomic-embed-text`) so chunks can be searched semantically. This is the slow part on CPU.
- **Chat answers use a small local LLM** (`llama3.2:1b`) to turn retrieved chunks into readable Markdown with source-row citations. It is not used for indexing.
- **Query repair is deterministic.** It expands known CLS beamline acronyms before retrieval; no hidden language model is called.

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
