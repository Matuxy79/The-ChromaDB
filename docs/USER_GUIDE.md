# CLS Synchrotron Research Query — User Guide

A walkthrough for the 1.5v prototype. The current default is temporary fast mode: deterministic retrieval only, keyword-first, with LLM augmentation disabled.

---

## 1. Start The App

```bash
./scripts/launch_cls.sh
```

Creates `.venv` on first run, installs packages, and opens the UI at `http://localhost:8501`. Does not start Ollama or pull any LLM.

Fast mode defaults:

```bash
export CLS_RETRIEVAL_ONLY=1
export CLS_KEYWORD_ONLY=1
```

Set both to `0` before launch to restore hybrid semantic retrieval and carrier synthesis.

### Inference Carrier (optional, Full App only)

The carrier is any OpenAI-compatible endpoint. OpenRouter + `openai/gpt-oss-120b` is the default. It is ignored while `CLS_RETRIEVAL_ONLY=1`. To test generation again:

```bash
export CLS_RETRIEVAL_ONLY=0
export CLS_KEYWORD_ONLY=0
# Cloud (OpenRouter)
export CLS_DLLM_API_KEY="sk-or-..."
# Local llama.cpp (no key):  llama-server -m model.gguf --port 8080
# export CLS_DLLM_API_URL="http://localhost:8080/v1"
```

### Endpoint Status (Full App)

| State | Meaning |
| --- | --- |
| `Streamlit -> embedded service` | Local Python service, no API bridge needed |
| `Streamlit -> FastAPI` | Calling the shared API at `CLS_API_URL` |
| `Retrieval-only mode active` | Generation, cleanup, parrot phrasing, and carrier proxy calls are blocked |
| `Inference carrier ... online` | Carrier is reachable; synthesis available when `CLS_RETRIEVAL_ONLY=0` |
| `Inference carrier ... offline` | RAG/CAG extraction still works; synthesis unavailable |

---

## 2. Choose A UI

The landing page offers two entry points:

- **Full App** — Admin / User roles, corpus upload, evidence rows, precision controls, optional LLM synthesis.
- **Ask Lane** — Bright chat interface (llama.cui-style). Type in the bottom bar, get a cited answer with source chips; the conversation stacks as chat bubbles. Retrieval-only — no LLM, for instant speed.

Use **🧹 Clear chat** to reset the conversation, or **← Home** to return to the landing page (both in the Ask Lane sidebar).

---

## 3. Index Documents

In the Full App workspace:

1. Open **Workspace** and use **Corpus admin** to click **Index default documents** for the local literature test corpus (`data/training_corpus/test_books` by default).
2. Use the full-width **Upload & index documents** panel to drag-and-drop PDF, TXT, MD, DOCX, HTML, CSV, TSV, and JSON files.
3. Assign a **beamline** in that upload panel before clicking **Index uploaded files**.

The upload writes to the same ChromaDB Evidence Store used by the query UI and the API.

For batch ingestion, `ingest_daemon.py` watches a folder and indexes new files automatically.

---

## 4. Asking A Question

In the **Ask Lane**, type in the bottom chat bar and press enter. In the **Full App**, type a query (or click a suggested-problem chip) and click **Search Documents**.

In temporary fast mode the system:

1. Repairs the query — strips conversational scaffolding ("how do I...", "can you tell me about...") so the meaningful keywords stay prominent.
2. Runs deterministic keyword retrieval filtered by the selected research scope.
3. Builds an extractive answer from cited source sentences (shown with source chips in the Ask Lane).

When `CLS_KEYWORD_ONLY=0`, the system restores the semantic path: MiniLM embedding, CAG cache lookup, and hybrid semantic+lexical retrieval. When `CLS_RETRIEVAL_ONLY=0`, the Full App can synthesize a direct answer from the same evidence rows.

The **Research scope** selector (sidebar) narrows retrieval to a single CLS beamline. Set it to **All beamlines** to search the full corpus.

For fastest lookup, use content words or exact keywords from the documents. Set `CLS_KEYWORD_ONLY=0` when you want paraphrase-heavy natural language queries.

---

## 5. Speed Expectations

| Action | Wall time |
| --- | --- |
| App cold-start | 2–4 s |
| Keyword-only query | usually milliseconds on the prototype corpus |
| First embed call when semantic mode is enabled | a few seconds, once per process |
| Indexing a 7 MB / 100-page PDF | seconds to tens of seconds |
| First semantic query after indexing (warm model) | usually under a second |
| Repeated cached query | near-instant (CAG hit) |

The inference carrier is blocked in retrieval-only mode, so it cannot slow down searches.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Inference carrier disabled | This is expected with `CLS_RETRIEVAL_ONLY=1`; set it to `0` only when testing generation |
| No indexed chunks found | Index documents from **Workspace** or run `ingest_daemon.py` |
| API bridge unavailable | Run `./scripts/launch_api.sh`, relaunch with `CLS_USE_API=1` |
| Answer is wrong or thin | Open **Retrieval evidence** and check whether the right text was indexed |
| Streamlit traceback | Re-run the launcher; it refreshes requirements when they drift |

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).
