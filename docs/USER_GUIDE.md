# CLS Synchrotron Research Query — User Guide

A walkthrough for the v1.2 prototype. Retrieval is local and semantic (sentence-transformers, offline on CPU); the optional inference carrier is API-only and used only in the Full App.

---

## 1. Start The App

```bash
./launch_cls.sh
```

Creates `.venv` on first run, installs packages, and opens the UI at `http://localhost:8501`. Does not start Ollama or pull any LLM. On first run the embedder (`all-MiniLM-L6-v2`, ~80 MB) downloads once to `~/.cache/huggingface/`; after that it runs fully offline.

### Inference Carrier (optional, Full App only)

The carrier is any OpenAI-compatible endpoint. OpenRouter + `openai/gpt-oss-120b` is the default; leave it unset for retrieval-only use. You can also point it at a local llama.cpp or Ollama server:

```bash
# Cloud (OpenRouter)
export CLS_DLLM_API_KEY="sk-or-..."
# Local llama.cpp (no key):  llama-server -m model.gguf --port 8080
# export CLS_DLLM_API_URL="http://localhost:8080/v1"
```

### Endpoint Status (Full App sidebar)

| State | Meaning |
| --- | --- |
| `Streamlit -> embedded service` | Local Python service, no API bridge needed |
| `Streamlit -> FastAPI` | Calling the shared API at `CLS_API_URL` |
| `Inference carrier ... online` | Carrier is reachable; synthesis available |
| `Inference carrier ... offline` | RAG/CAG extraction still works; synthesis unavailable |

---

## 2. Choose A UI

The landing page offers two entry points:

- **Full App** — Admin / Scientist / Staff / User roles, corpus upload, evidence rows, precision controls, optional LLM synthesis.
- **Ask Lane** — Bright chat interface (llama.cui-style). Type in the bottom bar, get a cited answer with source chips; the conversation stacks as chat bubbles. Retrieval-only — no LLM, for instant speed.

Use **🧹 Clear chat** to reset the conversation, or **← Home** to return to the landing page (both in the Ask Lane sidebar).

---

## 3. Index Documents

In the Full App sidebar:

1. Admins can click **Index default documents** to index the local literature test corpus (`Training for perfect in ui graded/test_books` by default).
2. Admins and Scientists can drag-and-drop PDF, TXT, MD, DOCX, HTML, CSV, TSV, and JSON files.
3. Assign a **Research domain** from the selector before clicking **Index uploaded files**.

The upload writes to the same ChromaDB Evidence Store used by the query UI and the API.

For batch ingestion, `ingest_daemon.py` watches a folder and indexes new files automatically.

---

## 4. Asking A Question

In the **Ask Lane**, type in the bottom chat bar and press enter. In the **Full App**, type a query (or click a suggested-problem chip) and click **Search Documents**.

Either way the system:

1. Repairs the query — strips conversational scaffolding ("how do I…", "can you tell me about…") so messy phrasing maps to the right vector.
2. Embeds the query with `all-MiniLM-L6-v2` (local, semantic).
3. Checks the CAG evidence cache.
4. On a cache miss, runs **hybrid retrieval** — semantic vector search plus a lexical keyword scan, merged — filtered by the selected research scope.
5. Builds an extractive answer from cited source sentences (shown with source chips in the Ask Lane).
6. *(Full App only)* If the carrier is keyed and toggled on, synthesizes a direct answer from the same evidence rows.

The **Research scope** selector (sidebar) narrows retrieval to a single domain. Set it to **All disciplines** to search the full corpus.

Because retrieval is now semantic, natural language works well: "I can't get my sample to stay in the holder" finds the same evidence as "sample mounting procedure".

---

## 5. Speed Expectations

| Action | Wall time |
| --- | --- |
| App cold-start | 2–4 s |
| First embed call (loads MiniLM) | a few seconds, once per process |
| Indexing a 7 MB / 100-page PDF | seconds to tens of seconds |
| First query after indexing (warm model) | usually under a second |
| Repeated cached query | near-instant (CAG hit) |

The inference carrier only affects synthesis and optional cleanup — it does not slow down retrieval.

---

## 6. Troubleshooting

| Symptom | Fix |
| --- | --- |
| Inference carrier offline | Set `CLS_DLLM_API_KEY` in environment, or disable synthesis |
| No indexed chunks found | Index documents from the sidebar or run `ingest_daemon.py` |
| API bridge unavailable | Run `./launch_api.sh`, relaunch with `CLS_USE_API=1` |
| Answer is wrong or thin | Open **Retrieval evidence** and check whether the right text was indexed |
| Streamlit traceback | Re-run the launcher; it refreshes requirements when they drift |

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).
