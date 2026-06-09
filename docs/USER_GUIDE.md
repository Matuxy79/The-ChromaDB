# CLS RAG+CAG — Scientist Guide

A short walkthrough for the v0.9 prototype. The important bit: the app does not pull local LLMs. The default answer path is local retrieval plus deterministic cleanup.

---

## 1. Start The App

```bash
./launch_cls.sh
```

The launcher creates `.venv` on first run, installs Python packages, and opens the chat UI on `http://localhost:8501`. It does not start Ollama and does not download `gpt-oss-120b` or any smaller local model.

On Linux desktops, double-clicking `CLS_RAG_CAG.desktop` does the same thing.

### Optional dLLM API

Set these only if you have an external OpenAI-compatible dLLM endpoint:

```bash
export CLS_DLLM_API_URL="https://your-dllm-endpoint.example/v1"
export CLS_DLLM_API_KEY="..."
export CLS_DLLM_MODEL="gpt-oss-120b"
```

Leave them unset for a fully local retrieval-only prototype.

### Endpoint Status

The sidebar shows the active frontend bridge and dLLM API state:

| State | What it means |
| ----- | ------------- |
| `Streamlit -> embedded service` | Streamlit is calling the local Python service directly. |
| `Streamlit -> FastAPI` | Streamlit is calling the shared API at `CLS_API_URL`. |
| `dLLM API online` | The configured external dLLM API is reachable. |
| `dLLM API offline` | Instant RAG/CAG answers still work; optional dLLM correction is unavailable. |

The app can answer indexed-manual questions without network access. Network is only needed when you enable the optional dLLM API correction.

---

## 2. Index Documents

Use the sidebar:

1. Admins can click **Index IVU manual** for the canonical manual.
2. Admins and Scientists can upload PDF, TXT, or MD files.
3. Click **Index uploaded files**.

The upload path writes into the same ChromaDB Evidence Store used by the chat UI and API.

The older `launch_indexer.sh` inbox daemon remains for batch experiments. In v0.9 it uses the no-download hash encoder and no local chat model, but the Streamlit sidebar is the normal path for this prototype.

---

## 3. Asking A Question

Type a question or click one of the suggested problem chips, then click **Search IVU Manual**.

The system:

1. Repairs common beamline acronym spacing/typos deterministically.
2. Embeds the query with `HashEmbedder`.
3. Checks the CAG evidence cache.
4. Searches ChromaDB on a cache miss.
5. Builds an extractive answer from cited source sentences.

The optional dLLM toggle appears only for roles that can use it. It is off by default and calls the configured dLLM API only to correct mechanical extraction artifacts.

---

## 4. Speed Expectations

On a typical CLS workstation using the deterministic local `HashEmbedder`:

| Action | Wall time |
| ------ | --------- |
| App cold-start | 2-4 s |
| Indexing a 7 MB / 100-page PDF | seconds to tens of seconds |
| First extractive question | usually under a second after indexing |
| Repeated cached question | near-instant |

The dLLM API only affects optional correction and direct dLLM chat endpoints.

---

## 5. Troubleshooting

| Symptom | First thing to try |
| ------- | ------------------ |
| dLLM API offline | Set `CLS_DLLM_API_URL` to an external OpenAI-compatible endpoint, or leave correction off. |
| Chat says no indexed chunks were found | Index the IVU manual or upload documents from the sidebar. |
| API bridge unavailable | Run `./launch_api.sh`, then relaunch Streamlit with `CLS_USE_API=1`. |
| Answer is wrong or thin | Open the source passages and check whether the right document text was indexed. |
| Streamlit shows a Python traceback | Re-run the launcher script; it refreshes requirements when they drift. |

For the technical layout, see [ARCHITECTURE.md](ARCHITECTURE.md).
