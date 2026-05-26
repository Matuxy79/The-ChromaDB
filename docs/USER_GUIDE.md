# CLS RAG+CAG — Scientist Guide

A short, opinionated walkthrough for using the chatbot. Read once; then keep the app open.

---

## 1. Start the app

```bash
./launch_cls.sh
```

The launcher creates `.venv` on first run, installs Python packages, starts Ollama if it is installed but not running, and opens the chat UI on `http://localhost:8501`.

On Linux desktops, double-clicking `CLS_RAG_CAG.desktop` does the same thing.

### Offline pill

Top right of the page shows one of two states:

| Pill                                  | What it means                                                       |
| ------------------------------------- | ------------------------------------------------------------------- |
| 🟢 `Offline-only · Ollama reachable`  | Ollama is responding on `127.0.0.1:11434`. The app is fully local.  |
| 🔴 `Ollama unreachable`               | The chat input is disabled. Start Ollama and click **Re-check**.    |

The app never makes outbound calls during a chat. If your network is off, the only thing that breaks is the first-time model download — not the running app.

---

## 2. Index documents

You have two paths:

### A. Drop files in the sidebar (quick, one-off)

1. Open the **Index maintenance** expander.
2. Drag PDFs / TXTs onto the uploader.
3. Pick **Index as lane** (the [Prism lane](#3-prism-lanes) the file belongs to) and **Document domain**.
4. Click **Index uploaded files**.

You'll see one card per file with four live progress bars:

```
📄 Extract → ✂️ Chunk → 🧠 Embed → 💾 Store
```

Each bar updates every few seconds. The final card shows:

```
✅ IVU manual.pdf
   108 page(s) · 287 chunk(s) · 51.4 s total · 5.6 chunks/s
   📄 Extract 3.8 s · ✂️ Chunk 0.1 s · 🧠 Embed 46.9 s · 💾 Store 0.6 s
```

### B. Drop files in `docs/inbox/` (batch, scriptable)

```bash
./launch_indexer.sh --lane green --domain beamline
```

Processed files move to `docs/processed/`, failures move to `docs/failed/`. Add `--watch --interval 10` to keep polling.

Optional per-file metadata via a sidecar JSON named `<file>.metadata.json`:

```json
{ "colour_code": "green", "domain": "beamline", "source_url": "IVU beamline manual" }
```

---

## 3. Prism lanes

Lanes are stored in chunk metadata and applied at query time. Pick one in the sidebar before asking a question.

| Lane     | Colour | Use it for                                      |
| -------- | ------ | ----------------------------------------------- |
| Research | purple | Publications, technical specs                   |
| Beamline | green  | Energy ranges, hardware, alignment, procedures  |
| Outreach | blue   | Public-facing info, tours, general facility     |
| Logistics| orange | User-program policies, administrative procedures|
| Education| yellow | Student resources, training material            |

Leave it on **None** to search across all lanes. If a lane is set and the app says it doesn't know, the UI shows a small blue card suggesting you try **None**.

---

## 4. Asking a question

Type into the chat box at the bottom. The system uses the embedding model for semantic retrieval, then returns matching source chunks directly. There is no chat LLM in the answer path.

Before searching, the app silently repairs a small set of known beamline acronym typos, such as `ivw` -> `IVU`. Open the retrieval trace to see the repaired search query.

Three things appear under each answer:

1. **Safety banner** (orange) — only if the question touches a safety topic. Lists emergency numbers and reminds the scientist to confirm procedures with staff. See [SAFETY.md](SAFETY.md).
2. **Low-confidence card** (yellow) — only if the best retrieval distance is past `0.55`. The answer is still shown, but treat it as a lead.
3. **Retrieval trace** — collapsible expander labelled `🔍 Retrieval trace — N hit(s) · X ms`. Open it to see source filename, lane, domain, vector distance, and a 120-character preview for each chunk the model used.

The trace is the citation. If the trace looks wrong, the answer is wrong.

---

## 5. Speed expectations

On a typical CLS workstation (CPU-only, `nomic-embed-text`):

| Action                              | Wall time         |
| ----------------------------------- | ----------------- |
| App cold-start                      | 2–4 s             |
| Indexing 7 MB / 100-page PDF        | 45–75 s           |
| First question after model load     | 2–6 s             |
| Subsequent questions                | usually under 2 s |

The embedding model is the floor on CPU. If you want true speedups, run Ollama on GPU or reduce chunk count.

---

## 6. Troubleshooting

| Symptom                                            | First thing to try                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Red offline pill                                   | `ollama serve` in a terminal, then click **Re-check**.                                  |
| Chat says "I don't know"                           | Switch lane to **None**; the doc may not be in the lane you selected.                   |
| Indexing bar stuck at "Embed"                      | Confirm `nomic-embed-text` is installed: `ollama list`. If not: `ollama pull nomic-embed-text`. |
| Answer is confidently wrong                        | Open the retrieval trace. If the sources are unrelated, the doc isn't indexed; add it.  |
| Streamlit shows a Python traceback                 | Re-run the launcher script; it reinstalls requirements when they drift.                 |

For anything else, see [ARCHITECTURE.md](ARCHITECTURE.md) for the technical layout.
