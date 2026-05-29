# dLLM — Optional Downstream Correction LLM (off by default)

## Why

[DocuSearch](https://indsci.clsi.ca/docu/) feels instant because there is **no model in the hot
path** — it retrieves and highlights. This app works the same way: the answer is **instant clean
parsed text** straight from the RAG/CAG dual layer, rendered with zero LLM latency and with query
terms highlighted (DocuSearch-style green hits).

**The only active model is the embedding Retrieval Encoder.** Artifact cleanup — hyphenation
breaks (`undula- tor`→`undulator`), stray spacing, duplicate sentences — is handled by
**deterministic code** (`clean_sentences` in `examples/cls_pipeline.py`), not a language model. No
LLM augments the displayed text by default.

A single optional **dLLM** stays wired in but **off by default** (sidebar toggle). It never
touches the text unless you turn it on; even then it only *corrects* (never rewrites/adds), and
only when the instant text still shows artifacts the deterministic pass could not fix.

## Where it sits

```
question
  → Retrieval Encoder (HashEmbedder)            ┐
  → CAG Layer (reuse) / Evidence Store (ChromaDB) │ System 2: RAG+CAG backend
  → deterministic clean parse                    ┘ (examples/cls_pipeline.py)
  → instant answer  ← what you see immediately, no LLM
  → (only if toggled on) dLLM correction, streamed in place
```

The codebase is split into four systems so these concerns don't tangle:

| # | System | Where |
| --- | --- | --- |
| 1 | UI/UX (Streamlit) | `app.py` |
| 2 | RAG+CAG instant backend | `examples/cls_pipeline.py` (`instant_answer`) |
| 3 | dLLM corrector (optional) | `examples/cls_dllm.py` |
| 4 | CAG cache | `examples/cls_cag_cache.py` |

## The activation gate (`examples/cls_dllm.py :: needs_correction`)

The gate is intentionally conservative so the dLLM **usually does not activate**. It returns
`True` only on clear artifacts in the extracted sentences (the trailing `[Source: …]` citation is
stripped before judging):

| Signal | Reason shown |
| --- | --- |
| `exam- ple` hyphenation breaks | joined hyphenation breaks |
| leftover `Source:/Section:/Page:` text | stripped leftover header text |
| single-char / table-OCR soup | tidied fragmented table text |
| > 35% non-alphanumeric characters | cleaned symbol-heavy fragment |
| long (>80 char) sentence cut mid-thought | completed a truncated fragment |
| duplicated sentences | removed duplicated sentence |

Most artifacts are already repaired by the deterministic `clean_sentences` pass, so even with the
toggle on the gate rarely has anything left to do.

## What it is allowed to do (only when toggled on)

The dLLM **streams its correction in place** into the already-shown instant card
(`stream_correction` + `CORRECTION_SYSTEM`), constrained to **mechanical correction only**:

1. No rewriting, summarising, reordering, or adding information — same facts, same order.
2. Every number and every `[Source: …, page …]` citation preserved verbatim.
3. Bullet structure kept (one line per input bullet).

When the stream finishes it is re-checked by `validate_correction` (= `numbers_grounded` **and**
`citations_preserved`). If the model invented any number or mangled a citation, the correction is
discarded and the instant text is restored. **Fabricated facts are never shown.**

## Provenance in the UI

- `⚡ instant — clean parsed text from RAG/CAG, no LLM.` — the default; no model call.
- `✎ dLLM corrected — <reason>.` — toggle on, gate fired, correction passed both guards.
- `⚡ instant — dLLM drifted, kept the grounded extraction.` — toggle on but the correction
  was rejected (or the model was unavailable).

## Encoder caveat

The dLLM is independent of the Retrieval Encoder. Today that encoder is the deterministic
`HashEmbedder`; swapping in `nomic-embed-text` would improve retrieval and CAG paraphrase
matching but does not change how the dLLM gate or correction behave.
