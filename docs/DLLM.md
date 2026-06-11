# Inference Carrier Cleanup — gpt-oss-120b

## Why

[DocuSearch](https://indsci.clsi.ca/docu/) feels instant because retrieval happens first: it retrieves and highlights. This app keeps that shape. The deterministic answer is **instant clean parsed text** from the RAG/CAG layer, rendered with query-term highlighting and cited retrieval evidence rows.

Artifact cleanup such as PDF hyphenation breaks, stray spacing, and duplicate sentences is handled first by deterministic code (`clean_sentences` in `examples/cls_pipeline.py`).

In v1.0 the primary external model path is the **inference carrier**: OpenRouter + `openai/gpt-oss-120b` by default. When a key is present, the UI can synthesize a direct answer from retrieved evidence rows. The secondary **cleanup** checkbox is downstream, off by default, and API-only:

- The launcher never starts Ollama.
- The launcher never pulls `gpt-oss-120b`.
- The UI calls `/v1/dllm/chat`, which proxies to `CLS_DLLM_API_URL`.
- If the carrier is offline or keyless, normal RAG/CAG extraction and retrieval evidence still work.

## Configuration

Use an external OpenAI-compatible endpoint. OpenRouter + `openai/gpt-oss-120b` is already the default; for that carrier only the key is required:

```bash
export CLS_DLLM_API_KEY="sk-or-..."
# export CLS_DLLM_API_URL="https://openrouter.ai/api/v1"
# export CLS_DLLM_MODEL="openai/gpt-oss-120b"
```

`CLS_DLLM_API_URL` must be the carrier provider/runtime base URL. Do not point it back at this app's own `http://127.0.0.1:8010/v1` bridge.

## Where It Sits

```text
question
  -> Retrieval Encoder (HashEmbedder)
  -> CAG Layer reuse or Evidence Store search (ChromaDB)
  -> deterministic clean parse
  -> instant answer shown immediately
  -> optional carrier synthesis from evidence rows
  -> optional carrier cleanup if toggled on and the gate fires
```

The codebase keeps the pieces separate:

| # | System | Where |
| --- | --- | --- |
| 1 | UI/UX | `app.py` |
| 2 | RAG+CAG instant backend | `cls_service.py`, `examples/cls_pipeline.py` |
| 3 | Inference carrier synthesis + cleanup | `cls_service.py`, `examples/cls_dllm.py`, `/v1/dllm/*` |
| 4 | CAG cache | `examples/cls_cag_cache.py` |

## Activation Gate

`examples/cls_dllm.py::needs_correction` is intentionally conservative. It returns `True` only for clear extraction artifacts:

| Signal | Reason shown |
| --- | --- |
| `exam- ple` hyphenation breaks | joined hyphenation breaks |
| leftover `Source:/Section:/Page:` text | stripped leftover header text |
| single-character or table-OCR fragments | tidied fragmented table text |
| more than 35% non-alphanumeric characters | cleaned symbol-heavy fragment |
| long sentence cut mid-thought | completed a truncated fragment |
| duplicated sentences | removed duplicated sentence |

Most artifacts are already repaired by deterministic cleanup, so even with cleanup enabled the carrier rarely has work to do.

## Allowed Correction

Carrier cleanup is constrained to mechanical correction only:

1. No rewriting, summarising, reordering, or adding information.
2. Every number and every `[Source: ..., page ...]` citation must be preserved verbatim.
3. Bullet structure stays one line per input bullet.

When the API returns, `validate_correction` checks `numbers_grounded` and `citations_preserved`. If either guard fails, the correction is discarded and the original instant extraction remains visible.

## UI Provenance

- `💬 Carrier synthesis · OpenRouter · gpt-oss-120b` means the model synthesized from retrieval evidence rows.
- `⚡ Deterministic extraction · RAG/CAG` means the cited extraction came from local retrieval and cleanup.
- `✎ Extraction cleaned by <carrier> — <reason>.` means cleanup was on, the gate fired, and both guards passed.
- `⚡ Carrier cleanup drifted/unavailable...` means the correction was rejected or failed, so deterministic extraction was kept.

## Encoder Caveat

The inference carrier is independent of retrieval. Today the retrieval encoder is deterministic `HashEmbedder`, which keeps the prototype offline and small. A future semantic encoder could improve paraphrase matching, but it is not required for v1.0.
