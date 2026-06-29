# Inference Carrier — gpt-oss-120b

## Why

[DocuSearch](https://indsci.clsi.ca/docu/) feels instant because retrieval happens first. This app keeps that shape. The deterministic answer is **instant clean parsed text** from the RAG/CAG layer, rendered with query-term highlighting and cited evidence rows.

Artifact cleanup (PDF hyphenation breaks, stray spacing, duplicate sentences) is handled first by deterministic code (`clean_sentences` in `cls_backend/pipeline.py`).

> Temporary fast mode: `CLS_RETRIEVAL_ONLY=1` is the default right now. In that mode all carrier synthesis, cleanup, self-debate, parrot phrasing, and proxy endpoints are disabled even if a carrier key is configured.

The **inference carrier** — OpenRouter + `openai/gpt-oss-120b` by default — can synthesize a direct answer from the retrieved evidence rows when a key is present. The secondary **cleanup** checkbox is downstream, off by default, and API-only:

- The launcher never starts Ollama.
- The launcher never pulls `gpt-oss-120b`.
- The UI calls `/v1/dllm/chat`, which proxies to `CLS_DLLM_API_URL`.
- If the carrier is offline or keyless, normal RAG/CAG extraction and retrieval evidence still work.

## Configuration

```bash
export CLS_RETRIEVAL_ONLY=0
export CLS_KEYWORD_ONLY=0
export CLS_DLLM_API_KEY="sk-or-..."
# export CLS_DLLM_API_URL="https://openrouter.ai/api/v1"
# export CLS_DLLM_MODEL="openai/gpt-oss-120b"
```

`CLS_DLLM_API_URL` must be the carrier provider's base URL. Do not point it back at this app's own `http://127.0.0.1:8010/v1` bridge.

## Where It Sits

```text
question
  -> query repair (strip NL scaffolding)
  -> keyword-only retrieval when CLS_KEYWORD_ONLY=1
  -> otherwise all-MiniLM-L6-v2 + CAG + hybrid semantic/lexical retrieval
  -> deterministic clean parse
  -> instant answer shown immediately
  -> optional carrier synthesis only when CLS_RETRIEVAL_ONLY=0
  -> optional carrier cleanup only when CLS_RETRIEVAL_ONLY=0
```

| # | System | Where |
| --- | --- | --- |
| 1 | UI/UX | `app.py` |
| 2 | RAG+CAG instant backend | `cls_service.py`, `cls_backend/pipeline.py` |
| 3 | Inference carrier synthesis + cleanup | `cls_service.py`, `cls_backend/dllm.py`, `/v1/dllm/*` |
| 4 | CAG cache | `cls_backend/cag_cache.py` |

## Activation Gate

`cls_backend/dllm.py::needs_correction` is intentionally conservative. It returns `True` only for clear extraction artifacts:

| Signal | Reason shown |
| --- | --- |
| `exam- ple` hyphenation breaks | joined hyphenation breaks |
| leftover `Source:/Section:/Page:` text | stripped leftover header text |
| single-character or table-OCR fragments | tidied fragmented table text |
| more than 35% non-alphanumeric characters | cleaned symbol-heavy fragment |
| long sentence cut mid-thought | completed a truncated fragment |
| duplicated sentences | removed duplicated sentence |

Most artifacts are already repaired by deterministic cleanup, so even with the checkbox on the carrier rarely has work to do.

## Allowed Correction

Carrier cleanup is constrained to mechanical correction only:

1. No rewriting, summarising, reordering, or adding information.
2. Every number and every `[Source: ..., page ...]` citation must be preserved verbatim.
3. Bullet structure stays one line per input bullet.

When the API returns, `validate_correction` checks `numbers_grounded` and `citations_preserved`. If either guard fails, the correction is discarded and the original instant extraction remains visible.

## UI Provenance Labels

- `💬 Carrier synthesis · OpenRouter · gpt-oss-120b` — model synthesized from retrieval evidence rows.
- `⚡ Deterministic extraction · RAG/CAG` — cited extraction came from local retrieval.
- `✎ Extraction cleaned by <carrier> — <reason>.` — cleanup was on, gate fired, both guards passed.
- `⚡ Carrier cleanup drifted/unavailable...` — correction rejected or failed; deterministic extraction kept.
