# dLLM API — gpt-oss-120b Correction Connection

## Why

[DocuSearch](https://indsci.clsi.ca/docu/) feels instant because there is no language model in the hot path: it retrieves and highlights. This app works the same way by default. The answer is **instant clean parsed text** from the RAG/CAG layer, rendered with query-term highlighting and cited source passages.

Artifact cleanup such as PDF hyphenation breaks, stray spacing, and duplicate sentences is handled first by deterministic code (`clean_sentences` in `examples/cls_pipeline.py`).

The optional **dLLM API** connection is downstream. It is off by default, and it is API-only in v0.9:

- The launcher never starts Ollama.
- The launcher never pulls `gpt-oss-120b`.
- The UI toggle calls `/v1/dllm/chat`, which proxies to `CLS_DLLM_API_URL`.
- If `CLS_DLLM_API_URL` is unset, the toggle stays offline and normal RAG/CAG answers still work.

## Configuration

Use an external OpenAI-compatible endpoint:

```bash
export CLS_DLLM_API_URL="https://your-dllm-endpoint.example/v1"
export CLS_DLLM_API_KEY="..."
export CLS_DLLM_MODEL="gpt-oss-120b"
```

`CLS_DLLM_API_URL` must be the dLLM provider/runtime base URL. Do not point it back at this app's own `http://127.0.0.1:8010/v1` bridge.

## Where It Sits

```text
question
  -> Retrieval Encoder (HashEmbedder)
  -> CAG Layer reuse or Evidence Store search (ChromaDB)
  -> deterministic clean parse
  -> instant answer shown immediately
  -> optional dLLM API correction if toggled on and the gate fires
```

The codebase keeps the pieces separate:

| # | System | Where |
| --- | --- | --- |
| 1 | UI/UX | `app.py` |
| 2 | RAG+CAG instant backend | `cls_service.py`, `examples/cls_pipeline.py` |
| 3 | dLLM API corrector | `cls_service.py`, `examples/cls_dllm.py`, `/v1/dllm/*` |
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

Most artifacts are already repaired by deterministic cleanup, so even with the toggle on the dLLM rarely has work to do.

## Allowed Correction

The dLLM API is constrained to mechanical correction only:

1. No rewriting, summarising, reordering, or adding information.
2. Every number and every `[Source: ..., page ...]` citation must be preserved verbatim.
3. Bullet structure stays one line per input bullet.

When the API returns, `validate_correction` checks `numbers_grounded` and `citations_preserved`. If either guard fails, the correction is discarded and the original instant extraction remains visible.

## UI Provenance

- `⚡ instant — clean parsed text from RAG/CAG, no LLM.` means no dLLM call happened.
- `✎ dLLM corrected — <reason>.` means the toggle was on, the gate fired, and both guards passed.
- `⚡ instant — dLLM drifted, kept the grounded extraction.` means the correction was rejected.
- `⚡ instant — dLLM unavailable, kept the grounded extraction.` means the API call failed.

## Encoder Caveat

The dLLM API is independent of retrieval. Today the retrieval encoder is deterministic `HashEmbedder`, which keeps the prototype offline and small. A future semantic encoder could improve paraphrase matching, but it is not required for v0.9.
