# CLS RAG+CAG Architecture Review

## Product Shape

The app is split into three active workflows:

- **Indexing:** extract text, chunk it, embed chunks with `HashEmbedder`, and store them in ChromaDB.
- **Scientist chat:** embed the question, retrieve matching chunks, and render a cited extractive answer instantly.
- **Shared API:** expose the same retrieval path to Streamlit and OpenAI-compatible frontends.

An optional fourth workflow, **dLLM API correction**, can mechanically clean extraction artifacts after the instant answer is already available. It is off by default.

## Model Roles

| Component | Model | Required? | Why |
| --------- | ----- | --------- | --- |
| Indexing embeddings | `HashEmbedder` | Yes | Converts document chunks into deterministic local vectors. |
| Query embeddings | `HashEmbedder` | Yes | Converts the scientist question into a vector for Chroma search. |
| dLLM API correction | `CLS_DLLM_MODEL` (`gpt-oss-120b` by default) | Optional | Corrects extraction artifacts through an external API only when toggled on. |

The launchers do not start Ollama and do not pull local models. Legacy Ollama adapters remain explicit-only helpers, but the v0.9 app path does not use or install them.

## Runtime Layers

```text
Streamlit UI or OpenAI-compatible client
  -> FastAPI bridge when CLS_USE_API=1, otherwise embedded cls_service
  -> Retrieval Encoder (HashEmbedder)
  -> CAG Layer cache
  -> Evidence Store search on cache miss
  -> deterministic answer builder
  -> optional dLLM API correction
```

`/v1/chat/completions` exposes two model routes:

- `cls-rag-cag-v0.9`: local RAG/CAG extraction.
- `CLS_DLLM_MODEL`: proxy to the configured external dLLM API.

`/v1/dllm/chat` is the direct dLLM API proxy used by the Streamlit correction toggle.

## CAG Layer

The app is **RAG + CAG**, not just RAG. The CAG Layer (`examples/cls_cag_cache.py`, `SemanticEvidenceCache`) is a second ChromaDB collection keyed by the embedding of a past question.

- **Lookup:** each question is encoded by the same Retrieval Encoder and matched against the cache with a cosine threshold.
- **Reuse granularity:** evidence only. A hit reuses stored evidence while the deterministic answer builder re-runs.
- **Invalidation:** every entry stores a `corpus_sig` derived from the Evidence Store's source hashes. Re-indexing or resetting the store makes stale cache entries unreachable.
- **Encoder caveat:** with `HashEmbedder`, the cache is strongest for near-identical questions. Paraphrase generalisation would need a future semantic encoder.

## Evidence Store

The active v0.9 Streamlit path stores chunks in `chroma_store` using the collection names in `cls_config.py`:

- `cls_ivu_manual_hash_v1`: indexed evidence chunks.
- `cls_cag_evidence_cache_v1`: cached query-to-evidence rows.

The older inbox daemon remains for batch experiments and now uses `HashEmbedder` as well, so it does not require local embedding models.

## Performance Notes

For a 7 MB / 100-page PDF on CPU:

- PDF extraction: a few seconds.
- Chunking: under a second.
- Embedding with `HashEmbedder`: seconds to tens of seconds depending on chunk count.
- Chroma storage: usually under a few seconds.

The dLLM API has no effect on indexing speed and only runs when correction is toggled on.

## Current Design Choice

The prototype favors a small, inspectable local retrieval path:

- User question -> deterministic query repair -> hash embedding -> Chroma vector search.
- CAG cache reuses prior evidence for repeated questions.
- The default UI renders extractive cited sentences, not generated prose.
- Source passages remain visible so the answer can be audited.
- dLLM correction is downstream, optional, guarded, and API-only.
