# CLS dsRAG Architecture Review (v1.1)

## Product Shape

The app is a domain-specific RAG (dsRAG) implementation using:

- **Indexing:** semantic sectioning and AutoContext (via dsRAG patterns), embedded with `nomic-embed-text` (Ollama), and stored in ChromaDB.
- **Research scopes:** metadata-gated retrieval domains for outreach, science, logistics, operations, and administration context.
- **Relevance audit:** a second-pass filter that keeps high-signal retrieved chunks before generation.
- **Scientist chat:** embed the question, optionally filter by research scope, retrieve matching chunks, optionally refine via the relevance audit, and render cited deterministic extraction instantly.
- **Shared API:** expose the same retrieval path to Streamlit and OpenAI-compatible frontends.
- **Inference carrier:** when keyed, OpenRouter + `openai/gpt-oss-120b` synthesizes a direct answer from the refined evidence rows.

A secondary **carrier cleanup** checkbox can mechanically clean extraction artifacts after the instant answer is already available. It is off by default.

## Model Roles

| Component | Model | Required? | Why |
| --------- | ----- | --------- | --- |
| Indexing embeddings | `nomic-embed-text` (Ollama) | Yes | High-quality 768d vectors for scientific retrieval. |
| Query embeddings | `nomic-embed-text` (Ollama) | Yes | Same encoder as indexing. |
| Relevance audit | `CLS_DLLM_MODEL` | Optional | Evidence relevance auditing of retrieved chunks. |
| Inference carrier | `CLS_DLLM_MODEL` (`openai/gpt-oss-120b` by default) | Optional | Synthesizes a direct answer from refined evidence rows when the carrier is keyed and toggled on. |
| Carrier cleanup | `CLS_DLLM_MODEL` | Optional | Corrects extraction artifacts through the same carrier only when the cleanup checkbox is on. |

The launchers do not start Ollama and do not pull local models. Make sure Ollama is running and `nomic-embed-text` is available before indexing or querying.

## Runtime Layers

```text
Streamlit UI (Research Scopes)
  -> FastAPI bridge when CLS_USE_API=1, otherwise embedded cls_service
  -> Retrieval Encoder (nomic-embed-text via Ollama)
  -> Metadata Filter (Research Scope)
  -> CAG Layer cache
  -> Evidence Store search on cache miss
  -> Relevance Audit (optional refinement)
  -> deterministic answer builder
  -> optional inference carrier synthesis
  -> optional carrier cleanup for extraction artifacts
```

`/v1/chat/completions` exposes two model routes:

- `cls-rag-cag-v1.0`: local RAG/CAG extraction.
- `CLS_DLLM_MODEL`: proxy to the configured external inference carrier.

`/v1/dllm/chat` is the direct carrier proxy used by the Streamlit synthesis, relevance audit, and cleanup controls.

## CAG Layer

The app is **RAG + CAG**, not just RAG. The CAG Layer (`cls_backend/cag_cache.py`, `SemanticEvidenceCache`) is a second ChromaDB collection keyed by the embedding of a past question.

- **Lookup:** each question is encoded by the same Retrieval Encoder and matched against the cache with a cosine threshold.
- **Reuse granularity:** evidence only. A hit reuses stored evidence while the deterministic answer builder re-runs.
- **Invalidation:** every entry stores a `corpus_sig` derived from the Evidence Store's source hashes. Re-indexing or resetting the store makes stale cache entries unreachable.
- **Encoder caveat:** with `nomic-embed-text`, the cache generalises better across paraphrases than the previous deterministic encoder, but near-identical questions still give the strongest hits.

## Evidence Store

The active v1.1 Streamlit path stores chunks in `chroma_store` using the collection names in `cls_config.py`:

- `cls_v1_dsrag_evidence`: indexed evidence chunks (768d).
- `cls_v1_dsrag_cag_cache`: cached query-to-evidence rows (768d).

Because the embedding model changed from v1.0, the v1.0 collections (`cls_ivu_manual_hash_v1`, `cls_cag_evidence_cache_v1`) are no longer queried. Use **Reset Chroma index** in the sidebar and re-index to start a clean v1.1 dsRAG store.

## Performance Notes

For a 7 MB / 100-page PDF on CPU:

- PDF extraction: a few seconds.
- Chunking: under a second.
- Embedding with `nomic-embed-text` via Ollama: depends on local GPU/CPU, typically seconds to tens of seconds.
- Chroma storage: usually under a few seconds.

The inference carrier has no effect on indexing speed. It runs only for relevance auditing, synthesis, or cleanup after retrieval has already produced evidence rows.

## Current Design Choice

The prototype favors a small, inspectable local retrieval path:

- User question -> deterministic query repair -> `nomic-embed-text` embedding -> optional research-scope metadata filter -> Chroma vector search.
- Optional relevance audit uses the inference carrier to drop irrelevant chunks.
- CAG cache reuses prior evidence for repeated questions.
- With a keyed carrier, the default UI shows carrier synthesis above deterministic cited extraction.
- Retrieval evidence rows remain visible so both answer modes can be audited.
- Carrier cleanup is downstream, optional, guarded, and API-only.
