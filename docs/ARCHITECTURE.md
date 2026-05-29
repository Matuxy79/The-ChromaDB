# CLS RAG+CAG Architecture Review

## Product Shape

The app is intentionally split into two workflows:

- **Offline indexing:** extract text, chunk it, embed chunks, and store them in ChromaDB.
- **Scientist chat:** embed the question, retrieve matching chunks, and let a small local chat model compose a cited answer from those chunks.

This keeps the scientist-facing UI fast after the one-time indexing cost is paid.

## Model Roles

| Component | Model | Required? | Why |
| --------- | ----- | --------- | --- |
| Indexing embeddings | `nomic-embed-text` | Yes for semantic search | Converts document chunks into vectors. This is the CPU-heavy stage. |
| Query embeddings | `nomic-embed-text` | Yes for semantic search | Converts the scientist question into a vector for Chroma search. |
| Answer composer | `llama3.2:1b` | Yes for default chat mode | Turns retrieved evidence rows into readable Markdown with citations. Not used for indexing. |

The **Evidence rows** UI mode bypasses the answer composer and renders retrieved chunks directly. That mode is useful for timing and debugging, but the default scientist experience uses the small local chat model.

## CAG Layer (semantic evidence cache)

The app is **RAG + CAG**, not just RAG. The CAG Layer (`examples/cls_cag_cache.py`,
`SemanticEvidenceCache`) is a *second* ChromaDB collection (`cls_cag_evidence_cache_v1`) keyed by
the embedding of a past question.

- **Lookup:** each question is encoded by the same Retrieval Encoder and matched against the cache
  with a cosine threshold (default min similarity `0.97`). A hit returns the previously retrieved
  evidence rows and skips the Evidence Store search.
- **Reuse granularity:** *evidence only* — a hit reuses the stored evidence but the Formatting LLM
  still re-runs, so prose stays fresh. We cache retrieval, not generation.
- **Invalidation:** every entry stores a `corpus_sig` (hash of the Evidence Store's distinct source
  hashes). Lookups filter by the current signature, so re-indexing or resetting the store makes old
  entries unreachable; reset also clears the cache outright.
- **Encoder caveat:** with the deterministic `HashEmbedder` the cache matches near-identical
  queries reliably; paraphrase generalisation would need a semantic encoder (`nomic-embed-text`).

> The embedding model acts as a dense retrieval encoder, the LLM as the grounded answer formatter,
> and the CAG layer is a semantic cache that reuses previously retrieved evidence when a similar
> query reappears.

## Prism Lanes

The lane selected during indexing is stored as chunk metadata:

```json
{ "colour_code": "green", "domain": "beamline" }
```

At query time, `examples/cls_filters.py` maps the selected lane to a ChromaDB `where` filter. This is cheap and does not call a language model.

Connected components:

- `app.py`: lets maintainers choose the lane when indexing and lets scientists filter by lane while asking questions.
- `examples/cls_filters.py`: converts a lane into a Chroma metadata filter.
- `ragandcag/database/vector/chroma_db.py`: passes that filter to ChromaDB.
- Retrieval trace: shows the stored lane for each returned chunk.

Recommendation: keep lanes as metadata, but present them as a source/indexing category rather than architecture terminology.

## Performance Notes

For a 7 MB / 100-page PDF on CPU:

- PDF extraction: a few seconds.
- Chunking: under a second.
- Embedding: tens of seconds.
- Chroma storage: usually under a few seconds.

The current UI shows progress per stage and batches embeddings so the user sees movement. True speedups come from GPU-backed Ollama, smaller documents, deduplication, or fewer chunks.

## Current Design Choice

The prototype is local RAG with a retrieval-only fallback:

- User question -> embedding -> Chroma vector search.
- Silent deterministic query repair handles known beamline acronym spacing/typos before embedding while preserving distinct concepts such as `IVU` and `IVW`.
- Optional lane filter -> narrower source set.
- Default UI streams an LLM-composed answer using only the top evidence chunks.
- UI always renders the retrieval trace so citations can be audited.

This keeps the more readable human-facing answer while preserving an inspectable source trail. For pure retrieval timing, switch the sidebar answer mode to **Evidence rows**.
