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
