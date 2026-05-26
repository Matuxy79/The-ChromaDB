# CLS RAG+CAG Architecture Review

## Product Shape

The app is intentionally split into two workflows:

- **Offline indexing:** extract text, chunk it, embed chunks, and store them in ChromaDB.
- **Scientist chat:** embed the question, retrieve matching chunks, and render source-backed evidence snippets.

This keeps the scientist-facing UI fast after the one-time indexing cost is paid.

## Model Roles

| Component | Model | Required? | Why |
| --------- | ----- | --------- | --- |
| Indexing embeddings | `nomic-embed-text` | Yes for semantic search | Converts document chunks into vectors. This is the CPU-heavy stage. |
| Query embeddings | `nomic-embed-text` | Yes for semantic search | Converts the scientist question into a vector for Chroma search. |

The chat LLM has been removed from the prototype. Removing embeddings would turn the system into keyword search, so the embedding model remains the one required local model.

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

The prototype is retrieval-only:

- User question -> embedding -> Chroma vector search.
- Silent deterministic query repair handles known beamline acronym typos before embedding.
- Optional lane filter -> narrower source set.
- UI renders the top evidence chunks and retrieval trace.

This removes generated prose, reduces latency, and makes the output easier to audit.
