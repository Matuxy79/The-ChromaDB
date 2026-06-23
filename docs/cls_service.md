# 2. Service — `cls_service.py`

`cls_service.py` is the **single public API surface** for the CLS retrieval backend.  
Every caller—`app.py`, `chat_lane.py`, `api.py`, and `ingest_daemon.py`—imports from here; nothing below this layer is reached directly.

The file has two responsibilities:

1. **Lazy singleton initialization** of expensive, process-scoped resources (embedder, Chroma client, collections, CAG cache).
2. **A small, stable public API** that hides the backend pipeline behind functions that are easy to call and safe to share between UIs, the daemon, and tests.

---

## 2.1 Lazy singletons

The following resources are created once per process and reused:

| Resource | Getter | Why it is lazy |
| --- | --- | --- |
| `SentenceTransformerEmbedder` | `get_embedder()` | Downloads/loads `all-MiniLM-L6-v2` (~80 MB) on first use only. |
| ChromaDB `PersistentClient` | `get_chroma_client()` | Opens the on-disk store in `chroma_store/` once. |
| Evidence collection | `get_collection()` | Creates or reopens `cls_v2_evidence`. |
| CAG cache collection | `get_cache_collection()` | Creates or reopens `cls_v2_cag_cache`. |
| `SemanticEvidenceCache` | `get_cache()` | Wraps the cache collection + embedder; built on demand. |

Each getter stores its instance in a module-level variable guarded by `_RESOURCE_LOCK`, an `RLock`.  Calling any getter is therefore thread-safe and idempotent:

```python
from cls_service import get_embedder, get_collection

# First call initializes; later calls return the same instance.
embedder = get_embedder()
collection = get_collection()
```

This matters because:

- Streamlit reruns the script on every interaction; lazy init avoids rebuilding the embedder each time.
- The Chainlit Ask Lane and FastAPI bridge may both import `cls_service` in the same process.
- Tests can reset or patch singletons without restarting the interpreter.

### Resetting singletons

`reset_collection()` wipes **all** collections in `chroma_store/` and clears the singleton references. The next call to `get_collection()` or `get_cache()` rebuilds a clean store. This is the backend for the Full App's **Reset Chroma index** button.

---

## 2.2 The public API

These are the functions other modules are expected to call.

### Retrieval

```python
def ask_manual(
    query: str,
    *,
    top_k: int = 16,
    cache_enabled: bool = True,
    min_similarity: float = 0.97,
    metadata_filter: dict | None = None,
    debate_enabled: bool = False,
    keyword_only: bool | None = None,
) -> dict:
```

The main entry point for a user question. It returns the dictionary produced by `pipeline.instant_answer`, which contains the repaired query, evidence rows, answer sentences, source chips, and retrieval mode.

- `metadata_filter` narrows retrieval by Chroma metadata (for example, `{"domain": "physics"}`).
- `keyword_only=None` falls back to `cls_config.KEYWORD_ONLY_RETRIEVAL`.
- `cache_enabled` toggles the CAG cache lookup.
- `min_similarity` controls how close a cached question must be to reuse its evidence.

### Ingestion

```python
def ingest_path(
    path: Path,
    source_hash: str,
    force: bool = False,
    extra_metadata: dict | None = None,
    source_name: str | None = None,
    embedder: Any | None = None,
) -> tuple[int, str]:
```

Indexes a single file into the evidence collection.

- Returns `(chunk_count, status)` where `status` is `"indexed"`, `"already indexed"`, or `"no readable text found"`.
- `source_hash` should be a stable file signature (see `file_signature`/`uploaded_signature`); it is used to skip duplicates and to delete stale chunks on re-ingest.
- `force=True` re-indexes an already-known file.
- Ingestion writes embeddings in batches of `INGEST_BATCH_SIZE` (256) to keep peak memory low.

```python
def build_chunks(path, source_hash, ...) -> list[Chunk]
def file_signature(path: Path) -> str
def uploaded_signature(data: bytes) -> str
```

Helper functions exposed mainly for tests and the daemon. `Chunk` is a small frozen dataclass holding `text`, `metadata`, and `chunk_id`.

### Status and diagnostics

```python
def service_status() -> dict
def dllm_status(timeout: float = 1.0) -> dict
def parrot_status(timeout: float = 1.0) -> dict
def evidence_breakdown(collection=None) -> list[dict]
def evaluate_retrieval() -> list[dict]
def warm_keyword_index() -> int
```

- `service_status()` returns counts, document breakdown, and carrier status for the Full App sidebar.
- `dllm_status()` probes the configured OpenAI-compatible carrier at `/models`.
- `parrot_status()` probes the local parrot endpoint (Ollama/llama.cpp).
- `evidence_breakdown()` groups indexed chunks by source and reports page spans.
- `evaluate_retrieval()` runs a small built-in benchmark against `EVAL_CASES`.
- `warm_keyword_index()` preloads the lexical keyword snapshot.

### Generative / carrier path

```python
def generate_answer(query, rows, *, model=..., timeout=60.0, grounded=True) -> str
def stream_generate_answer(query, rows, *, model=..., timeout=90.0, grounded=True) -> Generator[str, None, None]
def call_dllm_api(messages, *, system=None, model=..., timeout=60.0) -> str
```

- `generate_answer()` synthesizes a natural-language answer from retrieved evidence rows.
- `stream_generate_answer()` yields tokens for `st.write_stream()`.
- `call_dllm_api()` is the low-level OpenAI-compatible chat-completions proxy.

All three raise `RuntimeError` when `CLS_RETRIEVAL_ONLY=1` or the carrier is unconfigured.

### Parrot layer

```python
def parrot_answer(sentences: list[str], *, timeout=30.0) -> str | None
def parrot_stream(sentences: list[str], *, timeout=60.0) -> Generator[str, None, None]
def answer_text(sentences: list[str]) -> str
```

The Ask Lane uses `parrot_stream()` to rephrase grounded extractive bullets into one flowing paragraph. If the parrot model is unavailable or the output drifts from the evidence, the UI falls back to `answer_text()`, which formats the bullets deterministically.

---

## 2.3 Design rules

1. **No app state.** `cls_service.py` stores only the lazy singletons; it has no concept of sessions, UI widgets, or HTTP requests.
2. **No direct backend imports from callers.** All imports from `cls_backend.*` are concentrated here. If the pipeline interface changes, only this file and its tests need updating.
3. **Thread-safe initialization.** The `RLock` guarantees that even if two callers race to `get_embedder()`, the model loads exactly once.
4. **Deterministic fallbacks.** When the carrier or parrot is offline, the retrieval path still returns grounded evidence bullets.
5. **Configuration is read, not passed.** Constants come from `cls_config.py`; callers do not need to know collection names, model names, or paths.

---

## 2.4 Typical call chains

### Ask Lane query

```text
chat_lane.py
  -> ask_manual(query)
       -> get_collection(), get_embedder(), get_cache()
       -> pipeline.instant_answer()
       -> answer_text(sentences) OR parrot_stream(sentences)
```

### Full App upload

```text
app.py
  -> file_signature(tmp_path)
  -> ingest_path(path, source_hash, extra_metadata={"domain": domain})
       -> get_collection(), get_embedder()
       -> build_chunks() -> load_document() -> split_page_text()
       -> collection.add() / upsert()
       -> clear_lexical_index_cache()
```

### API query

```text
api.py
  -> ask_manual(query)
  -> optionally generate_answer() if model == CLS_DLLM_MODEL and not RETRIEVAL_ONLY
```

---

## 2.5 Testing the service

The test file `tests/test_cls_service.py` exercises the public API directly:

- `reset_collection()` is called in fixtures to guarantee a clean store.
- `ingest_path()` is used to index small fixture documents.
- `ask_manual()` verifies retrieval returns the expected evidence.
- `dllm_status()` and `parrot_status()` are mocked or run against empty carriers.

Because the singletons are module-level variables, tests that need a fresh state should call `reset_collection()` rather than trying to reinstantiate Chroma manually.
