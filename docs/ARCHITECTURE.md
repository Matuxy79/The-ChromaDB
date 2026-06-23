# CLS Synchrotron Research Query — Architecture (v1.2)

## Visual Architecture Map

> How the generic web-app layers map to **this** project, corrected for what actually exists.

```mermaid
flowchart TD

    %% ── CLIENT ─────────────────────────────────────────────────────────
    subgraph CLIENT["CLIENT  ·  Browser"]
        UI1["Streamlit Full App\napp.py\n— Admin + User roles, admin, eval, retrieval-only by default"]
        UI2["Chainlit Ask Lane\nchat_lane.py\n— fast cited chat, no LLM, no telemetry"]
    end

    %% ── ROLES NOTE ──────────────────────────────────────────────────────
    ROLES["⚠ ROLES  ·  UI presentation tiers only\nExplorer · Analyst · Engineer\ncontrols visible widgets — NOT a backend security boundary\n\nThis is where  Auth / Security  would live\nif real login / token enforcement were added"]

    %% ── API GATEWAY (optional) ──────────────────────────────────────────
    subgraph APIGW["API GATEWAY  ·  optional\napi.py  ·  FastAPI\nonly active when  CLS_USE_API=1"]
        R1["POST /v1/query"]
        R2["POST /v1/chat/completions"]
        R3["POST /v1/dllm/chat"]
    end

    %% ── SERVICES ────────────────────────────────────────────────────────
    subgraph SVC["SERVICES\ncls_service.py"]
        S1["ingest  ·  query  ·  cache\nLLM proxy blocked when CLS_RETRIEVAL_ONLY=1\nstatus  ·  answer format"]
    end

    %% ── RAG/CAG PIPELINE ────────────────────────────────────────────────
    subgraph PIPE["RAG / CAG PIPELINE\ncls_backend/pipeline.py"]
        P1["query_repair.py\nstrip NL scaffolding\n'can you tell me about X' → 'X'"]
        P2["MiniLM embed\nskipped when CLS_KEYWORD_ONLY=1\nall-MiniLM-L6-v2  ·  local CPU\n384-dim vector"]
        P3["scope filter\ndomain metadata gate"]
        P4{"CAG cache\nhit?"}
        P5["retrieve\nkeyword-only fast path OR\nsemantic vector + lexical keyword"]
        P6["relevance audit\nLLM-backed; disabled in retrieval-only mode"]
        P7["cited answer builder\ndeterministic extraction\nevidence rows + source chips"]
        P1 --> P2 --> P3 --> P4
        P4 -->|"cache miss"| P5 --> P6 --> P7
        P4 -->|"cache hit\nreuse stored evidence"| P7
    end

    %% ── DATABASE / CACHE ────────────────────────────────────────────────
    subgraph STORE["DATABASE / CACHE\nchroma_store/  ·  ChromaDB"]
        DB1["cls_v2_evidence\n384-dim evidence vectors\nindexed chunks from all documents"]
        DB2["cls_v2_cag_cache\nCAG semantic cache\ncosine-matched past questions"]
    end

    %% ── OPTIONAL LLM CARRIER ────────────────────────────────────────────
    subgraph CARRIER["OPTIONAL LLM CARRIER\ncls_backend/dllm.py"]
        L1["OpenRouter  ·  Ollama  ·  llama.cpp\nany OpenAI-compatible endpoint\ndefault: openai/gpt-oss-120b\ndisabled while CLS_RETRIEVAL_ONLY=1"]
    end

    %% ── EDGES ───────────────────────────────────────────────────────────
    UI1 & UI2 --> ROLES

    ROLES -->|"embedded mode\ndefault — direct Python call"| SVC
    ROLES -.->|"CLS_USE_API=1\nHTTP bridge"| APIGW

    APIGW --> SVC
    SVC --> PIPE
    PIPE <-->|"vector search\nCAG lookup + store"| STORE
    P7 -.->|"optional synthesis\nFull App only"| CARRIER
    CARRIER -.->|"grounded answer\nback into evidence rows"| P7
```

**Reading the diagram:**

- Solid arrows `→` are the default embedded code path (no HTTP hop).
- Dashed arrows `-.->` are optional / conditional paths.
- The **Roles** box shows where `Auth / Security` *would* live; right now it only gates UI widgets, not backend data.
- The **API Gateway** box is skipped entirely in the default run; it activates only with `CLS_USE_API=1`.

---

## Temporary Fast Mode

Current default is speed-first and retrieval-only:

```bash
export CLS_RETRIEVAL_ONLY=1  # disables synthesis, cleanup, self-debate, parrot, and API proxy calls
export CLS_KEYWORD_ONLY=1    # skips query embedding/cache lookup and ranks by keyword overlap
```

To restore the previous semantic+carrier behavior for an experiment:

```bash
export CLS_RETRIEVAL_ONLY=0
export CLS_KEYWORD_ONLY=0
```

In retrieval-only mode, `/v1/dllm/chat` and direct `CLS_DLLM_MODEL` chat completions return 403. `/v1/query` and the `cls-rag-cag-v1.0` chat route continue to return deterministic cited retrieval.

---

## Codebase Map — Every File and Its Role

### Entry Points

| File | What it is | How it starts |
| --- | --- | --- |
| [app.py](../app.py) | Streamlit Full App — Admin + User roles, corpus admin, upload, precision controls, eval, evidence rows; retrieval-only by default | `streamlit run app.py` |
| [chat_lane.py](../chat_lane.py) | Chainlit Ask Lane — instant grounded bullets (RAG/CAG); parrot prose is disabled while `CLS_RETRIEVAL_ONLY=1` | `chainlit run chat_lane.py -w` |
| [api.py](../api.py) | FastAPI bridge — optional HTTP interface for external/OpenAI-compatible frontends, activated with `CLS_USE_API=1` | `uvicorn api:app` |
| [ingest_daemon.py](../ingest_daemon.py) | Folder-watching ingestion daemon — polls `docs/inbox/`, ingests new files via sidecar metadata, moves to `processed/` or `failed/` | `python ingest_daemon.py` |

### Config

| File | What it is |
| --- | --- |
| [cls_config.py](../cls_config.py) | Single source of truth for all env vars, collection names, paths, chunk sizes, model defaults — every other module imports from here |
| [cls.env](../cls.env) | Local secrets and overrides (gitignored); copied from `cls.env.example` |

### Service Layer

| File | What it is |
| --- | --- |
| [cls_service.py](../cls_service.py) | Service façade — lazy singleton init of embedder, Chroma client, collections, CAG cache. Exposes `ask_manual`, `ingest_path`, `call_dllm_api`, `parrot_stream`, `generate_answer`, `reset_collection`, `evaluate_retrieval`, etc. All UI and API code calls this; nothing reaches `cls_backend/` directly. |

### Backend — Pure Python, No App State

| File | What it is |
| --- | --- |
| [cls_backend/pipeline.py](../cls_backend/pipeline.py) | Core retrieval engine: `SentenceTransformerEmbedder`, `instant_answer`, `retrieve` (semantic), `lexical_retrieve`, `_merge_hybrid`, `expand_cross_refs`, `collection_count`. No LLM in the hot path. |
| [cls_backend/cag_cache.py](../cls_backend/cag_cache.py) | `SemanticEvidenceCache` — second ChromaDB collection; cosine-matches past questions by embedding; stores/returns evidence rows; invalidated by `corpus_sig` on re-index. |
| [cls_backend/query_repair.py](../cls_backend/query_repair.py) | `repair_query` — strips NL scaffolding (`_FILLER_PREFIX`, `_FILLER_WORDS`) before embedding. Hooks for `TYPO_REPLACEMENTS` and `QUERY_EXPANSIONS` are empty; fill for deployment-specific acronyms. |
| [cls_backend/dllm.py](../cls_backend/dllm.py) | Carrier prompt builders and guards: `CORRECTION_SYSTEM`, `ANSWER_SYSTEM`, `PARROT_SYSTEM`, `needs_correction` (sparse gate), `numbers_grounded`, `relation_drift`, `parse_bullets`, `validate_correction`. Pure-Python — no HTTP calls here. |
| [cls_backend/readers.py](../cls_backend/readers.py) | Document normalisation: `load_pdf` (pymupdf), `load_text`, `load_docx`, `load_html`, `load_csv`, `load_json`. All return `(page_number, text)` tuples; chunking and embedding are unchanged per format. Gutenberg boilerplate stripping included. |
| [cls_backend/spectrum.py](../cls_backend/spectrum.py) | **Presentation only.** `classify_query` maps a query to a visible-spectrum category (contacts / procedure / specs / general). Returns colour hue, glyph, and label for the UI answer card. Never touches retrieval. |

### Storage

| Path | What it holds |
| --- | --- |
| `chroma_store/` | Persistent ChromaDB directory: two collections — evidence vectors (`cls_v2_evidence`, 384d) and CAG cache (`cls_v2_cag_cache`, 384d). |
| `docs/inbox/` | Drop zone for `ingest_daemon`. New files land here. |
| `docs/processed/` | Files successfully ingested by the daemon. |
| `docs/failed/` | Files the daemon could not ingest (unreadable format, extraction error). |

---

## Two Data Flows

### Query Path

```mermaid
flowchart LR
    Q([user question]) --> QR["query_repair.py\nstrip filler"]
    QR --> SP["spectrum.py\nclassify_query\ncolour + glyph\npresentation only"]
    QR --> MODE{"CLS_KEYWORD_ONLY=1?"}
    MODE -->|"yes"| LEXONLY["pipeline.lexical_retrieve\nkeyword overlap scan"]
    MODE -->|"no"| EMB["pipeline.py\nSentenceTransformerEmbedder\n384-dim vector"]
    EMB --> SCOPE["metadata filter\ndomain scope\n cls_config.RESEARCH_SCOPES"]
    SCOPE --> CAG{"cag_cache.py\nSemanticEvidenceCache\ncosine hit?"}
    CAG -->|"hit — reuse rows"| BUILD
    CAG -->|"miss"| SEM["pipeline.retrieve\nsemantic vector search"]
    CAG -->|"miss"| LEX["pipeline.lexical_retrieve\nterm-overlap keyword scan"]
    SEM & LEX --> MERGE["_merge_hybrid\nsemantic scores win\nlexical as safety net"]
    LEXONLY --> BUILD
    MERGE --> AUDIT["relevance audit\ndrop MIN_EXTRACTIVE_SCORE < 0.38"]
    AUDIT --> BUILD["cited answer builder\ndeterministic extraction\nevidence rows + source chips"]
    BUILD --> PARROT(["Pass 2 optional\nparrot_stream\nqwen2.5:0.5b · Ollama\nAsk Lane only"])
    BUILD --> CARRIER(["Pass 2 optional\ncarrier synthesis\nOpenRouter / Ollama / llama.cpp\nFull App only"])
    BUILD --> GUARD["dllm.py guards\nnumbers_grounded\nrelation_drift\nneeds_correction"]
    PARROT --> GUARD
    CARRIER --> GUARD
```

### Ingest Path

```mermaid
flowchart LR
    DROP(["file dropped\ndocs/inbox/\nor uploaded via UI"]) --> SIDECAR["ingest_daemon.py\nload_sidecar_metadata\n.metadata.json optional"]
    SIDECAR --> READ["readers.py\nload_pdf · load_text\nload_docx · load_html\nload_csv · load_json\n→ list[(page, text)]"]
    READ --> CHUNK["cls_service.py\nchunk_document\n1100 chars · 180 overlap"]
    CHUNK --> EMBED["pipeline.py\nSentenceTransformerEmbedder\nembed batch → 384-dim"]
    EMBED --> STORE[("chroma_store/\ncls_v2_dsrag_evidence\nchunk + metadata + vector")]
    STORE --> SIG["corpus_sig updated\nCAG cache entries with\nold sig become unreachable"]
    DROP -->|"success"| PROC["docs/processed/"]
    DROP -->|"error"| FAIL["docs/failed/"]
```

---

## Product Shape

- **Indexing:** semantic sectioning with AutoContext patterns, embedded with `all-MiniLM-L6-v2` (sentence-transformers, offline CPU), stored in ChromaDB.
- **Research scopes:** metadata-gated retrieval across six disciplines (Chemistry, Computer Science, Biology, Physics, Mathematics, Literature).
- **Query repair:** natural-language scaffolding is stripped before embedding so verbose human phrasing maps to the same vector as the keyword form.
- **Temporary keyword retrieval:** `CLS_KEYWORD_ONLY=1` skips query embedding and CAG lookup, then ranks by lexical term overlap for fastest deterministic searches.
- **Hybrid retrieval:** when `CLS_KEYWORD_ONLY=0`, semantic vector search runs alongside a lexical keyword scan; results merge so exact-keyword hits are never lost.
- **Relevance audit:** LLM-backed second-pass filter; disabled while `CLS_RETRIEVAL_ONLY=1`.
- **Query path:** repair the query → keyword-only retrieval or hybrid retrieve → deterministic cited extraction → optional carrier synthesis only when `CLS_RETRIEVAL_ONLY=0`.
- **Shared API:** same retrieval path exposed to Streamlit and OpenAI-compatible frontends.
- **Inference carrier:** when keyed, OpenRouter + `openai/gpt-oss-120b` synthesizes a direct answer from refined evidence rows (Full App only).

## Two UIs

| Surface | Entry | Visible controls |
| --- | --- | --- |
| Full App | Landing → "Full App" card | All roles, corpus admin, upload, precision controls, eval, evidence rows, optional carrier synthesis |
| Ask Lane | Landing → "Ask Lane" card | Bright llama.cui-style chat: scope selector, chat input, cited answer with source chips — no LLM, no engineering telemetry |

Both surfaces share the same retrieval backend. Session state keys are isolated (`lane_*` vs `last_*`) so switching between them is safe. The Ask Lane keeps a conversation history (`lane_messages`) and renders each turn as native chat bubbles with a source-chip row.

## Research Scopes

```python
RESEARCH_SCOPES = {
    "All disciplines":  None,
    "Chemistry":        {"domain": "chemistry"},
    "Computer Science": {"domain": "computer_science"},
    "Biology":          {"domain": "biology"},
    "Physics":          {"domain": "physics"},
    "Mathematics":      {"domain": "mathematics"},
    "Literature":       {"domain": "literature"},
}
```

`None` bypasses the Chroma metadata filter. Any other value is passed as `metadata_filter={"domain": "<value>"}` to the retrieval call. Documents are tagged at upload time with the matching domain string.

## Model Roles

| Component | Model | Required? | Why |
| --- | --- | --- | --- |
| Query / index embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Yes | Real semantic vectors, offline on CPU, ~80 MB one-time download |
| Relevance audit | `CLS_DLLM_MODEL` | Optional; blocked when `CLS_RETRIEVAL_ONLY=1` | Drop irrelevant chunks before synthesis |
| Inference carrier | `openai/gpt-oss-120b` (default) | Optional; blocked when `CLS_RETRIEVAL_ONLY=1` | Synthesizes grounded answer from evidence rows (Full App) |
| Carrier cleanup | `CLS_DLLM_MODEL` | Optional; blocked when `CLS_RETRIEVAL_ONLY=1` | Corrects PDF extraction artifacts; off by default |

The embedder downloads once to `~/.cache/huggingface/hub/`; after that it loads locally with no network. The launchers do not start Ollama and do not pull any LLM.

### Pluggable carrier

The carrier is any OpenAI-compatible `/v1/chat/completions` endpoint. Swap backends with env vars — no code change:

- **OpenRouter** (cloud, keyed): default `https://openrouter.ai/api/v1`.
- **llama.cpp** (local, offline): run `llama-server -m model.gguf --port 8080`, set `CLS_DLLM_API_URL=http://localhost:8080/v1`, unset the key.
- **Ollama** (local): point `CLS_DLLM_API_URL` at `http://localhost:11434/v1`.

## Runtime Layers

```text
Streamlit UI (Research Scopes)
  -> query repair (strip NL scaffolding)
  -> FastAPI bridge when CLS_USE_API=1, otherwise embedded cls_service
  -> if CLS_KEYWORD_ONLY=1: lexical keyword retrieval
  -> else: SentenceTransformerEmbedder + Metadata Filter + CAG Layer
  -> Hybrid retrieve on cache miss: semantic vector + lexical keyword, merged (ChromaDB)
  -> Relevance Audit (optional; blocked when CLS_RETRIEVAL_ONLY=1)
  -> deterministic answer builder
  -> optional inference carrier synthesis only when CLS_RETRIEVAL_ONLY=0
  -> optional carrier cleanup only when CLS_RETRIEVAL_ONLY=0
```

`/v1/chat/completions` exposes two model routes:

- `cls-rag-cag-v1.0`: local RAG/CAG extraction.
- `CLS_DLLM_MODEL`: proxy to the configured external inference carrier, omitted/blocked when `CLS_RETRIEVAL_ONLY=1`.

`/v1/dllm/chat` is the direct carrier proxy used by Streamlit synthesis, relevance audit, and cleanup. It returns 403 while `CLS_RETRIEVAL_ONLY=1`.

## Hybrid Retrieval

`instant_answer` always runs both retrievers and merges them in `_merge_hybrid`:

- **Semantic** (`retrieve`): MiniLM vector search — handles paraphrase and messy natural language.
- **Lexical** (`lexical_retrieve`): term-overlap keyword scan — catches an exact keyword the encoder ranked low.
- **Merge:** semantic scores win on chunks both retrievers return; lexical-only hits are appended as a safety net, then truncated to `top_k`.

If the embedder is unavailable, retrieval degrades cleanly to lexical-only (`retrieval_mode = "lexical_fallback"`).

## Query Repair

`cls_backend/query_repair.py::repair_query` strips conversational scaffolding before embedding — "can you tell me about the X-ray energy range" → "X-ray energy range". It runs on every query in `query_backend`. `TYPO_REPLACEMENTS` and `QUERY_EXPANSIONS` are empty hooks for deployment-specific acronym repair.

## CAG Layer

The CAG Layer (`cls_backend/cag_cache.py`, `SemanticEvidenceCache`) is a second ChromaDB collection keyed by the embedding of a past question.

- **Lookup:** question is encoded by MiniLM and matched against the cache with a cosine threshold. With a real semantic encoder, the cache now generalises across paraphrases, not just near-identical strings.
- **Reuse granularity:** evidence rows only. A hit reuses stored evidence while the deterministic answer builder re-runs.
- **Invalidation:** every entry stores a `corpus_sig` derived from Evidence Store source hashes. Re-indexing makes stale cache entries unreachable.

## Evidence Store

Collection names are in `cls_config.py`:

- `cls_v2_evidence` — indexed evidence chunks (384d, MiniLM).
- `cls_v2_cag_cache` — cached query-to-evidence rows (384d, MiniLM).

> **Migration note (v1.1 → v1.2):** the encoder changed from a 768d hash to 384d MiniLM, so the collections were renamed `v1` → `v2`. The old `cls_v1_*` collections are no longer queried. Use **Reset Chroma index** in the admin sidebar and re-index to build the v2 store.
>
> **Migration note (drop `dsrag` tag):** the collections were renamed `cls_v2_dsrag_*` → `cls_v2_*` (the `dsrag` label was dead naming — not the real `dsrag` library). After pulling this change the renamed collections start empty; re-index (admin sidebar **Reset Chroma index**, or `python scripts/ingest_corpus.py`) to populate them. The orphaned `cls_v2_dsrag_*` collections can be dropped from `chroma_store/`.

### Document readers

`cls_backend/readers.py` normalises extraction across formats so the rest of the pipeline stays document-type-agnostic. Supported formats:

- PDF (`pymupdf`)
- Plain text / Markdown
- DOCX (`python-docx`, small pure-Python dependency)
- HTML / HTM (stdlib parser, strips script/style/nav/footer/header)
- CSV / TSV (stdlib, each row becomes a pseudo-page with header context)
- JSON (stdlib, flattened and paginated)

All readers return `(page_number, text)` tuples; chunking, embedding, and indexing are unchanged.

## Performance Notes

For a 7 MB / 100-page PDF on CPU:

| Step | Typical time |
| --- | --- |
| PDF extraction | a few seconds |
| Chunking | under a second |
| MiniLM embedding (first call) | a few seconds (model load) |
| MiniLM embedding (warm) | well under a second |
| Chroma storage | under a few seconds |

The inference carrier runs only after retrieval; it has no effect on indexing speed.

## Design Philosophy

- Retrieval is instant and primary (DocuSearch-inspired).
- The Ask Lane is built for speed first: instant cited extraction, with an optional tiny local parrot (Qwen 2.5 0.5B via Ollama/llama.cpp) that rephrases evidence and is guarded by a deterministic relation-drift check.
- The deterministic cited extraction is always shown; carrier synthesis is a downstream Full-App option.
- The prototype favors inspectable local retrieval: every evidence row is visible and auditable.
- Carrier cleanup is downstream, optional, guarded, and API-only.
