# Test corpus — multi-discipline, multi-format

This is the project's **test file folder area**: a deliberately messy, realistic
research-facility corpus used to exercise the RAG+CAG pipeline end to end.

It is **synthetic and generated** — not committed to git (~600 MB; see the root
`.gitignore`). Only this README is tracked. Regenerate the files locally.

## Layout

One folder per research discipline. The folder name **is** the `domain`
metadata key used by the Research Scope filter in the UI
(`cls_config.RESEARCH_SCOPES`):

```
data/corpus/
  chemistry/         ~100 MB · 5 files
  computer_science/  ~100 MB · 5 files
  biology/           ~100 MB · 5 files
  physics/           ~100 MB · 5 files
  mathematics/       ~100 MB · 5 files
  literature/        ~100 MB · 5 files
```

## Format mix

Each discipline holds 5 files, and the formats are spread so the corpus as a
whole exercises **every reader** in `cls_backend/readers.py`
(`.pdf .docx .txt .md .html .csv .tsv .json`). Data-heavy formats (txt/md/html/
csv/tsv/json) carry most of the bytes — the "raw data dumps" — while `.pdf` and
`.docx` stay paper-sized, matching how real facilities accumulate documents.

Content is topic-coherent per discipline (its own term bank and sentence
templates), so per-domain retrieval returns sensible, on-topic evidence.

## Regenerate

```bash
python scripts/generate_test_corpus.py                # full ~600 MB build
python scripts/generate_test_corpus.py --scale 0.05   # ~30 MB smoke build
python scripts/generate_test_corpus.py --only physics chemistry
```

Generation is seeded (`--seed`, default 1729), so a given seed reproduces the
same corpus.

## Index it

`scripts/ingest_corpus.py` walks each discipline folder and tags every chunk
with the correct `domain`, so the Research Scope radio buttons filter correctly:

```bash
python scripts/ingest_corpus.py            # index everything in data/corpus
python scripts/ingest_corpus.py --only biology
python scripts/ingest_corpus.py --dry-run  # preview without touching the store
```

> Note: the full ~600 MB corpus is a heavy embedding load on CPU. For a quick
> demo, generate at `--scale 0.05` or ingest a single discipline with `--only`.

The real, git-tracked fixtures (the IVU beamline manual and the Project
Gutenberg literature set) still live under `data/training_corpus/` and remain
the app's default one-click corpus.
