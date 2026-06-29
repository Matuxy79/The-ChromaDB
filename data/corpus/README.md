# Corpus Layout — Beamline-Scoped, Multi-Format

This folder tracks the beamline-organized corpus layout used by the app. It is
not the older six-discipline demo. Subdirectories match the `domain` slugs in
`cls_config.RESEARCH_SCOPES`, and they can be populated incrementally as CLS
material arrives.

## Layout

One folder per beamline slug. Empty folders are normal between ingestion
cycles; this corpus grows episodically as beamline material is added.

```text
data/corpus/
  bioxas_imaging/
  bioxas_spectroscopy/
  bmit/
  bxds/
  cls_aps/
  cmcf/
  eiml/
  far_ir/
  hxma/
  ideas/
  mid_ir/
  qmsc/
  reixs/
  sgm/
  sm/
  sxrmb/
  sylmand/
  vespers/
  vls_pgm/
```

The checked-in `all/` folder is currently just an empty placeholder; it is not
one of the named beamline scopes in `RESEARCH_SCOPES`.

## Format mix

Beamline folders can contain any of the formats supported by
`cls_backend/readers.py` (`.pdf .docx .txt .md .html .csv .tsv .json`). Files
arrive episodically, so some beamlines may be sparse while others hold manuals,
runbooks, notes, or data exports.

## Index it

`scripts/ingest_corpus.py` walks each beamline folder and tags every chunk with
that folder slug as `domain`, which is the metadata key used by retrieval:

```bash
python scripts/ingest_corpus.py                # index everything in data/corpus
python scripts/ingest_corpus.py --only bmit cmcf
python scripts/ingest_corpus.py --dry-run      # preview without touching the store
```

The default one-click admin corpus still lives under `data/training_corpus/`.
