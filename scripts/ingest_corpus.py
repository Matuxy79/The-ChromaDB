#!/usr/bin/env python3
"""Ingest the beamline-organised corpus into the Evidence Store.

Walks <root>/<beamline_slug>/ and indexes every supported file, tagging each
chunk with `domain=<beamline_slug>` so the Research Scope filter in the UI
(cls_config.RESEARCH_SCOPES) returns the right slice. The folder name *is* the
beamline slug from the shared scope map — for example `bmit`, `cmcf`,
`reixs`, `vespers` — so there is no per-file metadata to maintain.

This is the multi-beamline counterpart to the app's corpus-admin flow: one
pass across every populated beamline folder with the correct tags.

Usage
-----
    python scripts/ingest_corpus.py                       # index data/corpus
    python scripts/ingest_corpus.py --root data/corpus --only bmit cmcf
    python scripts/ingest_corpus.py --force               # re-embed everything
    python scripts/ingest_corpus.py --dry-run             # list, don't index
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cls_backend.readers import is_supported  # noqa: E402
from cls_backend.pipeline import EMBED_DIM  # noqa: E402
from cls_config import RESEARCH_SCOPES  # noqa: E402
from cls_service import file_signature, ingest_path  # noqa: E402


class HashEmbedder:
    """Fast, deterministic placeholder embeddings — no model, no inference.

    Each chunk gets a distinct unit vector derived from a hash of its text. This is
    ~1000x faster than the real sentence-transformer and is correct for KEYWORD-ONLY
    retrieval, where the stored vectors are never read at query time (search ranks by
    lexical overlap). The vectors are NOT semantically meaningful — re-ingest with the
    real embedder (drop --no-embed) before enabling semantic/hybrid retrieval.
    """

    def embed(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            digest = hashlib.sha256((text or "").encode("utf-8", "ignore")).digest()  # 32 bytes
            base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            tiled = np.tile(base, -(-EMBED_DIM // base.size))[:EMBED_DIM]
            norm = np.linalg.norm(tiled) or 1.0
            rows.append((tiled / norm).tolist())
        return rows

# Valid beamline slugs advertised by the UI's Research Scope filter.
VALID_DOMAINS = {
    scope["domain"] for scope in RESEARCH_SCOPES.values() if scope
}


def scope_dirs(root: Path, only: list[str] | None) -> list[Path]:
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if only:
        wanted = set(only)
        dirs = [p for p in dirs if p.name in wanted]
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT / "data" / "corpus",
                        help="Corpus root containing one folder per beamline slug.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Limit to these beamline folders.")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed files that are already indexed.")
    parser.add_argument("--no-embed", action="store_true",
                        help="Use fast placeholder vectors instead of the real embedder. "
                             "Correct for keyword-only retrieval; re-ingest without this "
                             "before enabling semantic/hybrid mode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be indexed without touching the store.")
    args = parser.parse_args()

    embedder = HashEmbedder() if args.no_embed else None
    if args.no_embed:
        print("[fast] Using placeholder embeddings (keyword-only mode — vectors unused at query time).")

    if not args.root.exists():
        sys.exit(f"Corpus root not found: {args.root}\n"
                 f"Populate it first with beamline folders and documents.")

    totals = {"indexed": 0, "skipped": 0, "failed": 0, "chunks": 0}
    for folder in scope_dirs(args.root, args.only):
        domain = folder.name
        if domain not in VALID_DOMAINS:
            print(f"[warn] {domain!r} is not a known beamline scope slug "
                  f"({', '.join(sorted(VALID_DOMAINS))}); tagging it anyway.")
        files = sorted(p for p in folder.iterdir() if p.is_file() and is_supported(p))
        print(f"\n[{domain}] {len(files)} file(s)")
        for path in files:
            if args.dry_run:
                print(f"  would index {path.name}  ->  domain={domain}")
                continue
            try:
                count, message = ingest_path(
                    path,
                    file_signature(path),
                    force=args.force,
                    extra_metadata={"domain": domain},
                    embedder=embedder,
                )
            except Exception as exc:  # noqa: BLE001
                totals["failed"] += 1
                print(f"  [failed]  {path.name}: {type(exc).__name__}: {exc}")
                continue
            totals["chunks"] += count
            if message == "already indexed":
                totals["skipped"] += 1
            else:
                totals["indexed"] += 1
            print(f"  [{message}]  {path.name}  ({count} chunks)")

    if not args.dry_run:
        print(f"\nDone: {totals['indexed']} indexed, {totals['skipped']} skipped, "
              f"{totals['failed']} failed, {totals['chunks']} chunks added.")


if __name__ == "__main__":
    main()
