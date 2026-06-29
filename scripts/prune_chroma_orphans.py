"""Prune orphaned ChromaDB HNSW segments from chroma_store/.

Safe to run while the app is stopped.  Reads chroma.sqlite3 to find every
segment UUID that a live collection references, then deletes any sub-directory
inside chroma_store/ that is NOT in that set.

Both the Full App (app.py) and the Ask Lane (chat_lane.py) share the same two
collections (cls_v2_evidence + cls_v2_cag_cache), so this script preserves
whatever is live for both UIs.

Usage
-----
    python scripts/prune_chroma_orphans.py          # dry-run (prints what would be removed)
    python scripts/prune_chroma_orphans.py --delete  # actually deletes orphaned dirs
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

STORE = Path(__file__).resolve().parent.parent / "chroma_store"
DB    = STORE / "chroma.sqlite3"


def live_segment_uuids(db_path: Path) -> set[str]:
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT id FROM segments").fetchall()
        return {r[0] for r in rows}
    finally:
        con.close()


def collection_summary(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        print("Live collections:")
        for cid, name, dim in con.execute(
            "SELECT id, name, dimension FROM collections ORDER BY name"
        ).fetchall():
            count_row = con.execute(
                "SELECT COUNT(*) FROM embeddings WHERE collection_id = ?", (cid,)
            ).fetchone()
            n = count_row[0] if count_row else "?"
            print(f"  {name!r:40s}  dim={dim}  chunks={n}")
        print()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune orphaned ChromaDB HNSW segment directories.")
    parser.add_argument("--delete", action="store_true", help="Actually delete orphaned directories (default: dry-run).")
    args = parser.parse_args()

    if not DB.exists():
        print(f"ERROR: {DB} not found. Is chroma_store/ in the right place?")
        return

    collection_summary(DB)

    live = live_segment_uuids(DB)
    on_disk = {d.name for d in STORE.iterdir() if d.is_dir()}
    orphans = on_disk - live
    kept    = on_disk & live

    print(f"Referenced segments  : {len(live)}")
    print(f"On-disk directories  : {len(on_disk)}")
    print(f"Orphaned (deletable) : {len(orphans)}")
    print(f"Live (kept)          : {len(kept)}")

    if not orphans:
        print("\nNothing to prune — chroma_store is already clean.")
        return

    print(f"\n{'[DRY-RUN] Would delete' if not args.delete else 'Deleting'} {len(orphans)} orphaned dirs:")
    for name in sorted(orphans):
        path = STORE / name
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576
        print(f"  {name}  ({size_mb:.1f} MB)")
        if args.delete:
            shutil.rmtree(path)

    if args.delete:
        print("\nDone. Orphaned segments removed.")
    else:
        print("\nDry-run complete. Pass --delete to actually remove them.")


if __name__ == "__main__":
    main()
