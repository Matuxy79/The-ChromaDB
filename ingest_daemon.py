import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from cls_service import file_signature, ingest_path


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_sidecar_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(path.suffix + ".metadata.json"),
        path.with_suffix(".metadata.json"),
    ]

    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as file:
                return json.load(file)

    return {}


def sidecar_paths(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in (
            path.with_suffix(path.suffix + ".metadata.json"),
            path.with_suffix(".metadata.json"),
        )
        if candidate.exists()
    ]


def unique_destination(directory: Path, name: str) -> Path:
    destination = directory / name
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem}-{index}{suffix}"
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not create unique destination for {name}")


def move_with_sidecars(path: Path, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), unique_destination(directory, path.name))

    for sidecar in sidecar_paths(path):
        shutil.move(str(sidecar), unique_destination(directory, sidecar.name))


def iter_documents(inbox: Path) -> list[Path]:
    inbox.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in inbox.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def process_document(path: Path, args: argparse.Namespace) -> bool:
    metadata = {
        "colour_code": args.lane,
        "domain": args.domain,
        "source_url": path.name,
        "trust_level": args.trust_level,
    }
    metadata.update(load_sidecar_metadata(path))

    source_hash = file_signature(path)

    try:
        chunks_indexed, status = ingest_path(
            path, source_hash, extra_metadata=metadata
        )
    except Exception as exc:
        print(f"[failed] {path.name}: {exc}")
        move_with_sidecars(path, args.failed)
        return False

    if status == "no readable text found":
        print(f"[skipped] {path.name}: no extractable text")
        move_with_sidecars(path, args.failed)
        return False

    if status == "already indexed":
        print(f"[skipped] {path.name}: already indexed")
        move_with_sidecars(path, args.processed)
        return True

    print(
        f"[indexed] {path.name} lane={metadata.get('colour_code')} "
        f"domain={metadata.get('domain')} chunks={chunks_indexed}"
    )
    move_with_sidecars(path, args.processed)
    return True


def run_once(args: argparse.Namespace) -> int:
    documents = iter_documents(args.inbox)
    if not documents:
        print(f"[idle] No PDF/TXT files found in {args.inbox}")
        return 0

    count = 0
    for document in documents:
        if process_document(document, args):
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLS one-time/daemon document indexer")
    parser.add_argument("--inbox", type=Path, default=Path("docs/inbox"))
    parser.add_argument("--processed", type=Path, default=Path("docs/processed"))
    parser.add_argument("--failed", type=Path, default=Path("docs/failed"))
    parser.add_argument("--lane", default="green", choices=["purple", "green", "blue", "orange", "yellow"])
    parser.add_argument("--domain", default="beamline", choices=["beamline", "research", "outreach", "logistics", "education"])
    parser.add_argument("--trust-level", default="official_cls")
    parser.add_argument("--watch", action="store_true", help="Keep polling the inbox after the first pass.")
    parser.add_argument("--interval", type=int, default=10, help="Polling interval in seconds when --watch is set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.inbox.mkdir(parents=True, exist_ok=True)
    args.processed.mkdir(parents=True, exist_ok=True)
    args.failed.mkdir(parents=True, exist_ok=True)

    while True:
        indexed = run_once(args)
        if not args.watch:
            print(f"[done] Indexed {indexed} document(s).")
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
