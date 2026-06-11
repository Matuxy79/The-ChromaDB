import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

import fitz  # PyMuPDF


ProgressCallback = Callable[[str, int, int], None]


def _extract_page(doc_path: str, page_index: int) -> Tuple[int, str]:
    # Each thread opens its own fitz handle. PyMuPDF documents are not safe to
    # share across threads, but per-thread Document instances are fine and the
    # GIL is released during text extraction.
    with fitz.open(doc_path) as doc:
        return page_index, doc[page_index].get_text()


def extract_text_from_pdf(
    file_path: str,
    progress_callback: Optional[ProgressCallback] = None,
    max_workers: int = 4,
) -> Tuple[str, Dict[str, Any]]:
    start = time.perf_counter()

    try:
        with fitz.open(file_path) as doc:
            page_count = doc.page_count
    except Exception as exc:
        print(f"Error opening PDF: {exc}")
        return "", {"extract_seconds": 0.0, "pages": 0, "chars": 0}

    if page_count == 0:
        return "", {"extract_seconds": 0.0, "pages": 0, "chars": 0}

    if progress_callback:
        progress_callback("extract", 0, page_count)

    pages = [""] * page_count
    workers = max(1, min(max_workers, page_count))

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_extract_page, file_path, i) for i in range(page_count)]
            for done, future in enumerate(futures, start=1):
                index, text = future.result()
                pages[index] = text
                if progress_callback:
                    progress_callback("extract", done, page_count)
    except Exception as exc:
        print(f"Error extracting PDF text: {exc}")
        return "", {"extract_seconds": time.perf_counter() - start, "pages": page_count, "chars": 0}

    text = "".join(pages)
    return text, {
        "extract_seconds": time.perf_counter() - start,
        "pages": page_count,
        "chars": len(text),
    }


def ingest_document(
    kb,
    file_path: str,
    metadata: Dict[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[Dict[str, Any]]:
    lower_path = file_path.lower()
    stats: Dict[str, Any] = {"file": metadata.get("source_url", file_path)}

    if lower_path.endswith(".pdf"):
        text, extract_stats = extract_text_from_pdf(file_path, progress_callback=progress_callback)
        stats.update(extract_stats)
    else:
        if progress_callback:
            progress_callback("extract", 0, 1)
        read_start = time.perf_counter()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as exc:
            print(f"Error reading file {file_path}: {exc}")
            return None
        stats.update({
            "extract_seconds": time.perf_counter() - read_start,
            "pages": 1,
            "chars": len(text),
        })
        if progress_callback:
            progress_callback("extract", 1, 1)

    if not text.strip():
        return None

    add_stats = kb.add_document(text, metadata, progress_callback=progress_callback)
    if add_stats:
        stats.update(add_stats)

    return stats
