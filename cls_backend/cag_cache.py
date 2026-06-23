"""Semantic CAG layer — a Cache-Augmented Generation evidence cache.

This is the missing piece that makes the app genuinely RAG+CAG rather than RAG with
resource memoization. It is a *second* ChromaDB collection keyed by the embedding of a
past question. When a similar question reappears, the previously retrieved evidence is
returned and the slow evidence-store search is skipped (the formatter still re-runs, so
the prose stays fresh — we cache retrieval, not generation).

Architecture (names used across the app):
    Retrieval Encoder (sentence-transformers) -> Evidence Store (ChromaDB) -> deterministic answer builder
    CAG Layer = this semantic cache of prior (query -> retrieved evidence) results.

Design notes:
- Dependency-injected (collection + encoder) so it stays pure and testable without
  importing `app.py`. Any encoder exposing `.embed(iterable) -> list[list[float]]`
  works; the app passes its HashEmbedder.
- Entries are scoped by `corpus_sig` so a re-indexed Evidence Store never serves stale
  evidence. The cache holds JSON-serialised evidence rows in the document field.
- With a semantic encoder, near-identical and paraphrased queries can reuse prior
  evidence while the corpus signature prevents stale retrieval rows.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional


def _normalize(query: str) -> str:
    """Whitespace/case-insensitive key so trivial spacing variants share an entry."""
    return re.sub(r"\s+", " ", query.strip().lower())


class SemanticEvidenceCache:
    def __init__(self, collection, encoder, distance_max: float = 0.20) -> None:
        self.collection = collection
        self.encoder = encoder
        self.distance_max = distance_max

    def _embed(self, text: str) -> list[float]:
        return self.encoder.embed([text])[0]

    def _entry_id(self, query: str, corpus_sig: str) -> str:
        key = f"{corpus_sig}::{_normalize(query)}".encode("utf-8")
        return hashlib.sha256(key).hexdigest()[:24]

    def lookup(self, query: str, corpus_sig: str) -> Optional[dict[str, Any]]:
        """Return the cached evidence for a close-enough prior query, else None."""
        if not query.strip() or self.count() == 0:
            return None
        embedding = self._embed(query)
        try:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=1,
                where={"corpus_sig": corpus_sig},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return None
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        if not documents or not distances:
            return None
        distance = float(distances[0])
        if distance > self.distance_max:
            return None
        meta = metadatas[0] or {}
        return {
            "rows": json.loads(documents[0]),
            "similarity": max(0.0, min(1.0, 1.0 - distance)),
            "cached_query": meta.get("query", ""),
            "created_at": meta.get("created_at", 0.0),
        }

    def store(
        self,
        query: str,
        rows: list[dict],
        corpus_sig: str,
        category: str = "general",
        top_k: int = 0,
    ) -> None:
        """Upsert the retrieved evidence under this query (deterministic id dedups)."""
        if not query.strip() or not rows:
            return
        embedding = self._embed(query)
        try:
            self.collection.upsert(
                ids=[self._entry_id(query, corpus_sig)],
                embeddings=[embedding],
                documents=[json.dumps(rows)],
                metadatas=[
                    {
                        "query": query.strip(),
                        "category": category,
                        "corpus_sig": corpus_sig,
                        "top_k": int(top_k),
                        "created_at": time.time(),
                    }
                ],
            )
        except Exception:
            return

    def clear(self) -> None:
        try:
            existing = self.collection.get()
        except Exception:
            return
        ids = existing.get("ids") or []
        if ids:
            try:
                self.collection.delete(ids=ids)
            except Exception:
                return

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0
