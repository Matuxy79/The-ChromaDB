import hashlib
import unittest
import uuid

import chromadb

from cls_backend.cag_cache import SemanticEvidenceCache


class FakeEncoder:
    """Deterministic encoder: identical normalized text -> identical unit-ish vector.

    Maps the lowercased/space-collapsed text to a sparse 16-dim vector via hashing so
    distinct queries are (near-)orthogonal and repeats are identical.
    """

    DIM = 16

    def embed(self, texts):
        out = []
        for text in texts:
            key = " ".join(text.lower().split())
            vec = [0.0] * self.DIM
            digest = hashlib.sha256(key.encode()).digest()
            bucket = digest[0] % self.DIM
            vec[bucket] = 1.0
            out.append(vec)
        return out


def make_cache(distance_max=0.03):
    client = chromadb.EphemeralClient()
    col = client.create_collection(
        f"cag_test_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    return SemanticEvidenceCache(col, FakeEncoder(), distance_max=distance_max)


ROWS = [
    {"document": "Control room extension is 3570.", "metadata": {"source": "m.pdf", "page": 4}, "distance": 0.8, "score": 0.2},
]


class CagCacheTests(unittest.TestCase):
    def test_store_then_lookup_identical_is_hit(self):
        cache = make_cache()
        cache.store("control room phone?", ROWS, corpus_sig="sigA", category="contacts", top_k=8)
        hit = cache.lookup("control room phone?", corpus_sig="sigA")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["rows"], ROWS)
        self.assertGreaterEqual(hit["similarity"], 0.99)
        self.assertEqual(hit["cached_query"], "control room phone?")

    def test_normalized_variant_is_hit(self):
        cache = make_cache()
        cache.store("control room  phone?", ROWS, corpus_sig="sigA")
        self.assertIsNotNone(cache.lookup("control room phone?", corpus_sig="sigA"))

    def test_unrelated_query_is_miss(self):
        cache = make_cache()
        cache.store("control room phone?", ROWS, corpus_sig="sigA")
        # Different text hashes to a different bucket -> orthogonal -> distance ~1.0.
        self.assertIsNone(cache.lookup("cryostat warm up steps", corpus_sig="sigA"))

    def test_corpus_sig_mismatch_is_miss(self):
        cache = make_cache()
        cache.store("control room phone?", ROWS, corpus_sig="sigA")
        self.assertIsNone(cache.lookup("control room phone?", corpus_sig="sigB"))

    def test_upsert_dedups_repeated_query(self):
        cache = make_cache()
        cache.store("control room phone?", ROWS, corpus_sig="sigA")
        cache.store("control room phone?", ROWS, corpus_sig="sigA")
        self.assertEqual(cache.count(), 1)

    def test_clear_empties_cache(self):
        cache = make_cache()
        cache.store("control room phone?", ROWS, corpus_sig="sigA")
        cache.clear()
        self.assertEqual(cache.count(), 0)
        self.assertIsNone(cache.lookup("control room phone?", corpus_sig="sigA"))

    def test_empty_cache_lookup_is_none(self):
        self.assertIsNone(make_cache().lookup("anything", corpus_sig="sigA"))

    def test_empty_query_and_rows_are_ignored(self):
        cache = make_cache()
        cache.store("   ", ROWS, corpus_sig="sigA")
        cache.store("real query", [], corpus_sig="sigA")
        self.assertEqual(cache.count(), 0)


if __name__ == "__main__":
    unittest.main()
