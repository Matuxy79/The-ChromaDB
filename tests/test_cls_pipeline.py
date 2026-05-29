import unittest
import uuid

import chromadb

from examples.cls_cag_cache import SemanticEvidenceCache
from examples.cls_pipeline import (
    HashEmbedder,
    clean_sentence,
    clean_sentences,
    instant_answer,
    retrieve,
)


def fresh_collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(f"store_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


def make_cache():
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"cache_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    return SemanticEvidenceCache(col, HashEmbedder())


def doc(name, section, page, body):
    return f"Source: {name}\nSection: {section}\nPage: {page}\n\n{body}"


class CleanTextTests(unittest.TestCase):
    def test_rejoins_hyphenation_and_keeps_citation(self):
        out = clean_sentence("The undula- tor is aligned. [Source: m.pdf, page 4]")
        self.assertIn("undulator", out)
        self.assertNotIn("undula- tor", out)
        self.assertIn("[Source: m.pdf, page 4]", out)

    def test_collapses_space_before_punctuation(self):
        self.assertEqual(clean_sentence("the gap is set ."), "the gap is set.")

    def test_dedupes_case_insensitive(self):
        sents = ["The gap is 5 mm. [Source: m.pdf, page 1]", "the gap is 5 mm. [Source: m.pdf, page 9]"]
        self.assertEqual(len(clean_sentences(sents)), 1)


class RetrieveAndInstantAnswerTests(unittest.TestCase):
    def setUp(self):
        self.embedder = HashEmbedder()
        self.collection = fresh_collection()
        docs = [
            doc("IVU.pdf", "Contacts", 4, "The Undulator beamline phone is ext. 3832 for operators."),
            doc("IVU.pdf", "Optics", 7, "The undulator energy range spans several kiloelectronvolts."),
        ]
        metas = [
            {"source": "IVU.pdf", "source_hash": "h1", "page": 4, "section": "Contacts", "chunk_index": 0},
            {"source": "IVU.pdf", "source_hash": "h1", "page": 7, "section": "Optics", "chunk_index": 1},
        ]
        self.collection.add(
            ids=["h1:0", "h1:1"],
            documents=docs,
            metadatas=metas,
            embeddings=self.embedder.embed(docs),
        )

    def test_retrieve_returns_scored_rows(self):
        rows = retrieve(self.collection, self.embedder, "undulator phone number", n_results=2)
        self.assertTrue(rows)
        self.assertIn("score", rows[0])

    def test_retrieve_empty_collection(self):
        self.assertEqual(retrieve(fresh_collection(), self.embedder, "anything"), [])

    def test_instant_answer_fresh_then_cached(self):
        cache = make_cache()
        q = "What is the undulator beamline phone number?"
        first = instant_answer(q, collection=self.collection, cache=cache, embedder=self.embedder, top_k=4)
        self.assertTrue(first["rows"])
        self.assertFalse(first["from_cache"])
        self.assertEqual(cache.count(), 1)

        second = instant_answer(q, collection=self.collection, cache=cache, embedder=self.embedder, top_k=4)
        self.assertTrue(second["from_cache"])
        self.assertGreaterEqual(second["similarity"], 0.99)

    def test_instant_answer_cache_disabled_never_stores(self):
        cache = make_cache()
        instant_answer("anything", collection=self.collection, cache=cache,
                       embedder=self.embedder, top_k=4, cache_enabled=False)
        self.assertEqual(cache.count(), 0)


if __name__ == "__main__":
    unittest.main()
