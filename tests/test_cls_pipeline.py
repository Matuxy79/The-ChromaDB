import hashlib
import unittest
import uuid

import chromadb

from cls_backend.cag_cache import SemanticEvidenceCache
from cls_backend.pipeline import (
    EMBED_DIM,
    PARTIAL_MATCH_MIN_LEN,
    EmbeddingUnavailableError,
    build_extractive_answer,
    clean_sentence,
    clean_sentences,
    expand_query_terms,
    instant_answer,
    lexical_retrieve,
    partial_overlap,
    retrieve,
    warm_lexical_index,
)


class TestEmbedder:
    """Deterministic 768-dim embedder for unit tests (no Ollama required)."""

    def embed(self, texts):
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * EMBED_DIM
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            vector[i % EMBED_DIM] += byte / 255.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector
        return [v / norm for v in vector]


class OfflineEmbedder:
    def embed(self, texts):
        raise EmbeddingUnavailableError("offline")


def fresh_collection():
    client = chromadb.EphemeralClient()
    return client.create_collection(f"store_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})


def make_cache():
    client = chromadb.EphemeralClient()
    col = client.create_collection(f"cache_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    return SemanticEvidenceCache(col, TestEmbedder())


def doc(name, section, page, body):
    return f"Source: {name}\nSection: {section}\nPage: {page}\n\n{body}"


class CleanTextTests(unittest.TestCase):
    def test_rejoins_hyphenation_and_keeps_citation(self):
        out = clean_sentence("The synchro- tron is aligned. [Source: m.pdf, page 4]")
        self.assertIn("synchrotron", out)
        self.assertNotIn("synchro- tron", out)
        self.assertIn("[Source: m.pdf, page 4]", out)

    def test_collapses_space_before_punctuation(self):
        self.assertEqual(clean_sentence("the gap is set ."), "the gap is set.")

    def test_dedupes_case_insensitive(self):
        sents = ["The gap is 5 mm. [Source: m.pdf, page 1]", "the gap is 5 mm. [Source: m.pdf, page 9]"]
        self.assertEqual(len(clean_sentences(sents)), 1)


class RetrieveAndInstantAnswerTests(unittest.TestCase):
    def setUp(self):
        self.embedder = TestEmbedder()
        self.collection = fresh_collection()
        docs = [
            doc("manual.pdf", "Contacts", 4, "The CLS Control Room phone is ext. 3570 for operators."),
            doc("manual.pdf", "Specifications", 7, "The synchrotron X-ray energy range spans several kiloelectronvolts."),
        ]
        metas = [
            {"source": "manual.pdf", "source_hash": "h1", "page": 4, "section": "Contacts", "chunk_index": 0},
            {"source": "manual.pdf", "source_hash": "h1", "page": 7, "section": "Specifications", "chunk_index": 1},
        ]
        self.collection.add(
            ids=["h1:0", "h1:1"],
            documents=docs,
            metadatas=metas,
            embeddings=self.embedder.embed(docs),
        )

    def test_retrieve_returns_scored_rows(self):
        rows = retrieve(self.collection, self.embedder, "control room phone number", n_results=2)
        self.assertTrue(rows)
        self.assertIn("score", rows[0])

    def test_retrieve_empty_collection(self):
        self.assertEqual(retrieve(fresh_collection(), self.embedder, "anything"), [])

    def test_lexical_retrieve_reuses_warmed_snapshot_and_respects_filters(self):
        self.assertEqual(warm_lexical_index(self.collection), 2)

        rows = lexical_retrieve(
            self.collection,
            "control room",
            n_results=2,
            metadata_filter={"section": "Contacts"},
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("3570", rows[0]["document"])

    def test_instant_answer_uses_lexical_fallback_when_embedder_is_offline(self):
        result = instant_answer(
            "control room phone number",
            collection=self.collection,
            cache=make_cache(),
            embedder=OfflineEmbedder(),
            top_k=2,
            cache_enabled=False,
        )

        self.assertEqual(result["retrieval_mode"], "lexical_fallback")
        self.assertTrue(result["rows"])
        self.assertIn("3570", result["rows"][0]["document"])
        self.assertTrue(result["answer"])

    def test_instant_answer_keyword_only_skips_embedding(self):
        result = instant_answer(
            "control room phone number",
            collection=self.collection,
            cache=make_cache(),
            embedder=OfflineEmbedder(),
            top_k=2,
            cache_enabled=True,
            keyword_only=True,
        )

        self.assertEqual(result["retrieval_mode"], "keyword_only")
        self.assertFalse(result["from_cache"])
        self.assertTrue(result["rows"])
        self.assertIn("3570", result["rows"][0]["document"])

    def test_instant_answer_fresh_then_cached(self):
        cache = make_cache()
        q = "What is the CLS control room phone number?"
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

    def test_extractive_answer_keeps_adjacent_segment_page_citation(self):
        rows = [
            {
                "document": (
                    doc("manual.pdf", "Contacts", 4, "The CLS Control Room phone is ext. 3570.")
                    + "\n\n--- adjacent segment ---\n\n"
                    + doc("manual.pdf", "Specifications", 7, "The synchrotron X-ray energy range spans several kiloelectronvolts.")
                ),
                "metadata": {
                    "source": "manual.pdf",
                    "source_hash": "h1",
                    "page": 4,
                    "section": "Contacts",
                    "chunk_index": 0,
                },
                "score": 0.9,
            }
        ]

        answer = build_extractive_answer("X-ray energy range", rows)

        self.assertTrue(answer)
        self.assertIn("[Source: manual.pdf, page 7]", answer[0])
        self.assertNotIn("[Source: manual.pdf, page 4]", answer[0])


class PartialKeywordMatchTests(unittest.TestCase):
    """Keyword retrieval matches partial words / arbitrary fragments, not just whole tokens."""

    def setUp(self):
        self.embedder = TestEmbedder()
        self.collection = fresh_collection()
        docs = [
            doc("gene_table.csv", "Page 1", 1,
                "Header: gene | organism | chromosome\nRow: SOD7 | H. sapiens | 12"),
            doc("notes.txt", "Mitochondria", 3,
                "The mitochondrial membrane regulates osmoregulation near the chromosome."),
            doc("quantum.md", "QM", 5,
                "The wavefunction collapses into an eigenstate upon measurement."),
        ]
        metas = [
            {"source": "gene_table.csv", "source_hash": "b1", "page": 1, "domain": "biology", "chunk_index": 0},
            {"source": "notes.txt", "source_hash": "b2", "page": 3, "domain": "biology", "chunk_index": 0},
            {"source": "quantum.md", "source_hash": "p1", "page": 5, "domain": "physics", "chunk_index": 0},
        ]
        self.collection.add(
            ids=["b1:0", "b2:0", "p1:0"],
            documents=docs,
            metadatas=metas,
            embeddings=self.embedder.embed(docs),
        )

    def _sources(self, rows):
        return {row["metadata"].get("source") for row in rows}

    def test_partial_fragment_retrieves_full_word(self):
        # "sapi" is not a whole token in any document, yet it must find "sapiens".
        rows = lexical_retrieve(self.collection, "sapi", n_results=8)
        self.assertEqual(self._sources(rows), {"gene_table.csv"})

    def test_partial_fragment_surfaces_in_answer(self):
        rows = lexical_retrieve(self.collection, "chro", n_results=8)
        answer = build_extractive_answer("chro", rows)
        self.assertTrue(answer)
        self.assertTrue(any("chromosome" in sentence.lower() for sentence in answer))

    def test_partial_match_respects_scope_filter(self):
        # "sapien" lives only in the biology doc; filtering to physics yields nothing.
        self.assertEqual(
            lexical_retrieve(self.collection, "sapien", n_results=8, metadata_filter={"domain": "physics"}),
            [],
        )
        rows = lexical_retrieve(self.collection, "sapien", n_results=8, metadata_filter={"domain": "biology"})
        self.assertEqual(self._sources(rows), {"gene_table.csv"})

    def test_gibberish_returns_nothing(self):
        self.assertEqual(lexical_retrieve(self.collection, "zxqvw", n_results=8), [])

    def test_short_fragment_falls_back_to_exact(self):
        # Below the partial-match floor, fragments must not match by substring.
        self.assertLess(2, PARTIAL_MATCH_MIN_LEN + 1)
        vocab = frozenset({"chromosome", "eigenstate"})
        expanded = expand_query_terms({"ch"}, vocab)
        self.assertEqual(expanded["ch"], set())  # "ch" too short -> no substring expansion

    def test_partial_overlap_counts_fragments(self):
        self.assertEqual(partial_overlap({"sapi"}, {"sapiens", "organism"}), 1)
        self.assertEqual(partial_overlap({"chro", "quan"}, {"chromosome", "quantum", "biology"}), 2)

    def test_no_reverse_direction_match(self):
        # A long, specific query must NOT match short indexed fragments it merely contains
        # (the over-reach that made "metabolism" pull in unrelated literature chunks).
        self.assertEqual(partial_overlap({"metabolism"}, {"meta", "bol", "ism", "tab"}), 0)
        self.assertEqual(partial_overlap({"metabolism"}, {"metabolism", "metabolisms"}), 2)
        expanded = expand_query_terms({"metabolism"}, frozenset({"meta", "metabolism", "metabolisms"}))
        self.assertEqual(expanded["metabolism"], {"metabolism", "metabolisms"})

    def test_full_word_query_does_not_match_unrelated_docs(self):
        # "metabolism" appears in none of the setUp docs -> nothing comes back.
        self.assertEqual(lexical_retrieve(self.collection, "metabolism", n_results=8), [])


if __name__ == "__main__":
    unittest.main()
