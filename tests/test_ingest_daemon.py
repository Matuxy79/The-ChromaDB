import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import chromadb

from cls_service import file_signature, ingest_path


class IngestDaemonParityTests(unittest.TestCase):
    def setUp(self):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            f"ingest_test_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )

    def test_ingest_path_preserves_extra_metadata(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(
                "The IVU beamline phone extension is 3832.\n\n"
                "Contact Beatriz Moreno for operational support.\n"
            )
            temp_path = Path(handle.name)

        try:
            source_hash = file_signature(temp_path)
            extra = {"colour_code": "green", "domain": "beamline", "trust_level": "official_cls"}

            with patch("cls_service.get_collection", return_value=self.collection):
                chunks, status = ingest_path(
                    temp_path, source_hash, extra_metadata=extra
                )

            self.assertEqual(status, "indexed")
            self.assertGreater(chunks, 0)

            result = self.collection.get(include=["metadatas"])
            metas = result.get("metadatas", []) or []
            self.assertTrue(metas)

            for meta in metas:
                self.assertEqual(meta.get("colour_code"), "green")
                self.assertEqual(meta.get("domain"), "beamline")
                self.assertEqual(meta.get("trust_level"), "official_cls")
                self.assertIn("source", meta)
                self.assertIn("source_hash", meta)
                self.assertIn("page", meta)
                self.assertIn("section", meta)
                self.assertIn("chunk_index", meta)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_ingest_path_skips_already_indexed(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("Sample content for duplicate detection.\n")
            temp_path = Path(handle.name)

        try:
            source_hash = file_signature(temp_path)

            with patch("cls_service.get_collection", return_value=self.collection):
                first_chunks, first_status = ingest_path(temp_path, source_hash)
                self.assertEqual(first_status, "indexed")
                self.assertGreater(first_chunks, 0)

                second_chunks, second_status = ingest_path(temp_path, source_hash)
                self.assertEqual(second_status, "already indexed")
                self.assertEqual(second_chunks, 0)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
