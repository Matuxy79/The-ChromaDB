import unittest

from cls_config import KEYWORD_ONLY_RETRIEVAL, RETRIEVAL_ONLY
from cls_service import call_dllm_api, dllm_status, parrot_answer, parrot_status


class RetrievalOnlyModeTests(unittest.TestCase):
    @unittest.skipUnless(RETRIEVAL_ONLY, "retrieval-only mode is not enabled")
    def test_dllm_status_reports_disabled_without_network(self):
        status = dllm_status(timeout=0.01)

        self.assertFalse(status["online"])
        self.assertTrue(status["disabled"])
        self.assertTrue(status["retrieval_only"])
        self.assertIn("Retrieval-only mode", status["detail"])

    @unittest.skipUnless(RETRIEVAL_ONLY, "retrieval-only mode is not enabled")
    def test_dllm_calls_are_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "Retrieval-only mode"):
            call_dllm_api([{"role": "user", "content": "hello"}], timeout=0.01)

    @unittest.skipUnless(RETRIEVAL_ONLY, "retrieval-only mode is not enabled")
    def test_parrot_is_blocked(self):
        status = parrot_status(timeout=0.01)

        self.assertFalse(status["online"])
        self.assertTrue(status["disabled"])
        self.assertIsNone(parrot_answer(["The phone is ext. 3570."]))

    def test_keyword_only_default_is_documented_by_config(self):
        self.assertIsInstance(KEYWORD_ONLY_RETRIEVAL, bool)


if __name__ == "__main__":
    unittest.main()
