import unittest

from cls_backend.query_repair import repair_query


class QueryRepairTests(unittest.TestCase):
    def test_returns_required_keys(self):
        result = repair_query("sample mounting procedure")
        self.assertIn("original", result)
        self.assertIn("search", result)
        self.assertIn("changed", result)
        self.assertIn("notes", result)

    def test_passthrough_clean_query(self):
        q = "sample mounting alignment procedure"
        result = repair_query(q)
        self.assertEqual(result["search"], q)
        self.assertFalse(result["changed"])

    def test_strips_how_do_i(self):
        result = repair_query("how do I mount my sample")
        self.assertNotIn("how do i", result["search"].lower())
        self.assertIn("mount", result["search"].lower())

    def test_strips_can_you_tell_me(self):
        result = repair_query("can you tell me about the X-ray energy range")
        self.assertIn("X-ray energy range", result["search"])
        self.assertNotIn("can you tell me", result["search"].lower())

    def test_strips_i_was_wondering(self):
        result = repair_query("I was wondering about sample preparation protocol")
        self.assertIn("sample preparation protocol", result["search"])

    def test_strips_what_is_the(self):
        result = repair_query("what is the sample mounting procedure")
        self.assertNotIn("what is", result["search"].lower())
        self.assertIn("sample mounting", result["search"].lower())

    def test_strips_trailing_question_mark(self):
        result = repair_query("what are the radiation safety interlocks?")
        self.assertNotIn("?", result["search"])

    def test_empty_query_returns_empty(self):
        result = repair_query("")
        self.assertEqual(result["search"], "")

    def test_pure_filler_falls_back_to_original(self):
        # Normalization should not produce an empty string — fall back to original.
        result = repair_query("how do i")
        self.assertGreater(len(result["search"].strip()), 0)

    def test_original_always_preserved(self):
        q = "What is the SAXS beamline energy range?"
        result = repair_query(q)
        self.assertEqual(result["original"], q)

    def test_strips_leading_whitespace(self):
        result = repair_query("  X-ray diffraction resolution  ")
        self.assertEqual(result["search"], "X-ray diffraction resolution")


if __name__ == "__main__":
    unittest.main()
