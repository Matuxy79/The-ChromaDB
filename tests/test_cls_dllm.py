import unittest

from examples.cls_dllm import correction_user, needs_correction, parse_bullets, validate_correction
from examples.cls_spectrum import decorate


class GateStaysSparseTests(unittest.TestCase):
    def test_clean_answer_does_not_activate(self):
        clean = [
            "The Undulator beamline phone is ext. 3832. [Source: IVU manual.pdf, page 4]",
            "Contact Beatriz Moreno at 306-241-1999. [Source: IVU manual.pdf, page 2]",
        ]
        activate, reason = needs_correction(clean)
        self.assertFalse(activate)
        self.assertIsNone(reason)

    def test_empty_does_not_activate(self):
        self.assertEqual(needs_correction([]), (False, None))

    def test_citation_suffix_not_treated_as_artifact(self):
        # Trailing [Source: ...] must be stripped before judging; this is clean.
        activate, _ = needs_correction(["The gap is set correctly. [Source: m.pdf, page 1]"])
        self.assertFalse(activate)


class GateActivatesOnArtifactsTests(unittest.TestCase):
    def test_hyphenation_break_activates(self):
        activate, reason = needs_correction(["The undula- tor must be aligned before use."])
        self.assertTrue(activate)
        self.assertEqual(reason, "joined hyphenation breaks")

    def test_leftover_header_activates(self):
        activate, reason = needs_correction(["Section: Safety the shutter must be closed first."])
        self.assertTrue(activate)
        self.assertEqual(reason, "stripped leftover header text")

    def test_table_soup_activates(self):
        activate, reason = needs_correction(["E 1 2 a b 3 c value column row data"])
        self.assertTrue(activate)

    def test_truncated_fragment_activates(self):
        long_fragment = (
            "the beamline operates within a tightly controlled energy band that spans "
            "roughly seven to ninety kiloelectronvolts depending on the configuration chosen"
        )
        activate, reason = needs_correction([long_fragment])
        self.assertTrue(activate)
        self.assertEqual(reason, "completed a truncated fragment")

    def test_duplicate_sentences_activate(self):
        dup = ["The gap is 5 mm. [Source: m.pdf, page 1]", "The gap is 5 mm. [Source: m.pdf, page 9]"]
        activate, reason = needs_correction(dup)
        self.assertTrue(activate)
        self.assertEqual(reason, "removed duplicated sentence")


class CorrectionPromptTests(unittest.TestCase):
    def test_user_prompt_lists_bullets(self):
        out = correction_user(["a", "b"])
        self.assertIn("- a", out)
        self.assertIn("- b", out)

    def test_parse_bullets_strips_markers(self):
        self.assertEqual(parse_bullets("- one\n* two\n\n• three"), ["one", "two", "three"])


class ValidateCorrectionTests(unittest.TestCase):
    SRC = ["The phone is ext. 3832. [Source: m.pdf, page 4]"]

    def test_faithful_correction_passes(self):
        self.assertTrue(validate_correction(
            "The phone is ext. 3832. [Source: m.pdf, page 4]", self.SRC))

    def test_invented_number_rejected(self):
        self.assertFalse(validate_correction(
            "The phone is ext. 9999. [Source: m.pdf, page 4]", self.SRC))

    def test_mangled_citation_rejected(self):
        # The weak-model failure mode: citation rewritten into a section ref.
        self.assertFalse(validate_correction("The phone is ext. 3832. Section 1.1, p. 4", self.SRC))


class QueryHighlightTests(unittest.TestCase):
    def test_query_terms_become_hits(self):
        out = decorate("The undulator energy range.", "specs", query="energy range")
        self.assertIn("tok-hit", out)

    def test_acronym_beats_query_hit(self):
        # IVU is in the stoplist AND an acronym -> rendered as acronym, never as a plain hit.
        out = decorate("The IVU device.", "general", query="ivu device")
        self.assertIn("tok-acr", out)

    def test_stopwords_not_highlighted(self):
        out = decorate("what is the gap", "general", query="what is the")
        self.assertNotIn("tok-hit", out)


if __name__ == "__main__":
    unittest.main()
