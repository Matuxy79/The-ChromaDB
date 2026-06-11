import unittest

from cls_backend.spectrum import (
    SPECTRUM,
    SUGGESTED_PROBLEMS,
    classify_query,
    decorate,
    glow_css,
)


class ClassifyQueryTests(unittest.TestCase):
    def test_radiation_is_safety(self):
        self.assertEqual(classify_query("is the shutter open, radiation exposure?"), "safety")

    def test_safety_beats_other_keywords(self):
        # "phone" would be contacts, but a beryllium mention must win.
        self.assertEqual(classify_query("phone for beryllium window swap"), "safety")

    def test_phone_is_contacts(self):
        self.assertEqual(classify_query("what is the phone number for the beamline"), "contacts")

    def test_alignment_is_procedure(self):
        # Note: a "warm-up" query is intentionally safety (cryogenics) — safety-first.
        self.assertEqual(classify_query("how do I align the beamline optics"), "procedure")

    def test_energy_range_is_specs(self):
        self.assertEqual(classify_query("undulator energy range and optics"), "specs")

    def test_junk_is_general(self):
        self.assertEqual(classify_query("tell me a story about lunch"), "general")

    def test_empty_is_general(self):
        self.assertEqual(classify_query("   "), "general")


class DecorateTests(unittest.TestCase):
    def test_wraps_phone_number(self):
        out = decorate("Call ext. 3832 for the hutch.", "contacts")
        self.assertIn("tok-phone", out)
        self.assertIn("3832", out)

    def test_wraps_acronym(self):
        out = decorate("The IVU beamline optics.", "specs")
        self.assertIn("tok-acr", out)

    def test_html_escaped(self):
        out = decorate("danger < 5 mm gap", "general")
        self.assertNotIn("< 5", out)
        self.assertIn("&lt;", out)

    def test_safety_words_only_underlined_on_safety_category(self):
        text = "Check the radiation interlock."
        self.assertIn("tok-safety", decorate(text, "safety"))
        self.assertNotIn("tok-safety", decorate(text, "general"))


class GlowTests(unittest.TestCase):
    def test_high_score_brighter_than_low(self):
        hue = SPECTRUM["contacts"]["hue"]
        hi = glow_css(hue, 0.9)
        lo = glow_css(hue, 0.1)
        self.assertNotEqual(hi, lo)
        # Higher score => larger blur radius.
        hi_blur = int(hi.split("px")[0].split()[-1])
        lo_blur = int(lo.split("px")[0].split()[-1])
        self.assertGreater(hi_blur, lo_blur)

    def test_score_clamped(self):
        hue = SPECTRUM["safety"]["hue"]
        self.assertTrue(glow_css(hue, 5.0))  # does not raise


class SuggestedProblemsTests(unittest.TestCase):
    def test_chips_use_known_categories(self):
        for category, text in SUGGESTED_PROBLEMS:
            self.assertIn(category, SPECTRUM)
            self.assertTrue(text.strip())


if __name__ == "__main__":
    unittest.main()
