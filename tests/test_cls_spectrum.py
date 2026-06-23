import unittest

from cls_backend.spectrum import (
    SPECTRUM,
    classify_query,
    decorate,
    glow_css,
)


class ClassifyQueryTests(unittest.TestCase):
    def test_phone_is_contacts(self):
        self.assertEqual(classify_query("what is the phone number for the beamline"), "contacts")

    def test_alignment_is_procedure(self):
        self.assertEqual(classify_query("how do I align the sample stage optics"), "procedure")

    def test_protocol_is_procedure(self):
        self.assertEqual(classify_query("sample preparation protocol steps"), "procedure")

    def test_energy_range_is_specs(self):
        self.assertEqual(classify_query("X-ray energy range and flux at the beamline"), "specs")

    def test_diffraction_is_specs(self):
        self.assertEqual(classify_query("what is the diffraction resolution limit"), "specs")

    def test_junk_is_general(self):
        self.assertEqual(classify_query("tell me a story about lunch"), "general")

    def test_empty_is_general(self):
        self.assertEqual(classify_query("   "), "general")


class DecorateTests(unittest.TestCase):
    def test_wraps_phone_number(self):
        out = decorate("Call ext. 3570 for the control room.", "contacts")
        self.assertIn("tok-phone", out)
        self.assertIn("3570", out)

    def test_wraps_saxs_acronym(self):
        out = decorate("The SAXS experiment at the beamline.", "specs")
        self.assertIn("tok-acr", out)

    def test_html_escaped(self):
        out = decorate("danger < 5 mm gap", "general")
        self.assertNotIn("< 5", out)
        self.assertIn("&lt;", out)


class GlowTests(unittest.TestCase):
    def test_high_score_brighter_than_low(self):
        hue = SPECTRUM["contacts"]["hue"]
        hi = glow_css(hue, 0.9)
        lo = glow_css(hue, 0.1)
        self.assertNotEqual(hi, lo)
        hi_blur = int(hi.split("px")[0].split()[-1])
        lo_blur = int(lo.split("px")[0].split()[-1])
        self.assertGreater(hi_blur, lo_blur)

    def test_score_clamped(self):
        hue = SPECTRUM["contacts"]["hue"]
        self.assertTrue(glow_css(hue, 5.0))

if __name__ == "__main__":
    unittest.main()
