import unittest

from cls_backend.query_repair import repair_query


class QueryRepairTests(unittest.TestCase):
    def test_ivw_is_not_rewritten_to_ivu(self):
        repaired = repair_query("tell me about the ivw beamline")

        self.assertIn("IVW wiggler beamline", repaired["search"])
        self.assertNotIn("Normalized spaced IVW acronym.", repaired["notes"])
        self.assertNotIn("Corrected IVW to IVU.", repaired["notes"])
        self.assertNotIn("IVU beamline BXDS", repaired["search"])

    def test_ivu_is_not_treated_as_spaced_acronym(self):
        repaired = repair_query("tell me about the ivu beamline")

        self.assertIn("IVU beamline BXDS", repaired["search"])
        self.assertNotIn("Normalized spaced IVU acronym.", repaired["notes"])

    def test_spaced_ivw_is_normalized_to_ivw(self):
        repaired = repair_query("tell me about i v w")

        self.assertIn("IVW", repaired["search"])
        self.assertIn("Normalized spaced IVW acronym.", repaired["notes"])

    def test_spaced_ivu_is_normalized_to_ivu(self):
        repaired = repair_query("tell me about i v u")

        self.assertIn("IVU", repaired["search"])
        self.assertIn("Normalized spaced IVU acronym.", repaired["notes"])

    def test_bxd_is_corrected_to_bxds(self):
        repaired = repair_query("bxd optics")

        self.assertIn("BXDS", repaired["search"])
        self.assertIn("Corrected BXD to BXDS.", repaired["notes"])

    def test_wiggler_expands_toward_ivw_not_ivu(self):
        repaired = repair_query("wiggler beamline")

        self.assertIn("IVW low energy wiggler", repaired["search"])
        self.assertNotIn("IVU beamline BXDS", repaired["search"])


if __name__ == "__main__":
    unittest.main()
