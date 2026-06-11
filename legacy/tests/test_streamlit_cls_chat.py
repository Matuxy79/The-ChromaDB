import unittest

from examples.streamlit_cls_chat import clean_llm_answer, format_evidence_packet


class StreamlitChatFormattingTests(unittest.TestCase):
    def test_wiggler_packet_ignores_neighboring_undulator_sentence(self):
        hits = [
            {
                "text": (
                    "The optical components of the undulator beamline are arranged to "
                    "prevent collisions with the wiggler beamlines. "
                    "The wiggler beamline is divided into two branches: The Low Energy "
                    "Wiggler (LEW), which operates in the energy range of 7 to 23 keV, "
                    "and the High Energy Wiggler (HEW), spanning from 20 to 90 keV."
                ),
                "metadata": {
                    "source_url": "IVU beamline manual.pdf",
                    "colour_code": "purple",
                    "domain": "beamline",
                },
            },
            {
                "text": "The camera login is available from the beamline Windows desktop.",
                "metadata": {"source_url": "IVU beamline manual.pdf"},
            },
        ]

        packet = format_evidence_packet("wiggler beamline", hits)

        self.assertIn("Low Energy Wiggler", packet)
        self.assertIn("High Energy Wiggler", packet)
        self.assertNotIn("undulator beamline are arranged", packet)
        self.assertNotIn("prevent collisions", packet)
        self.assertNotIn("camera login", packet)

    def test_clean_llm_answer_removes_instruction_echo_and_clamps_citations(self):
        evidence_packet = "[1] Text:\nThe wiggler beamline has LEW and HEW branches."
        answer = (
            "1. The Low Energy Wiggler operates from 7 to 23 keV\n"
            "2. The High Energy Wiggler spans 20 to 90 keV [2]\n"
            "Available citations: [1]\n"
        )

        cleaned = clean_llm_answer(answer, evidence_packet)

        self.assertIn("1. The Low Energy Wiggler operates from 7 to 23 keV [1]", cleaned)
        self.assertIn("2. The High Energy Wiggler spans 20 to 90 keV [1]", cleaned)
        self.assertNotIn("Available citations", cleaned)


if __name__ == "__main__":
    unittest.main()
