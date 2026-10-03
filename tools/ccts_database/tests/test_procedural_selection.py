import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llm_measurement"))
from build_specificity_scoring_manifest import build_qa_units, qa_non_substantive_reasons
from build_ccts_proposed_analysis_sample import classify_block
from extract_ccts_ceo_qa_blocks import extract_event, speaker_key


class ProceduralSelectionTests(unittest.TestCase):
    def block(self, answer):
        rows = [
            {"event_id": "1", "sequence_id": "1", "analysis_text_type": "PRE",
             "text_name": "David Smith, Acme - CEO", "text_contents": "Sales grew this quarter."},
            {"event_id": "1", "sequence_id": "2", "analysis_text_type": "Q&A",
             "text_name": "Ann Jones, Bank - Analyst",
             "text_contents": "Can you explain revenue growth and the delivery outlook this quarter?"},
            {"event_id": "1", "sequence_id": "3", "analysis_text_type": "Q&A",
             "text_name": "David Smith, Acme - CEO", "text_contents": answer},
        ]
        _, blocks = extract_event(rows, validated_anchor_key=speaker_key(rows[0]))
        return blocks[0]

    def test_business_after_procedural_phrase_survives_both_selection_paths(self):
        for answer in (
            "Thank you all for joining the call. Revenue grew 12% to $500 million and delivery is in June.",
            "Thank you all for joining us. Revenue grew 12% this quarter.",
            "Could you repeat the question? Revenue grew 12% to $500 million this quarter.",
            "If I understand your question, you mean our outlook. Revenue grew 12%. Is that correct?",
            "Thank you (laughter). Revenue grew 12% this quarter.",
        ):
            with self.subTest(answer=answer):
                block = self.block(answer)
                self.assertEqual(block["ceo_answer"], answer)
                self.assertEqual(block["quality_tier"], "high")
                self.assertEqual(classify_block(block, {}, "david smith acme")["primary_block_candidate"], 1)
                event = dict.fromkeys(("episode_rank", "turnover_id", "turnover_side",
                                       "expected_execid", "start_date", "calendar_quarter"), "1")
                units, exclusions = build_qa_units([block], {"1": event})
                self.assertEqual(len(units), 1)
                self.assertEqual(exclusions, [])

    def test_procedural_only_controls_and_source_defects_still_excluded(self):
        for answer in ("Could you repeat the question?", "Thank you all for joining the call."):
            with self.subTest(answer=answer):
                block = self.block(answer)
                self.assertEqual(block["ceo_answer"], "")
                self.assertEqual(classify_block(block, {}, "david smith acme")["primary_block_candidate"], 0)
                self.assertTrue(qa_non_substantive_reasons(answer))
        for answer in ("Revenue increased (inaudible) this quarter.", "Revenue increased --"):
            self.assertEqual(self.block(answer)["quality_tier"], "review")
            self.assertTrue(qa_non_substantive_reasons(answer))
        self.assertEqual(qa_non_substantive_reasons("We cannot provide a forecast."), [])


if __name__ == "__main__":
    unittest.main()
