import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "llm_measurement"))
from build_specificity_scoring_manifest import build_qa_units, qa_non_substantive_reasons
from build_ccts_proposed_analysis_sample import classify_block
from extract_ccts_ceo_qa_blocks import extract_event, speaker_key
from ccts_qa_episodes import procedure_kind


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

    def test_whole_passage_clarification_controls_remain_excluded(self):
        for answer in (
            "I'm sorry, I missed the question. Would you please say it again?",
            "Well, I -- I'm not sure I understand the question.",
            "I'm sorry, could you repeat that?",
            "Sorry. I didn't hear your question. Can you repeat it, please?",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(procedure_kind(answer), 'clarification_request')
                block = self.block(answer)
                self.assertEqual(block['ceo_answer'], '')
                self.assertEqual(classify_block(block, {}, 'david smith acme')['primary_block_candidate'], 0)
                self.assertIn('procedural_clarification_request', qa_non_substantive_reasons(answer))
                event = dict.fromkeys(('episode_rank', 'turnover_id', 'turnover_side',
                                      'expected_execid', 'start_date', 'calendar_quarter'), '1')
                with self.assertRaisesRegex(SystemExit, 'non-high block'):
                    build_qa_units([block], {'1': event})
                # A mistakenly high-tier upstream row is screened again by the manifest.
                units, exclusions = build_qa_units(
                    [dict(block, quality_tier='high', ceo_answer=answer)], {'1': event})
                self.assertEqual(units, [])
                self.assertEqual(len(exclusions), 1)

    def test_clarification_followed_by_business_or_disclosure_limit_is_retained(self):
        controls = (
            "I'm sorry, I missed the question. Could you say it again? Revenue rose 8%.",
            "I'm sorry, could you repeat that? We cannot disclose customer-level margins.",
            "I'm not sure I understand the question, but deliveries start in May.",
            "We missed the revenue target this quarter.",
            "I didn't catch the exact revenue figure. We expect growth next quarter.",
        )
        for answer in controls:
            with self.subTest(answer=answer):
                self.assertEqual(procedure_kind(answer), '')
                block = self.block(answer)
                self.assertEqual(block['ceo_answer'], answer)
                self.assertEqual(classify_block(block, {}, 'david smith acme')['primary_block_candidate'], 1)
                self.assertEqual(qa_non_substantive_reasons(answer), [])


if __name__ == "__main__":
    unittest.main()
