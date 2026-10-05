import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/llm_measurement"))
from ceo_answer_policy import answer_policy, qa_word_coverage


class CEOAnswerPolicyTests(unittest.TestCase):
    def decision(self, strict, broad, status, reason=""):
        return dict(primary_block_candidate=str(strict), management_context_block_candidate=str(broad),
                    inclusion_status=status, inclusion_reason=reason)

    def test_strict_and_identified_management_are_included(self):
        self.assertEqual(answer_policy(self.decision(1, 1, "proposed_primary")),
                         dict(policy_primary=1, strict_robustness=1, added_identified_management=0))
        self.assertEqual(answer_policy(self.decision(0, 1, "sensitivity_only", "identified_same_issuer_management_context")),
                         dict(policy_primary=1, strict_robustness=0, added_identified_management=1))

    def test_review_and_procedure_stay_excluded(self):
        for status in ("source_or_boundary_review", "exclude_procedural_or_empty"):
            self.assertEqual(answer_policy(self.decision(0, 0, status))["policy_primary"], 0)

    def test_contradictory_and_unknown_decisions_rejected(self):
        for decision in (self.decision(1, 0, "proposed_primary"), self.decision(0, 1, "source_or_boundary_review"),
                         self.decision(0, 1, "sensitivity_only", "unknown_manager"), self.decision(0, 0, "sensitivity_only")):
            with self.assertRaises(ValueError):
                answer_policy(decision)
        for value in (True, 1.0, "true", " 1", 2):
            with self.assertRaises(ValueError):
                answer_policy(dict(primary_block_candidate=value, management_context_block_candidate=1,
                                   inclusion_status="proposed_primary"))

    def ledger(self):
        return [dict(event_id="e", sequence_id=str(i), category=c, word_count=w, block_id=b)
                for i, (c, w, b) in enumerate([
                    ("assigned_answer", 100, "a"), ("assigned_answer", 50, "b"),
                    ("post_session_closing", 40, ""), ("procedure_only", 10, ""), ("unresolved", 20, "")])]

    def test_distinct_denominators_and_pending_scoring(self):
        r = qa_word_coverage(self.ledger(), ["a"])
        self.assertEqual(r["raw_ceo_qa_words"], 220)
        self.assertEqual(r["raw_qa_word_share"], 100 / 220)
        self.assertEqual(r["answer_inclusion_coverage"], 100 / 150)
        self.assertEqual(r["answer_inclusion_if_all_unresolved_are_answers"], 100 / 170)
        self.assertIsNone(r["scoring_coverage"])
        self.assertIsNone(r["validly_scored_words"])

    def test_valid_unscorable_and_missing_scores_accounted(self):
        r = qa_word_coverage(self.ledger(), ["a", "b"], {"a": "valid", "b": "unscorable"})
        self.assertEqual(r["scoring_coverage"], 100 / 150)
        self.assertEqual(r["score_status_words"]["unscorable"], 50)
        r = qa_word_coverage(self.ledger(), ["a", "b"], {"a": "missing", "b": "invalid"})
        self.assertEqual(r["scoring_coverage"], 0)

    def test_empty_denominators_are_not_zero(self):
        r = qa_word_coverage([], [], {})
        self.assertIsNone(r["scoring_coverage"])
        self.assertIsNone(r["answer_inclusion_coverage"])

    def test_duplicate_identity_words_categories_and_blocks_rejected(self):
        cases = [self.ledger() + [self.ledger()[0]],
                 [dict(event_id="e", sequence_id="0", category="unresolved", word_count=True)],
                 [dict(event_id="e", sequence_id="0", category="unresolved", word_count=-1)],
                 [dict(event_id="e", sequence_id="0", category="unknown", word_count=1)],
                 [dict(event_id="e", sequence_id="0", category="assigned_answer", word_count=1)],
                 [dict(event_id="e", sequence_id="0", category="procedure_only", word_count=1, block_id="a")]]
        for ledger in cases:
            with self.assertRaises(ValueError):
                qa_word_coverage(ledger, [])
        with self.assertRaises(ValueError):
            qa_word_coverage(self.ledger(), ["absent"])
        with self.assertRaises(ValueError):
            qa_word_coverage(self.ledger(), ["a", "a"])
        with self.assertRaises(ValueError):
            qa_word_coverage(self.ledger(), ["a"], {})
        with self.assertRaises(ValueError):
            qa_word_coverage(self.ledger(), ["a"], {"a": "guessed_valid"})


if __name__ == "__main__":
    unittest.main()
