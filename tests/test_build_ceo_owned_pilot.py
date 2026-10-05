import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/llm_measurement"))
from build_ceo_owned_pilot import indexed, source_ledger
from extract_ccts_ceo_qa_blocks import speaker_key
from ceo_answer_policy import qa_word_coverage
from verify_ceo_owned_pilot import verify_coverage


class SourceLedgerTests(unittest.TestCase):
    def row(self, seq, text, speaker="A Person, Example - CEO", section="Q&A"):
        return {"event_id": "e", "sequence_id": str(seq), "text_contents": text,
                "text_name": speaker, "analysis_text_type": section}

    def block(self, seq, identity="b", words=2, status="proposed_primary"):
        block = {"block_id": identity, "ceo_answer_sequence_ids": json.dumps([str(seq)])}
        decision = {"inclusion_status": status, "inclusion_reason": "source_rule",
                    "original_ceo_target_word_count": words}
        return block, decision

    def test_unknown_unassigned_stays_visible(self):
        source = [self.row(1, "Sales increased."), self.row(2, "Something else.")]
        block, decision = self.block(1)
        result = source_ledger("e", source, [block], {"b": decision}, speaker_key(source[0]), {})
        self.assertEqual([r["category"] for r in result], ["assigned_answer", "unresolved"])
        self.assertEqual(sum(r["word_count"] for r in result), 4)

    def test_procedural_block_not_answer_denominator(self):
        source = [self.row(1, "Thank you.")]
        block, decision = self.block(1, status="exclude_procedural_or_empty")
        result = source_ledger("e", source, [block], {"b": decision}, speaker_key(source[0]), {})
        self.assertEqual(result[0]["category"], "procedure_only")
        self.assertEqual(result[0]["block_id"], "")
        self.assertEqual(result[0]["source_block_id"], "b")

    def test_closing_requires_exact_prior_audit_context(self):
        source = [self.row(0, "There are no more questions.", "Operator"), self.row(1, "Business remains strong.")]
        audit = {("e", "1"): {"category": "post_session_closing", "ceo_text": source[1]["text_contents"],
                             "words": "3", "previous_name": "Operator", "previous_text": source[0]["text_contents"]}}
        result = source_ledger("e", source, [], {}, speaker_key(source[1]), audit)
        self.assertEqual(result[0]["category"], "post_session_closing")
        audit[("e", "1")]["previous_text"] = "Invented evidence"
        with self.assertRaises(ValueError):
            source_ledger("e", source, [], {}, speaker_key(source[1]), audit)

    def test_audited_procedure_must_reproduce_pattern(self):
        source = [self.row(1, "Good morning.")]
        audit = {("e", "1"): {"category": "recognized_procedure", "ceo_text": "Good morning.",
                               "words": "2", "procedure_kind": "greeting_only"}}
        result = source_ledger("e", source, [], {}, speaker_key(source[0]), audit)
        self.assertEqual(result[0]["category"], "procedure_only")
        audit[("e", "1")]["procedure_kind"] = "closing_only"
        with self.assertRaises(ValueError):
            source_ledger("e", source, [], {}, speaker_key(source[0]), audit)

    def test_ownership_and_word_discrepancies_fail_closed(self):
        source = [self.row(1, "Sales increased.")]
        block, decision = self.block(1, words=3)
        with self.assertRaises(ValueError):
            source_ledger("e", source, [block], {"b": decision}, speaker_key(source[0]), {})
        with self.assertRaises(ValueError):
            source_ledger("e", source, [block, block], {"b": decision}, speaker_key(source[0]), {})
        block["ceo_answer_sequence_ids"] = '["2"]'
        with self.assertRaises(ValueError):
            source_ledger("e", source, [block], {"b": decision}, speaker_key(source[0]), {})

    def test_duplicate_csv_identity_rejected(self):
        with self.assertRaises(ValueError):
            indexed([{"id": "a"}, {"id": "a"}], "id")

    def test_separate_coverage_readback_and_pending_status(self):
        ledger = [{"event_id": "e", "sequence_id": "1", "category": "assigned_answer", "word_count": "2", "block_id": "b"},
                  {"event_id": "e", "sequence_id": "2", "category": "unresolved", "word_count": "3", "block_id": ""}]
        decisions = {"b": {"event_id": "e", "primary_block_candidate": "1", "management_context_block_candidate": "1",
                           "inclusion_status": "proposed_primary", "inclusion_reason": "high_tier_source_screen_pass"}}
        typed = [{**r, "word_count": int(r["word_count"])} for r in ledger]
        metrics = {**qa_word_coverage(typed, ["b"]), "event_id": "e", "lane": "primary_ceo_owned",
                   "eligible_blocks": 1, "review_candidate_answer_words": 0}
        result = verify_coverage(ledger, decisions, metrics)
        self.assertEqual(result["raw_qa_word_share"], 2 / 5)
        self.assertEqual(result["answer_inclusion_coverage"], 1)
        self.assertEqual(result["answer_inclusion_if_all_unresolved_are_answers"], 2 / 5)
        with self.assertRaises(ValueError):
            verify_coverage(ledger, decisions, {**metrics, "scoring_coverage": 0})
        with self.assertRaises(ValueError):
            verify_coverage(ledger, decisions, {**metrics, "answer_inclusion_coverage": 2 / 5})


if __name__ == "__main__":
    unittest.main()
