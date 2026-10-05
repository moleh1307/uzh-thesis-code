import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ccts_qa_episodes import analyst_question_candidate
from extract_ccts_ceo_qa_blocks import extract_event, is_analyst, speaker_key
from extract_execucomp_confirmed_ceo_qa_blocks import validate_blocks
from build_ccts_proposed_analysis_sample import classify_block


class QaRoleAndRequestRecognitionTests(unittest.TestCase):
    def row(self, sequence, section, name, text):
        return {"event_id": "1", "sequence_id": str(sequence),
                "analysis_text_type": section, "text_name": name, "text_contents": text}

    def research_label(self, title):
        return "Jordan Example, Example Securities, Research Division - " + title + " [12]"

    def event(self, title="Associate", question=None, management=False):
        ceo = "David Smith, Acme Inc - CEO"
        rows = [self.row(1, "PRE", ceo, "We expect deliveries in June."),
                self.row(2, "Q&A", self.research_label(title), question or
                         "Can you explain the delivery timetable for the next quarter?")]
        if management:
            rows.append(self.row(3, "Q&A", "Alex Brown, Acme Inc - CFO",
                                 "The cost estimate is unchanged."))
        rows.append(self.row(4, "Q&A", ceo,
                             "Signed orders cover 80 units and we expect deliveries in June."))
        return rows

    def extract_and_classify(self, rows):
        anchor = speaker_key(rows[0])
        _, blocks = extract_event(rows, validated_anchor_key=anchor)
        validate_blocks("1", rows, blocks, anchor)
        source = {r["sequence_id"]: r for r in rows}
        for block in blocks:
            block["validated_ceo_speaker"] = rows[0]["text_name"]
            block["validated_ceo_speaker_json"] = json.dumps([rows[0]["text_name"]])
        return blocks, [classify_block(b, source, anchor) for b in blocks]

    def test_research_division_role_variants(self):
        for title in ("Associate", "Research Associate", "Senior Research Associate",
                      "Co-Head of Research", "Head of Telecommunication Services Equity Research",
                      "Head of Technology Equity Research", "Director of Healthcare Research",
                      "MD", "Managing Director", "MD & Senior Research Analyst"):
            with self.subTest(title=title):
                self.assertTrue(is_analyst({"text_name": self.research_label(title)}))

    def test_sector_qualified_equity_research_without_division_suffix(self):
        for title in ("Head of Technology Equity Research", "Head of Equity Research",
                      "Managing Director of Equity Research", "Co-Head of Equity Research"):
            with self.subTest(title=title):
                self.assertTrue(is_analyst({"text_name": "Jordan Example, Example Securities - " + title}))

    def test_generic_financial_titles_are_not_research_roles(self):
        for title in ("Associate", "Senior Associate", "Director", "Managing Director",
                      "Co-Head of Research", "Head of Research", "Head of Investment Banking"):
            with self.subTest(title=title):
                self.assertFalse(is_analyst({"text_name": "Jordan Example, Example Securities - " + title}))
        for title in ("Head of Investment Banking", "Sales Associate", "Corporate Finance Associate"):
            self.assertFalse(is_analyst({"text_name": self.research_label(title)}))

    def test_plural_and_title_token_boundaries(self):
        for title in ("Analyst", "Analysts", "Senior Research Analyst", "Research Analysts"):
            self.assertTrue(is_analyst({"text_name": "Jordan Example, Example Securities - " + title}))
        for title in ("Analytical Director", "Psychoanalyst", "AnalystRelations", "Analytics Associate"):
            self.assertFalse(is_analyst({"text_name": self.research_label(title)}))

    def test_title_not_person_or_firm_supplies_role_evidence(self):
        for name in ("Analyst Example, Analyst Research Inc - CFO",
                     "Jordan Analyst, Example Securities - Director",
                     "Jordan Example, Analysts Securities - Associate",
                     "Research Division Example, Example Securities - Associate"):
            self.assertFalse(is_analyst({"text_name": name}))

    def test_issuer_management_and_unknown_new_roles_not_promoted(self):
        for title in ("CEO & Research Analyst", "CFO & Research Analyst",
                      "Chief Financial Officer and Former Analyst", "Director of Investor Relations",
                      "Investor Relations Analyst", "Corporate Finance Analyst",
                      "Former Research Analyst", "Retired Analyst", "Ex-Research Analyst",
                      "VP and Former Equity Research Analyst", "Analyst Assistant",
                      "Assistant to the Research Analyst"):
            self.assertFalse(is_analyst({"text_name": self.research_label(title)}))
        for name in ("Unidentified Participant", "Unknown Speaker", "Operator"):
            self.assertFalse(is_analyst({"text_name":
                             name + ", Example Securities, Research Division - Associate"}))
        self.assertFalse(is_analyst({"text_name": "Jordan Example, Acme Inc - Associate"}))
        for name in ("Jordan Example, - Head of Equity Research",
                     "Jordan Example, Unknown Firm, Research Division - Associate"):
            self.assertFalse(is_analyst({"text_name": name}))

    def test_issuer_affiliation_checked_consistently_across_consumers(self):
        for title in ("Analyst", "Associate", "Head of Technology Equity Research"):
            rows = self.event(title)
            rows[1]["text_name"] = "Jordan Example, Acme Inc, Research Division - " + title
            self.assertFalse(is_analyst(rows[1], issuer_labels=[rows[0]["text_name"]]))
            audit, blocks = extract_event(rows, validated_anchor_key=speaker_key(rows[0]))
            self.assertEqual(blocks, [])
            self.assertEqual(audit["qa_analyst_turn_count"], 0)
        # A forged same-issuer question must also be rejected by reconstruction.
        original = self.event()
        blocks, _ = self.extract_and_classify(original)
        original[1]["text_name"] = "Jordan Example, Acme Inc, Research Division - Associate"
        with self.assertRaisesRegex(ValueError, "non-analyst question turn"):
            validate_blocks("1", original, blocks, speaker_key(original[0]))

    def test_issuer_name_suffix_and_company_commas_do_not_change_affiliation(self):
        issuer = "David Smith, Jr., Acme, Inc. - CEO [1]"
        self.assertFalse(is_analyst(
            {"text_name": "Jordan Example, Acme, Inc., Research Division - Associate"},
            issuer_labels=[issuer]))
        # Do not collapse distinct comma-bearing company names to their legal suffix.
        self.assertTrue(is_analyst(
            {"text_name": "Jordan Example, Example Securities, Inc., Research Division - Associate"},
            issuer_labels=[issuer]))

    def test_existing_explicit_role_only_labels_remain_supported(self):
        for name in ("Analyst", "Analysts", "Unidentified Analyst", "Unidentified Research Analyst",
                     "Senior Research Analyst", "Unidentified Senior Equity Research Analyst"):
            self.assertTrue(is_analyst({"text_name": name}))
        for name in ("Operator", "Unidentified Company Representative", "Analyst Research Inc"):
            self.assertFalse(is_analyst({"text_name": name}))

    def test_each_role_recovers_source_exact_primary_episode(self):
        for title in ("Associate", "Research Associate", "Co-Head of Research",
                      "Head of Technology Equity Research", "Analysts"):
            with self.subTest(title=title):
                rows = self.event(title)
                blocks, decisions = self.extract_and_classify(rows)
                self.assertEqual(len(blocks), 1)
                self.assertEqual(blocks[0]["analyst_question"], rows[1]["text_contents"])
                self.assertEqual(blocks[0]["ceo_answer"], rows[-1]["text_contents"])
                self.assertEqual(json.loads(blocks[0]["question_sequence_ids"]), ["2"])
                self.assertEqual(json.loads(blocks[0]["ceo_answer_sequence_ids"]), ["4"])
                self.assertEqual(decisions[0]["inclusion_status"], "proposed_primary")
                self.assertIn("research_requests_v3_20261005", blocks[0]["extraction_version"])

    def test_changed_questioner_splits_episodes_without_duplicate_answers(self):
        rows = self.event("Associate")
        rows.extend([self.row(5, "Q&A", "Sam Example, Other Securities - Analysts",
                              "What is the contract renewal timetable?"),
                     self.row(6, "Q&A", rows[0]["text_name"],
                              "The contracts renew in September and cover 90 units.")])
        blocks, decisions = self.extract_and_classify(rows)
        self.assertEqual(len(blocks), 2)
        self.assertEqual([json.loads(b["question_sequence_ids"]) for b in blocks], [["2"], ["5"]])
        self.assertEqual([json.loads(b["ceo_answer_sequence_ids"]) for b in blocks], [["4"], ["6"]])
        self.assertTrue(all(d["primary_block_candidate"] == 1 for d in decisions))

    def test_same_research_questioner_followup_keeps_distinct_pairs(self):
        rows = self.event("Research Associate")
        rows.extend([self.row(5, "Q&A", rows[1]["text_name"], "And when will these contracts renew?"),
                     self.row(6, "Q&A", rows[0]["text_name"], "The contracts renew in September.")])
        blocks, _ = self.extract_and_classify(rows)
        self.assertEqual([json.loads(b["ceo_answer_sequence_ids"]) for b in blocks], [["4"], ["6"]])

    def test_management_sensitivity_rule_and_ceo_only_target_unchanged(self):
        rows = self.event("Associate", management=True)
        blocks, decisions = self.extract_and_classify(rows)
        self.assertEqual(decisions[0]["inclusion_status"], "sensitivity_only")
        self.assertEqual(blocks[0]["ceo_answer"], rows[-1]["text_contents"])
        self.assertNotIn(rows[2]["text_contents"], blocks[0]["ceo_answer"])
        self.assertIn(rows[2]["text_contents"], blocks[0]["answer_context"])

    def test_bounded_indirect_detail_requests(self):
        for text in ("I was looking for a little more detail on the expansion timetable.",
                     "Good morning. I was looking for just kind of a little more detail on the project.",
                     "We were hoping for some additional color on delivery timing.",
                     "I am just looking for an update on the delivery schedule.",
                     "I was looking for clarification on the margin outlook."):
            with self.subTest(text=text):
                self.assertTrue(analyst_question_candidate(text))
                blocks, decisions = self.extract_and_classify(self.event(question=text))
                self.assertEqual(blocks[0]["analyst_question"], text)
                self.assertEqual(decisions[0]["inclusion_status"], "proposed_primary")

    def test_acknowledgments_congratulations_and_confirmation_stay_in_review(self):
        for text in ("Good morning.", "Thanks for taking the call.", "Congratulations on the quarter.",
                     "Understood. Congratulations on a good quarter.",
                     "If that is the case, you will report closer to 20 percent growth.",
                     "I was looking for a new job.", "We were looking for stronger revenue growth.",
                     "I was not looking for more detail.",
                     "I was looking for more detail last month, but I found it. Thanks.",
                     "We are looking for some color for our office walls.",
                     "I was looking for more detail. On the project, we are satisfied."):
            with self.subTest(text=text):
                self.assertFalse(analyst_question_candidate(text))
                _, decisions = self.extract_and_classify(self.event(question=text))
                self.assertEqual(decisions[0]["inclusion_status"], "source_or_boundary_review")

    def test_source_gaps_and_trailing_fragments_are_not_cleared_by_request_rule(self):
        for text, applicable in (
            ("I was looking for more detail (inaudible) on the delivery schedule.", False),
            ("I was looking for more detail on the delivery schedule (inaudible).", True),
            ("I was looking for more detail on the delivery schedule --", True),
        ):
            self.assertEqual(analyst_question_candidate(text), applicable)
            _, decisions = self.extract_and_classify(self.event(question=text))
            self.assertEqual(decisions[0]["inclusion_status"], "source_or_boundary_review")

    def test_operator_handoff_and_closing_do_not_join_the_preceding_answer(self):
        rows = self.event()
        rows.extend([self.row(5, "Q&A", "Operator", "There are no further questions. Any closing remarks?"),
                     self.row(6, "Q&A", rows[0]["text_name"], "We expect 30 new orders next month.")])
        blocks, _ = self.extract_and_classify(rows)
        self.assertEqual(json.loads(blocks[0]["ceo_answer_sequence_ids"]), ["4"])
        self.assertNotIn(rows[-1]["text_contents"], blocks[0]["ceo_answer"])


if __name__ == "__main__":
    unittest.main()
