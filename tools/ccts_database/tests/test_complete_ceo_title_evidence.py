import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_ccts_ceo_presentation_representation as pre
import build_ccts_execucomp_speaker_gate as gate
import extract_ccts_ceo_qa_blocks as qa
from ceo_title_evidence import SPECIAL_CEO_RE, evidence_labels, single_anchor
from test_build_ccts_execucomp_speaker_gate import source


def turn(index, section, title="CEO", analyst=False):
    return {"event_id": "1", "sample_rank": "1", "year": "2020", "start_date": "2020-03-15",
            "company_name": "Acme Corp", "sequence_id": str(index), "analysis_text_type": section,
            "text_name": ("Jordan Jones, Bank - Analyst" if analyst else f"David Smith, Acme Corp - {title}") + f" [{index}]",
            "text_contents": ("What is the sales outlook next quarter?" if analyst else
                              "We expect sales to grow ten percent next quarter based on signed contracts.")}


class CompleteCeoTitleEvidenceTests(unittest.TestCase):
    def classify(self, rows):
        return gate.classify_event(source(), "David Smith",
            gate.make_candidate_map([r for r in rows if r["analysis_text_type"] == "PRE"]),
            gate.make_candidate_map([r for r in rows if r["analysis_text_type"] == "Q&A"]), {})

    def test_special_title_in_either_section_and_either_order_is_retained(self):
        for special in ("Interim CEO", "Acting CEO", "Co-CEO", "Co-Chief Executive Officer"):
            for section in ("PRE", "Q&A"):
                for special_first in (True, False):
                    with self.subTest(special=special, section=section, first=special_first):
                        titles = [special, "CEO"] if special_first else ["CEO", special]
                        rows = [turn(1, "PRE"), turn(2, "Q&A", analyst=True), turn(3, "Q&A")]
                        rows += [turn(i + 4, section, title) for i, title in enumerate(titles)]
                        result = self.classify(rows)
                        self.assertEqual(result["event_speaker_gate_pass"], 0)
                        self.assertIn("SPECIAL_CEO_TITLE", result["identity_flags"])
                        key = "pre_ceo_speakers" if section == "PRE" else "qa_ceo_speakers"
                        self.assertIn(special, result[key])
                        self.assertEqual(result["shared_ceo_candidate_count"], 1)
                        audit, blocks = qa.extract_event(sorted(rows, key=lambda r: int(r["sequence_id"])))
                        self.assertIn("SPECIAL_CEO_TITLE", audit["event_flags"])
                        self.assertTrue(all("SPECIAL_CEO_TITLE" in b["block_flags"] for b in blocks))

    def test_maps_are_order_independent_and_count_people_not_title_variants(self):
        rows = [turn(1, "PRE", "Acting CEO"), turn(2, "PRE"), turn(3, "PRE", "Chief Executive Officer")]
        for builder in (gate.make_candidate_map, pre.candidate_map, qa.candidate_map):
            original = builder(rows)
            self.assertEqual(original, builder(list(reversed(rows))))
            self.assertEqual(len(original), 1)
            self.assertEqual(len(evidence_labels(original)), 3)

    def test_acting_and_interim_cfo_not_mistaken_for_special_ceo(self):
        for title in ("CEO, Acting CFO, President", "CEO and Acting CFO", "Interim CFO, CEO", "CEO, Interim CFO"):
            rows = [turn(1, "PRE", title), turn(2, "Q&A", analyst=True), turn(3, "Q&A")]
            result = self.classify(rows)
            self.assertEqual(result["event_speaker_gate_pass"], 1, title)
            self.assertNotIn("SPECIAL_CEO_TITLE", result["identity_flags"])
            self.assertNotIn("SPECIAL_CEO_TITLE", qa.extract_event(rows)[0]["event_flags"])

    def test_concurrent_roles_with_special_ceo_are_review(self):
        for title in ("CFO, Acting CEO", "Interim President and CEO", "Acting as Chief Executive Officer and CFO",
                      "Interim Group CEO", "Acting Global Chief Executive Officer"):
            result = self.classify([turn(1, "PRE", title), turn(2, "Q&A")])
            self.assertEqual(result["event_speaker_gate_pass"], 0, title)
            self.assertIn("SPECIAL_CEO_TITLE", result["identity_flags"])

    def test_ordinary_ceo_title_variants_still_pass(self):
        result = self.classify([turn(1, "PRE", "President and CEO"), turn(2, "Q&A", "Chief Executive Officer")])
        self.assertEqual(result["event_speaker_gate_pass"], 1)
        self.assertIn("President and CEO", result["shared_ceo_speakers"])
        self.assertIn("Chief Executive Officer", result["shared_ceo_speakers"])

    def test_single_anchor_checks_every_label_not_just_first(self):
        self.assertEqual(single_anchor("David Smith, Acme Corp - CEO; David Smith, Acme Corp - President and CEO", qa.speaker_key),
                         "david smith acme corp")
        with self.assertRaises(ValueError):
            single_anchor("David Smith, Acme Corp - CEO; Other Person, Acme Corp - CEO", qa.speaker_key)

    def test_presentation_cli_uses_complete_title_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            turns = root / "turns.csv"
            rows = [turn(1, "PRE", "Interim CEO"), turn(2, "PRE"), turn(3, "Q&A")]
            with turns.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            audit = root / "audit.csv"
            audit.write_text("event_id,event_audit_status\n1,usable\n")
            out = root / "pre"
            result = subprocess.run([sys.executable, str(Path(pre.__file__)), "--turns", str(turns),
                "--block-event-audit", str(audit), "--output-dir", str(out)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (out / "ceo_presentation_event_table.csv").open(newline="") as f:
                event = next(csv.DictReader(f))
            self.assertEqual(event["strict_identity_ready"], "0")
            self.assertIn("SPECIAL_CEO_TITLE", event["identity_flags"])
            self.assertIn("Interim CEO", event["pre_ceo_speakers"])


if __name__ == "__main__":
    unittest.main()
