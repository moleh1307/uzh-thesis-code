import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_execucomp_confirmed_ceo_qa_blocks import run_extraction
from ceo_title_evidence import GATE_VERSION


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gate_row(event_id: str, *, passed: bool) -> dict[str, str]:
    return {
        "event_id": event_id,
        "turn_rows": "4" if passed else "2",
        "pre_turn_rows": "1",
        "qa_turn_rows": "3" if passed else "1",
        "shared_ceo_candidate_count": "1" if passed else "0",
        "shared_ceo_speakers": "David Smith, Acme Inc - CEO" if passed else "",
        "matched_shared_speakers": "David Smith, Acme Inc - CEO" if passed else "",
        "within_call_identity_ready": "1" if passed else "0",
        "external_name_match_quality": "exact_full_name" if passed else "no_match",
        "external_speaker_validation_status": "confirmed_external_ceo_shared_pre_qa" if passed else "not_confirmed_external_ceo_not_title_labelled",
        "event_speaker_gate_pass": "1" if passed else "0",
        "gvkey": "000001",
        "expected_execid": "1" if passed else "2",
        "tenure_episode": "1",
        "expected_ceo_name": "David Smith" if passed else "Other Person",
        "event_date": "2020-03-15",
        "calendar_quarter": "2020Q1",
        "turnover_ids": "turnover_1",
    }


class ConfirmedCeoQaExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.turns = self.root / "turns.csv"
        self.event_gate = self.root / "event_gate.csv"
        self.episode_gate = self.root / "episode_gate.csv"
        self.turnover_gate = self.root / "turnover_gate.csv"
        self.gate_summary = self.root / "gate_summary.json"
        self.output = self.root / "blocks"

        turn_rows = [
            {"sample_rank": "1", "event_id": "100", "start_date": "2020-03-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q1 call", "sequence_id": "1", "raw_sequence_id": "1", "source_event_company_name": "Acme Inc", "text_type": "PRE", "analysis_text_type": "PRE", "text_name": "David Smith, Acme Inc - CEO [1]", "text_contents": "We expect sales to grow ten percent this year.", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
            {"sample_rank": "1", "event_id": "100", "start_date": "2020-03-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q1 call", "sequence_id": "10", "raw_sequence_id": "10", "source_event_company_name": "Acme Inc", "text_type": "Q&A", "analysis_text_type": "Q&A", "text_name": "Jordan Jones, Bank - Analyst [10]", "text_contents": "What is the revenue outlook for the next quarter?", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
            {"sample_rank": "1", "event_id": "100", "start_date": "2020-03-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q1 call", "sequence_id": "11", "raw_sequence_id": "11", "source_event_company_name": "Acme Inc", "text_type": "Q&A", "analysis_text_type": "Q&A", "text_name": "David Smith, Acme Inc - Chief Executive Officer [11]", "text_contents": "We expect revenue to grow about ten percent next quarter based on signed contracts.", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
            {"sample_rank": "1", "event_id": "100", "start_date": "2020-03-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q1 call", "sequence_id": "12", "raw_sequence_id": "12", "source_event_company_name": "Acme Inc", "text_type": "Q&A", "analysis_text_type": "Q&A", "text_name": "Operator [12]", "text_contents": "The next question comes from another analyst.", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
            {"sample_rank": "2", "event_id": "101", "start_date": "2020-06-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q2 call", "sequence_id": "1", "raw_sequence_id": "1", "source_event_company_name": "Acme Inc", "text_type": "PRE", "analysis_text_type": "PRE", "text_name": "Other Person, Acme Inc - CEO [1]", "text_contents": "We expect sales to grow.", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
            {"sample_rank": "2", "event_id": "101", "start_date": "2020-06-15", "year": "2020", "company_name": "Acme Inc", "company_ticker": "ACM", "event_title": "Q2 call", "sequence_id": "10", "raw_sequence_id": "10", "source_event_company_name": "Acme Inc", "text_type": "Q&A", "analysis_text_type": "Q&A", "text_name": "Jordan Jones, Bank - Analyst [10]", "text_contents": "What is the outlook?", "deduplication_status": "unchanged", "source_event_company_names_json": "[\"Acme Inc\"]"},
        ]
        write_csv(self.turns, turn_rows)
        write_csv(self.event_gate, [gate_row("100", passed=True), gate_row("101", passed=False)])
        write_csv(self.episode_gate, [
            {"gvkey": "000001", "expected_execid": "1", "tenure_episode": "1", "episode_analysis_gate_pass": "1"},
            {"gvkey": "000001", "expected_execid": "2", "tenure_episode": "1", "episode_analysis_gate_pass": "0"},
        ])
        write_csv(self.turnover_gate, [{
            "turnover_id": "turnover_1", "gvkey": "000001", "old_execid": "1", "old_tenure_episode": "1",
            "new_execid": "2", "new_tenure_episode": "1", "turnover_analysis_gate_pass": "0",
        }])
        self.gate_summary.write_text(json.dumps({
            "status": "complete",
            "gate_version": GATE_VERSION,
            "candidate_events_total": 2,
            "event_gate_pass": 1,
            "turn_rows_scanned": 6,
            "episode_gate_pass": 1,
            "episode_gate_fail": 1,
            "turnover_gate_pass": 0,
            "turnover_gate_fail": 1,
            "provenance": {"turns_sha256_verified": hashlib.sha256(self.turns.read_bytes()).hexdigest()},
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_streams_only_gate_pass_calls_and_reconciles_source(self) -> None:
        result = run_extraction(
            self.turns, self.event_gate, self.episode_gate, self.turnover_gate,
            self.gate_summary, self.output, manual_audit_blocks=10,
        )
        self.assertEqual(result["scope"]["events_extracted"], 1)
        self.assertEqual(result["scope"]["all_input_events_reconciled"], 2)
        self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)
        self.assertTrue((self.output / "manual_audit_sample.md").is_file())
        with (self.output / "ceo_qa_blocks.csv").open(encoding="utf-8", newline="") as handle:
            blocks = list(csv.DictReader(handle))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["event_id"], "100")
        self.assertEqual(blocks[0]["validated_ceo_speaker"], "David Smith, Acme Inc - CEO")
        self.assertIn("ten percent", blocks[0]["ceo_answer"])

    def test_refuses_gate_anchor_inconsistent_with_shared_speaker(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["matched_shared_speakers"] = "Different Person, Acme Inc - CEO"
        write_csv(self.event_gate, rows)
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            run_extraction(
                self.turns, self.event_gate, self.episode_gate, self.turnover_gate,
                self.gate_summary, self.output, manual_audit_blocks=0,
            )
        self.assertFalse(self.output.exists())

    def test_preserved_title_variants_resolve_to_one_anchor(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = "David Smith, Acme Inc - CEO; David Smith, Acme Inc - Chief Executive Officer"
        rows[0]["matched_shared_speakers"] = labels
        rows[0]["shared_ceo_speakers"] = labels
        write_csv(self.event_gate, rows)
        result = run_extraction(self.turns, self.event_gate, self.episode_gate,
            self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)

    def test_rejects_second_person_hidden_after_first_anchor(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["matched_shared_speakers"] += "; Other Person, Acme Inc - CEO"
        write_csv(self.event_gate, rows)
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())

    def test_old_last_label_gate_is_not_recertified(self) -> None:
        summary = json.loads(self.gate_summary.read_text())
        summary["gate_version"] = "v1.2"
        self.gate_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "expected gate"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
