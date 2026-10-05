import csv
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_execucomp_confirmed_ceo_qa_blocks import run_extraction
import extract_execucomp_confirmed_ceo_qa_blocks as consumer
from ceo_title_evidence import GATE_VERSION, encode_labels
import build_ccts_execucomp_speaker_gate as producer
from speaker_gate_contract import INPUT_NAMES, sha256_file


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
        assignments = []
        for row in (gate_row("100", passed=True), gate_row("101", passed=False)):
            assignments.append({"event_id": row["event_id"], "gvkey": row["gvkey"],
                                "execid": row["expected_execid"], "tenure_episode": "1",
                                "event_date": "2020-03-15" if row["event_id"] == "100" else "2020-06-15",
                                "calendar_quarter": "2020Q1" if row["event_id"] == "100" else "2020Q2",
                                "turnover_ids": "turnover_1"})
        source_rows = {
            "event_assignments": assignments,
            "episode_manifest": [{"gvkey": "000001", "execid": str(i), "tenure_episode": "1",
                                  "calls_in_source_panel": "1", "distinct_quarters_in_source_panel": "1"}
                                 for i in (1, 2)],
            "turnover_episode_map": [{"turnover_id": "turnover_1", "side": side,
                                      "gvkey": "000001", "execid": str(i), "tenure_episode": "1"}
                                     for i, side in ((1, "old_episode"), (2, "new_episode"))],
            "turnover_pairs": [{
            "turnover_id": "turnover_1", "gvkey": "000001", "old_execid": "1", "old_tenure_episode": "1",
            "new_execid": "2", "new_tenure_episode": "1", "transition_gap_days": "90",
            }],
            "execucomp_normalized": [{"gvkey": "000001", "execid": str(i), "tenure_episode": "1",
                                      "person_name": name}
                                     for i, name in ((1, "David Smith"), (2, "Other Person"))],
            "call_sequences": [{**row, "ceo_name": name} for row, name in
                               zip(assignments, ("David Smith", "Other Person"))],
            "fetch_audit": [{"event_id": str(i), "fetch_status": "ok", "fetch_notes": ""}
                            for i in (100, 101)],
        }
        parameters = {"turns": self.turns, "output_dir": self.root / "gate",
                      "minimum_confirmed_calls": 1, "minimum_confirmed_quarters": 1,
                      "limit_events": None}
        for name, rows in source_rows.items():
            parameters[name] = self.root / (name + ".csv")
            write_csv(parameters[name], rows)
        parameters["resolution_audit"] = self.root / "resolution.csv"
        parameters["resolution_audit"].write_text("event_id\n")
        parameters["dedup_summary"] = self.root / "dedup.json"
        parameters["dedup_summary"].write_text(json.dumps({
            "status": "complete",
            "input_events": 2, "output_rows": 6, "unique_event_sequence_keys": 6,
            "raw_input_modified": False, "review_event_ids_resolved": [],
            "resolution_audit_rows": 0, "duplicate_rows_collapsed": 0,
            "output_sha256": hashlib.sha256(self.turns.read_bytes()).hexdigest(),
        }))
        self.producer_args = SimpleNamespace(**parameters)
        with contextlib.redirect_stdout(io.StringIO()):
            producer.run(self.producer_args)
        self.event_gate = parameters["output_dir"] / "event_speaker_gate.csv"
        self.episode_gate = parameters["output_dir"] / "episode_speaker_gate.csv"
        self.turnover_gate = parameters["output_dir"] / "turnover_speaker_gate.csv"
        self.gate_summary = parameters["output_dir"] / "speaker_gate_summary.json"

    def refresh_artifact_hashes(self):
        summary = json.loads(self.gate_summary.read_text())
        summary["artifact_sha256"] = {name: sha256_file(path) for name, path in (
            ("event_speaker_gate_csv", self.event_gate), ("episode_speaker_gate_csv", self.episode_gate),
            ("turnover_speaker_gate_csv", self.turnover_gate))}
        self.gate_summary.write_text(json.dumps(summary))

    def rebuild_title_gate(self, pre_title, qa_title=None):
        rows = producer.read_rows(self.turns)
        for row in rows:
            if row["event_id"] == "100" and row["text_name"].startswith("David Smith,"):
                title = pre_title if row["analysis_text_type"] == "PRE" else (qa_title or pre_title)
                row["text_name"] = f"David Smith, Acme Inc - {title} [{row['sequence_id']}]"
        write_csv(self.turns, rows)
        summary = json.loads(self.producer_args.dedup_summary.read_text())
        summary["output_sha256"] = sha256_file(self.turns)
        self.producer_args.dedup_summary.write_text(json.dumps(summary))
        self.producer_args.output_dir = self.root / "structured_title_gate"
        with contextlib.redirect_stdout(io.StringIO()):
            producer.run(self.producer_args)
        root = self.producer_args.output_dir
        self.event_gate = root / "event_speaker_gate.csv"
        self.episode_gate = root / "episode_speaker_gate.csv"
        self.turnover_gate = root / "turnover_speaker_gate.csv"
        self.gate_summary = root / "speaker_gate_summary.json"

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
        self.assertEqual(blocks[0]["validated_ceo_speaker"],
                         "David Smith, Acme Inc - CEO; David Smith, Acme Inc - Chief Executive Officer")
        self.assertIn("ten percent", blocks[0]["ceo_answer"])

    def test_research_roles_and_indirect_request_in_bound_publication(self):
        for index, title in enumerate(("Associate", "Research Associate", "Co-Head of Research",
                                       "Head of Technology Equity Research", "Analysts")):
            with self.subTest(title=title):
                rows = producer.read_rows(self.turns)
                rows[1]["text_name"] = "Jordan Jones, Example Securities, Research Division - " + title
                rows[1]["text_contents"] = "I was looking for a little more detail on the revenue outlook."
                write_csv(self.turns, rows)
                summary = json.loads(self.producer_args.dedup_summary.read_text())
                summary["output_sha256"] = sha256_file(self.turns)
                self.producer_args.dedup_summary.write_text(json.dumps(summary))
                root = self.root / f"research_gate_{index}"
                self.producer_args.output_dir = root
                with contextlib.redirect_stdout(io.StringIO()):
                    producer.run(self.producer_args)
                destination = self.root / f"research_blocks_{index}"
                result = run_extraction(
                    self.turns, root / "event_speaker_gate.csv", root / "episode_speaker_gate.csv",
                    root / "turnover_speaker_gate.csv", root / "speaker_gate_summary.json",
                    destination, manual_audit_blocks=0,
                )
                self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)
                self.assertEqual(result["script_version"], consumer.SCRIPT_VERSION)
                blocks = producer.read_rows(destination / "ceo_qa_blocks.csv")
                self.assertEqual(blocks[0]["quality_tier"], "high")
                self.assertEqual(blocks[0]["analyst_question"], rows[1]["text_contents"])
                self.assertEqual(blocks[0]["ceo_answer"], rows[2]["text_contents"])
                self.assertIn("research_requests_v3_20261005", blocks[0]["extraction_version"])

    def test_refuses_gate_anchor_inconsistent_with_shared_speaker(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["matched_shared_speakers"] = "Different Person, Acme Inc - CEO"
        rows[0]["matched_shared_speakers_json"] = encode_labels([rows[0]["matched_shared_speakers"]])
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            run_extraction(
                self.turns, self.event_gate, self.episode_gate, self.turnover_gate,
                self.gate_summary, self.output, manual_audit_blocks=0,
            )
        self.assertFalse(self.output.exists())

    def test_preserved_title_variants_resolve_to_one_anchor(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        labels = ["David Smith, Acme Inc - CEO", "David Smith, Acme Inc - Chief Executive Officer"]
        for field in ("matched_shared_speakers", "shared_ceo_speakers"):
            rows[0][field] = "; ".join(labels)
            rows[0][field + "_json"] = encode_labels(labels)
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        result = run_extraction(self.turns, self.event_gate, self.episode_gate,
            self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)

    def test_rejects_second_person_hidden_after_first_anchor(self) -> None:
        with self.event_gate.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["matched_shared_speakers"] += "; Other Person, Acme Inc - CEO"
        rows[0]["matched_shared_speakers_json"] = encode_labels(
            json.loads(rows[0]["matched_shared_speakers_json"]) + ["Other Person, Acme Inc - CEO"])
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())

    def test_semicolon_title_producer_to_extraction_round_trip(self):
        self.rebuild_title_gate("Chairman; President; CEO", 'Chief Executive Officer; "Global" President')
        result = run_extraction(self.turns, self.event_gate, self.episode_gate,
            self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)
        rows = producer.read_rows(self.output / "ceo_qa_blocks.csv")
        labels = json.loads(rows[0]["validated_ceo_speaker_json"])
        self.assertEqual(len(labels), 2)
        self.assertIn("David Smith, Acme Inc - Chairman; President; CEO", labels)
        self.assertIn('David Smith, Acme Inc - Chief Executive Officer; "Global" President', labels)
        self.assertEqual(rows[0]["ceo_answer"], producer.read_rows(self.turns)[2]["text_contents"])

    def test_invalid_structured_gate_evidence_is_rejected(self):
        original = producer.read_rows(self.event_gate)
        for payload in ('not json', '"CEO"', '[null]', '[1]', '[true]', '[""]', '[]',
                        '["David Smith, Acme Inc - CEO", "David Smith, Acme Inc - CEO"]'):
            with self.subTest(payload=payload):
                rows = [dict(row) for row in original]
                rows[0]["matched_shared_speakers_json"] = payload
                write_csv(self.event_gate, rows)
                self.refresh_artifact_hashes()
                with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
                    consumer.load_speaker_gate(self.event_gate, self.episode_gate,
                        self.turnover_gate, self.gate_summary, self.turns)

    def test_structured_labels_require_matching_display_evidence(self):
        rows = producer.read_rows(self.event_gate)
        rows[0]["shared_ceo_speakers"] += "; President"
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            consumer.load_speaker_gate(self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.turns)

    def test_missing_structured_column_is_not_recovered_from_display(self):
        rows = producer.read_rows(self.event_gate)
        for row in rows:
            row.pop("matched_shared_speakers_json")
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "missing columns"):
            consumer.load_speaker_gate(self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.turns)

    def test_matched_label_must_be_present_in_shared_evidence(self):
        rows = producer.read_rows(self.event_gate)
        invented = "David Smith, Acme Inc - President and CEO"
        rows[0]["matched_shared_speakers"] = invented
        rows[0]["matched_shared_speakers_json"] = encode_labels([invented])
        write_csv(self.event_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "inconsistent CEO anchor"):
            consumer.load_speaker_gate(self.event_gate, self.episode_gate,
                self.turnover_gate, self.gate_summary, self.turns)

    def test_producer_binds_all_inputs_and_outputs(self):
        summary = json.loads(self.gate_summary.read_text())
        self.assertEqual(set(summary["provenance"]["input_files"]), set(INPUT_NAMES))
        for name, record in summary["provenance"]["input_files"].items():
            self.assertEqual(record["sha256"], sha256_file(getattr(self.producer_args, name)))
        self.assertEqual(len(summary["artifact_sha256"]), 3)

    def test_rejects_gate_table_byte_substitution(self):
        for path in (self.event_gate, self.episode_gate, self.turnover_gate):
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "artifact hash differs"):
                    run_extraction(self.turns, self.event_gate, self.episode_gate,
                                   self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
                self.assertFalse(self.output.exists())
            finally:
                path.write_bytes(original)

    def test_rejects_unbound_historical_bundle(self):
        summary = json.loads(self.gate_summary.read_text())
        del summary["bundle_version"]
        self.gate_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "unbound"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())

    def test_rejects_rehashed_external_identity_substitution(self):
        for path, field in ((self.event_gate, "expected_execid"),
                            (self.episode_gate, "expected_execid"), (self.turnover_gate, "old_execid")):
            rows = producer.read_rows(path)
            rows[0][field] = "99"
            write_csv(path, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "expected_execid differs from bound assignment"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())

    def test_episode_flags_reconcile_even_with_same_pass_total(self):
        rows = producer.read_rows(self.episode_gate)
        rows[0]["episode_analysis_gate_pass"], rows[1]["episode_analysis_gate_pass"] = "0", "1"
        write_csv(self.episode_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "episode gate does not reconcile"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_turnover_flag_cannot_override_failed_episode(self):
        rows = producer.read_rows(self.turnover_gate)
        rows[0]["turnover_analysis_gate_pass"] = "1"
        write_csv(self.turnover_gate, rows)
        self.refresh_artifact_hashes()
        summary = json.loads(self.gate_summary.read_text())
        summary.update(turnover_gate_pass=1, turnover_gate_fail=0)
        self.gate_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "turnover gate does not reconcile"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_duplicate_episode_identity_is_rejected(self):
        rows = producer.read_rows(self.episode_gate)
        rows[1] = dict(rows[0])
        write_csv(self.episode_gate, rows)
        self.refresh_artifact_hashes()
        with self.assertRaisesRegex(ValueError, "duplicate episode gate identity"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_external_input_changed_after_gate_creation_is_rejected(self):
        self.producer_args.execucomp_normalized.write_text("different data\n")
        with self.assertRaisesRegex(ValueError, "input hash differs: execucomp_normalized"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_producer_rejects_input_changed_during_scan(self):
        self.producer_args.output_dir = self.root / "new_gate"
        original = producer.scan_turns
        def changing_scan(*args):
            result = original(*args)
            with self.producer_args.execucomp_normalized.open("a") as handle:
                handle.write("\n")
            return result
        with patch.object(producer, "scan_turns", side_effect=changing_scan):
            with self.assertRaisesRegex(ValueError, "input hash differs: execucomp_normalized"):
                producer.run(self.producer_args)
        self.assertFalse(self.producer_args.output_dir.exists())

    def test_consumer_rejects_input_changed_during_metadata_read(self):
        original = consumer.validate_metadata_inputs
        def changing_read(*args):
            result = original(*args)
            with self.producer_args.execucomp_normalized.open("a") as handle:
                handle.write("\n")
            return result
        with patch.object(consumer, "validate_metadata_inputs", side_effect=changing_read):
            with self.assertRaisesRegex(ValueError, "input hash differs: execucomp_normalized"):
                run_extraction(self.turns, self.event_gate, self.episode_gate,
                               self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())

    def test_missing_input_binding_is_not_recognized_as_complete(self):
        summary = json.loads(self.gate_summary.read_text())
        del summary["provenance"]["input_files"]["resolution_audit"]
        self.gate_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "input bindings are missing or incomplete"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_pass_counts_reconcile_with_actual_episode_flags(self):
        summary = json.loads(self.gate_summary.read_text())
        summary.update(episode_gate_pass=2, episode_gate_fail=0)
        self.gate_summary.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "episode gate pass count differs"):
            run_extraction(self.turns, self.event_gate, self.episode_gate,
                           self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)

    def test_relocated_same_bytes_with_relative_input_paths_remain_valid(self):
        summary = json.loads(self.gate_summary.read_text())
        for record in summary["provenance"]["input_files"].values():
            record["path"] = "../" + Path(record["path"]).name
        self.gate_summary.write_text(json.dumps(summary))
        result = run_extraction(self.turns, self.event_gate, self.episode_gate,
                               self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertEqual(result["blocks"]["candidate_blocks_total"], 1)

    def test_old_last_label_gate_is_not_recertified(self) -> None:
        summary = json.loads(self.gate_summary.read_text())
        for old_version in ("v1.2", "v1.3_complete_title_evidence_20261003"):
            summary["gate_version"] = old_version
            self.gate_summary.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "expected gate"):
                run_extraction(self.turns, self.event_gate, self.episode_gate,
                    self.turnover_gate, self.gate_summary, self.output, manual_audit_blocks=0)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
