import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/llm_measurement"))
import build_specificity_scoring_manifest as builder


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ManifestPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.presentation = dict(event_id="1", episode_rank="1", turnover_id="T1", turnover_side="old",
            expected_execid="10", expected_ceo_name="Synthetic CEO", company_id="100", company_name="Synthetic",
            start_date="2015-01-01", calendar_quarter="2015Q1", ceo_presentation_text="Revenue grew twelve percent.",
            pre_ceo_word_count="4", qa_ceo_word_count="20")
        self.qa_fields = ["block_id", "event_id", "quality_tier", "analyst_question", "ceo_answer"]
        self.output = self.root / "output"
        write_csv(self.root / "pre.csv", [self.presentation], list(self.presentation))
        write_csv(self.root / "qa.csv", [], self.qa_fields)
        write_csv(self.root / "turnovers.csv", [], ["turnover_analysis_gate_pass", "old_episode_rank", "new_episode_rank"])

    def run_cli(self, *extra):
        argv = ["manifest", "--presentations", str(self.root / "pre.csv"), "--high-qa-blocks", str(self.root / "qa.csv"),
                "--turnover-gate", str(self.root / "turnovers.csv"), "--contract-dir", str(ROOT / "configs"),
                "--output-dir", str(self.output), *extra]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            builder.main()
        return json.loads((self.output / "specificity_manifest_summary.json").read_text())

    def test_empty_qa_publishes_complete_diagnostic_with_stable_headers(self):
        result = self.run_cli()
        self.assertEqual(result["counts"]["qa_scoring_units"], 0)
        self.assertEqual(result["counts"]["primary_eligible_calls"], 0)
        self.assertEqual(len(list(self.output.iterdir())), 6)
        with (self.output / "qa_specificity_scoring_units.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, builder.QA_FIELDS)
            self.assertEqual(list(reader), [])

    def test_all_filtered_qa_exclusions_are_published(self):
        write_csv(self.root / "qa.csv", [{"block_id": "b", "event_id": "1", "quality_tier": "high",
                  "analyst_question": "What changed?", "ceo_answer": "Thank you all for joining the call."}], self.qa_fields)
        result = self.run_cli()
        self.assertEqual(result["counts"]["qa_input_blocks_excluded_non_substantive"], 1)
        self.assertEqual(result["counts"]["qa_scoring_units"], 0)
        self.assertTrue((self.output / "qa_specificity_excluded_non_substantive.csv").is_file())

    def test_header_only_presentations_rejected_without_partial_package(self):
        write_csv(self.root / "pre.csv", [], list(self.presentation))
        with self.assertRaisesRegex(SystemExit, "no presentation"):
            self.run_cli()
        self.assertFalse(self.output.exists())

    def test_blank_pre_targets_have_complete_zero_support_headers(self):
        row = dict(self.presentation, ceo_presentation_text="", pre_ceo_word_count="0")
        write_csv(self.root / "pre.csv", [row], list(row))
        result = self.run_cli()
        self.assertEqual(result["counts"]["total_scoring_units"], 0)
        with (self.output / "pre_specificity_scoring_units.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, builder.PRE_FIELDS)
            self.assertEqual(list(reader), [])

    def test_failed_late_write_leaves_no_published_package(self):
        original = builder.write_csv
        def failing_write(path, *args):
            if path.name == "specificity_call_coverage.csv":
                raise OSError("synthetic failure after PRE and QA")
            return original(path, *args)
        with patch.object(builder, "write_csv", side_effect=failing_write), self.assertRaises(OSError):
            self.run_cli()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".output.*")))

    def test_existing_package_unchanged_even_with_legacy_force(self):
        self.run_cli()
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        for extra in ((), ("--force",)):
            with self.assertRaises(FileExistsError):
                self.run_cli(*extra)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})


if __name__ == "__main__":
    unittest.main()
