import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/llm_measurement/audit_local_specificity_run.py"


def completed(custom_id="u1", score=4, ok=1, **changes):
    parsed = {"ok": ok, "specificity": score}
    row = dict(custom_id=custom_id, status="completed", parsed=parsed,
               raw_output=json.dumps(parsed), validation_error=None)
    row.update(changes)
    return row


def failed(custom_id="u1"):
    return dict(custom_id=custom_id, status="error", parsed=None,
                raw_output=None, validation_error="Synthetic inference failure")


class AuditRunStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_audit(self, outputs, repeats=None, ids=("u1",), expect_report=True):
        def jsonl(name, rows):
            path = self.root / name
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return path

        request = jsonl("input.jsonl", [{"custom_id": key} for key in ids])
        output = jsonl("output.jsonl", outputs)
        original_output = output.read_bytes()
        manifest = self.root / "units.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["custom_id", "unit_type"])
            writer.writeheader()
            writer.writerows({"custom_id": key, "unit_type": "qa"} for key in ids)
        run = self.root / "run.json"
        run.write_text("{}")
        out = self.root / "audit"
        args = [sys.executable, str(SCRIPT), "--input-jsonl", str(request),
                "--manifest-csv", str(manifest), "--output-jsonl", str(output),
                "--run-manifest", str(run), "--output-dir", str(out)]
        if repeats is not None:
            args.extend(["--repeat-output-jsonl", str(jsonl("repeat.jsonl", repeats))])
        result = subprocess.run(args, text=True, capture_output=True, timeout=15)
        self.assertEqual(output.read_bytes(), original_output)
        if not expect_report:
            return result, None, None
        self.assertTrue((out / "dry_run_comparison.json").exists(), result.stderr)
        report = json.loads((out / "dry_run_comparison.json").read_text())
        with (out / "dry_run_audit.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return result, report, rows

    def test_valid_run_passes(self):
        result, report, rows = self.run_audit([completed()])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["strict_valid_outputs"], 1)
        self.assertEqual(rows[0]["technical_valid"], "1")

    def test_all_errors_fail_and_produce_diagnostics(self):
        result, report, rows = self.run_audit([failed()])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["status"], "failed_technical_validation")
        self.assertEqual(report["strict_valid_outputs"], 0)
        self.assertEqual(report["row_errors"], 1)
        self.assertEqual(report["invalid_output_ids"], ["u1"])
        self.assertIn("Synthetic inference failure", rows[0]["validation_error"])

    def test_equal_failed_repeats_never_count_as_agreement(self):
        result, report, rows = self.run_audit([failed()], [failed()])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["repeat_valid_pairs"], 0)
        self.assertEqual(report["repeat_exact_raw_outputs"], 0)
        self.assertEqual(report["repeat_exact_parsed_scores"], 0)
        self.assertEqual(report["repeat_excluded_pairs"], 1)
        self.assertEqual(rows[0]["repeat_exact_output"], "")
        self.assertEqual(rows[0]["repeat_exact_score"], "")

    def test_one_failed_repeat_fails_otherwise_valid_run(self):
        result, report, _ = self.run_audit([completed()], [failed()])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["strict_valid_outputs"], 1)
        self.assertEqual(report["repeat_strict_valid_outputs"], 0)
        self.assertEqual(report["repeat_row_errors"], 1)

    def test_valid_repeats_have_explicit_denominators(self):
        result, report, rows = self.run_audit([completed()], [completed()])
        self.assertEqual(result.returncode, 0)
        for field in ("repeat_valid_pairs", "repeat_raw_comparable_pairs", "repeat_score_comparable_pairs",
                      "repeat_exact_raw_outputs", "repeat_exact_parsed_scores"):
            self.assertEqual(report[field], 1)
        self.assertEqual(report["repeat_excluded_pairs"], 0)
        self.assertEqual(rows[0]["repeat_exact_score"], "1")

    def test_different_valid_scores_are_diagnostic_not_technical_failure(self):
        result, report, _ = self.run_audit([completed(score=4)], [completed(score=3)])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["repeat_score_comparable_pairs"], 1)
        self.assertEqual(report["repeat_exact_parsed_scores"], 0)

    def test_valid_abstention_is_separate_from_numeric_agreement(self):
        row = completed(score=0, ok=0)
        result, report, _ = self.run_audit([row], [row])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["status"], "passed_with_manual_edge_case")
        self.assertEqual(report["ok0_custom_ids"], ["u1"])
        self.assertEqual(report["repeat_valid_pairs"], 1)
        self.assertEqual(report["repeat_score_comparable_pairs"], 0)
        self.assertEqual(report["repeat_exact_parsed_scores"], 0)
        self.assertEqual(sum(report["score_distribution_ok1"].values()), 0)

    def test_repeat_only_abstention_is_reported(self):
        result, report, _ = self.run_audit([completed()], [completed(score=0, ok=0)])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["status"], "passed_with_manual_edge_case")
        self.assertEqual(report["repeat_ok0_custom_ids"], ["u1"])
        self.assertEqual(report["repeat_score_comparable_pairs"], 0)

    def test_error_with_valid_parsed_score_is_not_scored(self):
        result, report, _ = self.run_audit([completed(status="error")])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["strict_valid_outputs"], 0)
        self.assertEqual(sum(report["score_distribution_ok1"].values()), 0)

    def test_invalid_schema_types_fail_without_distribution_crash(self):
        result, report, _ = self.run_audit([completed(parsed={"ok": 1, "specificity": "bad"})])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["strict_valid_outputs"], 0)

    def test_truncated_unknown_or_validation_error_outputs_fail(self):
        for changes in ({"output_truncated": True}, {"finish_reason": "unknown"},
                        {"validation_error": "Synthetic schema error"}):
            with self.subTest(changes=changes):
                result, report, _ = self.run_audit([completed(**changes)])
                self.assertEqual(result.returncode, 1)
                self.assertEqual(report["strict_valid_outputs"], 0)

    def test_missing_and_unexpected_outputs_have_coverage_ledger(self):
        result, report, rows = self.run_audit([completed("extra")])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["coverage"]["missing_output_ids"], ["u1"])
        self.assertEqual(report["coverage"]["unexpected_output_ids"], ["extra"])
        self.assertEqual(rows[0]["status"], "missing")

    def test_missing_repeat_has_coverage_and_exclusion_ledger(self):
        result, report, rows = self.run_audit([completed()], [])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["coverage"]["missing_repeat_ids"], ["u1"])
        self.assertEqual(report["repeat_excluded_pairs"], 1)
        self.assertEqual(rows[0]["repeat_validation_error"], "missing output")

    def test_partial_failures_keep_requested_denominator(self):
        result, report, _ = self.run_audit([completed(), failed("u2")],
                                          [completed(), failed("u2")], ids=("u1", "u2"))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["input_requests"], 2)
        self.assertEqual(report["repeat_valid_pairs"], 1)
        self.assertEqual(report["repeat_excluded_pairs"], 1)
        self.assertEqual(report["repeat_exact_parsed_scores"], 1)

    def test_missing_raw_values_do_not_count_as_equal_raw(self):
        row = completed(raw_output=None)
        result, report, _ = self.run_audit([row], [row])
        self.assertEqual(result.returncode, 0)  # Whole-raw verification remains issue #7.
        self.assertEqual(report["repeat_raw_comparable_pairs"], 0)
        self.assertEqual(report["repeat_exact_raw_outputs"], 0)

    def test_duplicate_ids_and_empty_inputs_stop_with_nonzero_exit(self):
        for outputs, ids in (([completed(), completed()], ("u1",)), ([], ())):
            with self.subTest(ids=ids):
                result, _, _ = self.run_audit(outputs, ids=ids, expect_report=False)
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
