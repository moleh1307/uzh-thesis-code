import json
import unittest
from unittest.mock import patch

from runner_fixtures import FakeRunnerCase, runner


class RunnerStatusTests(FakeRunnerCase):
    def test_valid_run_succeeds_and_records_atomic_progress(self):
        self.requests(2)
        snapshots = []
        code, manifest, calls = self.run_fake(observe=snapshots)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["status"], "completed_local_run")
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 2)
        self.assertEqual(manifest["counts"]["missing_outputs"], 0)
        self.assertEqual(snapshots[0]["status"], "running_local_run")
        self.assertEqual(snapshots[0]["stage"], "loading_model")
        running = [row for row in snapshots if row["status"] == "running_local_run"]
        self.assertEqual([row["counts"]["processed_this_run"] for row in running], [0, 0, 1, 2])
        self.assertFalse(list(self.root.glob(".*.tmp")))
        self.assertTrue(all(call["do_sample"] is False for call in calls))

    def test_all_schema_failures_return_failure_and_keep_raw_rows(self):
        code, manifest, _ = self.run_fake(text='{"ok":1,"specificity":0}')
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "failed_local_run")
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 0)
        self.assertEqual(manifest["counts"]["invalid_or_error_outputs"], 1)
        self.assertEqual(json.loads(self.output.read_text())["raw_output"], '{"ok":1,"specificity":0}')

    def test_all_inference_errors_return_failure(self):
        code, manifest, _ = self.run_fake(error=True)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["status"], "failed_local_run")
        self.assertEqual(manifest["counts"]["row_errors"], 1)
        self.assertIn("Synthetic inference failure", json.loads(self.output.read_text())["validation_error"])

    def test_partial_failure_returns_failure_and_retains_both_rows(self):
        self.requests(2)
        code, manifest, _ = self.run_fake(error_on_call=2)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)
        self.assertEqual(manifest["counts"]["processed_this_run"], 2)
        self.assertEqual(len(self.output.read_text().splitlines()), 2)

    def test_valid_abstention_is_not_a_technical_failure(self):
        code, manifest, _ = self.run_fake(text='{"ok":0,"specificity":0}')
        self.assertEqual(code, 0)
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)

    def test_truncation_and_unknown_termination_fail(self):
        for max_tokens in (2, 16):
            with self.subTest(max_tokens=max_tokens):
                self.output.unlink(missing_ok=True)
                code, manifest, _ = self.run_fake(eos=False, max_tokens=max_tokens)
                self.assertEqual(code, 1)
                self.assertEqual(manifest["counts"]["strict_valid_outputs"], 0)
                self.assertEqual(manifest["status"], "failed_local_run")

    def test_model_load_failure_writes_failed_manifest(self):
        code, manifest, calls = self.run_fake(load_error=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(manifest["status"], "failed_local_run")
        self.assertEqual(manifest["counts"]["missing_outputs"], 1)
        self.assertIn("Synthetic model-load failure", manifest["fatal_error"])

    def test_interruption_preserves_checkpoint_and_reports_missing_ids(self):
        self.requests(2)
        code, manifest, _ = self.run_fake(interrupt_on_call=2)
        self.assertEqual(code, 130)
        self.assertEqual(manifest["status"], "interrupted_local_run")
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)
        self.assertEqual(manifest["counts"]["missing_outputs"], 1)
        self.assertEqual(len(self.output.read_text().splitlines()), 1)

    def test_compatible_valid_resume_succeeds_without_regeneration(self):
        self.run_fake()
        original = self.output.read_bytes()
        code, manifest, calls = self.run_fake(resume=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(manifest["counts"]["processed_this_run"], 0)
        self.assertEqual(self.output.read_bytes(), original)

    def test_resumed_error_without_validation_error_is_still_failed(self):
        self.run_fake()
        row = json.loads(self.output.read_text())
        row["status"] = "error"
        row["validation_error"] = None
        self.output.write_text(json.dumps(row) + "\n")
        code, manifest, _ = self.run_fake(resume=True)
        self.assertEqual(code, 1)
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 0)

    def test_empty_scoring_input_stops_before_model_loading(self):
        self.requests(0)
        with self.assertRaises(SystemExit):
            self.run_fake()
        self.assertFalse(self.manifest.exists())

    def test_atomic_writer_failure_retains_old_status(self):
        runner.atomic_json_write(self.manifest, {"status": "running_local_run"})
        before = self.manifest.read_bytes()
        with patch.object(runner.os, "replace", side_effect=OSError("Synthetic replace failure")):
            with self.assertRaises(OSError):
                runner.atomic_json_write(self.manifest, {"status": "completed_local_run"})
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertFalse(list(self.root.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
