import json
import unittest

from runner_fixtures import FakeRunnerCase
from specificity_validation import reconcile_attempts, technical_retry_reason


class TechnicalRetryTests(FakeRunnerCase):
    def rows(self):
        return [json.loads(line) for line in self.output.read_text().splitlines()]

    def test_resume_does_not_retry_without_explicit_flag(self):
        self.run_fake(error=True)
        before = self.output.read_bytes()
        code, manifest, calls = self.run_fake(resume=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(self.output.read_bytes(), before)
        self.assertEqual(manifest["counts"]["row_errors"], 1)

    def test_failed_to_valid_retains_original_evidence(self):
        self.run_fake(error=True)
        first = self.output.read_bytes()
        code, manifest, calls = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.output.read_bytes().startswith(first))
        self.assertEqual([r["attempt_number"] for r in self.rows()], [1, 2])
        self.assertEqual(self.rows()[1]["retry_reason"], "inference_error")
        self.assertEqual(manifest["counts"]["requested"], 1)
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)
        self.assertEqual(manifest["counts"]["row_errors"], 0)
        self.assertEqual(manifest["retried_ids"], ["FIXTURE_0"])

    def test_second_failure_and_no_third_attempt(self):
        self.run_fake(error=True)
        code, _, calls = self.run_fake(resume=True, technical_retry=True, error=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        first_two = self.output.read_bytes()
        code, manifest, calls = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(self.output.read_bytes(), first_two)
        self.assertEqual(manifest["counts"]["row_errors"], 1)

    def test_successful_ids_and_low_scores_never_regenerated(self):
        self.requests(2)
        self.run_fake(text='{"ok":1,"specificity":1}', error_on_call=2)
        first = self.rows()[0]
        code, manifest, calls = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.rows()[0], first)
        self.assertEqual(self.rows()[-1]["custom_id"], "FIXTURE_1")
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 2)

    def test_no_failures_is_optional_noop(self):
        self.run_fake()
        before = self.output.read_bytes()
        code, _, calls = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertEqual(self.output.read_bytes(), before)

    def test_schema_failure_can_recover_once(self):
        self.run_fake(text='{"ok":1,"specificity":0}')
        code, _, _ = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 0)
        self.assertEqual(self.rows()[-1]["retry_reason"], "output_validation_failure")

    def test_truncation_can_recover_once(self):
        self.run_fake(eos=False, max_tokens=2)
        code, _, _ = self.run_fake(resume=True, technical_retry=True, max_tokens=2)
        self.assertEqual(code, 0)
        self.assertEqual(self.rows()[-1]["retry_reason"], "output_truncation")

    def test_runtime_drift_rejects_retry_before_writes(self):
        self.run_fake(error=True)
        before = self.output.read_bytes(), self.manifest.read_bytes()
        with self.assertRaises(SystemExit):
            self.run_fake(resume=True, technical_retry=True, version="changed")
        self.assertEqual((self.output.read_bytes(), self.manifest.read_bytes()), before)

    def test_retry_requires_resume(self):
        with self.assertRaisesRegex(SystemExit, "requires --resume"):
            self.run_fake(technical_retry=True)
        self.assertFalse(self.output.exists())

    def test_missing_ids_must_be_finished_by_resume_first(self):
        self.requests(2)
        self.run_fake(error=True, interrupt_on_call=2)
        before = self.output.read_bytes(), self.manifest.read_bytes()
        with self.assertRaisesRegex(SystemExit, "finish missing first attempts"):
            self.run_fake(resume=True, technical_retry=True)
        self.assertEqual((self.output.read_bytes(), self.manifest.read_bytes()), before)
        code, manifest, calls = self.run_fake(resume=True)
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(manifest["counts"]["missing_outputs"], 0)

    def test_interrupted_retry_reservation_prevents_another_invocation(self):
        self.run_fake(error=True)
        code, manifest, calls = self.run_fake(resume=True, technical_retry=True, interrupt_on_call=1)
        self.assertEqual(code, 130)
        self.assertEqual(len(calls), 1)
        self.assertEqual(manifest["reserved_retries_without_result"], ["FIXTURE_0"])
        code, _, calls = self.run_fake(resume=True, technical_retry=True)
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertEqual(len(self.rows()), 1)

    def test_context_rejection_is_not_eligible_for_same_settings_retry(self):
        self.assertIsNone(technical_retry_reason({"status": "error", "input_context_rejected": True}))

    def test_reconciliation_rejects_success_retry_duplicate_and_third_attempt(self):
        self.run_fake()
        first = self.rows()[0]
        for row in (dict(first), dict(first, attempt_number=2, retry_reason="output_validation_failure"),
                    dict(first, attempt_number=3)):
            with self.assertRaises(ValueError):
                reconcile_attempts([first, row])


if __name__ == "__main__":
    unittest.main()
