"""Remaining runner defects, using fake inference rather than a real model."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runner_fixtures import FakeRunnerCase


class RunnerReview(FakeRunnerCase):
    def test_characterize_resume_skips_failed_rows(self):
        self.run_fake(error=True)
        code, manifest, calls = self.run_fake(resume=True)
        self.assertEqual(calls, [])
        self.assertEqual(manifest["counts"]["row_errors"], 1)
        self.assertEqual(code, 1)  # Reporting is fixed; technical retry remains unresolved.

    def test_characterize_resume_accepts_runtime_drift(self):
        self.run_fake()
        _, manifest, calls = self.run_fake(resume=True, version="changed-runtime-fixture")
        self.assertEqual(calls, [])
        self.assertEqual(manifest["runtime"]["torch"], "changed-runtime-fixture")
        self.assertNotIn("runtime", manifest["run_binding"])

    def test_characterize_special_prefix_removed_before_validation(self):
        _, manifest, _ = self.run_fake(special_prefix=True)
        row = json.loads(self.output.read_text())
        self.assertTrue(row["raw_output_with_special_tokens"].startswith("<|im_start|>"))
        self.assertEqual(row["raw_output"], '{"ok":1,"specificity":4}')
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
