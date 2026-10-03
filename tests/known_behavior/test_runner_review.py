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

if __name__ == "__main__":
    unittest.main(verbosity=2)
