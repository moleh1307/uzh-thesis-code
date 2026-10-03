import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_execucomp_turnover_windows as windows


class TurnoverRosterBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tenures = [
            {"gvkey": "001", "execid": "10", "tenure_episode": "1", "person_name": "Old Person",
             "company_name": "Acme", "ceo_start_date": "2017-01-01", "ceo_end_date": "2019-03-31",
             "normalization_status": "exact_tenure_dates"},
            {"gvkey": "001", "execid": "20", "tenure_episode": "1", "person_name": "New Person",
             "company_name": "Acme", "ceo_start_date": "2019-04-01", "ceo_end_date": "2021-12-31",
             "normalization_status": "exact_tenure_dates"},
        ]
        self.events = []
        for tenure, dates in zip(self.tenures, (
            ("2018-01-15", "2018-04-15", "2018-07-15", "2018-10-15", "2019-01-15"),
            ("2019-04-15", "2019-07-15", "2019-10-15", "2020-01-15", "2020-04-15"),
        )):
            for event_date in dates:
                self.events.append({**tenure, "event_id": str(len(self.events) + 1),
                    "ceo_name": tenure["person_name"], "event_date": event_date,
                    "issuer_match_status": "exact_cusip8_single_gvkey",
                    "ceo_coverage_status": "covered_exact_ceo_tenure", "ceo_candidate_count": "1"})

    def run_cli(self):
        for name, rows in (("panel", self.events), ("roster", self.tenures)):
            with (self.root / (name + ".csv")).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        output = self.root / "output"
        cfg = SimpleNamespace(exact_event_panel=self.root / "panel.csv",
            execucomp_tenures=self.root / "roster.csv", output_dir=output,
            max_call_gap_days=120, audit_size=100, force=False)
        with patch.object(windows, "args", return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            windows.main()
        return json.loads((output / "ceo_longitudinal_turnover_summary.json").read_text())

    def test_consistent_five_quarter_pair_is_ready(self):
        self.assertEqual(self.run_cli()["counts"]["anchor_turnover_ready"], 1)

    def test_old_after_end_and_new_before_start_reject_before_output(self):
        for index, field, value in ((0, "ceo_end_date", "2018-12-31"),
                                    (1, "ceo_start_date", "2019-05-01")):
            original = deepcopy(self.tenures)
            self.tenures[index][field] = value
            with self.assertRaisesRegex(ValueError, "boundary differs"):
                self.run_cli()
            self.assertFalse((self.root / "output").exists())
            self.tenures = original

    def test_internal_call_with_changed_panel_boundaries_cannot_count(self):
        self.events[2]["ceo_start_date"] = "2017-02-01"
        with self.assertRaisesRegex(ValueError, "boundary differs"):
            self.run_cli()
        self.assertFalse((self.root / "output").exists())

    def test_internal_call_date_outside_matching_boundaries_rejects(self):
        self.events[2]["event_date"] = "2016-07-15"
        with self.assertRaisesRegex(ValueError, "outside its roster tenure"):
            self.run_cli()

    def test_missing_duplicate_and_nonexact_roster_episodes_reject(self):
        for change, message in (
            (lambda rows: rows.pop(), "no matching roster episode"),
            (lambda rows: rows.append(dict(rows[0])), "duplicate roster episode"),
            (lambda rows: rows[0].update(normalization_status="annual_flag_date_bounds"), "nonexact roster episode"),
        ):
            rows = deepcopy(self.tenures)
            change(rows)
            with self.assertRaisesRegex(ValueError, message):
                windows.validate_panel_against_roster(self.events, rows)

    def test_duplicate_calls_cannot_inflate_support(self):
        self.events.append(dict(self.events[0]))
        with self.assertRaisesRegex(ValueError, "duplicate exact-panel event"):
            self.run_cli()

    def test_unobserved_nonexact_adjacent_episode_remains_review(self):
        self.tenures.insert(1, {**self.tenures[0], "execid": "15", "person_name": "Interim Person",
            "ceo_start_date": "2019-03-15", "ceo_end_date": "2019-04-15",
            "normalization_status": "annual_flag_date_bounds"})
        result = self.run_cli()
        self.assertEqual(result["counts"]["anchor_turnover_ready"], 0)
        self.assertEqual(result["counts"]["turnover_status"], {"review_nonexact_adjacent_tenure": 2})


if __name__ == "__main__":
    unittest.main()
