import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_ccts_event_master as master


class EventMasterPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.census = self.root / "census.csv"
        self.source("1")

    def source(self, event_id):
        row = dict.fromkeys(master.BASE_COLUMNS, "")
        row.update(event_id=event_id, year="2015", total_text_rows="2", pre_rows="1", qa_rows="1",
                   has_both_pre_and_qa="1", start_date="2015-01-01")
        with self.census.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=master.BASE_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

    def test_relative_and_absolute_latest_targets_resolve(self):
        cwd = Path.cwd()
        try:
            os.chdir(self.root)
            for index, root in enumerate((Path("data/master"), self.root / "absolute")):
                result = master.build_master(self.census, root, f"run{index}")
                latest = root / "latest"
                self.assertTrue(latest.exists())
                self.assertEqual(latest.resolve(), (root / f"run{index}").resolve())
                self.assertTrue(Path(result["artifacts"]["master_csv"]).is_file())
                self.assertNotIn(".stage-", str(result))
        finally:
            os.chdir(cwd)

    def test_existing_run_rejected_byte_identically(self):
        root = self.root / "master"
        master.build_master(self.census, root, "same")
        before = {p.name: p.read_bytes() for p in (root / "same").iterdir()}
        latest = os.readlink(root / "latest")
        self.source("2")
        with self.assertRaises(FileExistsError):
            master.build_master(self.census, root, "same")
        self.assertEqual(before, {p.name: p.read_bytes() for p in (root / "same").iterdir()})
        self.assertEqual(os.readlink(root / "latest"), latest)

    def test_failure_after_first_stage_write_never_publishes(self):
        root = self.root / "master"
        master.build_master(self.census, root, "good")
        with patch.object(master, "write_by_year", side_effect=OSError("synthetic write failure")):
            with self.assertRaises(OSError):
                master.build_master(self.census, root, "failed")
        self.assertFalse((root / "failed").exists())
        self.assertEqual((root / "latest").resolve(), (root / "good").resolve())
        self.assertEqual(sorted(p.name for p in root.iterdir()), ["good", "latest"])

    def test_real_latest_path_is_preserved(self):
        root = self.root / "master"
        latest = root / "latest"
        latest.mkdir(parents=True)
        (latest / "keep").write_text("saved evidence")
        with self.assertRaises(SystemExit):
            master.build_master(self.census, root, "new")
        self.assertEqual((latest / "keep").read_text(), "saved evidence")
        self.assertFalse((root / "new").exists())

    def test_empty_or_reserved_run_paths_cannot_be_reused(self):
        root = self.root / "master"
        (root / "empty").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            master.build_master(self.census, root, "empty")
        for name in ("../escape", ".", "latest", "/absolute"):
            with self.assertRaises(ValueError):
                master.build_master(self.census, root, name)


if __name__ == "__main__":
    unittest.main()
