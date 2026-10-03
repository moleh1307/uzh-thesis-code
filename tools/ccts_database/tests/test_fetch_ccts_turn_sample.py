from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import fetch_ccts_turn_sample as fetcher


class FakeCursor:
    def __init__(self, counts, rows):
        self.counts = counts
        self.rows = rows
        self.result = []
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        if "COUNT(*)" in sql:
            self.result = [
                (event_id, values[0], values[1], values[2])
                for event_id, values in self.counts.items()
                if event_id in params[0]
            ]
        elif "text_contents" in sql:
            self.result = [row for row in self.rows if row[0] in params[0]]
        else:
            self.result = []

    def fetchall(self):
        return self.result


class FetchCctsTurnSampleTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"event_id": "101"},
            {"event_id": "102"},
        ]
        self.rows = [
            (101, 1, "Acme Corp", "PRE", "Operator", "Prepared remarks"),
            (101, 2, "Acme Corp", "Q&A", "Analyst", "Question and answer"),
        ]
        self.cursor = FakeCursor({101: (2, 1, 2)}, self.rows)

    def test_id_only_manifest_fetches_and_audits_without_false_zero_mismatch(self):
        turns, audits = fetcher.fetch_batch(self.cursor, self.events)

        self.assertEqual(len(self.cursor.calls), 2)
        self.assertEqual(len(turns), 2)
        self.assertEqual([row["event_id"] for row in audits], [101, 102])
        self.assertEqual(audits[0]["fetch_status"], "ok")
        self.assertEqual(audits[0]["expected_pre_rows"], "")
        self.assertEqual(audits[0]["expected_qa_rows"], "")
        self.assertEqual(audits[1]["fetch_status"], "review")
        self.assertEqual(audits[1]["fetch_notes"], "no_database_text_rows")

    def test_optional_manifest_counts_are_still_checked(self):
        event = {"event_id": "101", "pre_rows": "2", "qa_rows": "1"}
        turns, audits = fetcher.fetch_batch(self.cursor, [event])

        self.assertEqual(len(turns), 2)
        self.assertEqual(audits[0]["fetch_status"], "review")
        self.assertIn("pre_count_mismatch", audits[0]["fetch_notes"])

    def test_manifest_rejects_duplicates_and_assigns_ranks(self):
        normalized = fetcher.normalize_manifest([{"event_id": "101"}], None)
        self.assertEqual(normalized[0]["selection_rank"], "1")
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            fetcher.normalize_manifest([{"event_id": "101"}, {"event_id": "101"}], None)

    def test_checkpoint_is_resumable_and_checksum_verified(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            turns, audits = fetcher.fetch_batch(self.cursor, [self.events[0]])
            key = fetcher.checkpoint_key(["101"])
            fetcher.save_checkpoint(checkpoint_dir, key, ["101"], turns, audits)

            self.assertIsNotNone(fetcher.read_valid_checkpoint(checkpoint_dir, key, {"101"}))
            turns_path, _, _ = fetcher.checkpoint_paths(checkpoint_dir, key)
            with turns_path.open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            self.assertIsNone(fetcher.read_valid_checkpoint(checkpoint_dir, key, {"101"}))

    def test_materialized_csv_contains_only_valid_completed_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            turns, audits = fetcher.fetch_batch(self.cursor, [self.events[0]])
            key = fetcher.checkpoint_key(["101"])
            fetcher.save_checkpoint(checkpoint_dir, key, ["101"], turns, audits)
            checkpoints = fetcher.load_valid_checkpoints(checkpoint_dir, ["101", "102"])
            output = root / "ccts_turns.csv"

            count = fetcher.materialize_csv(
                output, fetcher.TURN_COLUMNS, checkpoint_dir, checkpoints, "turns"
            )

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(count, 2)
            self.assertEqual([row["event_id"] for row in rows], ["101", "101"])

    def test_materializer_accepts_long_transcript_text_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            body = "x" * 150_000
            turns = [{"event_id": 101, "sequence_id": 1, "text_contents": body}]
            audits = [{"event_id": 101, "fetch_status": "ok"}]
            key = fetcher.checkpoint_key(["101"])
            fetcher.save_checkpoint(checkpoint_dir, key, ["101"], turns, audits)
            checkpoints = fetcher.load_valid_checkpoints(checkpoint_dir, ["101"])
            output = root / "ccts_turns.csv"

            fetcher.materialize_csv(
                output, fetcher.TURN_COLUMNS, checkpoint_dir, checkpoints, "turns"
            )

            with output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows[0]["text_contents"]), len(body))

    def test_completed_checkpoint_survives_a_smaller_resume_batch_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.csv"
            manifest_path.write_text("event_id\n101\n102\n103\n", encoding="utf-8")
            manifest = fetcher.normalize_manifest(fetcher.read_csv(manifest_path), None)
            output_dir = root / "output"
            checkpoint_dir, _ = fetcher.prepare_run_config(output_dir, manifest_path, manifest)
            turns, audits = fetcher.fetch_batch(self.cursor, manifest[:2])
            ids = [event["event_id"] for event in manifest[:2]]
            fetcher.save_checkpoint(checkpoint_dir, fetcher.checkpoint_key(ids), ids, turns, audits)

            checkpoints = fetcher.load_valid_checkpoints(
                checkpoint_dir, [event["event_id"] for event in manifest]
            )
            completed = {
                event_id
                for checkpoint in checkpoints
                for event_id in checkpoint["event_ids"]
            }
            remaining = [event for event in manifest if event["event_id"] not in completed]

            self.assertEqual(len(checkpoints), 1)
            self.assertEqual([event["event_id"] for event in remaining], ["103"])
            self.assertEqual([len(batch) for batch in fetcher.chunks(remaining, 1)], [1])

            smoke_summary = fetcher.write_outputs(
                output_dir,
                manifest_path,
                manifest,
                checkpoint_dir,
                ["101", "102"],
            )
            self.assertEqual(smoke_summary["status"], "complete")
            self.assertTrue(smoke_summary["limited_scope"])
            self.assertEqual(smoke_summary["events_in_output"], 2)

            full_summary = fetcher.write_outputs(
                output_dir,
                manifest_path,
                manifest,
                checkpoint_dir,
                ["101", "102", "103"],
            )
            self.assertEqual(full_summary["status"], "partial")
            self.assertFalse(full_summary["limited_scope"])
            self.assertEqual(full_summary["events_complete"], 2)


if __name__ == "__main__":
    unittest.main()
