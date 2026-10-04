import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import derive_ccts_turns_unique_sequence as derive
from derive_ccts_turns_unique_sequence import build_derived_table


TURN_FIELDS = [
    "event_id",
    "sequence_id",
    "source_event_company_name",
    "text_type",
    "text_name",
    "text_contents",
]
AUDIT_FIELDS = ["event_id", "fetched_rows", "fetch_status", "fetch_notes"]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def turn(event_id: str, sequence_id: str, company: str, text: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "sequence_id": sequence_id,
        "source_event_company_name": company,
        "text_type": "Q&A",
        "text_name": "Jordan Smith - Analyst [1]",
        "text_contents": text,
    }


class DeriveCctsTurnsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_root.name)
        self.input_csv = self.root / "turns.csv"
        self.audit_csv = self.root / "fetch_audit.csv"
        self.output_dir = self.root / "derived_v1"

    def tearDown(self) -> None:
        self.temp_root.cleanup()

    def run_build(
        self, turns: list[dict[str, str]], audit: list[dict[str, str]]
    ) -> dict:
        write_csv(self.input_csv, TURN_FIELDS, turns)
        write_csv(self.audit_csv, AUDIT_FIELDS, audit)
        return build_derived_table(self.input_csv, self.audit_csv, self.output_dir)

    def test_collapses_only_exact_alias_duplicate_and_preserves_aliases(self) -> None:
        summary = self.run_build(
            [
                turn("1", "0", "Northwind Inc", "Opening remarks"),
                turn("2", "0", "Zeta Corp", "Question and answer"),
                turn("2", "0", "Alpha Corp", "Question and answer"),
            ],
            [
                {
                    "event_id": "1",
                    "fetched_rows": "1",
                    "fetch_status": "ok",
                    "fetch_notes": "",
                },
                {
                    "event_id": "2",
                    "fetched_rows": "2",
                    "fetch_status": "review",
                    "fetch_notes": "duplicate_sequence_id",
                },
            ],
        )

        with (self.output_dir / "ccts_turns_analysis_v1.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        merged = rows[1]
        self.assertEqual(merged["source_event_company_name"], "Alpha Corp")
        self.assertEqual(
            json.loads(merged["source_event_company_names_json"]),
            ["Alpha Corp", "Zeta Corp"],
        )
        self.assertEqual(
            merged["deduplication_status"],
            "collapsed_exact_name_alias_duplicate",
        )
        self.assertEqual(summary["input_rows"], 3)
        self.assertEqual(summary["output_rows"], 2)
        self.assertEqual(summary["duplicate_rows_collapsed"], 1)
        self.assertEqual(summary["resolution_audit_rows"], 1)
        self.assertTrue((self.output_dir / "deduplication_summary.json").is_file())

    def test_rejects_duplicate_with_different_transcript_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.run_build(
                [
                    turn("2", "0", "Zeta Corp", "First text"),
                    turn("2", "0", "Alpha Corp", "Different text"),
                ],
                [
                    {
                        "event_id": "2",
                        "fetched_rows": "2",
                        "fetch_status": "review",
                        "fetch_notes": "duplicate_sequence_id",
                    }
                ],
            )
        self.assertFalse(self.output_dir.exists())

    def test_rejects_unflagged_duplicate(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unflagged duplicate"):
            self.run_build(
                [
                    turn("2", "0", "Zeta Corp", "Same text"),
                    turn("2", "0", "Alpha Corp", "Same text"),
                ],
                [
                    {
                        "event_id": "2",
                        "fetched_rows": "2",
                        "fetch_status": "ok",
                        "fetch_notes": "",
                    }
                ],
            )
        self.assertFalse(self.output_dir.exists())

    def test_rejects_review_event_without_duplicate_key(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not match fetch-audit"):
            self.run_build(
                [turn("2", "0", "Alpha Corp", "Single row")],
                [
                    {
                        "event_id": "2",
                        "fetched_rows": "1",
                        "fetch_status": "review",
                        "fetch_notes": "duplicate_sequence_id",
                    }
                ],
            )
        self.assertFalse(self.output_dir.exists())

    def seed_single_row(self):
        write_csv(self.input_csv, TURN_FIELDS, [turn("1", "1", "Example Corp", "Original assertion")])
        write_csv(self.audit_csv, AUDIT_FIELDS,
                  [{"event_id": "1", "fetched_rows": "1", "fetch_status": "ok", "fetch_notes": ""}])

    def test_rejects_duplicate_transcript_columns_before_text_is_overwritten(self):
        self.seed_single_row()
        with self.input_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(TURN_FIELDS + ["text_contents"])
            writer.writerow(["1", "1", "Example Corp", "Q&A", "Example Speaker",
                             "FIRST ORIGINAL ASSERTION", "SECOND DIFFERENT ASSERTION"])
        with self.assertRaisesRegex(ValueError, "duplicate CSV columns"):
            build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_duplicate_audit_columns_before_count_is_overridden(self):
        self.seed_single_row()
        with self.audit_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["event_id", "fetched_rows", "fetched_rows", "fetch_status", "fetch_notes"])
            writer.writerow(["1", "99", "1", "ok", ""])
        with self.assertRaisesRegex(ValueError, "duplicate CSV columns"):
            build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_blank_and_missing_headers(self):
        for kind in ("raw", "audit"):
            for header in ("", "event_id, ,sequence_id\n", "event_id,wrong_column\n"):
                with self.subTest(kind=kind, header=header):
                    self.seed_single_row()
                    (self.input_csv if kind == "raw" else self.audit_csv).write_text(header)
                    with self.assertRaises(ValueError):
                        build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
                    self.assertFalse(self.output_dir.exists())

    def test_audit_change_during_processing_prevents_publication(self):
        self.seed_single_row()
        original = derive.make_event_rows

        def change_audit(*args):
            result = original(*args)
            self.audit_csv.write_text(self.audit_csv.read_text().replace(",1,ok,", ",99,ok,"))
            return result

        with patch.object(derive, "make_event_rows", side_effect=change_audit):
            with self.assertRaisesRegex(ValueError, "Fetch audit CSV changed"):
                build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
        self.assertFalse(self.output_dir.exists())
        self.assertFalse(list(self.root.glob(".derived_v1.tmp-*")))

    def test_raw_change_with_same_size_and_restored_mtime_prevents_publication(self):
        self.seed_single_row()
        before = self.input_csv.stat()
        original = derive.make_event_rows

        def change_raw(*args):
            result = original(*args)
            self.input_csv.write_bytes(self.input_csv.read_bytes().replace(b"Original", b"Modified"))
            os.utime(self.input_csv, ns=(before.st_atime_ns, before.st_mtime_ns))
            return result

        with patch.object(derive, "make_event_rows", side_effect=change_raw):
            with self.assertRaisesRegex(ValueError, "Raw input CSV changed"):
                build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
        self.assertFalse(self.output_dir.exists())

    def test_unchanged_inputs_have_exact_before_and_after_bindings(self):
        self.seed_single_row()
        before = (derive.sha256_file(self.input_csv), derive.sha256_file(self.audit_csv))
        summary = build_derived_table(self.input_csv, self.audit_csv, self.output_dir)
        self.assertEqual((summary["input_sha256"], summary["fetch_audit_sha256"]), before)
        self.assertEqual((derive.sha256_file(self.input_csv), derive.sha256_file(self.audit_csv)), before)
        self.assertEqual(summary["derivation_version"], derive.DERIVATION_VERSION)


if __name__ == "__main__":
    unittest.main()
