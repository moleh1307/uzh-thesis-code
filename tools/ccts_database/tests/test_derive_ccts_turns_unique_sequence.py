import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


if __name__ == "__main__":
    unittest.main()
