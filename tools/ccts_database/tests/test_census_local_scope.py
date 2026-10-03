import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_ccts_event_level_census as census


class LocalScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "metadata.csv"
        self.rows = []
        for event, year, kind, selected in [(5, 2010, census.STANDARD_EARNING_TYPE, 1),
                (2, 1999, census.STANDARD_EARNING_TYPE, 1), (3, 2012, "Other", 1),
                (4, 2008, census.STANDARD_EARNING_TYPE, 0), (1, 2005, census.STANDARD_EARNING_TYPE, 1)]:
            row = dict.fromkeys(census.METADATA_COLUMNS, "")
            row.update(event_id=str(event), year=str(year), start_date=f"{year}-01-01 00:00:00",
                       event_type_name=kind, selected=str(selected))
            self.rows.append(row)
        self.write()

    def write(self):
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[*census.METADATA_COLUMNS, "selected"])
            writer.writeheader()
            writer.writerows(self.rows)

    def test_local_and_database_id_only_scope_agree_and_limit_after_sort(self):
        counts = {}
        local = census.read_filtered_metadata_rows_from_csv(self.path, "selected", "__truthy__", 1, 2001, counts)
        fixture = self.rows
        class Cursor:
            def execute(self, sql, params):
                ids, kind, min_year = params
                self.result = [tuple(int(row[col]) if col in {"event_id", "year"} else row[col]
                                     for col in census.METADATA_COLUMNS) for row in fixture
                               if int(row["event_id"]) in ids and row["event_type_name"] == kind
                               and int(row["year"]) >= min_year]
            def fetchall(self):
                return self.result
        ids = census.read_event_id_filter(self.path, "selected", "__truthy__")
        database = census.fetch_metadata_for_event_ids(Cursor(), 2001, sorted(ids), 1, 10)
        self.assertEqual([int(row[0]) for row in local], [row[0] for row in database])
        self.assertEqual([int(row[0]) for row in local], [1])
        self.assertEqual(counts, dict(input_rows=5, excluded_by_event_id_filter=1,
            excluded_nonstandard_event_type=1, excluded_before_min_year=1,
            eligible_before_limit=2, excluded_by_limit=1))

    def test_nondefault_year_filter_and_no_eligible_support(self):
        self.assertEqual([int(row[0]) for row in census.read_filtered_metadata_rows_from_csv(
            self.path, None, "__truthy__", None, 2009)], [5])
        self.assertEqual(census.read_filtered_metadata_rows_from_csv(self.path, None, "__truthy__", None, 2020), [])

    def test_invalid_date_year_and_duplicate_eligible_identity_rejected(self):
        for field, value in [("year", "2009"), ("start_date", "corrupted")]:
            original = self.rows[0][field]
            self.rows[0][field] = value
            self.write()
            with self.assertRaisesRegex(ValueError, "date/year"):
                census.read_filtered_metadata_rows_from_csv(self.path, None, "__truthy__", None)
            self.rows[0][field] = original
        self.rows.append(dict(self.rows[0]))
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            census.read_filtered_metadata_rows_from_csv(self.path, None, "__truthy__", None)

    def test_main_reports_actual_local_scope_and_refuses_old_selector_binding(self):
        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, *args): pass
            def fetchone(self): return (5,)
        class Connection:
            def cursor(self): return Cursor()
            def close(self): pass
        captured = []
        def count_stub(**kwargs):
            captured.append(kwargs["event_ids"])
            columns = kwargs["columns"]
            with kwargs["output_path"].open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for event in kwargs["event_ids"]:
                    writer.writerow(dict(event_id=event, **{field: "" if field == "text_types" else "0" for field in columns[1:]}))
        output = self.path.parent / "run"
        argv = ["census", "--event-id-csv", str(self.path), "--event-id-filter-column", "selected",
                "--min-year", "2009", "--section-only", "--output-dir", str(output)]
        creds = dict(host="fixture", dbname="fixture", user="reader", password="unused")
        with patch.object(sys, "argv", argv), patch.object(census, "read_credentials", return_value=creds), \
             patch.object(census.psycopg2, "connect", return_value=Connection()), \
             patch.object(census, "count_rows_for_events", side_effect=count_stub), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(census.main(), 0)
        self.assertEqual(captured, [[5], [5]])
        summary = json.loads((output / "event_level_census_summary.json").read_text())
        self.assertEqual(summary["scope"]["events_in_census"], 1)
        self.assertEqual(summary["scope"]["local_metadata_selection_counts"]["excluded_before_min_year"], 2)
        with (output / "event_level_section_census.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["event_id"] for row in rows], ["5"])
        config = output / "census_run_config.json"
        payload = json.loads(config.read_text())
        payload["binding"]["runner_sha256"] = "previous-selector-code"
        config.write_text(json.dumps(payload))
        before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        with patch.object(sys, "argv", argv), patch.object(census, "read_credentials", return_value=creds), \
             patch.object(census.psycopg2, "connect") as connect, self.assertRaises(ValueError):
            census.main()
        connect.assert_not_called()
        self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
