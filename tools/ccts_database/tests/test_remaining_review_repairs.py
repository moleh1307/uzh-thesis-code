"""Synthetic regressions for review issues 9-13 and historical issue 15."""
import csv
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import census_checkpoint as cp
import build_ccts_event_level_census as census
import build_ccts_execucomp_coverage as coverage
import fetch_ccts_turn_sample as fetcher
import derive_ccts_turns_unique_sequence as derive
import build_ccts_execucomp_speaker_gate as gate
import validate_ccts_execucomp_speaker_identity as speaker


def write(path, rows, columns=None):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cli(script, *args):
    return subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                          capture_output=True, text=True, timeout=30, cwd="/tmp")


class CoverageTests(unittest.TestCase):
    def test_issuer_ambiguity_and_multiple_ceos_stay_out_of_exact_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tenures = []
            def tenure(cusip, gvkey, execid, start="2000-01-01", end="2010-12-31"):
                tenures.append(dict(cusip=cusip, gvkey=gvkey, execid=execid,
                    ceo_start_date=start, ceo_end_date=end, normalization_status="exact_tenure_dates"))
            tenure("11111111", "1", "10")
            tenure("11111111", "2", "20", "2012-01-01", "2013-12-31")
            tenure("22222222", "3", "30")
            tenure("22222222", "3", "31")
            tenure("33333333", "4", "40")
            tenure("44444444", "", "50")
            write(root / "tenures.csv", tenures)
            write(root / "events.csv", [dict(event_id=str(i), start_date="2005-01-01", cusip=key,
                has_both_pre_and_qa="1") for i, key in enumerate(
                ["11111111", "22222222", "33333333", "", "99999999", "44444444"], 1)])
            result = cli("build_ccts_execucomp_coverage.py", "--ccts-census", root / "events.csv",
                         "--execucomp-tenures", root / "tenures.csv", "--output-dir", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            exact = read(root / "out/ccts_execucomp_exact_ceo_coverage.csv")
            review = read(root / "out/ccts_execucomp_match_review.csv")
            self.assertEqual([row["event_id"] for row in exact], ["3"])
            self.assertEqual([row["event_id"] for row in review], ["1", "2", "6"])
            self.assertEqual(review[0]["ceo_candidate_count"], "1")
            self.assertEqual(review[0]["match_confidence"], "review")
            self.assertFalse(coverage.is_exact_panel_row(review[0]))
            tampered = dict(exact[0], issuer_match_status="ambiguous_cusip8_multiple_gvkeys")
            write(root / "tampered.csv", [tampered])
            result = cli("build_execucomp_turnover_windows.py", "--exact-event-panel", root / "tampered.csv",
                         "--execucomp-tenures", root / "tenures.csv", "--output-dir", root / "windows")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not an unambiguous", result.stderr)
            self.assertFalse((root / "windows/ceo_call_sequences.csv").exists())


class NormalizationTests(unittest.TestCase):
    def normalize(self, rows):
        root = Path(self.directory.name)
        write(root / "raw.csv", rows)
        result = cli("normalize_execucomp_ceo_tenure.py", "--input", root / "raw.csv",
                     "--output-dir", root / "out")
        self.assertEqual(result.returncode, 0, result.stderr)
        return read(root / "out/execucomp_ceo_tenure_normalized.csv"), read(root / "out/execucomp_ceo_tenure_normalization_audit.csv")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def row(self, year, start="2000-01-01", end="2001-12-31", execid="1"):
        return dict(gvkey="1", execid=execid, year=str(year), ceoann="CEO",
                    exec_fullname="Synthetic Person", becameceo=start, leftofc=end)

    def test_equivalent_dates_are_exact_and_raw_evidence_survives(self):
        rows, audit = self.normalize([self.row(2000), self.row(2001, "01/01/2000", "12/31/2001")])
        self.assertEqual(rows[0]["normalization_status"], "exact_tenure_dates")
        self.assertIn("01/01/2000", audit[0]["becameceo_values"])

    def test_conflicting_boundaries_are_review_not_min_max_resolution(self):
        rows, audit = self.normalize([self.row(2000), self.row(2001, "2000-02-01", "2001-11-30")])
        self.assertEqual(rows[0]["normalization_status"], "source_boundary_conflict_review")
        self.assertIn("2000-02-01", audit[0]["becameceo_values"])
        self.assertNotEqual(rows[0]["start_date_source"], "becameceo_exact")

    def test_missing_boundaries_remain_annual_review(self):
        rows, _ = self.normalize([self.row(2000, "", "")])
        self.assertEqual(rows[0]["normalization_status"], "annual_flag_date_bounds")

    def test_returning_executive_episodes_do_not_pool_dates(self):
        rows, audit = self.normalize([self.row(2000, end="2000-12-31"),
            self.row(2005, "2005-01-01", "2005-12-31")])
        self.assertEqual([row["ceo_start_date"] for row in rows], ["2000-01-01", "2005-01-01"])
        self.assertTrue(all(row["normalization_status"] == "exact_tenure_dates" for row in rows))
        self.assertNotIn("2000", audit[1]["becameceo_values"])

    def test_invalid_boundary_is_not_dropped_into_exact_interval(self):
        rows, _ = self.normalize([self.row(2000), self.row(2001, "not-a-date")])
        self.assertEqual(rows[0]["normalization_status"], "source_boundary_conflict_review")


class CensusCheckpointTests(unittest.TestCase):
    columns = ["event_id", "total_text_rows", "text_types"]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "counts.csv"

    def store(self, columns=None, ids=None, source=None):
        return cp.CountCheckpoint(self.output, columns or self.columns,
                                  ids or [1, 2, 3], source or {"mode": "full"})

    def test_compatible_resume_and_measured_zero(self):
        self.store().commit([1], {1: {"total_text_rows": 2, "text_types": "PRE"}})
        resumed = self.store()
        resumed.commit([2, 3], {})
        resumed.materialize()
        self.assertEqual(list(self.store().counts), [1, 2, 3])
        self.assertEqual(read(self.output)[1]["total_text_rows"], "0")

    def test_incompatible_mode_scope_schema_or_source_preserves_bytes(self):
        self.store().commit([1], {})
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        for kwargs in ({"source": {"mode": "section"}}, {"ids": [1, 2]},
                       {"columns": ["event_id", "different"]}, {"source": {"source": "other"}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.store(**kwargs)
        self.assertEqual(original, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_unbound_legacy_csv_is_rejected_without_modification(self):
        self.output.write_text("event_id,total_text_rows\n1,2\n")
        before = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "unbound historical"):
            self.store()
        self.assertEqual(self.output.read_bytes(), before)

    def test_malformed_duplicate_and_noninteger_rows_are_rejected(self):
        good = {"event_id": "1", "total_text_rows": "2", "text_types": "PRE"}
        for rows in ([good, good], [{**good, "total_text_rows": None}],
                     [{**good, "total_text_rows": ""}], [{**good, "total_text_rows": 1.5}],
                     [{**good, "total_text_rows": -1}], [{**good, "event_id": "9"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                cp.validate_rows(rows, self.columns, {1, 2, 3})
        self.output.write_text("event_id,total_text_rows\n1,2\n")
        with self.assertRaisesRegex(ValueError, "schema"):
            cp.read_counts(self.output, self.columns, {1})

    def test_interrupted_uncommitted_shard_is_refetched_not_counted(self):
        store = self.store()
        with patch.object(cp, "atomic_json", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                store.commit([1], {})
        self.assertEqual(self.store().counts, {})
        self.store().commit([1], {})
        self.assertEqual(set(self.store().counts), {1})

    def test_committed_hash_corruption_is_rejected(self):
        store = self.store()
        store.commit([1], {})
        next(store.shards.glob("*.csv")).write_text("damaged")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.store()

    def test_missing_query_cell_or_out_of_batch_id_is_not_zero(self):
        store = self.store()
        for values in ({1: {"total_text_rows": 1}}, {9: {"total_text_rows": 1, "text_types": "PRE"}}):
            with self.assertRaises(ValueError):
                store.commit([1], values)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            store.materialize()
        self.assertFalse(self.output.exists())

    def test_run_binding_drift_is_rejected_before_writing(self):
        path = self.root / "run.json"
        cp.prepare_run(path, {"mode": "full", "scope": [1]}, [])
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            cp.prepare_run(path, {"mode": "section", "scope": [1]}, [])
        self.assertEqual(path.read_bytes(), before)

    def test_metadata_publication_can_recover_after_config_commit(self):
        stage, path, config_path = self.root / "metadata.tmp", self.root / "metadata.csv", self.root / "run.json"
        stage.write_text("event_id\n1\n")
        config = {"binding": {"source": "synthetic"}, "metadata_sha256": None}
        with patch.object(cp, "recover_metadata", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                cp.publish_metadata(path, stage, config_path, config)
        saved = json.loads(config_path.read_text())
        cp.recover_metadata(path, stage, config_path, saved)
        self.assertEqual(path.read_text(), "event_id\n1\n")
        self.assertFalse(saved["metadata_publish_pending"])

    def test_section_year_diagnostic_is_blank_not_zero(self):
        from collections import Counter
        path = self.root / "year.csv"
        census.write_by_year(path, {2000: Counter(events=1)}, section_only=True)
        self.assertEqual(read(path)[0]["possible_qa_dialogue_encoded_as_pre"], "")

    def test_final_table_rejects_missing_stage_before_replacing_output(self):
        self.output.write_text("previous evidence")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            census.build_final_table([[1]], {}, {}, self.output, section_only=False)
        self.assertEqual(self.output.read_text(), "previous evidence")

    def test_census_cli_section_resume_and_full_mode_rejection_before_connect(self):
        class Cursor:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, *args): pass
            def fetchone(self): return (1,)
        class Connection:
            def cursor(self): return Cursor()
            def close(self): pass
        metadata = [[1, "2000-01-01", 2000] + [""] * (len(census.METADATA_COLUMNS) - 3)]
        def counts(cur, batch):
            return {event: {column: "PRE;Q&A" if column == "text_types" else 1
                for column in census.TEXT_SECTION_COUNT_COLUMNS[1:]} for event in batch}
        argv = [str(SCRIPTS / "build_ccts_event_level_census.py"), "--output-dir", str(self.root / "run"),
                "--section-only", "--min-year", "2000"]
        with patch.object(sys, "argv", argv), patch.object(census, "read_credentials", return_value={
                "host": "synthetic.invalid", "dbname": "synthetic", "user": "synthetic", "password": "unused"}), \
             patch.object(census.psycopg2, "connect", return_value=Connection()) as connect, \
             patch.object(census, "fetch_metadata", return_value=metadata), \
             patch.object(census, "query_text_section_counts", side_effect=counts) as query, \
             patch.object(census, "query_participant_section_counts", return_value={1: {"participant_rows": 1}}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(census.main(), 0)
            self.assertEqual(census.main(), 0)
            self.assertEqual(query.call_count, 1)
            connect.reset_mock()
            with patch.object(sys, "argv", [item for item in argv if item != "--section-only"]):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    census.main()
            connect.assert_not_called()
        summary = json.loads((self.root / "run/event_level_census_summary.json").read_text())
        self.assertIsNone(summary["percentages"]["pct_possible_qa_dialogue_encoded_as_pre"])
        self.assertNotIn("text_name_ceo_signal_events", summary["counts"])


class FragmentedFetchTests(unittest.TestCase):
    def test_fragmented_resume_is_byte_equal_to_uninterrupted_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            order = list(map(str, range(1, 7)))
            for name, batches in (("fragmented", [["3", "4"], ["1", "2", "5", "6"]]),
                                  ("full", [order])):
                checkpoint_dir = root / name
                for ids in batches:
                    turns = [dict(event_id=event, sequence_id=sequence, text_contents=f"{event}:{sequence}",
                                  source_event_company_name="Synthetic", text_type="PRE" if sequence == 1 else "Q&A",
                                  analysis_text_type="PRE" if sequence == 1 else "Q&A",
                                  text_name="Alice Example, Synthetic - CEO")
                             for event in ids for sequence in [1, 2]]
                    audits = [dict(event_id=event, fetched_rows=2, fetch_status="ok", fetch_notes="") for event in ids]
                    fetcher.save_checkpoint(checkpoint_dir, fetcher.checkpoint_key(ids), ids, turns, audits)
                shards = fetcher.load_valid_checkpoints(checkpoint_dir, order)
                for kind, columns in (("turns", fetcher.TURN_COLUMNS), ("audit", fetcher.AUDIT_COLUMNS)):
                    fetcher.materialize_csv(root / f"{name}_{kind}.csv", columns, checkpoint_dir, shards, kind, order)
            for kind in ("turns", "audit"):
                self.assertEqual((root / f"fragmented_{kind}.csv").read_bytes(), (root / f"full_{kind}.csv").read_bytes())
            events = [row["event_id"] for row in read(root / "fragmented_turns.csv")]
            self.assertEqual(events, [event for event in order for _ in range(2)])
            assignments = {event: dict(event_id=event, gvkey="1", execid="10", tenure_episode="1",
                event_date="2000-01-01", calendar_quarter="2000Q1", turnover_ids="T") for event in order}
            gated = []
            for name in ("fragmented", "full"):
                derive.build_derived_table(root / f"{name}_turns.csv", root / f"{name}_audit.csv", root / f"{name}_derived")
                rows, scanned = gate.scan_turns(root / f"{name}_derived/ccts_turns_analysis_v1.csv", order,
                                               assignments, {("1", "10", "1"): "Alice Example"})
                self.assertEqual(scanned, 12)
                self.assertTrue(all(row["event_speaker_gate_pass"] == 1 for row in rows))
                gated.append(rows)
            self.assertEqual(gated[0], gated[1])

    def test_invalidated_middle_shard_and_changed_batches_keep_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            order = ["1", "2", "3", "4", "5", "6"]
            def save(ids):
                fetcher.save_checkpoint(root, fetcher.checkpoint_key(ids), ids,
                    [dict(event_id=event, sequence_id=1, text_contents=event) for event in ids],
                    [dict(event_id=event, fetch_status="ok") for event in ids])
            for ids in (["1", "2"], ["3", "4"], ["5", "6"]):
                save(ids)
            turns, _, _ = fetcher.checkpoint_paths(root, fetcher.checkpoint_key(["3", "4"]))
            turns.write_text("damaged")
            shards = fetcher.load_valid_checkpoints(root, order)
            self.assertEqual(len(shards), 2)
            save(["3"])
            save(["4"])
            shards = fetcher.load_valid_checkpoints(root, order)
            fetcher.materialize_csv(root / "out.csv", fetcher.TURN_COLUMNS, root, shards, "turns", order)
            self.assertEqual([row["event_id"] for row in read(root / "out.csv")], order)


class HistoricalPairTests(unittest.TestCase):
    def row(self, side="old", event="1", execid="10", name="Alice Example", date="2000-01-01"):
        return dict(turnover_side=side, event_id=event, expected_execid=execid, expected_ceo_name=name,
            event_date=date, external_speaker_validation_status="confirmed_external_ceo_shared_pre_qa")

    def test_both_distinct_confirmed_ceos_in_correct_date_order_are_required(self):
        old, new = self.row(), self.row("new", "2", "20", "Bob Example", "2001-01-01")
        self.assertEqual(speaker.pair_validation([old, new])["pair_validation_status"], "confirmed_both_sides")
        cases = ([old], [new], [old, old], [old, dict(new, event_id="1")],
                 [old, dict(new, expected_execid="10")], [old, dict(new, event_date="1999-01-01")],
                 [old, dict(new, turnover_side="wrong")], [old, dict(new, event_id="")],
                 [old, dict(new, external_speaker_validation_status="review")],
                 [old, new, self.row("new", "3", "21", "Charlie Other", "2001-02-01")])
        for rows in cases:
            with self.subTest(rows=rows):
                self.assertEqual(speaker.pair_validation(rows)["pair_validation_status"], "review_or_mismatch")

    def test_old_only_cli_does_not_confirm_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = dict(selection_rank="1", turnover_rank="1", turnover_id="T", turnover_side="old",
                audit_stratum="test", event_id="1", start_date="2000-01-01", company_name="Synthetic",
                expected_execid="10", expected_ceo_name="Alice Example")
            representation = dict(event_id="1", pre_ceo_speakers="Alice Example", qa_ceo_speakers="Alice Example",
                shared_ceo_speakers="Alice Example", strict_identity_ready="1", identity_status="strict")
            write(root / "manifest.csv", [manifest])
            write(root / "representation.csv", [representation])
            result = cli("validate_ccts_execucomp_speaker_identity.py", "--event-manifest", root / "manifest.csv",
                "--representation-events", root / "representation.csv", "--output-dir", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            pair = read(root / "out/ccts_execucomp_turnover_pair_validation.csv")[0]
            self.assertEqual(pair["pair_validation_status"], "review_or_mismatch")
            self.assertEqual(pair["pair_review_reason"], "missing_or_invalid_old_new_sides")


if __name__ == "__main__":
    unittest.main()
