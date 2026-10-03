import csv
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PATH = Path(__file__).resolve().parents[1] / "tools/llm_measurement/human_audit_dashboard/server.py"
spec = importlib.util.spec_from_file_location("dashboard_persistence", PATH)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class DashboardPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.csv"
        self.progress = self.root / "progress.json"
        self.coded = self.root / "coded.csv"
        self.fields = dashboard.CONTENT_CLASS_AUDIT_COLUMNS
        self.rows = [dict.fromkeys(self.fields, "") for _ in range(2)]
        for index, row in enumerate(self.rows, 1):
            row.update(audit_id=f"u{index}", unit_type="qa", ceo_answer="Synthetic business statement")
        self.write(self.source, self.rows)

    def write(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    def read(self):
        with self.coded.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def store(self, **kwargs):
        return dashboard.AuditStore(self.source, self.progress, self.coded, **kwargs)

    def label(self, audit_id="u1", score=2):
        return dict(audit_id=audit_id, human_content_class="substantive",
                    human_ok=1, human_specificity=score, human_notes="Synthetic annotation")

    def edit(self, score=4):
        rows = self.read()
        rows[0]["human_specificity"] = str(score)
        self.write(self.coded, rows)

    def snapshot(self):
        return {path: path.read_bytes() if path.exists() else None
                for path in (self.source, self.progress, self.coded)}

    def test_conflicting_restart_preserves_every_file(self):
        self.store().save(self.label())
        self.edit()
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.read()[0]["human_specificity"], "4")

    def test_missing_progress_does_not_blank_csv_labels(self):
        self.store().save(self.label())
        self.progress.unlink()
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store()
        self.assertEqual(self.snapshot(), before)

    def test_explicit_import_backs_up_progress_and_preserves_other_rows(self):
        self.store().save(self.label())
        old_progress = self.progress.read_bytes()
        self.edit()
        original_csv = self.coded.read_bytes()
        store = self.store(label_source="coded-csv")
        self.assertEqual(self.coded.read_bytes(), original_csv)
        self.assertEqual(store.labels["u1"]["human_specificity"], 4)
        backups = list(self.root.glob("progress.json.backup-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), old_progress)
        store.save(self.label("u2", 3))
        self.assertEqual([r["human_specificity"] for r in self.read()], ["4", "3"])
        self.assertEqual(self.store().labels["u1"]["human_specificity"], 4)

    def test_explicit_import_without_progress(self):
        self.store().save(self.label())
        self.progress.unlink()
        store = self.store(label_source="coded-csv")
        store.save(self.label("u2", 5))
        self.assertEqual([r["human_specificity"] for r in self.read()], ["2", "5"])

    def test_consistent_restart_does_not_rewrite_files(self):
        self.store().save(self.label())
        before = self.snapshot()
        self.assertEqual(self.store().labels["u1"]["human_specificity"], 2)
        self.assertEqual(self.snapshot(), before)

    def test_external_change_blocks_save_and_export(self):
        store = self.store()
        store.save(self.label())
        self.edit()
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            store.save(self.label("u2", 3))
        with self.assertRaises(dashboard.PersistenceConflict):
            store.export_csv_text()
        self.assertEqual(self.snapshot(), before)
        self.assertNotIn("u2", store.labels)

    def test_external_progress_change_blocks_save(self):
        store = self.store()
        store.save(self.label())
        state = json.loads(self.progress.read_text())
        state["labels"]["u1"]["human_specificity"] = 5
        self.progress.write_text(json.dumps(state))
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            store.save(self.label("u2", 3))
        self.assertEqual(self.snapshot(), before)

    def test_wrong_source_or_id_cannot_be_imported(self):
        for field, value in (("ceo_answer", "Changed source"), ("audit_id", "unknown")):
            with self.subTest(field=field):
                self.store().save(self.label())
                rows = self.read()
                rows[0][field] = value
                self.write(self.coded, rows)
                before = self.snapshot()
                with self.assertRaises((dashboard.PersistenceConflict, SystemExit)):
                    self.store(label_source="coded-csv")
                self.assertEqual(self.snapshot(), before)
                self.write(self.coded, self.rows)
                self.progress.unlink()

    def test_invalid_csv_score_cannot_be_imported(self):
        self.store().save(self.label())
        self.edit(score=9)
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store(label_source="coded-csv")
        self.assertEqual(self.snapshot(), before)

    def test_invalid_progress_label_is_not_exported(self):
        self.store().save(self.label())
        state = json.loads(self.progress.read_text())
        state["labels"]["u1"]["human_specificity"] = 9
        self.progress.write_text(json.dumps(state))
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store()
        self.assertEqual(self.snapshot(), before)

    def test_boolean_progress_score_is_rejected_without_rewriting(self):
        self.store().save(self.label())
        state = json.loads(self.progress.read_text())
        state["labels"]["u1"]["human_specificity"] = True
        self.progress.write_text(json.dumps(state))
        before = self.snapshot()
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store()
        self.assertEqual(self.snapshot(), before)

    def test_http_save_and_export_report_conflict_after_external_edit(self):
        store = self.store()
        server = dashboard.DashboardServer(("127.0.0.1", 0), PATH.parent, store)
        original_log = server.RequestHandlerClass.log_message
        server.RequestHandlerClass.log_message = lambda *args: None
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            request = Request(base + "/api/label", data=json.dumps(self.label()).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
            with urlopen(base + "/api/export", timeout=2) as response:
                self.assertIn(b"Synthetic annotation", response.read())
            self.edit()
            before = self.snapshot()
            request = Request(base + "/api/label", data=json.dumps(self.label("u2", 3)).encode(),
                              headers={"Content-Type": "application/json"})
            for operation in (request, base + "/api/export"):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(operation, timeout=2)
                self.assertEqual(caught.exception.code, 409)
                self.assertIn("annotation files changed", json.loads(caught.exception.read())["error"])
                caught.exception.close()
            self.assertEqual(self.snapshot(), before)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            server.RequestHandlerClass.log_message = original_log
            self.assertFalse(thread.is_alive())

    def test_source_hash_mismatch_preserves_files(self):
        self.store().save(self.label())
        self.rows[0]["ceo_answer"] = "Changed input"
        self.write(self.source, self.rows)
        before = self.snapshot()
        with self.assertRaises(SystemExit):
            self.store()
        self.assertEqual(self.snapshot(), before)

    def test_missing_csv_reconstructed_only_from_matching_progress(self):
        self.store().save(self.label())
        self.coded.unlink()
        self.store()
        self.assertEqual(self.read()[0]["human_specificity"], "2")

    def test_input_output_alias_rejected(self):
        before = self.source.read_bytes()
        with self.assertRaises(dashboard.PersistenceConflict):
            dashboard.AuditStore(self.source, self.progress, self.source)
        self.assertEqual(self.source.read_bytes(), before)

    def test_explicit_import_requires_existing_csv(self):
        with self.assertRaises(dashboard.PersistenceConflict):
            self.store(label_source="coded-csv")
        self.assertFalse(self.progress.exists())
        self.assertFalse(self.coded.exists())

    def test_legacy_and_recode_schemas_remain_supported(self):
        for fields, key in ((dashboard.AUDIT_COLUMNS, "audit_id"), (dashboard.RECODE_COLUMNS, "recode_id")):
            with self.subTest(key=key):
                self.fields = fields
                rows = [dict.fromkeys(fields, "")]
                rows[0].update({key: "u1", "unit_type": "pre", "ceo_presentation_segment": "Synthetic business statement"})
                self.write(self.source, rows)
                store = self.store()
                store.save(dict(audit_id="u1", human_ok=1, human_specificity=3))
                self.assertEqual(self.store().labels["u1"]["human_specificity"], 3)
                self.progress.unlink()
                self.coded.unlink()

    def source_missing(self):
        return dict(audit_id="u1", human_content_class="uncertain", human_ok="",
                    human_specificity="", human_notes="Source incomplete")

    def test_source_missing_preserved_in_progress_csv_export_and_restart(self):
        store = self.store()
        stats = store.save(self.source_missing())
        self.assertEqual(stats["reviewed"], 1)
        self.assertEqual(stats["scored"], 0)
        self.assertEqual(stats["source_missing"], 1)
        self.assertEqual(stats["remaining"], 1)
        exported = list(csv.DictReader(store.export_csv_text().splitlines()))[0]
        for key, value in self.source_missing().items():
            self.assertEqual(exported[key], value)
        before = self.snapshot()
        restarted = self.store()
        self.assertEqual(restarted.labels["u1"]["annotation_state"], "source_uninterpretable")
        self.assertEqual(self.snapshot(), before)

    def test_source_missing_csv_import_preserves_blank_values(self):
        self.store()
        rows = self.read()
        rows[0].update(self.source_missing())
        self.write(self.coded, rows)
        store = self.store(label_source="coded-csv")
        store.save(self.label("u2", 4))
        self.assertEqual(self.read()[0]["human_ok"], "")
        self.assertEqual(self.read()[0]["human_specificity"], "")
        self.assertEqual(self.store().labels["u1"]["human_notes"], "Source incomplete")

    def test_source_missing_requires_reason_and_exact_blank_status_score(self):
        store = self.store()
        for change in ({"human_notes": " "}, {"human_notes": None}, {"human_notes": 123},
                       {"human_ok": 0}, {"human_specificity": 0},
                       {"human_content_class": "substantive"}, {"human_ok": False}):
            before = self.snapshot()
            with self.assertRaises(ValueError):
                store.save(dict(self.source_missing(), **change))
            self.assertEqual(self.snapshot(), before)
        store.save(dict(self.source_missing(), human_ok=None, human_specificity=None))
        self.assertEqual(self.read()[0]["human_specificity"], "")

    def test_source_missing_clear_is_unfinished_not_zero(self):
        store = self.store()
        store.save(self.source_missing())
        store.save(dict(audit_id="u1", clear=True))
        self.assertEqual(store.stats()["reviewed"], 0)
        self.assertEqual(store.stats()["remaining"], 2)
        self.assertNotIn("u1", store.labels)
        self.assertTrue(all(self.read()[0][key] == "" for key in
                            ("human_content_class", "human_ok", "human_specificity", "human_notes")))

    def test_procedural_zero_is_distinct_from_source_missing(self):
        store = self.store()
        store.save(self.source_missing())
        store.save(dict(audit_id="u2", human_content_class="procedural_only", human_ok=0,
                        human_specificity=0, human_notes="Routine handoff"))
        self.assertEqual(store.stats()["reviewed"], 2)
        self.assertEqual(store.stats()["scored"], 0)
        self.assertEqual(store.stats()["source_missing"], 1)
        self.assertEqual(store.stats()["procedural_only"], 1)
        self.assertEqual([r["human_specificity"] for r in self.read()], ["", "0"])

    def test_interpretable_uncertain_scored_state_remains_supported(self):
        store = self.store()
        store.save(dict(self.label(), human_content_class="uncertain"))
        self.assertEqual(self.store().labels["u1"]["human_specificity"], 2)
        self.assertEqual(store.stats()["scored"], 1)
        self.assertEqual(store.stats()["source_missing"], 0)


if __name__ == "__main__":
    unittest.main()
