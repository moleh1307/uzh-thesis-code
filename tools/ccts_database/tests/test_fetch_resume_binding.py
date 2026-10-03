import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_ccts_turn_sample as fetcher


class ResumeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.csv"
        self.manifest.write_text("event_id\n1\n2\n")
        self.output = self.root / "output"
        self.source = dict(host="fixture-A", dbname="fixture", user="reader", port=5432, sslmode="disable")
        self.rows = fetcher.normalize_manifest(fetcher.read_csv(self.manifest), None)
        self.checkpoints, self.config = fetcher.prepare_run_config(self.output, self.manifest, self.rows, self.source)
        fetcher.save_checkpoint(self.checkpoints, fetcher.checkpoint_key(["1"]), ["1"],
            [dict(event_id="1", sequence_id="1", text_contents="Synthetic business text")],
            [dict(event_id="1", fetch_status="ok")], self.config["run_binding_sha256"])
        fetcher.write_outputs(self.output, self.manifest, self.rows, self.checkpoints, ["1"], self.config["run_binding_sha256"])

    def snapshot(self):
        return {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}

    def invoke(self, source=None, password="rotated-password", extra=()):
        creds = {k: (source or self.source)[k] for k in ("host", "dbname", "user")}
        creds["password"] = password
        args = ["fetch", "--manifest", str(self.manifest), "--output-dir", str(self.output), *extra]
        with patch.object(sys, "argv", args), patch.object(fetcher, "read_credentials", return_value=creds), \
             patch.object(fetcher.psycopg2, "connect") as connect, contextlib.redirect_stdout(io.StringIO()):
            try:
                result = fetcher.main()
                return result, connect
            except Exception:
                self.assertFalse(connect.called)
                raise

    def test_source_changes_rejected_before_contact_or_output_mutation(self):
        before = self.snapshot()
        for field in ("host", "dbname", "user"):
            source = dict(self.source, **{field: "fixture-B"})
            with self.assertRaisesRegex(ValueError, "different manifest, source"):
                self.invoke(source)
            self.assertEqual(self.snapshot(), before)
        for extra in [("--port", "5433"), ("--sslmode", "require")]:
            with self.assertRaisesRegex(ValueError, "different manifest, source"):
                self.invoke(extra=extra)
            self.assertEqual(self.snapshot(), before)

    def test_password_rotation_and_completed_scope_reuse_without_connection(self):
        result, connect = self.invoke(extra=("--limit-events", "1"))
        self.assertEqual(result, 0)
        connect.assert_not_called()
        self.assertNotIn("rotated-password", (self.output / "fetch_run_config.json").read_text())
        self.assertNotIn("password", (self.output / "fetch_run_config.json").read_text())

    def test_same_bound_source_extends_partial_scope_and_preserves_first_shard(self):
        old_shard = {p.name: p.read_bytes() for p in self.checkpoints.iterdir()}
        def synthetic_batch(cur, events):
            self.assertEqual([row["event_id"] for row in events], ["2"])
            return ([dict(event_id="2", sequence_id="1", text_contents="Synthetic second event")],
                    [dict(event_id="2", fetch_status="ok")])
        with patch.object(fetcher, "fetch_batch", side_effect=synthetic_batch):
            result, connect = self.invoke()
        self.assertEqual(result, 0)
        connect.assert_called_once()
        self.assertEqual(connect.call_args.kwargs["password"], "rotated-password")
        self.assertEqual([row["event_id"] for row in fetcher.read_csv(self.output / "ccts_turn_fetch_audit.csv")], ["1", "2"])
        for name, content in old_shard.items():
            self.assertEqual((self.checkpoints / name).read_bytes(), content)

    def test_shared_csv_contract_change_rejects_resume(self):
        before = self.snapshot()
        original = fetcher.sha256_file
        with patch.object(fetcher, "sha256_file", side_effect=lambda path:
                "changed-shared-code" if Path(path).name == "csv_contract.py" else original(path)), \
             self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.snapshot(), before)

    def test_extractor_change_and_legacy_config_rejected_without_mutation(self):
        before = self.snapshot()
        original = fetcher.sha256_file
        def modified(path):
            return "changed-code" if Path(path) == Path(fetcher.__file__) else original(path)
        with patch.object(fetcher, "sha256_file", side_effect=modified), self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.snapshot(), before)
        config_path = self.output / "fetch_run_config.json"
        config_path.write_text(json.dumps({"schema": 1, "manifest": str(self.manifest)}))
        before = self.snapshot()
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.snapshot(), before)

    def test_foreign_shard_binding_is_hard_failure_not_silent_refetch(self):
        marker = next(self.checkpoints.glob("*.done.json"))
        payload = json.loads(marker.read_text())
        payload["run_binding_sha256"] = "foreign-source"
        marker.write_text(json.dumps(payload))
        before = self.snapshot()
        with self.assertRaises(fetcher.CheckpointBindingError):
            self.invoke()
        self.assertEqual(self.snapshot(), before)

    def test_source_identity_rejects_secrets(self):
        with self.assertRaisesRegex(ValueError, "only host"):
            fetcher.prepare_run_config(self.root / "new", self.manifest, self.rows,
                dict(self.source, password="must-not-be-recorded"))
        self.assertFalse((self.root / "new").exists())


if __name__ == "__main__":
    unittest.main()
