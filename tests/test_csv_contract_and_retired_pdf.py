"""Fresh-process CSV contracts and the fail-closed historical PDF entry point."""
import ast
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CsvContractTests(unittest.TestCase):
    def test_every_csv_entry_point_configures_its_own_bound(self):
        count = 0
        for area in ("ccts_database", "llm_measurement"):
            for path in (ROOT / "tools" / area).rglob("*.py"):
                if "tests" in path.parts or path.name == "census_checkpoint.py":
                    continue
                tree = ast.parse(path.read_text())
                if not any(isinstance(node, ast.Import) and any(alias.name == "csv" for alias in node.names)
                           for node in tree.body):
                    continue
                with self.subTest(path=path.name):
                    self.assertTrue(any(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == "configure_csv"
                        for node in tree.body))
                count += 1
        self.assertEqual(count, 24)

    def test_standalone_readers_accept_long_fields_without_fetcher_import(self):
        cases = [("ccts_database/build_ccts_execucomp_speaker_gate.py", "read_rows"),
                 ("ccts_database/extract_execucomp_confirmed_ceo_qa_blocks.py", "read_csv_rows"),
                 ("ccts_database/build_ccts_proposed_analysis_sample.py", None),
                 ("ccts_database/derive_ccts_turns_unique_sequence.py", None),
                 ("llm_measurement/build_specificity_scoring_manifest.py", "read_csv")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.csv"
            body = "Synthetic quoted text,\n" + "x" * 150_000
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["unit_id", "text"])
                writer.writerow(["one", body])
            for module, function in cases:
                read_expression = (f"ns[{function!r}](Path(sys.argv[1]))" if function else
                                   "list(ns['csv'].DictReader(Path(sys.argv[1]).open(newline='')))")
                code = ("import runpy,sys; from pathlib import Path; "
                        f"sys.path.insert(0,{str((ROOT / 'tools' / module).parent)!r}); "
                        f"ns=runpy.run_path({str(ROOT / 'tools' / module)!r}); "
                        f"rows={read_expression}; "
                        "assert rows[0]['text']==('Synthetic quoted text,\\n'+'x'*150000); "
                        "assert 'fetch_ccts_turn_sample' not in sys.modules")
                result = subprocess.run([sys.executable, "-c", code, str(path)],
                    capture_output=True, text=True, cwd=directory, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_explicit_upper_bound_fails_without_truncating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.csv"
            with path.open("w") as handle:
                handle.write("text\n")
                for _ in range(64):
                    handle.write("x" * (1024 * 1024))
                handle.write("x\n")
            before = path.stat().st_size
            code = ("import csv,io,runpy,sys; from pathlib import Path; "
                f"sys.path.insert(0,{str(ROOT / 'tools/ccts_database')!r}); "
                f"ns=runpy.run_path({str(ROOT / 'tools/ccts_database/extract_execucomp_confirmed_ceo_qa_blocks.py')!r}); "
                "assert len(next(csv.reader(io.StringIO('x'*67108864)))[0])==67108864; "
                "print('exact bound accepted',flush=True); "
                "ns['read_csv_rows'](Path(sys.argv[1]))")
            result = subprocess.run([sys.executable, "-c", code, str(path)],
                capture_output=True, text=True, cwd=directory, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("field larger than field limit (67108864)", result.stderr)
            self.assertIn("exact bound accepted", result.stdout)
            self.assertEqual(path.stat().st_size, before)

    def test_absolute_derive_cli_preserves_multiline_long_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = "Synthetic, quoted\n" + "x" * 150_000
            with (root / "turns.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["event_id", "sequence_id", "source_event_company_name", "text_type", "text_name", "text_contents"])
                writer.writerow([1, 1, "Synthetic", "PRE", "Alice Example - CEO", body])
            (root / "audit.csv").write_text("event_id,fetched_rows,fetch_status,fetch_notes\n1,1,ok,\n")
            result = subprocess.run([sys.executable, str(ROOT / "tools/ccts_database/derive_ccts_turns_unique_sequence.py"),
                "--input-csv", str(root / "turns.csv"), "--fetch-audit-csv", str(root / "audit.csv"),
                "--output-dir", str(root / "out")], cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            # The parent deliberately uses the documented bound rather than a prior fetcher import.
            old_limit = csv.field_size_limit(64 * 1024 * 1024)
            try:
                with (root / "out/ccts_turns_analysis_v1.csv").open(newline="") as handle:
                    self.assertEqual(next(csv.DictReader(handle))["text_contents"], body)
            finally:
                csv.field_size_limit(old_limit)


class RetiredPdfTests(unittest.TestCase):
    def test_old_resume_commands_cannot_execute_or_reuse_outputs(self):
        script = ROOT / "tools/historical/ecc_run_parser_batches.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batch.csv").write_text('text\n"multiline\nvalue"\n')
            (root / "paths.txt").write_text("second.pdf\nfirst.pdf\n")
            original = {path.name: path.read_bytes() for path in root.iterdir()}
            for args in ([], ["--force"], ["--dry-run"],
                         ["--input", str(root / "paths.txt"), "--output-dir", str(root)],
                         ["--batch-size", "1", "--resume"]):
                result = subprocess.run([sys.executable, str(script), *args],
                    cwd=directory, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2)
                self.assertIn("workflow is disabled", result.stderr)
                self.assertEqual(original, {path.name: path.read_bytes() for path in root.iterdir()})


if __name__ == "__main__":
    unittest.main()
