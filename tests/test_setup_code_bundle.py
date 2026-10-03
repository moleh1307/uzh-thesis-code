import ast
import contextlib
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/llm_measurement"))
import freeze_fresh_specificity_setup as freezer
import prepare_fresh_evaluation_human_package as package_builder


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SetupCodeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package, self.evaluation = self.root / "package", self.root / "evaluation"
        self.proposal = self.root / "definition/v2"
        for path in (self.package, self.evaluation, self.proposal, self.proposal.parent / "v1"):
            path.mkdir(parents=True)
        blinded = self.package / "fresh_evaluation_human_blinded.csv"
        rows = []
        for index in range(100):
            row = dict.fromkeys(package_builder.BLINDED_FIELDS, "")
            row.update(audit_id=f"SYNTHETIC_{index}", unit_type="pre" if index < 40 else "qa",
                       ceo_presentation_segment="Synthetic business fact." if index < 40 else "",
                       analyst_question="What changed?" if index >= 40 else "",
                       ceo_answer="Synthetic business fact." if index >= 40 else "")
            rows.append(row)
        write_csv(blinded, rows)
        key = self.package / "key.csv"
        write_csv(key, [{"audit_id": "synthetic"}])
        artifacts = {"blinded_csv": blinded, "private_key_csv": key}
        (self.package / "fresh_evaluation_human_package_manifest.json").write_text(json.dumps({
            "artifacts": {name: str(path) for name, path in artifacts.items()},
            "artifacts_sha256": {name: freezer.sha256_path(path) for name, path in artifacts.items()}}))
        sources = {}
        for name in ("pre_units_csv", "pre_lineage_csv", "qa_blocks_csv", "qa_lanes_csv"):
            path = self.evaluation / (name + ".csv")
            write_csv(path, [{"synthetic": "only"}])
            sources[name] = {"path": str(path), "sha256": freezer.sha256_path(path)}
        (self.evaluation / "evaluation_manifest_summary.json").write_text(json.dumps({"sources": sources}))
        for name in ("evaluation_unit_manifest.csv", "evaluation_call_manifest.csv"):
            write_csv(self.evaluation / name, [{"synthetic": "only"}])
        prompt = (ROOT / "configs/specificity-system-prompt.txt").read_text()
        (self.proposal / "specificity-system-prompt.txt").write_text(prompt)
        (self.proposal.parent / "v1/specificity-system-prompt.txt").write_text(prompt)
        (self.proposal / "review-plan.md").write_text("Synthetic fixture. No scoring authorization.")
        for source, target in (("specificity-settings.reference.json", "proposed-settings.json"),
                               ("specificity-production-schema.json", "specificity-production-schema.json")):
            (self.proposal / target).write_bytes((ROOT / "configs" / source).read_bytes())
        self.output = self.root / "freeze"
        with contextlib.redirect_stdout(io.StringIO()):
            self.manifest = freezer.freeze(self.proposal, self.package, self.evaluation, self.output)

    def isolated(self, script, *args):
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        return subprocess.run([sys.executable, str(self.output / script), *map(str, args)],
                              cwd=self.root, env=env, capture_output=True, text=True, timeout=15)

    def test_all_declared_executables_start_outside_checkout(self):
        for script in self.manifest["code_entrypoints"]:
            result = self.isolated(script, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
        runner = self.manifest["code_entrypoints"][0]
        result = self.isolated(runner, "--validate-only", "--input-jsonl",
                              self.output / "specificity_requests_100.jsonl", "--output-jsonl",
                              self.root / "not_written.jsonl", "--run-manifest", self.root / "not_written.json",
                              "--model", "not_loaded")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["requests"], 100)
        self.assertFalse((self.root / "not_written.jsonl").exists())
        self.assertFalse((self.root / "not_written.json").exists())

    def test_hashes_cover_nested_code_and_local_import_closure(self):
        files = {str(path.relative_to(self.output)): path for path in self.output.rglob("*")
                 if path.is_file() and path.name != "freeze_manifest.json"}
        self.assertEqual(set(files), set(self.manifest["artifact_sha256"]))
        for name, path in files.items():
            self.assertEqual(freezer.sha256_path(path), self.manifest["artifact_sha256"][name])
        local_names = {path.stem for path in (ROOT / "tools").rglob("*.py")}
        bundled_names = {path.stem for name, path in files.items() if name.endswith(".py")}
        for name, path in files.items():
            if not name.endswith(".py"):
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                modules = ([node.module] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if module in local_names:
                        self.assertIn(module, bundled_names, (name, module))

    def test_existing_archive_refused_and_unchanged(self):
        before = {str(p.relative_to(self.output)): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            freezer.freeze(self.proposal, self.package, self.evaluation, self.output)
        self.assertEqual(before, {str(p.relative_to(self.output)): p.read_bytes()
                                 for p in self.output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
