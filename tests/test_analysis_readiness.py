"""Offline readiness regressions; no transcript, model, SSH or API execution."""
import csv
import contextlib
import hashlib
import io
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools/llm_measurement"
sys.path.insert(0, str(TOOLS))
import aggregate_local_specificity as aggregate
import analyze_specificity_length_robustness as length
import launch_local_specificity as launcher
from annotation_contract import human_reference, CODING_RULES
from prepare_scoring_unit_metadata import derive, FIELDS
from specificity_validation import SCHEMA
from runner_fixtures import FakeRunnerCase


def csv_write(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference(label="substantive", ok="1", score="3", note=""):
    return dict(human_content_class=label, human_ok=ok, human_specificity=score, human_notes=note)


class AnnotationTests(unittest.TestCase):
    def test_reference_states(self):
        self.assertEqual(human_reference(reference())["reference_state"], "scored")
        self.assertIsNone(human_reference(reference("procedural_only", "0", "0"))["human_specificity"])
        self.assertIsNone(human_reference(reference("uncertain", "", "", "bad source"))["human_ok"])

    def test_unfinished_and_inconsistent_are_rejected(self):
        for row in (reference("uncertain", "", ""), reference("procedural_only", "1", "3"),
                    reference("substantive", "0", "0"), reference("", "", ""), reference(score="6")):
            with self.assertRaises(ValueError):
                human_reference(row)

    def test_rules_include_current_boundary_and_anchors(self):
        self.assertIn("Brief thanks or greetings alone are incidental", CODING_RULES)
        self.assertIn("leave human_ok and human_specificity blank", CODING_RULES)
        self.assertIn("no materially bounding detail", CODING_RULES)


class AggregationTests(unittest.TestCase):
    def unit(self, uid, kind, score, event="1", words=100):
        return dict(custom_id=uid, unit_type=kind, event_id=event, start_date="2020-01-01",
                    unit_word_count=words, scored=score is not None, model_ok=1 if score is not None else 0,
                    model_specificity=score if score is not None else "", status="completed")

    def test_sign_weighting_and_paired_population(self):
        units = [self.unit("p", "pre", 1), self.unit("p2", "pre", 3, words=300),
                 self.unit("q", "qa", 5), self.unit("p3", "pre", 1, event="2")]
        calls = aggregate.build_call_rows(units, {})
        self.assertEqual(calls[0]["pre_word_weighted_mean"], 2.5)
        self.assertEqual(calls[0]["qa_minus_pre_word_weighted_mean"], 2.5)
        self.assertIsNone(calls[1]["qa_minus_pre_word_weighted_mean"])
        metric = aggregate.aggregate_call_metrics(calls)["word_weighted_mean"]
        self.assertEqual(metric["pre_call_mean"], 2.5)
        self.assertEqual(metric["n_pre"], metric["n_qa"])
        self.assertEqual(metric["paired_event_ids"], ["1"])
        self.assertEqual(metric["available_pre_call_mean"], 1.75)

    def test_common_support_excludes_source_missing_and_procedural(self):
        units = [self.unit("p", "pre", 2), self.unit("q", "qa", 5),
                 self.unit("z", "qa", 4), self.unit("bad", "qa", 5), self.unit("no", "pre", None)]
        human = {uid: dict(audit_id=uid, unit_type=unit["unit_type"], **reference())
                 for uid, unit in ((row["custom_id"], row) for row in units)}
        human["z"].update(reference("procedural_only", "0", "0"))
        human["bad"].update(reference("uncertain", "", "", "corrupt"))
        key = {uid: {"audit_id": uid} for uid in human}
        crosswalk = aggregate.build_human_crosswalk(units, human, key)
        common = [row for row in crosswalk if row["human_scored"] and row["model_scored"]]
        self.assertEqual({row["custom_id"] for row in common}, {"p", "q"})
        metric = aggregate.agreement_metrics(crosswalk)
        self.assertEqual(metric["n"], 2)
        model_calls = aggregate.build_call_rows([row for row in units if row["custom_id"] in {"p", "q"}], {})
        human_calls = aggregate.build_human_call_rows(common, {})
        self.assertEqual(model_calls[0]["qa_unit_count_scored"], human_calls[0]["qa_unit_count_scored"])

    def test_human_unit_and_event_mismatch_rejected(self):
        unit = self.unit("p", "pre", 3)
        for human, key in ((dict(unit_type="qa", **reference()), {"audit_id": "p"}),
                           (dict(unit_type="pre", **reference()), {"audit_id": "p", "event_id": "other"})):
            with self.assertRaises(ValueError):
                aggregate.build_human_crosswalk([unit], {"p": human}, {"p": key})


class MetadataTests(unittest.TestCase):
    def fixture(self, kind="pre"):
        target = {"unit_type": kind}
        text = "Synthetic business statement"
        if kind == "pre":
            target["ceo_presentation_segment"] = text
        else:
            target.update(analyst_question="What changed?", ceo_answer=text)
        content = text if kind == "pre" else "What changed?\n" + text
        request = {"custom_id": "u", "response_schema": SCHEMA,
                   "messages": [{"role": "system", "content": "synthetic"},
                                {"role": "user", "content": json.dumps(target)}]}
        key = dict(unit_type=kind, target_word_count="3", question_word_count="0" if kind == "pre" else "2",
                   source_content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                   event_id="1", start_date="2020-01-01", period_bin="early", source_unit_id="s",
                   human_specificity="5")
        return request, key

    def test_label_free_pre_and_qa_metadata(self):
        for kind in ("pre", "qa"):
            request, key = self.fixture(kind)
            if kind == "pre":
                key["question_word_count"] = ""
            row = derive([request], {"u": key})[0]
            self.assertEqual(set(row), set(FIELDS))
            self.assertNotIn("human_specificity", row)
            self.assertEqual(row["unit_word_count"], 3)

    def test_mismatched_source_and_identity_rejected(self):
        request, key = self.fixture()
        for change in ({"target_word_count": "4"}, {"unit_type": "qa"},
                       {"source_content_sha256": "bad"}, {"event_id": ""}):
            with self.assertRaises(ValueError):
                derive([request], {"u": dict(key, **change)})
        for requests, keys in (([request, request], {"u": key}), ([request], {})):
            with self.assertRaises(ValueError):
                derive(requests, keys)


class AggregationCliTests(FakeRunnerCase):
    def test_valid_bound_run_with_current_reference_and_model_length(self):
        self.requests(4)
        requests = [json.loads(line) for line in self.input.read_text().splitlines()]
        for index, request in enumerate(requests):
            if index:
                request["messages"][1]["content"] = json.dumps(dict(unit_type="qa",
                    analyst_question="Synthetic question", ceo_answer="Synthetic business statement"))
        self.input.write_text("".join(json.dumps(row) + "\n" for row in requests))
        csv_write(self.units, [dict(custom_id=row["custom_id"], unit_type="pre" if index == 0 else "qa",
                                    event_id="1", start_date="2020-01-01", unit_word_count="3", period_bin="early")
                              for index, row in enumerate(requests)])
        code, _, _ = self.run_fake(text=['{"ok":1,"specificity":2}', '{"ok":1,"specificity":5}',
                                         '{"ok":1,"specificity":4}', '{"ok":0,"specificity":0}'])
        self.assertEqual(code, 0)
        human = self.root / "human.csv"
        rows = [dict(audit_id=row["custom_id"], unit_type="pre" if index == 0 else "qa", **reference(score="3"))
                for index, row in enumerate(requests)]
        rows[2].update(reference("uncertain", "", "", "synthetic corruption"))
        csv_write(human, rows)
        output = self.root / "aggregate"
        result = subprocess.run([sys.executable, str(TOOLS / "aggregate_local_specificity.py"),
            "--input-manifest", str(self.units), "--input-jsonl", str(self.input),
            "--output-jsonl", str(self.output), "--run-manifest", str(self.manifest),
            "--human-coded", str(human), "--output-dir", str(output)], text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((output / "specificity_call_level_summary.json").read_text())
        self.assertEqual(summary["human_crosswalk"]["common_scored_custom_ids"], ["FIXTURE_0", "FIXTURE_1"])
        model = aggregate.read_csv_rows(output / "specificity_model_common_support_call_level.csv")[0]
        reference_call = aggregate.read_csv_rows(output / "specificity_human_call_level.csv")[0]
        self.assertEqual(model["qa_unit_count_scored"], reference_call["qa_unit_count_scored"])
        self.assertEqual(model["qa_minus_pre_word_weighted_mean"], "3.0")
        args = SimpleNamespace(lane="model", coded_csv=None, key_csv=self.units,
            unit_results_csv=output / "specificity_unit_results.csv", aggregation_summary=output / "specificity_call_level_summary.json")
        scored, excluded = length.load_diagnostic_rows(args)
        self.assertEqual(len(scored), 3)
        self.assertEqual(excluded[0]["audit_id"], "FIXTURE_3")


class LengthTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.coded, self.key = self.root / "human.csv", self.root / "key.csv"
        self.rows = [dict(audit_id="p", unit_type="pre", ceo_presentation_segment="business fact",
                          ceo_answer="", **reference(score="2")),
                     dict(audit_id="q", unit_type="qa", ceo_presentation_segment="",
                          ceo_answer="business fact", **reference(score="4")),
                     dict(audit_id="z", unit_type="pre", ceo_presentation_segment="thanks all",
                          ceo_answer="", **reference("procedural_only", "0", "0")),
                     dict(audit_id="bad", unit_type="qa", ceo_presentation_segment="",
                          ceo_answer="broken text", **reference("uncertain", "", "", "corruption"))]
        csv_write(self.coded, self.rows)
        csv_write(self.key, [dict(audit_id=row["audit_id"], unit_type=row["unit_type"],
                                 period_bin="early", target_word_count="2") for row in self.rows])
        self.args = SimpleNamespace(lane="human_reference", coded_csv=self.coded, key_csv=self.key,
                                    unit_results_csv=None, aggregation_summary=None)

    def test_arbitrary_counts_and_exclusion_ledger(self):
        rows, excluded = length.load_diagnostic_rows(self.args)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["reference_state"] for row in excluded}, {"procedural_only", "source_uninterpretable"})

    def test_cli_undefined_support_is_null_and_repeatable(self):
        summaries = []
        for index in range(2):
            out = self.root / str(index)
            result = subprocess.run([sys.executable, str(TOOLS / "analyze_specificity_length_robustness.py"),
                "--coded-csv", str(self.coded), "--key-csv", str(self.key), "--output-dir", str(out),
                "--bootstrap-draws", "50"], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = (out / "specificity_length_robustness_summary.json").read_text()
            self.assertNotIn("NaN", raw)
            summary = json.loads(raw)
            self.assertIsNone(summary["common_support"]["qa_minus_pre"])
            self.assertEqual(summary["hc3_ols_additive"]["status"], "insufficient_support")
            summaries.append(summary["period_and_length_matching"])
        self.assertEqual(summaries[0], summaries[1])
        self.assertEqual(summaries[0]["15"]["mean_paired_qa_minus_pre"], 2)

    def test_empty_scored_support_and_constant_scores(self):
        csv_write(self.coded, self.rows[2:])
        csv_write(self.key, [dict(audit_id=r["audit_id"], unit_type=r["unit_type"], period_bin="early",
                                 target_word_count="2") for r in self.rows[2:]])
        rows, excluded = length.load_diagnostic_rows(self.args)
        self.assertEqual(rows, [])
        self.assertEqual(len(excluded), 2)
        self.assertEqual(length.regression(rows)["status"], "insufficient_support")
        self.assertEqual(length.json_safe(length.bootstrap_mean_ci([], random.Random(0), 10)), [None, None])
        constant = [dict(unit_type="pre" if i % 2 else "qa", period_bin="a", word_count=80+i*10, score=3)
                    for i in range(12)]
        self.assertEqual(length.regression(constant)["status"], "insufficient_support")

    def test_model_receipt_schema_and_word_identity(self):
        model = self.root / "model.csv"
        receipt = self.root / "summary.json"
        rows = [dict(custom_id=r["audit_id"], unit_type=r["unit_type"], status="completed",
                     model_validation_error="", model_ok="1", model_specificity="3", scored="True",
                     unit_word_count="2") for r in self.rows]
        csv_write(model, rows)
        def bind():
            receipt.write_text(json.dumps(dict(status="completed_diagnostic_aggregation",
                output_sha256={"specificity_unit_results.csv": length.sha256(model)})))
        bind()
        self.args = SimpleNamespace(lane="model", coded_csv=None, key_csv=self.key,
                                    unit_results_csv=model, aggregation_summary=receipt)
        self.assertEqual(len(length.load_diagnostic_rows(self.args)[0]), 4)
        model.write_text(model.read_text() + "\n")
        with self.assertRaises(ValueError):
            length.load_diagnostic_rows(self.args)
        rows[0]["model_specificity"] = "0"
        csv_write(model, rows)
        bind()
        with self.assertRaises(ValueError):
            length.load_diagnostic_rows(self.args)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        self.settings = self.root / "settings.json"
        self.settings.write_text(json.dumps({"model": {"revision": "fixed"}, "decoding": {"max_new_tokens": 16}}))
        self.input = self.root / "input.jsonl"
        self.input.write_text("synthetic")
        self.config = dict(python=sys.executable, runner=str(TOOLS / "run_local_specificity.py"),
                           input_jsonl=str(self.input), unit_manifest_csv=str(self.input),
                           model=str(self.model), revision="fixed", settings=str(self.settings), contract_version="test")

    def test_validate_capture_commands_and_no_writes(self):
        for stage, flag in (("validate", "--validate-only"), ("capture", "--capture-execution-profile")):
            plan = launcher.prepare(self.config, stage, self.root / "job", "uzh-specificity-fixture")
            self.assertIn(flag, plan["command"])
            self.assertNotIn("--execution-profile", plan["command"])
        self.assertFalse((self.root / "job").exists())

    def test_score_requires_reviewed_profile_and_revision(self):
        with self.assertRaises(ValueError):
            launcher.prepare(self.config, "score", self.root / "job", "uzh-specificity-fixture")
        profile = self.root / "profile.json"
        profile.write_text('{"status":"candidate_requires_review"}')
        with self.assertRaises(ValueError):
            launcher.prepare(dict(self.config, execution_profile=str(profile)), "score", self.root / "job", "uzh-specificity-fixture")
        with self.assertRaises(ValueError):
            launcher.prepare(dict(self.config, revision="other"), "validate", self.root / "job", "uzh-specificity-fixture")

    def test_worker_rejects_changed_inputs_before_subprocess(self):
        job = self.root / "job"
        job.mkdir()
        plan = launcher.prepare(self.config, "validate", job, "uzh-specificity-fixture")
        (job / "execution.json").write_text(json.dumps(plan))
        self.input.write_text("changed")
        with patch.object(launcher.subprocess, "Popen") as popen:
            self.assertEqual(launcher.worker(job), 1)
            popen.assert_not_called()
        self.assertEqual(json.loads((job / "status.json").read_text())["status"], "failed")

    def test_capture_success_stops_at_candidate_not_approval(self):
        job = self.root / "job"
        job.mkdir()
        plan = launcher.prepare(self.config, "capture", job, "uzh-specificity-fixture")
        profile = job / "candidate-execution-profile.json"
        plan["command"] = [sys.executable, "-c",
            "from pathlib import Path; Path(" + repr(str(profile)) + ").write_text('{\"status\":\"candidate_requires_review\"}')"]
        (job / "execution.json").write_text(json.dumps(plan))
        self.assertEqual(launcher.worker(job), 0)
        status = json.loads((job / "status.json").read_text())
        self.assertEqual(status["status"], "candidate_requires_review")
        self.assertIn("finished_at_utc", status)
        self.assertFalse((job / "results.jsonl").exists())

    def test_interrupted_worker_terminates_owned_child(self):
        job = self.root / "job"
        job.mkdir()
        (job / "execution.json").write_text(json.dumps(launcher.prepare(self.config, "validate", job, "uzh-specificity-fixture")))
        with patch.object(launcher.subprocess, "Popen") as popen:
            process = popen.return_value
            process.pid = 12345
            process.wait.side_effect = [KeyboardInterrupt(), 0]
            process.poll.return_value = None
            self.assertEqual(launcher.worker(job), 1)
            process.terminate.assert_called_once()
            self.assertEqual(process.wait.call_count, 2)

    def test_score_cli_requires_explicit_confirmation(self):
        result = subprocess.run([sys.executable, str(TOOLS / "launch_local_specificity.py"),
                                 "--stage", "score"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--confirm-scoring", result.stderr)

    def test_parent_uses_forking_screen_daemon(self):
        config = self.root / "job-config.json"
        config.write_text(json.dumps(self.config))
        job = self.root / "job"
        argv = ["launcher", "--config", str(config), "--stage", "validate", "--job-dir", str(job),
                "--session", "uzh-specificity-fixture"]
        with patch.object(sys, "argv", argv), patch.object(launcher.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="")
            self.assertEqual(launcher.main(), 0)
        self.assertEqual(run.call_args_list[1].args[0][1], "-dmS")
        self.assertNotIn("-DmS", run.call_args_list[1].args[0])
        self.assertEqual(json.loads((job / "status.json").read_text())["status"], "queued")
