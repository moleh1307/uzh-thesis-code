import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_audit_run_status import completed
from specificity_validation import SCHEMA, decode_evidence, digest, file_hash, strict_json, technical_retry_reason


TOOLS = Path(__file__).resolve().parents[1] / "tools/llm_measurement"


class ConsumerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.run = self.root / "run.json"
        self.units = self.root / "units.csv"
        self.requests = [{"custom_id": "u1", "response_schema": SCHEMA,
            "messages": [{"role": "system", "content": "Synthetic fixture"},
                         {"role": "user", "content": json.dumps({"unit_type": "pre",
                          "ceo_presentation_segment": " ".join(["business"] * 100)})}]}]
        self.input.write_text(json.dumps(self.requests[0]) + "\n")
        self.units.write_text("custom_id,unit_type,event_id,start_date,unit_word_count\n"
                              "u1,pre,100,2020-01-01,100\n")
        self.rows = [completed()]
        self.bind()

    def bind(self):
        binding = {"input_sha256": file_hash(self.input), "unit_manifest_sha256": file_hash(self.units),
                   "execution_identity": {"synthetic": True}, "contract": "synthetic", "limit": None,
                   "response_schema_hashes": {r["custom_id"]: digest(r["response_schema"]) for r in self.requests}}
        self.rows = [dict(row, run_binding_sha256=digest(binding)) for row in self.rows]
        self.output.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.run.write_text(json.dumps({"run_binding": binding, "run_binding_sha256": digest(binding),
            "contract": "synthetic", "input_sha256": file_hash(self.input),
            "output_sha256": file_hash(self.output), "retry_reservations": {
                row["custom_id"]: {"attempt_number": 2, "retry_reason": row["retry_reason"]}
                for row in self.rows if row.get("attempt_number") == 2}}))

    def consume(self, name):
        out = self.root / name
        script = "audit_local_specificity_run.py" if name == "audit" else "aggregate_local_specificity.py"
        args = [sys.executable, str(TOOLS / script), "--input-jsonl", str(self.input),
                "--output-jsonl", str(self.output), "--run-manifest", str(self.run),
                "--output-dir", str(out)]
        args.extend(["--manifest-csv" if name == "audit" else "--input-manifest", str(self.units)])
        if name != "audit":
            args.append("--force")
        result = subprocess.run(args, text=True, capture_output=True, timeout=15)
        summary_path = out / ("dry_run_comparison.json" if name == "audit" else "specificity_call_level_summary.json")
        return result, json.loads(summary_path.read_text()) if summary_path.exists() else None

    def assert_invalid_both(self):
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(summary["status"], "failed_technical_validation")
            field = "invalid_output_ids" if name == "audit" else "invalid_or_error_custom_ids"
            self.assertEqual(summary[field], ["u1"])

    def assert_provenance_rejected_both(self):
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("provenance rejected", result.stderr)
            self.assertIsNone(summary)

    def test_valid_output_accepted_by_both_consumers(self):
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(summary["provenance"]["verified"])

    def retry_rows(self, second_valid=True):
        original = completed(raw_output="invalid", raw_output_with_special_tokens="invalid<eos>")
        second = completed() if second_valid else dict(original)
        self.rows = [dict(original, attempt_number=1, retry_reason=None),
                     dict(second, attempt_number=2, retry_reason=technical_retry_reason(original))]
        self.bind()

    def test_retry_recovery_retains_attempt_ledger_without_double_counting(self):
        self.retry_rows()
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 0, result.stderr)
            ledger = summary["provenance"]["attempt_ledger"]
            self.assertEqual(ledger["total_attempt_records"], 2)
            self.assertEqual(ledger["unique_request_ids"], 1)
            self.assertEqual(ledger["first_attempt_invalid_ids"], ["u1"])
            self.assertEqual(ledger["retried_ids"], ["u1"])
            field = "score_distribution_ok1" if name == "audit" else "model_score_distribution_ok1"
            self.assertEqual(sum(summary[field].values()), 1)

    def test_failed_retry_is_not_salvaged(self):
        self.retry_rows(second_valid=False)
        self.assert_invalid_both()

    def test_retry_requires_manifest_reservation(self):
        self.retry_rows()
        manifest = json.loads(self.run.read_text())
        manifest["retry_reservations"] = {}
        self.run.write_text(json.dumps(manifest))
        self.assert_provenance_rejected_both()

    def test_first_attempt_binding_checked_even_after_recovery(self):
        self.retry_rows()
        self.rows[0]["run_binding_sha256"] = "unbound original"
        self.output.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        manifest = json.loads(self.run.read_text())
        manifest["output_sha256"] = file_hash(self.output)
        self.run.write_text(json.dumps(manifest))
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertNotEqual(result.returncode, 0)
            self.assertIsNone(summary)

    def test_interrupted_retry_reservation_visible_without_new_output(self):
        self.retry_rows()
        self.rows = self.rows[:1]
        self.bind()
        manifest = json.loads(self.run.read_text())
        manifest["retry_reservations"] = {"u1": {
            "attempt_number": 2, "retry_reason": technical_retry_reason(self.rows[0])}}
        self.run.write_text(json.dumps(manifest))
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(summary["provenance"]["attempt_ledger"]["reserved_retries_without_result"], ["u1"])

    def test_reservation_for_successful_output_rejected(self):
        manifest = json.loads(self.run.read_text())
        manifest["retry_reservations"] = {"u1": {"attempt_number": 2, "retry_reason": "inference_error"}}
        self.run.write_text(json.dumps(manifest))
        self.assert_provenance_rejected_both()

    def test_raw_wrappers_trailing_duplicates_nonfinite_and_extra_fields_rejected(self):
        for raw in ('Answer: {"ok":1,"specificity":4}', '```json\n{"ok":1,"specificity":4}\n```',
                    '<think>x</think>{"ok":1,"specificity":4}', '{"ok":1,"specificity":4} extra',
                    '{"ok":1,"specificity":3,"specificity":4}', '{"ok":1,"specificity":NaN}',
                    '{"ok":1,"specificity":4,"reason":"extra"}'):
            with self.subTest(raw=raw):
                self.rows = [completed(raw_output=raw, raw_output_with_special_tokens=raw + "<eos>")]
                self.bind()
                self.assert_invalid_both()

    def test_parsed_raw_type_or_value_mismatch_rejected(self):
        for parsed in ({"ok": 1, "specificity": 3}, {"ok": True, "specificity": 4},
                       {"ok": 1, "specificity": 4.0}):
            self.rows = [completed(parsed=parsed)]
            self.bind()
            self.assert_invalid_both()

    def test_missing_raw_or_missing_token_evidence_rejected(self):
        for field in ("raw_output", "generated_token_ids", "terminal_eos_text"):
            self.rows = [completed()]
            del self.rows[0][field]
            self.bind()
            self.assert_invalid_both()

    def test_bad_eos_special_tokens_and_truncation_rejected(self):
        for change in ({"generated_token_ids": [8, 3, 9]}, {"generated_token_ids": [3, 9, 3, 9]},
                       {"terminal_eos_token_id": 8}, {"output_truncated": True},
                       {"raw_output_with_special_tokens": "changed"}):
            self.rows = [completed(**change)]
            self.bind()
            self.assert_invalid_both()

    def test_missing_output_has_exclusion_ledger_in_both(self):
        self.rows = []
        self.bind()
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(summary["coverage"]["missing_output_ids"], ["u1"])

    def test_abstention_is_valid_but_not_a_numerical_score(self):
        self.rows = [completed(ok=0, score=0)]
        self.bind()
        for name in ("audit", "aggregate"):
            result, summary = self.consume(name)
            self.assertEqual(result.returncode, 0, result.stderr)
            field = "score_distribution_ok1" if name == "audit" else "model_score_distribution_ok1"
            self.assertEqual(sum(summary[field].values()), 0)

    def test_output_bytes_changed_after_manifest_rejected(self):
        self.output.write_text(self.output.read_text() + "\n")
        self.assert_provenance_rejected_both()

    def test_input_changed_after_manifest_rejected(self):
        self.input.write_text(self.input.read_text() + "\n")
        self.assert_provenance_rejected_both()

    def test_unit_metadata_changed_after_manifest_rejected(self):
        self.units.write_text(self.units.read_text().replace(",100\n", ",200\n"))
        self.assert_provenance_rejected_both()

    def test_row_identity_mismatch_rejected(self):
        row = json.loads(self.output.read_text())
        row["run_binding_sha256"] = "other run"
        self.output.write_text(json.dumps(row) + "\n")
        manifest = json.loads(self.run.read_text())
        manifest["output_sha256"] = file_hash(self.output)
        self.run.write_text(json.dumps(manifest))
        self.assert_provenance_rejected_both()

    def test_contract_mismatch_or_missing_execution_provenance_rejected(self):
        for change in ({"contract": "other contract"}, {"run_binding_sha256": "bad"}):
            self.bind()
            manifest = json.loads(self.run.read_text())
            manifest.update(change)
            self.run.write_text(json.dumps(manifest))
            self.assert_provenance_rejected_both()

    def test_incompatible_historical_schema_not_upgraded(self):
        self.requests[0]["response_schema"] = {"type": "object", "properties": {"reason": {"type": "string"}}}
        self.input.write_text(json.dumps(self.requests[0]) + "\n")
        self.bind()
        self.assert_provenance_rejected_both()

    def test_missing_provenance_not_reconstructed(self):
        self.run.write_text("{}")
        self.assert_provenance_rejected_both()


class TokenEvidenceTests(unittest.TestCase):
    def test_nonfinite_exponents_rejected(self):
        for text in ('1e999', '{"nested":[-1e999]}'):
            with self.assertRaises(ValueError):
                strict_json(text)

    def test_embedded_special_token_not_removed_by_decoder(self):
        class Tokenizer:
            all_special_ids = [8, 9]
            def decode(self, tokens, **kwargs):
                self_outer.assertFalse(kwargs["skip_special_tokens"])
                self_outer.assertFalse(kwargs["clean_up_tokenization_spaces"])
                return "".join({3: '{"ok":1,', 4: '"specificity":4}', 8: '<control>', 9: '<eos>'}[i] for i in tokens)
        self_outer = self
        evidence = decode_evidence(Tokenizer(), [3, 8, 4, 9], [9])
        self.assertIn("<control>", evidence["raw_output"])
        self.assertEqual(evidence["unexpected_special_token_ids"], [8])
        self.assertIn("token_evidence_error", evidence)


if __name__ == "__main__":
    unittest.main()
