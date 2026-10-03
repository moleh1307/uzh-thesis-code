"""Characterize deployed-runner gaps with fake inference, never a real model."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools/llm_measurement/run_local_specificity.py"
spec = importlib.util.spec_from_file_location("reviewed_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Tensor:
    def __init__(self, tokens, matrix=False):
        self.tokens = tokens
        self.shape = (1, len(tokens)) if matrix else (len(tokens),)

    def to(self, device):
        return self

    def tolist(self):
        return self.tokens

    def __getitem__(self, index):
        return Tensor(self.tokens[index[1]])


class RunnerReview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.manifest = self.root / "manifest.json"
        request = {"custom_id": "REVIEW_FIXTURE_1", "response_schema": {},
                   "messages": [{"role": "system", "content": "Review fixture"},
                                {"role": "user", "content": json.dumps({"unit_type": "pre", "ceo_presentation_segment": "Fixture business statement"})}]}
        self.input.write_text(json.dumps(request) + "\n")

    def run_fake(self, text='{"ok":1,"specificity":4}', error=False,
                 special_prefix=False, resume=False, version="2.8.0+cu128"):
        calls = []

        class Tokenizer:
            eos_token_id = 9
            model_max_length = 32768

            def apply_chat_template(self, *args, **kwargs):
                return "fixture"

            def __call__(self, *args, **kwargs):
                return {"input_ids": Tensor([1, 2], matrix=True)}

            def decode(self, tokens, skip_special_tokens):
                if skip_special_tokens:
                    return text
                return ("<|im_start|>" if special_prefix else "") + text + "<|im_end|>"

        class Model:
            config = SimpleNamespace(max_position_embeddings=32768)
            generation_config = SimpleNamespace(eos_token_id=9, temperature=.7, top_p=.8, top_k=20)

            def eval(self):
                pass

            def parameters(self):
                return iter([SimpleNamespace(device="fixture")])

            def generate(self, **kwargs):
                calls.append(kwargs)
                if error:
                    raise RuntimeError("Simulated CUDA out of memory")
                return Tensor([1, 2] + ([8] if special_prefix else []) + [3, 4, 9])

        torch = SimpleNamespace(__version__=version, bfloat16="fixture", manual_seed=lambda _: None,
                                cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
                                inference_mode=contextlib.nullcontext)
        transformers = SimpleNamespace(__version__="4.51.3",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model()))
        argv = [str(RUNNER_PATH), "--input-jsonl", str(self.input), "--output-jsonl", str(self.output),
                "--run-manifest", str(self.manifest), "--model", "/fixture/model", "--revision", "fixed"]
        if resume:
            argv.append("--resume")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            exit_code = runner.main()
        return exit_code, json.loads(self.manifest.read_text()), calls

    def test_characterize_zero_valid_schema_returns_success(self):
        code, manifest, _ = self.run_fake(text='{"ok":1,"specificity":0}')
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 0)
        self.assertEqual(manifest["status"], "completed_local_run")
        self.assertEqual(code, 0)

    def test_characterize_all_inference_errors_return_success(self):
        code, manifest, _ = self.run_fake(error=True)
        self.assertEqual(manifest["counts"]["row_errors"], 1)
        self.assertEqual(code, 0)

    def test_characterize_resume_skips_failed_rows(self):
        self.run_fake(error=True)
        code, manifest, calls = self.run_fake(resume=True)
        self.assertEqual(calls, [])
        self.assertEqual(manifest["counts"]["row_errors"], 1)
        self.assertEqual(code, 0)

    def test_characterize_resume_accepts_runtime_drift(self):
        self.run_fake()
        _, manifest, calls = self.run_fake(resume=True, version="changed-runtime-fixture")
        self.assertEqual(calls, [])
        self.assertEqual(manifest["runtime"]["torch"], "changed-runtime-fixture")
        self.assertNotIn("runtime", manifest["run_binding"])

    def test_characterize_special_prefix_removed_before_validation(self):
        _, manifest, _ = self.run_fake(special_prefix=True)
        row = json.loads(self.output.read_text())
        self.assertTrue(row["raw_output_with_special_tokens"].startswith("<|im_start|>"))
        self.assertEqual(row["raw_output"], '{"ok":1,"specificity":4}')
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
