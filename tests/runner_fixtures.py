"""Synthetic inference harness; no model, GPU, downloads or API calls."""

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


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools/llm_measurement/run_local_specificity.py"
spec = importlib.util.spec_from_file_location("synthetic_runner", RUNNER_PATH)
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


class FakeRunnerCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.manifest = self.root / "manifest.json"
        self.requests(1)

    def requests(self, count):
        rows = [{"custom_id": f"FIXTURE_{index}", "response_schema": {},
                 "messages": [{"role": "system", "content": "Synthetic scoring fixture"},
                              {"role": "user", "content": json.dumps({"unit_type": "pre",
                               "ceo_presentation_segment": "Synthetic business statement"})}]}
                for index in range(count)]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def run_fake(self, text='{"ok":1,"specificity":4}', error=False,
                 special_prefix=False, resume=False, version="2.8.0+cu128",
                 error_on_call=None, interrupt_on_call=None, load_error=False,
                 eos=True, max_tokens=16, observe=None):
        calls = []
        snapshots = []
        original_write = runner.atomic_json_write

        def observed_write(path, value):
            original_write(path, value)
            snapshots.append(json.loads(path.read_text()))

        class Tokenizer:
            eos_token_id = 9
            model_max_length = 32768

            def apply_chat_template(self, *args, **kwargs):
                return "fixture"

            def __call__(self, *args, **kwargs):
                return {"input_ids": Tensor([1, 2], matrix=True)}

            def decode(self, tokens, skip_special_tokens):
                content = text[len(calls) - 1] if isinstance(text, list) else text
                return content if skip_special_tokens else (
                    ("<|im_start|>" if special_prefix else "") + content + ("<|im_end|>" if eos else ""))

        class Model:
            config = SimpleNamespace(max_position_embeddings=32768)
            generation_config = SimpleNamespace(eos_token_id=9, temperature=.7, top_p=.8, top_k=20)

            def eval(self):
                pass

            def parameters(self):
                return iter([SimpleNamespace(device="fixture")])

            def generate(self, **kwargs):
                calls.append(kwargs)
                if interrupt_on_call == len(calls):
                    raise KeyboardInterrupt()
                if error or error_on_call == len(calls):
                    raise RuntimeError("Synthetic inference failure")
                return Tensor([1, 2] + ([8] if special_prefix else []) + [3, 4] + ([9] if eos else []))

        def load_model(*args, **kwargs):
            if load_error:
                raise RuntimeError("Synthetic model-load failure")
            return Model()

        torch = SimpleNamespace(__version__=version, bfloat16="fixture", manual_seed=lambda _: None,
                                cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
                                inference_mode=contextlib.nullcontext)
        transformers = SimpleNamespace(__version__="4.51.3",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
        argv = [str(RUNNER_PATH), "--input-jsonl", str(self.input), "--output-jsonl", str(self.output),
                "--run-manifest", str(self.manifest), "--model", "/fixture/model", "--revision", "fixed",
                "--max-new-tokens", str(max_tokens)]
        if resume:
            argv.append("--resume")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
             patch.object(sys, "argv", argv), patch.object(runner, "atomic_json_write", observed_write), \
             contextlib.redirect_stdout(io.StringIO()):
            exit_code = runner.main()
        if observe is not None:
            observe.extend(snapshots)
        return exit_code, json.loads(self.manifest.read_text()), calls
