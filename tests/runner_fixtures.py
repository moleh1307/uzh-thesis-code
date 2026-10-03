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
sys.path.insert(0, str(RUNNER_PATH.parent))
from specificity_validation import SCHEMA
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
        self.model_dir = self.root / "checkpoint"
        self.model_dir.mkdir()
        for name in ("config.json", "generation_config.json", "tokenizer_config.json", "model.safetensors"):
            (self.model_dir / name).write_text("synthetic fixture")
        self.settings = self.root / "settings.json"
        self.profile = self.root / "profile.json"
        self.units = self.root / "units.csv"
        self.requests(1)

    def requests(self, count):
        rows = [{"custom_id": f"FIXTURE_{index}", "response_schema": SCHEMA,
                 "messages": [{"role": "system", "content": "Synthetic scoring fixture"},
                              {"role": "user", "content": json.dumps({"unit_type": "pre",
                               "ceo_presentation_segment": "Synthetic business statement"})}]}
                for index in range(count)]
        self.input.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.units.write_text("custom_id\n" + "".join(f"FIXTURE_{i}\n" for i in range(count)))

    def run_fake(self, text='{"ok":1,"specificity":4}', error=False,
                 special_prefix=False, resume=False, version="2.8.0+cu128",
                 error_on_call=None, interrupt_on_call=None, load_error=False,
                 eos=True, max_tokens=16, observe=None, penalty=1.05, beams=1,
                 device="cpu", template="fixture", capture=False, min_new_tokens=0, technical_retry=False):
        calls = []
        snapshots = []
        original_write = runner.atomic_json_write

        def observed_write(path, value):
            original_write(path, value)
            snapshots.append(json.loads(path.read_text()))

        class Tokenizer:
            eos_token_id = 9
            all_special_ids = [8, 9]
            model_max_length = 32768

            def get_chat_template(self):
                return template

            def get_vocab(self):
                return {"fixture": 3}

            def apply_chat_template(self, *args, **kwargs):
                return "fixture"

            def __call__(self, *args, **kwargs):
                return {"input_ids": Tensor([1, 2], matrix=True)}

            def decode(self, tokens, skip_special_tokens, clean_up_tokenization_spaces=False):
                content = text[len(calls) - 1] if isinstance(text, list) else text
                return (("<|im_start|>" if 8 in tokens else "") +
                        (content if 3 in tokens else "") + ("<|im_end|>" if 9 in tokens else ""))

        class GenerationConfig(SimpleNamespace):
            def to_dict(self):
                return dict(vars(self))

        class Model:
            config = SimpleNamespace(max_position_embeddings=32768)
            dtype = "torch.bfloat16"
            hf_device_map = {"": device}

            def __init__(self):
                self.generation_config = GenerationConfig(eos_token_id=9, temperature=.7,
                    top_p=.8, top_k=20, repetition_penalty=penalty, num_beams=beams,
                    num_return_sequences=1, min_new_tokens=min_new_tokens)

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
                body = [3] if eos and max_tokens == 2 else [3, 4]
                return Tensor([1, 2] + ([8] if special_prefix else []) + body + ([9] if eos else []))

        def load_model(*args, **kwargs):
            assert kwargs["local_files_only"] and kwargs["trust_remote_code"] is False
            assert kwargs["use_safetensors"]
            if load_error:
                raise RuntimeError("Synthetic model-load failure")
            return Model()

        torch = SimpleNamespace(__version__=version, version=SimpleNamespace(cuda="fixture"),
                                bfloat16="fixture", manual_seed=lambda _: None,
                                cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
                                inference_mode=contextlib.nullcontext)
        transformers = SimpleNamespace(__version__="4.51.3",
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model))
        argv = [str(RUNNER_PATH), "--input-jsonl", str(self.input), "--output-jsonl", str(self.output),
                "--run-manifest", str(self.manifest), "--model", str(self.model_dir), "--revision", "fixed",
                "--max-new-tokens", str(max_tokens), "--settings", str(self.settings),
                "--execution-profile", str(self.profile), "--unit-manifest-csv", str(self.units)]
        if not self.settings.exists():
            self.settings.write_text(json.dumps({"model": {"id": "synthetic", "revision": "fixed",
                "precision": "bfloat16", "quantization": "none", "device_map": "auto"},
                "decoding": {"max_new_tokens": max_tokens, "do_sample": False, "temperature": None,
                "top_p": None, "top_k": None, "seed": 0, "input_truncation": False}, "runtime": {"reference_versions":
                {"python": "fixture", "torch": "2.8.0+cu128", "transformers": "4.51.3"}},
                "migration_notes": {"inherited_checkpoint_generation": {"repetition_penalty": 1.05}}}))
        if resume:
            argv.append("--resume")
        if technical_retry:
            argv.append("--technical-retry")
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
             patch.object(sys, "argv", argv), patch.object(runner, "atomic_json_write", observed_write), \
             patch("local_execution_identity.platform.python_version", return_value="fixture"), \
             patch("local_execution_identity.importlib.metadata.version", return_value="fixture"), \
             contextlib.redirect_stdout(io.StringIO()):
            if not self.profile.exists():
                # Manufacture a reviewed synthetic profile, not a real model approval.
                with patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=Model()):
                    args = SimpleNamespace(model=str(self.model_dir), settings=self.settings,
                        execution_profile=None, capture_execution_profile=True,
                        revision="fixed", max_new_tokens=max_tokens)
                    identity = runner.prepare_backend(args)[2]
                self.profile.write_text(json.dumps({"status": "approved", "identity": identity}))
            if capture:
                argv.extend(["--capture-execution-profile", str(self.root / "candidate.json")])
            exit_code = runner.main()
        if observe is not None:
            observe.extend(snapshots)
        return exit_code, json.loads(self.manifest.read_text()) if self.manifest.exists() else {}, calls
