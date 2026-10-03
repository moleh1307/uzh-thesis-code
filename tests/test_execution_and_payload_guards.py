import json
import unittest

from runner_fixtures import FakeRunnerCase


class ExecutionAndPayloadTests(FakeRunnerCase):
    def assert_rejected_without_write(self, **kwargs):
        original_output = self.output.read_bytes()
        original_manifest = self.manifest.read_bytes()
        with self.assertRaises(SystemExit):
            self.run_fake(resume=True, **kwargs)
        self.assertEqual(self.output.read_bytes(), original_output)
        self.assertEqual(self.manifest.read_bytes(), original_manifest)

    def test_runtime_drift_rejected_before_resume_write(self):
        self.run_fake()
        self.assert_rejected_without_write(version="changed-runtime")

    def test_checkpoint_content_drift_rejected(self):
        self.run_fake()
        (self.model_dir / "model.safetensors").write_text("changed synthetic weights")
        self.assert_rejected_without_write()

    def test_tokenizer_file_drift_rejected(self):
        self.run_fake()
        (self.model_dir / "tokenizer_config.json").write_text("changed synthetic tokenizer")
        self.assert_rejected_without_write()

    def test_effective_template_drift_rejected(self):
        self.run_fake()
        self.assert_rejected_without_write(template="changed template")

    def test_inherited_penalty_not_silently_overridden(self):
        self.run_fake()
        self.assert_rejected_without_write(penalty=1.0)

    def test_beam_search_rejected(self):
        self.run_fake()
        self.assert_rejected_without_write(beams=2)

    def test_actual_device_placement_drift_rejected(self):
        self.run_fake()
        self.assert_rejected_without_write(device="cuda:1")

    def test_effective_generation_config_drift_rejected(self):
        self.run_fake()
        self.assert_rejected_without_write(min_new_tokens=2)

    def test_profile_approval_required(self):
        self.run_fake()
        profile = json.loads(self.profile.read_text())
        profile["status"] = "candidate_requires_review"
        self.profile.write_text(json.dumps(profile))
        self.assert_rejected_without_write()

    def test_settings_drift_rejected(self):
        self.run_fake()
        settings = json.loads(self.settings.read_text())
        settings["decoding"]["do_sample"] = True
        self.settings.write_text(json.dumps(settings))
        self.assert_rejected_without_write()

    def test_first_run_provenance_preserved(self):
        _, first, _ = self.run_fake()
        _, resumed, _ = self.run_fake(resume=True)
        self.assertEqual(first["created_at_utc"], resumed["created_at_utc"])
        self.assertEqual(first["first_run_provenance"], resumed["first_run_provenance"])
        self.assertEqual(resumed["effective_generation_config"]["repetition_penalty"], 1.05)
        self.assertEqual(resumed["effective_generation_config"]["num_beams"], 1)

    def test_special_prefix_is_preserved_and_rejected(self):
        code, manifest, _ = self.run_fake(special_prefix=True)
        row = json.loads(self.output.read_text())
        self.assertEqual(code, 1)
        self.assertTrue(row["raw_output"].startswith("<|im_start|>"))
        self.assertEqual(row["unexpected_special_token_ids"], [8])
        self.assertEqual(row["generated_token_ids"], [8, 3, 4, 9])
        self.assertEqual(manifest["counts"]["strict_valid_outputs"], 0)

    def test_only_terminal_eos_is_removed(self):
        code, _, _ = self.run_fake()
        row = json.loads(self.output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(row["raw_output_with_special_tokens"], row["raw_output"] + "<|im_end|>")
        self.assertEqual(row["terminal_eos_token_id"], 9)

    def test_capture_does_not_score_or_approve(self):
        code, _, calls = self.run_fake(capture=True)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.manifest.exists())
        candidate = json.loads((self.root / "candidate.json").read_text())
        self.assertEqual(candidate["status"], "candidate_requires_review")

    def test_external_weight_shard_is_rejected(self):
        self.run_fake()
        index = self.model_dir / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"fixture": "../outside.safetensors"}}))
        self.assert_rejected_without_write()


if __name__ == "__main__":
    unittest.main()
