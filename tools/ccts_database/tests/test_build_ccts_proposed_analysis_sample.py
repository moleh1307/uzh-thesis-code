import csv
import json
import shutil
import unittest

import test_extract_execucomp_confirmed_ceo_qa_blocks as extraction_fixture
from extract_execucomp_confirmed_ceo_qa_blocks import run_extraction
from build_ccts_proposed_analysis_sample import classify_block, content_decision, coverage, known_management_context, run


class ProposedSampleTests(unittest.TestCase):
    def test_short_indirect_and_mixed_targets_remain(self):
        for text in (
            "Yes.", "We cannot provide that forecast at this stage.",
            "We remain optimistic despite the uncertainty.",
            "Thank you. Revenue increased ten percent this quarter.",
        ):
            self.assertTrue(content_decision(text)[2], text)
        self.assertFalse(content_decision("")[2])
        self.assertFalse(content_decision("Thank you.")[2])

    def test_management_context_requires_named_same_issuer_role(self):
        block = {"validated_ceo_speaker": "David Smith, Acme Inc - CEO", "context_sequence_ids": '["11"]'}
        block["validated_ceo_speaker_json"] = json.dumps([block["validated_ceo_speaker"]])
        source = {"11": {"text_name": "Jane Jones, Acme Inc - CFO [11]"}}
        self.assertTrue(known_management_context(block, source, "david smith, acme inc"))
        for speaker in ("Jane Jones, Other Inc - CFO", "Unidentified Company Representative", "Jane Jones, Acme Inc - Guest"):
            source["11"]["text_name"] = speaker
            self.assertFalse(known_management_context(block, source, "david smith, acme inc"))

    def test_block_decisions_keep_context_and_review_separate(self):
        block = {
            "block_id": "b1", "validated_ceo_speaker": "David Smith, Acme Inc - CEO",
            "context_sequence_ids": '["11"]', "ceo_answer": "Yes.",
            "ceo_answer_word_count": "1", "analyst_question": "Has revenue increased?",
            "block_flags": "SHORT_CEO_ANSWER", "quality_tier": "high",
        }
        block["validated_ceo_speaker_json"] = json.dumps([block["validated_ceo_speaker"]])
        source = {"11": {"text_name": "Jane Jones, Acme Inc - CFO [11]"}}
        decision = classify_block(block, source, "david smith, acme inc")
        self.assertEqual(decision["primary_block_candidate"], 1)
        self.assertEqual(decision["model_ok"], "")
        block.update(quality_tier="usable_with_flag", block_flags="SHORT_CEO_ANSWER;NON_CEO_MANAGEMENT_CONTEXT")
        decision = classify_block(block, source, "david smith, acme inc")
        self.assertEqual(decision["primary_block_candidate"], 0)
        self.assertEqual(decision["management_context_block_candidate"], 1)
        block["analyst_question"] = "Has revenue increased..."
        self.assertEqual(classify_block(block, source, "david smith, acme inc")["inclusion_status"], "source_or_boundary_review")
        block.update(ceo_answer="Thank you.", ceo_answer_word_count="2")
        self.assertEqual(classify_block(block, source, "david smith, acme inc")["inclusion_status"], "exclude_procedural_or_empty")

    def test_support_requires_distinct_quarters_and_both_ceos(self):
        episodes = [{"gvkey": "1", "expected_execid": str(i), "tenure_episode": "1", "episode_analysis_gate_pass": "1"} for i in (1, 2)]
        turnovers = [{"gvkey": "1", "old_execid": "1", "new_execid": "2", "old_tenure_episode": "1", "new_tenure_episode": "1", "turnover_id": "t1", "turnover_analysis_gate_pass": "1"}]
        calls = []
        for ceo in (1, 2):
            for q in range(5):
                calls.append({"gvkey": "1", "expected_execid": str(ceo), "tenure_episode": "1", "calendar_quarter": f"{2020 + q}Q1" if ceo == 1 else "2020Q1", "primary_call_candidate": 1, "management_context_call_candidate": 1, "qa_coverage_call_candidate": 0})
        ep, tr = coverage(calls, episodes, turnovers)
        self.assertEqual([r["primary_episode_candidate"] for r in ep], [1, 0])
        self.assertEqual(tr[0]["primary_turnover_candidate"], 0)
        for q, call in enumerate(calls[5:]):
            call["calendar_quarter"] = f"{2021 + q}Q1"
        ep, tr = coverage(calls, episodes, turnovers)
        self.assertEqual(tr[0]["primary_turnover_candidate"], 1)
        self.assertEqual(sum(c["primary_turnover_sample_candidate"] for c in calls), 10)

    def test_end_to_end_source_reconciliation_and_tamper_rejection(self):
        fixture = extraction_fixture.ConfirmedCeoQaExtractionTests()
        fixture.setUp()
        try:
            fixture.rebuild_title_gate("Chairman; President; CEO")
            run_extraction(fixture.turns, fixture.event_gate, fixture.episode_gate, fixture.turnover_gate, fixture.gate_summary, fixture.output, manual_audit_blocks=0)
            root = fixture.root / "fetch"
            (root / "derived_v1").mkdir(parents=True)
            gate = root / "speaker_gate_v1_2"
            gate.mkdir()
            shutil.copyfile(fixture.turns, root / "derived_v1/ccts_turns_analysis_v1.csv")
            for source, name in ((fixture.event_gate, "event_speaker_gate.csv"), (fixture.episode_gate, "episode_speaker_gate.csv"), (fixture.turnover_gate, "turnover_speaker_gate.csv"), (fixture.gate_summary, "speaker_gate_summary.json")):
                shutil.copyfile(source, gate / name)
            blocks = root / "ceo_qa_blocks_speaker_gate_v1_20260928"
            shutil.copytree(fixture.output, blocks)
            output = fixture.root / "sample"
            result = run(root, output, minimum_calls=1, minimum_quarters=1)
            self.assertEqual(result["candidate_calls"], 2)
            self.assertEqual(result["source_turns_reconciled"], 6)
            self.assertEqual(result["qa_blocks_reconstructed"], 1)
            self.assertEqual(result["lanes"]["primary"]["paired_calls"], 1)
            self.assertEqual(result["lanes"]["primary"]["supported_turnovers"], 0)
            with (output / "call_level_sample.csv").open(newline="") as handle:
                call = next(csv.DictReader(handle))
            self.assertEqual(call["issuer_label_from_ceo_speaker"], "Acme Inc")
            self.assertEqual(json.loads(call["validated_ceo_speaker_json"]),
                             ["David Smith, Acme Inc - Chairman; President; CEO"])
            with (output / "ceo_pre_units.csv").open(newline="") as handle:
                units = list(csv.DictReader(handle))
            self.assertEqual(units[0]["ceo_presentation_segment"], "We expect sales to grow ten percent this year.")
            self.assertEqual(units[0]["proposed_unit_candidate"], "1")
            second = fixture.root / "sample_repeat"
            run(root, second, minimum_calls=1, minimum_quarters=1)
            for name in ("call_level_sample.csv", "episode_sample.csv", "turnover_sample.csv", "qa_block_decisions.csv", "ceo_pre_units.csv", "ceo_presentations.csv"):
                self.assertEqual((output / name).read_bytes(), (second / name).read_bytes())
            block_file = blocks / "ceo_qa_blocks.csv"
            summary_file = blocks / "block_extraction_summary.json"
            original_blocks, original_summary = block_file.read_bytes(), summary_file.read_bytes()
            altered = extraction_fixture.producer.read_rows(block_file)
            altered[0]["validated_ceo_speaker_json"] = '["Different Person, Acme Inc - CEO"]'
            extraction_fixture.write_csv(block_file, altered)
            rebound = json.loads(summary_file.read_text())
            rebound["artifact_sha256"]["ceo_qa_blocks.csv"] = extraction_fixture.sha256_file(block_file)
            summary_file.write_text(json.dumps(rebound))
            with self.assertRaisesRegex(ValueError, "structured speaker evidence disagreement"):
                run(root, fixture.root / "rejected_speaker_evidence")
            self.assertFalse((fixture.root / "rejected_speaker_evidence").exists())
            block_file.write_bytes(original_blocks)
            summary_file.write_bytes(original_summary)
            with (blocks / "ceo_qa_blocks.csv").open("a") as handle:
                handle.write("\n")
            rejected = fixture.root / "rejected"
            with self.assertRaisesRegex(ValueError, "hash differs"):
                run(root, rejected)
            self.assertFalse(rejected.exists())
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
