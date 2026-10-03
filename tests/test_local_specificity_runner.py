import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tools/llm_measurement/run_local_specificity.py"
sys.path.insert(0, str(path.parent))
spec = importlib.util.spec_from_file_location("specificity_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LocalSpecificityRunnerTests(unittest.TestCase):
    def test_valid_json_and_score_combinations(self):
        for ok in (0, 1):
            for score in range(6):
                parsed, error = runner.extract_json(f' {{"ok":{ok},"specificity":{score}}} \n')
                self.assertIsNone(error)
                self.assertEqual(runner.validate_output(parsed)[0], (ok == 0 and score == 0) or (ok == 1 and score > 0))

    def test_no_salvage_duplicates_or_nonfinite_values(self):
        for text in (
            '```json\n{"ok":1,"specificity":3}\n```',
            '<think>reasoning</think>{"ok":1,"specificity":3}',
            'Answer: {"ok":1,"specificity":3}',
            '{"ok":1,"specificity":3} trailing',
            '{"ok":1,"specificity":3,"specificity":5}',
            '{"ok":1,"specificity":NaN}',
            '{"ok":1,"specificity":Infinity}',
            '', '{"ok":1',
        ):
            self.assertIsNotNone(runner.extract_json(text)[1], text)

    def test_invalid_shapes_and_types(self):
        for value in ([], 3, None, {"ok":True,"specificity":3}, {"ok":1,"specificity":True},
                      {"ok":1,"specificity":3.0}, {"ok":1,"specificity":3,"reason":"extra"}):
            self.assertFalse(runner.validate_output(value)[0])

    def test_termination_at_exact_limit(self):
        self.assertFalse(runner.termination_metadata([1,2,9], [8,9], 3)["output_truncated"])
        self.assertEqual(runner.termination_metadata([1,2,3], [8,9], 3)["finish_reason"], "length")
        self.assertEqual(runner.termination_metadata([1], 9, 3)["finish_reason"], "unknown")
        self.assertEqual(runner.termination_metadata([], None, 3)["finish_reason"], "unknown")

    def test_context_limit_ignores_unbounded_tokenizer_sentinel(self):
        self.assertEqual(runner.context_limit(32768, 10**30), 32768)
        self.assertEqual(runner.context_limit(32768, 8192), 8192)
        with self.assertRaises(ValueError):
            runner.context_limit(None, 10**30)


if __name__ == "__main__":
    unittest.main()
