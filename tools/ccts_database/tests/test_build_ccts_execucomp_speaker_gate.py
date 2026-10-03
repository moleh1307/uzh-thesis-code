import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_ccts_execucomp_speaker_gate import classify_event, make_candidate_map


def source(event_id: str = "1") -> dict[str, str]:
    return {
        "event_id": event_id,
        "gvkey": "000001",
        "execid": "00001",
        "tenure_episode": "1",
        "event_date": "2020-03-15",
        "calendar_quarter": "2020Q1",
        "turnover_ids": "turnover_1",
    }


def turn(name: str, section: str = "PRE") -> dict[str, str]:
    return {"text_name": name, "analysis_text_type": section}


class CctsExecucompSpeakerGateTests(unittest.TestCase):
    def test_exact_shared_external_ceo_passes(self) -> None:
        pre = make_candidate_map([turn("David Smith, Acme Corp - CEO [1]")])
        qa = make_candidate_map([turn("David Smith, Acme Corp - Chief Executive Officer [9]", "Q&A")])
        result = classify_event(source(), "David P. Smith", pre, qa, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["external_speaker_validation_status"], "confirmed_external_ceo_shared_pre_qa")
        self.assertEqual(result["event_speaker_gate_pass"], 1)
        self.assertEqual(result["within_call_identity_ready"], 1)

    def test_matching_name_not_same_shared_speaker_is_review(self) -> None:
        pre = make_candidate_map([turn("David Smith, Acme Corp - CEO [1]")])
        qa = make_candidate_map([turn("David Smith, Acme Holdings - CEO [9]", "Q&A")])
        result = classify_event(source(), "David P. Smith", pre, qa, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["external_speaker_validation_status"], "review_external_ceo_present_both_sections_not_strict_shared")
        self.assertEqual(result["event_speaker_gate_pass"], 0)

    def test_interim_ceo_is_not_strictly_confirmed(self) -> None:
        label = "David Smith, Acme Corp - Interim CEO [1]"
        candidates = make_candidate_map([turn(label)])
        result = classify_event(source(), "David P. Smith", candidates, candidates, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["within_call_identity_status"], "review_special_ceo_title")
        self.assertEqual(result["external_speaker_validation_status"], "review_external_ceo_present_both_sections_not_strict_shared")
        self.assertEqual(result["event_speaker_gate_pass"], 0)

    def test_acting_ceo_is_not_strictly_confirmed(self) -> None:
        candidates = make_candidate_map([turn("David Smith, Acme Corp - Acting CEO [1]")])
        result = classify_event(source(), "David P. Smith", candidates, candidates, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["within_call_identity_status"], "review_special_ceo_title")
        self.assertEqual(result["event_speaker_gate_pass"], 0)

    def test_acting_cfo_does_not_make_ceo_title_special(self) -> None:
        candidates = make_candidate_map([turn("David Smith, Acme Corp - CEO, Acting CFO, President [1]")])
        result = classify_event(source(), "David P. Smith", candidates, candidates, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["within_call_identity_status"], "exact_single_shared_speaker_key")
        self.assertEqual(result["event_speaker_gate_pass"], 1)

    def test_same_initial_and_surname_requires_review(self) -> None:
        candidates = make_candidate_map([turn("D. Smith, Acme Corp - CEO [1]")])
        result = classify_event(source(), "David P. Smith", candidates, candidates, {"all": 2, "PRE": 1, "Q&A": 1, "other": 0})
        self.assertEqual(result["external_speaker_validation_status"], "review_external_ceo_initial_last_name")
        self.assertEqual(result["event_speaker_gate_pass"], 0)


if __name__ == "__main__":
    unittest.main()
