#!/usr/bin/env python3
"""Read back a CEO-owned pilot without inference or writes to its artifacts."""
import argparse
import json
from collections import Counter
from pathlib import Path

from build_ceo_owned_pilot import historical_code_hash, indexed, load, rows, require, sha
from ceo_answer_policy import POLICY_ID, answer_policy
from prepare_scoring_unit_metadata import derive
from run_local_specificity import validate_request, validate_only


def verify_coverage(ledger, decisions, metrics):
    """Separate arithmetic read-back; does not call the coverage producer."""
    event = metrics["event_id"]
    source = [r for r in ledger if r["event_id"] == event]
    unique = indexed(source, "sequence_id")
    require(len(unique) == len(source), "Duplicate ledger turn")
    counts = Counter()
    block_words = Counter()
    review = 0
    for row in source:
        words = int(row["word_count"])
        require(words >= 0, "Negative source word count")
        category = row["category"]
        require(category in {"assigned_answer", "procedure_only", "post_session_closing", "unresolved"},
                "Unknown source category")
        counts[category] += words
        block = row["block_id"]
        require(bool(block) == (category == "assigned_answer"), "Ledger block/category mismatch")
        if block:
            require(block in decisions and decisions[block]["event_id"] == event, "Ledger decision mismatch")
            block_words[block] += words
            if decisions[block]["inclusion_status"] == "source_or_boundary_review":
                review += words
    field = {"primary_ceo_owned": "policy_primary", "strict_robustness": "strict_robustness"}[metrics["lane"]]
    selected = {key for key, value in decisions.items() if value["event_id"] == event and answer_policy(value)[field]}
    require(selected <= set(block_words), "Eligible block missing from source")
    numerator = sum(block_words[k] for k in selected)
    raw = sum(counts.values())
    assigned = counts["assigned_answer"]
    unresolved = counts["unresolved"]
    expected = {"raw_ceo_qa_words": raw, "assigned_answer_words": assigned,
                "procedure_only_words": counts["procedure_only"],
                "post_session_closing_words": counts["post_session_closing"], "unresolved_words": unresolved,
                "eligible_blocks": len(selected), "eligible_answer_words": numerator,
                "review_candidate_answer_words": review,
                "raw_qa_word_share": numerator / raw if raw else None,
                "answer_inclusion_coverage": numerator / assigned if assigned else None,
                "answer_inclusion_if_all_unresolved_are_answers": numerator / (assigned + unresolved) if assigned + unresolved else None}
    for key, value in expected.items():
        require(metrics[key] == value, "Coverage read-back differs: " + key)
    require(metrics["scoring_status"] == "not_run" and all(metrics[k] is None
        for k in ("validly_scored_words", "scoring_coverage", "score_status_words")), "Fabricated scoring coverage")
    return expected


def verify(packet):
    receipt = load(packet / "preparation_manifest.json")
    require(receipt["status"] == "verified_local_preparation_no_model_run" and not receipt["model_run"], "Wrong packet state")
    require(receipt["answer_policy_id"] == POLICY_ID, "Policy mismatch")
    for name, expected in receipt["artifacts"].items():
        require(sha(packet / name) == expected, "Packet file changed: " + name)
    for name, expected in receipt["protected_source_sha256"].items():
        require(sha(Path(name)) == expected, "Protected input changed")
    root = Path(__file__).resolve().parents[2]
    for name, expected in receipt["code_sha256"].items():
        require(sha(root / name) == expected, "Producer code changed: " + name)
    historical = receipt["historical_code_binding"]
    for name, expected in historical["sha256"].items():
        require(historical_code_hash(root, historical["commit"], name) == expected, "Historical producer binding differs")
    plan = load(packet / "pilot-plan.json")
    prior = Path(plan["historical_strict_packet"])
    original = (prior / "specificity_requests.jsonl").read_bytes()
    payload = (packet / "specificity_requests.jsonl").read_bytes()
    require(payload.startswith(original), "Original request prefix differs")
    requests = [json.loads(line) for line in payload.splitlines()]
    ids = indexed(requests, "custom_id")
    keys = indexed(rows(packet / "source_key.csv"), "audit_id")
    metadata = [{k: int(v) if k in {"unit_word_count", "question_word_count"} else v for k, v in r.items()}
                for r in rows(packet / "scoring_unit_metadata.csv")]
    require(derive(requests, keys) == metadata, "Metadata not exact request derivation")
    old_meta = list(rows(prior / "scoring_unit_metadata.csv"))
    require(list(rows(packet / "scoring_unit_metadata.csv"))[:len(old_meta)] == old_meta, "Old metadata cells changed")
    members = indexed(rows(packet / "policy_membership.csv"), "custom_id")
    review = indexed(rows(packet / "pilot_inputs_for_review.csv"), "custom_id")
    decisions = indexed(rows(packet / "qa_policy_decisions.csv"), "block_id")
    require(set(ids) == set(members) == set(review), "Request/member/review scope mismatch")
    old_ids = {json.loads(line)["custom_id"] for line in original.splitlines()}
    require(old_ids == {k for k, v in members.items() if v["strict_robustness"] == "1"}, "Strict subset changed")
    system = (prior / "specificity-system-prompt.txt").read_text(encoding="utf-8")
    schema = load(prior / "specificity-production-schema.json")
    qa_ids = set()
    for request in requests:
        validate_request(request)
        require(request["messages"][0]["content"] == system and request["response_schema"] == schema, "Prompt/schema differs")
        target = json.loads(request["messages"][1]["content"])
        key, member, readable = keys[request["custom_id"]], members[request["custom_id"]], review[request["custom_id"]]
        require(member["primary_ceo_owned"] == "1" and member["answer_policy_id"] == POLICY_ID, "Wrong primary membership")
        require(all(member[k] == key[k] for k in ("event_id", "unit_type", "source_unit_id")), "Member source differs")
        require(readable["ceo_target"] == target.get("ceo_answer", target.get("ceo_presentation_segment"))
                and readable["analyst_question"] == target.get("analyst_question", ""), "Readable input differs")
        if key["unit_type"] == "qa":
            source_id = key["source_unit_id"]
            require(source_id not in qa_ids, "Duplicate QA target")
            qa_ids.add(source_id)
            policy = answer_policy(decisions[source_id])
            require(policy["policy_primary"] and str(policy["strict_robustness"]) == member["strict_robustness"]
                    and str(policy["added_identified_management"]) == member["added_identified_management"], "QA policy differs")
        else:
            require(request["custom_id"] in old_ids, "Unexpected new PRE target")
    require(qa_ids == {k for k, d in decisions.items() if answer_policy(d)["policy_primary"]}, "Primary QA scope incomplete")
    ledger = list(rows(packet / "qa_source_turn_ledger.csv"))
    coverage = load(packet / "qa_coverage_by_call.json")
    pairs = {(r["event_id"], r["lane"]) for r in coverage}
    events = set(plan["selection"]["selected_event_ids"])
    require(len(pairs) == len(coverage) == 2 * len(events)
            and pairs == {(e, lane) for e in events for lane in ("primary_ceo_owned", "strict_robustness")}, "Coverage scope differs")
    for value in coverage:
        verify_coverage(ledger, decisions, value)
    csv_metrics = list(rows(packet / "qa_coverage_by_call.csv"))
    require(len(csv_metrics) == len(coverage), "CSV coverage scope differs")
    for csv_row, value in zip(csv_metrics, coverage):
        for key, item in value.items():
            require(csv_row[key] == ("" if item is None else str(item)), "CSV and JSON coverage differ")
    old_settings = load(prior / "execution-settings.json")
    settings = load(packet / "execution-settings.json")
    for key in ("model", "runtime", "decoding", "packaging", "migration_notes"):
        require(settings[key] == old_settings[key], "Model settings changed")
    require(settings["evaluation"]["id"] == plan["packet_id"]
            and settings["evaluation"]["units"] == len(requests), "Evaluation identity differs")
    overlays = {name: list(rows(packet / name)) for name in (
        "call_level_sample_policy_overlay.csv", "episode_sample_policy_overlay.csv", "turnover_sample_policy_overlay.csv")}
    indexed(overlays["call_level_sample_policy_overlay.csv"], "event_id")
    indexed(overlays["turnover_sample_policy_overlay.csv"], "turnover_id")
    support = load(packet / "pre_score_policy_support.json")
    for lane, prefix in (("primary_ceo_owned", "policy_primary_"), ("strict_robustness", "strict_robustness_")):
        calculated = {"paired_calls": sum(int(r[prefix + "call_candidate"]) for r in overlays["call_level_sample_policy_overlay.csv"]),
                      "supported_ceo_episodes": sum(int(r[prefix + "episode_candidate"]) for r in overlays["episode_sample_policy_overlay.csv"]),
                      "supported_turnovers": sum(int(r[prefix + "turnover_candidate"]) for r in overlays["turnover_sample_policy_overlay.csv"]),
                      "calls_in_supported_turnovers": sum(int(r[prefix + "turnover_sample_candidate"]) for r in overlays["call_level_sample_policy_overlay.csv"])}
        require(calculated == support[lane], "Pre-score support counts differ")
    require(dict(Counter(r["unit_type"] for r in metadata)) == settings["evaluation"]["units_by_type"], "Section counts differ")
    validate_only(packet / "specificity_requests.jsonl")
    return {"status": "read_back_verified_no_model_run", "counts": receipt["counts"],
            "original_request_prefix_and_metadata_preserved": True, "coverage_arithmetic_verified": True,
            "policy_overlay_support_verified": True, "manifest_sha256": sha(packet / "preparation_manifest.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.packet_dir), indent=2))


if __name__ == "__main__":
    main()
