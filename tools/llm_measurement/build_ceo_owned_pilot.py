#!/usr/bin/env python3
"""Expand a reviewed strict pilot with source-valid, CEO-owned management answers.

Offline only. Requires a reviewed source publication and a bounded, previously
audited source trace. Does not infer post-session boundaries for the full corpus.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ccts_database"))
from csv_contract import configure_csv
configure_csv()
from build_ccts_proposed_analysis_sample import classify_block, PROCEDURAL_KINDS
from ccts_qa_episodes import procedure_kind
from ceo_title_evidence import label_evidence, single_anchor
from extract_ccts_ceo_qa_blocks import clean, speaker_key, word_count
from extract_execucomp_confirmed_ceo_qa_blocks import validate_blocks
from ceo_answer_policy import POLICY_ID, answer_policy, binary, qa_word_coverage
from prepare_scoring_unit_metadata import derive, FIELDS
from run_local_specificity import validate_request, validate_only


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def historical_code_hash(root, commit, relative):
    result = subprocess.run(["git", "-C", str(root), "show", commit + ":" + relative],
                            check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and all(reader.fieldnames)
                and len(reader.fieldnames) == len(set(reader.fieldnames)), "Invalid CSV header")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Invalid CSV width")
            yield row


def indexed(values, field):
    result = {}
    for row in values:
        key = row[field]
        require(key and key not in result, "Duplicate/empty " + field)
        result[key] = row
    return result


def write_csv(path, values, fields=None):
    values = list(values)
    require(fields or values, "Empty CSV needs an explicit schema")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_ledger(event, source_rows, blocks, decisions, anchor, audited_unassigned):
    """Reconcile raw CEO turns with block ownership and explicit boundary evidence."""
    owners = {}
    for block in blocks:
        for seq in json.loads(block["ceo_answer_sequence_ids"]):
            key = str(seq)
            require(key not in owners, "Duplicate CEO ownership")
            owners[key] = block["block_id"]
    raw = [r for r in source_rows if r["analysis_text_type"] == "Q&A" and speaker_key(r) == anchor]
    raw_ids = {r["sequence_id"] for r in raw}
    require(set(owners) <= raw_ids, "Answer ownership outside anchored CEO Q&A")
    indexed(source_rows, "sequence_id")
    result = []
    for row in raw:
        seq, text = row["sequence_id"], row["text_contents"]
        block = owners.get(seq, "")
        evidence = ""
        if block:
            category = ("procedure_only" if decisions[block]["inclusion_status"] ==
                        "exclude_procedural_or_empty" else "assigned_answer")
            evidence = decisions[block]["inclusion_reason"]
        else:
            category = "unresolved"
            audit = audited_unassigned.get((event, seq))
            if audit:
                require(clean(audit["ceo_text"]) == clean(text)
                        and int(audit["words"]) == word_count(text), "Audit text/word mismatch")
                if audit["category"] == "post_session_closing":
                    # A prior bounded source read, not a new lexical closing classifier.
                    position = source_rows.index(row)
                    require(position > 0, "Closing lacks preceding source context")
                    previous = source_rows[position - 1]
                    require(clean(previous["text_name"]) == clean(audit["previous_name"])
                            and clean(previous["text_contents"]) == clean(audit["previous_text"]),
                            "Closing audit context mismatch")
                    category, evidence = "post_session_closing", "previous_bounded_source_audit"
                elif audit["category"] == "recognized_procedure":
                    kind = procedure_kind(text)
                    require(kind == audit["procedure_kind"] and kind in PROCEDURAL_KINDS,
                            "Procedure audit not reproduced")
                    category, evidence = "procedure_only", "exact_procedural_pattern_" + kind
        result.append({"event_id": event, "sequence_id": seq, "category": category,
                       "word_count": word_count(text),
                       "block_id": block if category == "assigned_answer" else "",
                       "source_block_id": block, "evidence": evidence,
                       "source_inclusion_status": decisions[block]["inclusion_status"] if block else ""})
    for block in blocks:
        counted = sum(r["word_count"] for r in result if r["source_block_id"] == block["block_id"])
        require(counted == int(decisions[block["block_id"]]["original_ceo_target_word_count"]),
                "Block and turn word ledgers differ")
    return result


def build(args):
    prior, source, trace = args.strict_packet, args.source_dir, args.trace_dir
    out = args.output_dir
    require(not out.exists() and not out.is_symlink(), "Output already exists")
    require(args.evaluation_id.strip() and args.evaluation_id == args.evaluation_id.strip(), "Invalid evaluation ID")
    protected, historical_code = {}, {}
    code = Path(__file__).resolve().parents[2]

    def bind(path, expected=None):
        value = sha(path)
        require(expected is None or value == expected, "Bound source changed: " + str(path))
        protected[str(path.resolve())] = value

    receipt = load(prior / "source_review_receipt.json")
    require(receipt["status"] == "source_and_packet_review_passed_no_model_run"
            and not receipt["model_run"] and not receipt["forwarded"], "Reviewed unrun strict packet required")
    bind(prior / "source_review_receipt.json")
    for name, expected in receipt["artifact_sha256"].items():
        bind(prior / name, expected)
    plan = load(prior / "pilot-plan.json")
    for name, expected in load(prior / "preparation_manifest.json")["protected_source_sha256"].items():
        path = Path(name).resolve()
        if path.is_relative_to(code):
            # Old receipts bound the mutable checkout. Verify its original Git
            # blob, not today's documentation/tests, without editing the receipt.
            relative = str(path.relative_to(code))
            require(historical_code_hash(code, plan["source_code_commit"], relative) == expected,
                    "Historical producer code binding changed: " + relative)
            historical_code[relative] = expected
        else:
            bind(path, expected)
    require(Path(plan["source_publication"]).resolve() == source.resolve(), "Different source publication")
    events = plan["selection"]["selected_event_ids"]
    require(events and len(events) == len(set(events)), "Invalid selected calls")
    selected = set(events)
    summary = load(source / "rebuild_summary.json")
    require(summary["status"] == "complete_verified_proposed_pre_score_sample", "Incomplete source publication")
    bind(source / "rebuild_summary.json")
    bind(source / "artifact_sha256.json")
    publication_hashes = load(source / "artifact_sha256.json")
    for name in ("proposed_sample/call_level_sample.csv", "proposed_sample/episode_sample.csv",
                 "proposed_sample/turnover_sample.csv", "proposed_sample/qa_block_decisions.csv",
                 "ceo_qa_blocks/ceo_qa_blocks.csv"):
        bind(source / name, publication_hashes[name])
    bind(trace / "checks.json")
    trace_hashes = load(trace / "checks.json")["source_trace_artifact_sha256"]
    for name in ("turns.json", "gates.json", "unassigned_ceo_turns.csv"):
        bind(trace / name, trace_hashes[name])
    trace_rows = load(trace / "turns.json")
    require({r["event_id"] for r in trace_rows} == selected, "Trace call scope differs")
    grouped = defaultdict(list)
    for row in trace_rows:
        grouped[row["event_id"]].append(row)
    for event_rows in grouped.values():
        indexed(event_rows, "sequence_id")
        event_rows.sort(key=lambda r: int(r["sequence_id"]))
    gates = indexed(load(trace / "gates.json"), "event_id")
    require(set(gates) == selected, "Gate call scope differs")
    audited = {}
    for row in rows(trace / "unassigned_ceo_turns.csv"):
        key = row["event_id"], row["sequence_id"]
        require(key not in audited, "Duplicate audited turn")
        audited[key] = row
    blocks = indexed((r for r in rows(source / "ceo_qa_blocks/ceo_qa_blocks.csv")
                      if r["event_id"] in selected), "block_id")
    decisions = indexed((r for r in rows(source / "proposed_sample/qa_block_decisions.csv")
                         if r["event_id"] in selected), "block_id")
    require(set(blocks) == set(decisions), "Source decision/block scope differs")
    calls_all = list(rows(source / "proposed_sample/call_level_sample.csv"))
    calls = indexed((r for r in calls_all if r["event_id"] in selected), "event_id")
    require(set(calls) == selected, "Call missing from publication")
    policies, ledger, coverage = {}, [], []
    for event in events:
        require(binary(gates[event]["event_speaker_gate_pass"]) == 1, "Speaker gate failed")
        anchor = single_anchor(label_evidence(gates[event], "matched_shared_speakers"), speaker_key)
        event_blocks = [b for b in blocks.values() if b["event_id"] == event]
        validate_blocks(event, grouped[event], event_blocks, anchor)
        source_map = indexed(grouped[event], "sequence_id")
        for block in event_blocks:
            decision = decisions[block["block_id"]]
            require(decision["event_id"] == event, "Decision event mismatch")
            calculated = classify_block(block, source_map, anchor)
            for key, value in calculated.items():
                require(str(decision[key]) == str(value), "Source classification differs: " + key)
            policies[block["block_id"]] = answer_policy(decision)
        event_ledger = source_ledger(event, grouped[event], event_blocks, decisions, anchor, audited)
        ledger.extend(event_ledger)
        for lane, field in (("primary_ceo_owned", "policy_primary"), ("strict_robustness", "strict_robustness")):
            ids = [b["block_id"] for b in event_blocks if policies[b["block_id"]][field]]
            metrics = qa_word_coverage(event_ledger, ids)
            metrics["review_candidate_answer_words"] = sum(r["word_count"] for r in event_ledger
                if r["category"] == "assigned_answer" and r["source_inclusion_status"] == "source_or_boundary_review")
            metrics["denominator_interpretation"] = "source_linked_candidate_answers_not_verified_valid_answer_recall"
            require(metrics["raw_ceo_qa_words"] == int(calls[event]["source_qa_words"]), "Raw word ledger differs")
            expected_field = "management_context_qa_words" if field == "policy_primary" else "primary_qa_words"
            require(metrics["eligible_answer_words"] == int(calls[event][expected_field]), "Lane word ledger differs")
            coverage.append({"event_id": event, "lane": lane, "eligible_blocks": len(ids), **metrics})
    raw_prior = (prior / "specificity_requests.jsonl").read_bytes()
    require(raw_prior.endswith(b"\n"), "Prior JSONL lacks final newline")
    requests = [json.loads(line) for line in raw_prior.splitlines()]
    request_lookup = indexed(requests, "custom_id")
    keys = indexed(rows(prior / "source_key.csv"), "audit_id")
    prior_meta = list(rows(prior / "scoring_unit_metadata.csv"))
    require(set(keys) == set(request_lookup), "Prior request/key scope differs")
    system = (prior / "specificity-system-prompt.txt").read_text(encoding="utf-8")
    schema = load(prior / "specificity-production-schema.json")
    qa_prior = set()
    for request in requests:
        validate_request(request)
        require(request["messages"][0]["content"] == system and request["response_schema"] == schema,
                "Prior prompt/schema differs")
        key = keys[request["custom_id"]]
        require(key["event_id"] in selected, "Prior target outside selected calls")
        if key["unit_type"] == "qa":
            block_id = key["source_unit_id"]
            require(block_id not in qa_prior and policies[block_id]["strict_robustness"], "Prior QA not unique strict lane")
            qa_prior.add(block_id)
            target = json.loads(request["messages"][1]["content"])
            require(target == {"unit_type": "qa", "analyst_question": blocks[block_id]["analyst_question"],
                               "ceo_answer": blocks[block_id]["ceo_answer"]}, "Prior source target differs")
    require(qa_prior == {k for k, p in policies.items() if p["strict_robustness"]}, "Prior strict QA incomplete")
    require(derive(requests, keys) == [{k: int(v) if k in {"unit_word_count", "question_word_count"} else v
                                     for k, v in r.items()} for r in prior_meta], "Prior metadata differs")
    new_requests = []
    ordered_blocks = sorted((b for b in blocks.values() if policies[b["block_id"]]["added_identified_management"]),
                            key=lambda b: (events.index(b["event_id"]),
                                           tuple(int(s) for s in json.loads(b["question_sequence_ids"])), b["block_id"]))
    for index, block in enumerate(ordered_blocks, 1):
        custom_id = "CEOOWNED_ADD_" + f"{index:04d}"
        require(custom_id not in request_lookup, "New custom ID collision")
        target = {"unit_type": "qa", "analyst_question": block["analyst_question"], "ceo_answer": block["ceo_answer"]}
        request = {"custom_id": custom_id, "messages": [{"role": "system", "content": system},
                   {"role": "user", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))}],
                   "response_schema": schema}
        validate_request(request)
        event_keys = [k for k in keys.values() if k["event_id"] == block["event_id"]]
        require(event_keys and len({(k["start_date"], k["period_bin"]) for k in event_keys}) == 1, "Call metadata ambiguous")
        example = event_keys[0]
        keys[custom_id] = {"audit_id": custom_id, "unit_type": "qa", "event_id": block["event_id"],
                          "start_date": example["start_date"], "period_bin": example["period_bin"],
                          "source_unit_id": block["block_id"], "target_word_count": word_count(block["ceo_answer"]),
                          "question_word_count": word_count(block["analyst_question"]),
                          "source_content_sha256": hashlib.sha256((block["analyst_question"] + "\n" + block["ceo_answer"]).encode()).hexdigest()}
        new_requests.append(request)
    all_requests = requests + new_requests
    metadata = derive(all_requests, keys)
    require({keys[r["custom_id"]]["source_unit_id"] for r in all_requests
             if keys[r["custom_id"]]["unit_type"] == "qa"} == {k for k, p in policies.items() if p["policy_primary"]},
            "Broader primary QA incomplete")
    old_settings = load(prior / "execution-settings.json")
    require(args.evaluation_id != old_settings["evaluation"]["id"], "New evaluation ID required")
    settings = json.loads(json.dumps(old_settings))
    settings["status"] = "prepared_diagnostic_pilot_not_run_not_population_approved"
    settings["historical_strict_pilot_evaluation"] = settings["evaluation"]
    settings["evaluation"] = {**settings["evaluation"], "id": args.evaluation_id,
                              "calls": len(events), "units": len(all_requests), "units_by_type": dict(Counter(r["unit_type"] for r in metadata)),
                              "answer_policy_id": POLICY_ID, "primary_lane": "primary_ceo_owned",
                              "strict_robustness_units": len(requests), "model_run": False}
    overlays = {}
    for name, identities in (("call_level_sample.csv", ["event_id", "gvkey", "expected_execid", "tenure_episode"]),
                             ("episode_sample.csv", ["gvkey", "expected_execid", "tenure_episode"]),
                             ("turnover_sample.csv", ["turnover_id", "gvkey", "old_execid", "old_tenure_episode", "new_execid", "new_tenure_episode"])):
        values = calls_all if name == "call_level_sample.csv" else list(rows(source / "proposed_sample" / name))
        result = []
        for row in values:
            item = {k: row[k] for k in identities}
            item["answer_policy_id"] = POLICY_ID
            for key, value in row.items():
                for prefix in ("", "old_", "new_"):
                    broad = prefix + "management_context_"
                    strict = prefix + "primary_"
                    if key.startswith(broad):
                        item[prefix + "policy_primary_" + key[len(broad):]] = value
                    elif key.startswith(strict):
                        item[prefix + "strict_robustness_" + key[len(strict):]] = value
            require(any(k.startswith("policy_primary_") for k in item), "Missing overlay support")
            result.append(item)
        overlays[name.replace(".csv", "_policy_overlay.csv")] = result
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".ceo-owned-pilot-", dir=out.parent))
    try:
        payload = raw_prior + b"".join((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
                                     for r in new_requests)
        (temporary / "specificity_requests.jsonl").write_bytes(payload)
        write_csv(temporary / "scoring_unit_metadata.csv", metadata, FIELDS)
        write_csv(temporary / "source_key.csv", keys.values())
        review = list(rows(prior / "pilot_inputs_for_review.csv"))
        additions = [{"custom_id": r["custom_id"], "event_id": keys[r["custom_id"]]["event_id"], "unit_type": "qa",
                      "source_unit_id": keys[r["custom_id"]]["source_unit_id"], "ceo_target_words": keys[r["custom_id"]]["target_word_count"],
                      "analyst_question": json.loads(r["messages"][1]["content"])["analyst_question"],
                      "ceo_target": json.loads(r["messages"][1]["content"])["ceo_answer"]} for r in new_requests]
        require(len(review) == len(requests), "Review CSV count differs")
        write_csv(temporary / "pilot_inputs_for_review.csv", review + additions)
        write_csv(temporary / "added_management_qa_inputs.csv", additions, list(review[0]))
        write_csv(temporary / "qa_policy_decisions.csv", [{**d, **policies[k], "answer_policy_id": POLICY_ID}
                                                        for k, d in decisions.items()])
        write_csv(temporary / "qa_source_turn_ledger.csv", ledger)
        write_csv(temporary / "qa_coverage_by_call.csv", coverage)
        write_json(temporary / "qa_coverage_by_call.json", coverage)
        write_csv(temporary / "policy_membership.csv", [
            {"custom_id": r["custom_id"], "event_id": keys[r["custom_id"]]["event_id"],
             "unit_type": keys[r["custom_id"]]["unit_type"], "source_unit_id": keys[r["custom_id"]]["source_unit_id"],
             "primary_ceo_owned": 1, "strict_robustness": int(r in requests),
             "added_identified_management": int(r in new_requests), "answer_policy_id": POLICY_ID}
            for r in all_requests])
        call_review = list(rows(prior / "pilot_calls.csv"))
        for row in call_review:
            primary = next(c for c in coverage if c["event_id"] == row["event_id"] and c["lane"] == "primary_ceo_owned")
            row["strict_qa_units"] = row.pop("qa_units")
            row["strict_qa_included_words"] = row.pop("qa_included_words")
            row["strict_raw_qa_word_share"] = row.pop("qa_word_coverage")
            row["qa_sensitivity_only_words"] = "0"
            row["qa_units"] = primary["eligible_blocks"]
            row["qa_included_words"] = primary["eligible_answer_words"]
            row["raw_qa_word_share"] = primary["raw_qa_word_share"]
            row["answer_policy_id"] = POLICY_ID
        write_csv(temporary / "pilot_calls.csv", call_review)
        for name, values in overlays.items():
            write_csv(temporary / name, values)
        write_json(temporary / "execution-settings.json", settings)
        for name in ("specificity-system-prompt.txt", "specificity-production-schema.json"):
            shutil.copyfile(prior / name, temporary / name)
        new_plan = {**plan, "packet_id": args.evaluation_id, "answer_policy_id": POLICY_ID,
                    "status": "prepared_not_deployed_not_run", "historical_strict_packet": str(prior.resolve()),
                    "policy_decision": "User explicitly authorized identified-management-inclusive CEO-owned answers as primary; strict-only preserved for robustness, before new scoring.",
                    "scope": "Same calls, all existing PRE and strict QA plus every source-valid identified-management QA block; CEO text only, unchanged whole mixed answers. Review and procedure/empty blocks remain excluded.",
                    "authorization": "Local policy/packet/coverage implementation only. No inference, deployment or population-scoring approval.",
                    "counts": {"calls": len(events), "requests": len(all_requests),
                               "pre": sum(r["unit_type"] == "pre" for r in metadata),
                               "qa": sum(r["unit_type"] == "qa" for r in metadata),
                               "strict_requests": len(requests), "management_additions": len(new_requests)}}
        write_json(temporary / "pilot-plan.json", new_plan)
        write_json(temporary / "pre_score_policy_support.json", {
            "status": "availability_only_not_scored", "answer_policy_id": POLICY_ID,
            "primary_ceo_owned": summary["lanes_verified"]["management_context"],
            "strict_robustness": summary["lanes_verified"]["primary"],
            "source_lane_mapping": {"primary_ceo_owned": "management_context", "strict_robustness": "primary"},
            "no_new_coverage_threshold": True, "source_tables_unchanged": True})
        (temporary / "README.md").write_text("\n".join([
            "# CEO-Owned Answer Pilot", "", "Prepared only; no deployment or scoring.", "",
            f"Requests: {len(all_requests)}; PRE: {new_plan['counts']['pre']}; QA: {new_plan['counts']['qa']}; added management QA: {len(new_requests)}.", "",
            "## Decision", "", new_plan["policy_decision"], "",
            "The primary measure scores the CEO's own words in attributable analyst-answer exchanges, even when a named same-issuer manager participates. Manager speech is neither scored nor added to the model input. Unidentified or otherwise defective contexts remain review-only. Old source-lane names are preserved; explicit policy overlays select the new primary lane without relabeling historical tables.", "",
            "The old strict request file is an exact byte prefix of this packet. Strict robustness uses its original IDs and PRE units. No model/prompt/decoding change or trimming of mixed answers. New custom IDs are packet-scoped and must be joined with the evaluation identity.", "",
            "## Coverage", "",
            "1. Raw Q&A word share = eligible CEO answer words / all raw anchored CEO Q&A words. This is descriptive, not answer recall.",
            "2. Answer-inclusion coverage = eligible CEO answer words / source-linked nonprocedural candidate CEO answer words, including review-excluded candidates. Procedure/empty turns and separately audited post-session closings are outside this denominator. Review candidates may have uncertain question boundaries; their words are separately reported, so this ratio is not verified valid-answer recall.",
            "3. Scoring coverage = technically validated scorable CEO target words / eligible CEO target words. Before inference it is null (blank in CSV), not zero.", "",
            "Unassigned speech without supported classification stays unresolved. A second inclusion ratio treats all unresolved words as answers; this is a denominator sensitivity, not imputed text or scores. Full raw-word accounting does not prove semantic completeness. Closing classifications reuse a bounded prior source audit and are not generalized to the population. Full-corpus coverage remains a later source-audit task, not inferred from these calls.", "",
            "qa_coverage_by_call.json preserves explicit nulls; qa_source_turn_ledger.csv preserves every raw CEO Q&A turn. QA source and eligibility decisions are retained separately. No new arbitrary coverage exclusion threshold.", "",
            "## Instruction", "", "Exact unchanged system prompt:", "", "```text", system.rstrip(), "```", "",
            "## Limits And Next", "",
            "The same previously observed diagnostic calls are retained. Failed sampled-gap criteria remain failed. This packet is not independent validation or population-scoring approval. Human references remain unchanged; no new annotation round. Intended settings are preserved, but a fresh remote execution profile is required before a separately authorized launch. A long run should use a named detached screen with logs/status, then be retrieved later.", ""]), encoding="utf-8")
        require((temporary / "specificity_requests.jsonl").read_bytes()[:len(raw_prior)] == raw_prior, "Prior request prefix changed")
        validate_only(temporary / "specificity_requests.jsonl")
        for key in ("model", "runtime", "decoding", "packaging", "migration_notes"):
            require(load(temporary / "execution-settings.json")[key] == old_settings[key], "Execution settings changed")
        for name, expected in protected.items():
            require(sha(Path(name)) == expected, "Protected input changed during build")
        code_hashes = {str(p.relative_to(code)): sha(p) for p in sorted((code / "tools").rglob("*.py"))}
        write_json(temporary / "preparation_manifest.json", {
            "status": "verified_local_preparation_no_model_run", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "counts": new_plan["counts"], "answer_policy_id": POLICY_ID,
            "protected_source_sha256": protected, "code_sha256": code_hashes,
            "historical_code_binding": {"commit": plan["source_code_commit"], "sha256": historical_code},
            "artifacts": {p.name: sha(p) for p in sorted(temporary.iterdir()) if p.is_file()},
            "original_requests_preserved_byte_for_byte": True, "source_classification_and_ownership_verified": True,
            "source_review_note": "Same-assistant computational/source checks, not independent human validation.",
            "model_run": False, "forwarded": False})
        require(not out.exists() and not out.is_symlink(), "Output appeared during preparation")
        temporary.rename(out)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return new_plan["counts"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-packet", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-id", required=True)
    args = parser.parse_args()
    print(json.dumps({"status": "prepared_not_run", "counts": build(args), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
