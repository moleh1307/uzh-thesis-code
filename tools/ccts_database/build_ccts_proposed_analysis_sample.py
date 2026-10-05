#!/usr/bin/env python3
"""Build a proposed, pre-score call/episode/turnover sample from verified CCTS."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()

from ccts_qa_episodes import procedure_kind, source_quality_reasons
from extract_ccts_ceo_qa_blocks import clean, is_analyst, is_operator, join_turns, label, speaker_key, word_count
from extract_execucomp_confirmed_ceo_qa_blocks import load_speaker_gate, episode_key, sha256_file, validate_blocks
from ceo_title_evidence import label_evidence, single_anchor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llm_measurement"))
from build_specificity_scoring_manifest import split_long_turn

VERSION = "proposed_analysis_sample_v3_research_roles_20261005"
LANES = ("primary", "management_context", "qa_coverage")
BENIGN_FLAGS = {
    "SHORT_ANALYST_QUESTION", "SHORT_CEO_ANSWER",
    "ANALYST_ACKNOWLEDGEMENT_RETAINED_IN_CONTEXT", "OPERATOR_CLOSING_BOUNDARY",
}
PROCEDURAL_KINDS = {
    "greeting_only", "audio_check", "lookup_deferral", "delegation_only",
    "clarification_request", "logistics_question", "logistics_reply",
    "question_invitation", "closing_only",
}
MANAGEMENT_RE = re.compile(
    r"\b(?:chief|CFO|COO|CTO|CMO|president|treasurer|controller|"
    r"vice president|VP|EVP|SVP|director|head)\b", re.I
)
CASE_EVENT = "16742069"
CASE_BLOCK = "ccts_16742069_episode_16"


def content_decision(text):
    if not clean(text):
        return "procedural_only", "empty_target", False
    kind = procedure_kind(text)
    if kind in PROCEDURAL_KINDS:
        return "procedural_only", "exact_procedural_pattern_" + kind, False
    # Unclassified content is retained under the active specification.
    return "uncertain", "nonprocedural_text_retained_without_semantic_classification", True


def issuer_label(speaker):
    left = re.split(r"\s+-\s+", speaker, maxsplit=1)[0]
    return clean(left.rsplit(",", 1)[-1]) if "," in left else ""


def known_management_context(block, source, anchor):
    labels = label_evidence(block, "validated_ceo_speaker")
    single_anchor(labels, speaker_key)
    issuer = issuer_label(labels[0]).casefold()
    others = []
    for seq in json.loads(block["context_sequence_ids"]):
        turn = source[str(seq)]
        if speaker_key(turn) == anchor or is_analyst(turn, issuer_labels=labels) or is_operator(turn):
            continue
        parts = re.split(r"\s+-\s+", label(turn), maxsplit=1)
        if (len(parts) != 2 or not issuer
                or issuer_label(label(turn)).casefold() != issuer
                or not MANAGEMENT_RE.search(parts[1])
                or re.search(r"\bunidentified|unknown\b", parts[0], re.I)):
            return False
        others.append(turn)
    return bool(others)


def classify_block(block, source, anchor):
    words = word_count(block["ceo_answer"])
    if words != int(block["ceo_answer_word_count"]):
        raise ValueError("CEO answer word count mismatch: " + block["block_id"])
    flags = set(filter(None, block["block_flags"].split(";")))
    cls, content_reason, content_ready = content_decision(block["ceo_answer"])
    reasons = set(source_quality_reasons(block["ceo_answer"]))
    reasons.update("question_" + r for r in source_quality_reasons(block["analyst_question"]))
    if not clean(block["analyst_question"]):
        reasons.add("empty_question")
    primary = bool(content_ready and block["quality_tier"] == "high"
                   and not (flags - BENIGN_FLAGS) and not reasons)
    management = primary
    if (content_ready and not reasons and block["quality_tier"] == "usable_with_flag"
            and flags <= BENIGN_FLAGS | {"NON_CEO_MANAGEMENT_CONTEXT"}
            and "NON_CEO_MANAGEMENT_CONTEXT" in flags):
        management = known_management_context(block, source, anchor)
    if not content_ready:
        status, reason = "exclude_procedural_or_empty", content_reason
    elif primary:
        status, reason = "proposed_primary", "high_tier_source_screen_pass"
    elif management:
        status, reason = "sensitivity_only", "identified_same_issuer_management_context"
    else:
        status = "source_or_boundary_review"
        reason = ";".join(sorted(reasons | (flags - BENIGN_FLAGS))) or "tier_requires_review"
    return {
        "content_class": cls, "content_reason_code": content_reason,
        "original_ceo_target_word_count": words, "primary_block_candidate": int(primary),
        "management_context_block_candidate": int(management),
        "inclusion_status": status, "inclusion_reason": reason,
        "semantic_relevance_filter_applied": 0,
        "model_ok": "", "specificity": "", "model_unscorable_reason": "",
    }


def coverage(calls, episodes, turnovers, minimum_calls=5, minimum_quarters=5):
    grouped = defaultdict(list)
    for call in calls:
        grouped[episode_key(call)].append(call)
    episode_rows, lookup = [], {}
    for episode in episodes:
        row = dict(episode)
        key = episode_key(episode)
        eligible = grouped.get(key, [])
        for lane in LANES:
            selected = [c for c in eligible if int(c[lane + "_call_candidate"])]
            quarters = {c["calendar_quarter"] for c in selected}
            row[lane + "_calls"] = len(selected)
            row[lane + "_quarters"] = len(quarters)
            row[lane + "_episode_candidate"] = int(
                bool(int(episode["episode_analysis_gate_pass"]))
                and len(selected) >= minimum_calls and len(quarters) >= minimum_quarters
            )
            row[lane + "_reason"] = (
                "meets_support" if row[lane + "_episode_candidate"] else
                "speaker_episode_gate_failed" if not int(episode["episode_analysis_gate_pass"]) else
                "insufficient_paired_calls_or_distinct_quarters"
            )
        row["minimum_paired_calls"] = minimum_calls
        row["minimum_paired_quarters"] = minimum_quarters
        lookup[key] = row
        episode_rows.append(row)
    turnover_rows = []
    memberships = {lane: set() for lane in LANES}
    for turnover in turnovers:
        row = dict(turnover)
        sides = {}
        for side in ("old", "new"):
            key = (row["gvkey"], str(int(row[side + "_execid"])), row[side + "_tenure_episode"])
            if key not in lookup:
                raise ValueError("turnover refers to absent CEO episode: " + row["turnover_id"])
            sides[side] = key
            ep = lookup[key]
            for lane in LANES:
                for measure in ("calls", "quarters", "episode_candidate"):
                    row[side + "_" + lane + "_" + measure] = ep[lane + "_" + measure]
        for lane in LANES:
            ready = bool(int(row["turnover_analysis_gate_pass"])) and all(
                row[side + "_" + lane + "_episode_candidate"] for side in ("old", "new")
            )
            row[lane + "_turnover_candidate"] = int(ready)
            row[lane + "_reason"] = (
                "both_sides_meet_support" if ready else
                "speaker_turnover_gate_failed" if not int(row["turnover_analysis_gate_pass"]) else
                "one_or_both_sides_below_paired_call_support"
            )
            if ready:
                memberships[lane].update(sides.values())
        turnover_rows.append(row)
    for call in calls:
        ep = lookup[episode_key(call)]
        for lane in LANES:
            call[lane + "_episode_candidate"] = ep[lane + "_episode_candidate"]
            call[lane + "_turnover_sample_candidate"] = int(
                int(call[lane + "_call_candidate"]) and episode_key(call) in memberships[lane]
            )
    return episode_rows, turnover_rows


def write_rows(path, rows):
    if not rows:
        raise ValueError("empty output table: " + str(path))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def make_writer(handle, fields):
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    return writer


def run(root, output, minimum_calls=5, minimum_quarters=5):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("existing output: " + str(output))
    gate_dir = root / "speaker_gate_v1_2"
    block_dir = root / "ceo_qa_blocks_speaker_gate_v1_20260928"
    turns = root / "derived_v1/ccts_turns_analysis_v1.csv"
    blocks_path = block_dir / "ceo_qa_blocks.csv"
    extraction_summary_path = block_dir / "block_extraction_summary.json"
    summary = json.loads(extraction_summary_path.read_text())
    if summary["status"] != "complete":
        raise ValueError("incomplete extraction summary")
    print("Verifying source and extraction checksums...", flush=True)
    gate, episodes, turnovers, gate_summary, turns_hash = load_speaker_gate(
        gate_dir / "event_speaker_gate.csv", gate_dir / "episode_speaker_gate.csv",
        gate_dir / "turnover_speaker_gate.csv", gate_dir / "speaker_gate_summary.json", turns,
    )
    blocks_hash = sha256_file(blocks_path)
    if blocks_hash != summary["artifact_sha256"]["ceo_qa_blocks.csv"]:
        raise ValueError("block CSV hash differs from extraction summary")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix="." + output.name + ".tmp-", dir=output.parent))
    calls, seen, all_block_ids = [], set(), set()
    total_turns = total_blocks = 0
    block_status = Counter()
    pre_status = Counter()
    case_rows, case_blocks = [], []
    qa_fields = [
        "block_id", "event_id", "gvkey", "expected_execid", "tenure_episode",
        "event_date", "quality_tier", "block_flags", "question_sequence_ids",
        "ceo_answer_sequence_ids", "context_sequence_ids", "content_class",
        "content_reason_code", "original_ceo_target_word_count", "primary_block_candidate",
        "management_context_block_candidate", "inclusion_status", "inclusion_reason",
        "semantic_relevance_filter_applied", "model_ok", "specificity", "model_unscorable_reason",
    ]
    pre_fields = [
        "unit_id", "event_id", "sequence_id", "raw_sequence_id", "chunk_index",
        "word_count", "content_class", "content_reason_code", "source_review_reasons",
        "proposed_unit_candidate", "inclusion_reason", "ceo_presentation_segment",
    ]
    presentation_fields = [
        "event_id", "gvkey", "expected_execid", "tenure_episode", "event_date",
        "validated_ceo_speaker", "validated_ceo_speaker_json", "pre_sequence_ids", "source_pre_word_count",
        "ceo_presentation_text",
    ]
    try:
        with (temp / "qa_block_decisions.csv").open("w", newline="", encoding="utf-8") as qout, \
                (temp / "ceo_pre_units.csv").open("w", newline="", encoding="utf-8") as pout, \
                (temp / "ceo_presentations.csv").open("w", newline="", encoding="utf-8") as presentations, \
                turns.open(newline="", encoding="utf-8-sig") as source_handle, \
                blocks_path.open(newline="", encoding="utf-8-sig") as block_handle:
            qw = make_writer(qout, qa_fields)
            pw = make_writer(pout, pre_fields)
            presentation_writer = make_writer(presentations, presentation_fields)
            block_groups = iter(groupby(csv.DictReader(block_handle), key=lambda r: r["event_id"]))
            next_blocks = next(block_groups, None)
            source_reader = csv.DictReader(source_handle)
            for event_id, rows_iter in groupby(source_reader, key=lambda r: r["event_id"]):
                rows = list(rows_iter)
                if event_id in seen or event_id not in gate:
                    raise ValueError("repeated or unexpected source event " + event_id)
                seen.add(event_id)
                g = gate[event_id]
                if len(rows) != int(g["turn_rows"]):
                    raise ValueError("source count differs from gate for " + event_id)
                total_turns += len(rows)
                source = {r["sequence_id"]: r for r in rows}
                if len(source) != len(rows) or [int(r["sequence_id"]) for r in rows] != sorted(int(r["sequence_id"]) for r in rows):
                    raise ValueError("source sequences not unique and ordered")
                event_blocks = []
                if next_blocks is not None and next_blocks[0] == event_id:
                    event_blocks = list(next_blocks[1])
                    next_blocks = next(block_groups, None)
                counts = Counter()
                pre_rows = []
                passed = g["_gate_pass"] == "1"
                labels = label_evidence(g, "matched_shared_speakers")
                anchor = single_anchor(labels, speaker_key) if passed else ""
                if not passed and event_blocks:
                    raise ValueError("blocks for a failing speaker gate")
                if passed:
                    pre_rows = [r for r in rows if r["analysis_text_type"] == "PRE" and speaker_key(r) == anchor]
                    all_qa_rows = [r for r in rows if r["analysis_text_type"] == "Q&A" and speaker_key(r) == anchor]
                    counts["source_pre_words"] = word_count(join_turns(pre_rows))
                    counts["source_qa_words"] = word_count(join_turns(all_qa_rows))
                    for turn in pre_rows:
                        chunks = split_long_turn(turn["text_contents"], 150, 50)
                        if clean(" ".join(chunks)) != clean(turn["text_contents"]):
                            raise ValueError("PRE chunks do not reconstruct source")
                        for index, text in enumerate(chunks, start=1):
                            cls, reason, content_ready = content_decision(text)
                            review = source_quality_reasons(turn["text_contents"])
                            ready = content_ready and not review
                            status = "proposed" if ready else "review" if review else "procedural"
                            pre_status[status] += 1
                            words = word_count(text)
                            counts["pre_units"] += 1
                            counts["pre_" + status + "_words"] += words
                            if ready:
                                counts["pre_proposed_units"] += 1
                            pw.writerow({
                                "unit_id": f"pre_{event_id}_s{turn['sequence_id']}_{index:03d}",
                                "event_id": event_id, "sequence_id": turn["sequence_id"],
                                "raw_sequence_id": turn["raw_sequence_id"], "chunk_index": index,
                                "word_count": words, "content_class": cls, "content_reason_code": reason,
                                "source_review_reasons": ";".join(review), "proposed_unit_candidate": int(ready),
                                "inclusion_reason": "source_valid_nonprocedural_or_uncertain" if ready else ";".join(review) or reason,
                                "ceo_presentation_segment": text,
                            })
                    validate_blocks(event_id, rows, event_blocks, anchor)
                    owned = set()
                    for block in event_blocks:
                        if block["block_id"] in all_block_ids:
                            raise ValueError("duplicate block ID")
                        all_block_ids.add(block["block_id"])
                        for field in ("gvkey", "expected_execid", "tenure_episode", "event_date", "calendar_quarter"):
                            if block[field] != g[field]:
                                raise ValueError("block/gate metadata disagreement")
                        if (block.get("validated_ceo_speaker") != g["matched_shared_speakers"]
                                or block.get("validated_ceo_speaker_json") != g["matched_shared_speakers_json"]):
                            raise ValueError("block/gate structured speaker evidence disagreement")
                        ids = set(json.loads(block["ceo_answer_sequence_ids"]))
                        if ids & owned:
                            raise ValueError("CEO source words duplicated across blocks")
                        owned.update(ids)
                        decision = classify_block(block, source, anchor)
                        qw.writerow({**block, **decision})
                        counts["qa_blocks"] += 1
                        counts["qa_" + decision["inclusion_status"] + "_blocks"] += 1
                        block_status[decision["inclusion_status"]] += 1
                        words = decision["original_ceo_target_word_count"]
                        counts["qa_candidate_words"] += words
                        for lane, field in (("primary", "primary_block_candidate"), ("management_context", "management_context_block_candidate")):
                            if decision[field]:
                                counts[lane + "_qa_blocks"] += 1
                                counts[lane + "_qa_words"] += words
                        total_blocks += 1
                        if event_id == CASE_EVENT:
                            case_blocks.append({**block, **decision})
                    if counts["qa_candidate_words"] > counts["source_qa_words"]:
                        raise ValueError("candidate CEO answer words exceed raw anchored Q&A")
                    if counts["source_pre_words"] != sum(counts["pre_" + s + "_words"] for s in ("proposed", "review", "procedural")):
                        raise ValueError("PRE word ledger does not reconcile")
                    presentation_writer.writerow({
                        **g, "validated_ceo_speaker": g["matched_shared_speakers"],
                        "validated_ceo_speaker_json": g["matched_shared_speakers_json"],
                        "pre_sequence_ids": json.dumps([r["sequence_id"] for r in pre_rows]),
                        "source_pre_word_count": counts["source_pre_words"],
                        "ceo_presentation_text": join_turns(pre_rows),
                    })
                pre_ready = counts["pre_proposed_units"] > 0
                primary = passed and pre_ready and counts["primary_qa_blocks"] > 0
                management = passed and pre_ready and counts["management_context_qa_blocks"] > 0
                quality = primary and counts["primary_qa_blocks"] >= 2 and counts["primary_qa_words"] >= 100 and counts["primary_qa_words"] >= .25 * counts["source_qa_words"]
                call = {k: g[k] for k in (
                    "event_id", "gvkey", "expected_execid", "tenure_episode", "expected_ceo_name",
                    "event_date", "calendar_quarter", "turnover_ids", "external_speaker_validation_status",
                )}
                call.update({
                    "issuer_label_from_ceo_speaker": issuer_label(labels[0]) if passed else "",
                    "validated_ceo_speaker": g["matched_shared_speakers"] if passed else "",
                    "validated_ceo_speaker_json": g["matched_shared_speakers_json"] if passed else "[]",
                    "speaker_gate_pass": int(passed), "pre_sequence_ids": json.dumps([r["sequence_id"] for r in pre_rows]),
                    "source_pre_words": counts["source_pre_words"], "proposed_pre_words": counts["pre_proposed_words"],
                    "review_pre_words": counts["pre_review_words"], "procedural_pre_words": counts["pre_procedural_words"],
                    "proposed_pre_units": counts["pre_proposed_units"], "source_qa_words": counts["source_qa_words"],
                    "candidate_qa_blocks": counts["qa_blocks"], "candidate_qa_words": counts["qa_candidate_words"],
                    "unassigned_ceo_qa_words": counts["source_qa_words"] - counts["qa_candidate_words"],
                    "primary_qa_blocks": counts["primary_qa_blocks"], "primary_qa_words": counts["primary_qa_words"],
                    "management_context_qa_blocks": counts["management_context_qa_blocks"],
                    "management_context_qa_words": counts["management_context_qa_words"],
                    "source_review_qa_blocks": counts["qa_source_or_boundary_review_blocks"],
                    "procedural_qa_blocks": counts["qa_exclude_procedural_or_empty_blocks"],
                    "primary_qa_word_coverage": counts["primary_qa_words"] / counts["source_qa_words"] if counts["source_qa_words"] else "",
                    "primary_call_candidate": int(primary), "management_context_call_candidate": int(management),
                    "qa_coverage_call_candidate": int(quality), "measurement_validation_pending": 1,
                    "call_decision_reason": (
                        "speaker_gate_failed" if not passed else
                        "no_source_screened_nonprocedural_pre_unit" if not pre_ready else
                        "proposed_primary_pair" if primary else
                        "sensitivity_pair_only" if management else
                        "no_extracted_ceo_qa_block" if not event_blocks else
                        "no_primary_or_management_context_qa_block"
                    ),
                })
                calls.append(call)
                if event_id == CASE_EVENT:
                    case_rows = rows
                if len(seen) % 2000 == 0:
                    print(f"Reconstructed {len(seen):,}/{len(gate):,} calls; {total_blocks:,} blocks", flush=True)
            if next_blocks is not None or seen != set(gate):
                raise ValueError("source/block event reconciliation failed")
        if total_blocks != summary["blocks"]["candidate_blocks_total"] or total_turns != summary["scope"]["turn_rows_scanned"]:
            raise ValueError("run counts differ from extraction summary")
        ep_rows, tr_rows = coverage(calls, episodes, turnovers, minimum_calls, minimum_quarters)
        write_rows(temp / "call_level_sample.csv", calls)
        write_rows(temp / "episode_sample.csv", ep_rows)
        write_rows(temp / "turnover_sample.csv", tr_rows)
        lanes = {}
        for lane in LANES:
            lanes[lane] = {
                "paired_calls": sum(c[lane + "_call_candidate"] for c in calls),
                "supported_ceo_episodes": sum(e[lane + "_episode_candidate"] for e in ep_rows),
                "supported_turnovers": sum(t[lane + "_turnover_candidate"] for t in tr_rows),
                "calls_in_supported_turnovers": sum(c[lane + "_turnover_sample_candidate"] for c in calls),
            }
        result = {
            "status": "complete_proposed_pre_score_sample", "version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate_calls": len(calls), "speaker_confirmed_calls": sum(c["speaker_gate_pass"] for c in calls),
            "source_turns_reconciled": total_turns, "qa_blocks_reconstructed": total_blocks,
            "qa_block_decisions": dict(block_status), "pre_unit_decisions": dict(pre_status),
            "call_reasons": dict(Counter(c["call_decision_reason"] for c in calls)),
            "lanes": lanes,
            "word_ledger": {field: sum(c[field] for c in calls) for field in (
                "source_pre_words", "proposed_pre_words", "review_pre_words", "procedural_pre_words",
                "source_qa_words", "candidate_qa_words", "unassigned_ceo_qa_words",
                "primary_qa_words", "management_context_qa_words",
            )},
            "parameters": {"minimum_paired_calls": minimum_calls, "minimum_distinct_quarters": minimum_quarters,
                           "pre_chunk_words": 150, "pre_tail_merge_words": 50},
            "provenance": {"turns_csv": str(turns), "turns_sha256": turns_hash,
                           "blocks_csv": str(blocks_path), "blocks_sha256": blocks_hash,
                           "input_files_sha256": {
                               str(p): sha256_file(p) for p in (
                                   extraction_summary_path, gate_dir / "event_speaker_gate.csv",
                                   gate_dir / "episode_speaker_gate.csv", gate_dir / "turnover_speaker_gate.csv",
                                   gate_dir / "speaker_gate_summary.json", Path(__file__),
                                   Path(__file__).with_name("ccts_qa_episodes.py"),
                                   Path(__file__).with_name("extract_ccts_ceo_qa_blocks.py"),
                                   Path(__file__).resolve().parents[1] / "llm_measurement/build_specificity_scoring_manifest.py",
                               )
                           }},
            "limitations": [
                "Proposed source-screened availability, not final model-validated scores or thesis inference.",
                "Nonprocedural text is retained as content_class=uncertain without claiming semantic validation.",
                "No topical relevance or directness threshold is applied.",
                "Management-context sensitivity requires identified same-issuer management context; source-review blocks remain held.",
                "Candidate universe is the selected ExecuComp turnover frame, not all US earnings calls.",
            ],
        }
        write_case_report(temp / "flagged_exchange_followup.md", case_rows, case_blocks)
        write_report(temp / "sample_report.md", result, output)
        artifact_names = [p.name for p in sorted(temp.iterdir()) if p.is_file()]
        result["artifacts"] = {name: str(output / name) for name in artifact_names}
        result["artifact_sha256"] = {name: sha256_file(temp / name) for name in artifact_names}
        (temp / "sample_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temp, output)
        return result
    except BaseException:
        shutil.rmtree(temp)
        raise


def write_case_report(path, rows, blocks):
    lines = [
        "# Follow-up: Question Relevance and Extraction Validity", "",
        "Review date: 2026-10-02. Scope: source event 16742069 and block ccts_16742069_episode_16.",
        "", "## Finding", "",
        "The raw sequence is analyst question 16, CFO response 17, CEO response 18, then analyst follow-up 19.",
        "No new question, unidentified speaker, or operator handoff intervenes between 16 and 18.",
        "The analyst's next turn still refers to the same margin discussion. The CEO changes the topic to demand.",
        "This verifies the CEO turn's location within the response episode; it does not establish why the CEO answered that way.",
        "Topical divergence alone is not evidence of a parser misjoin and must not become an exclusion rule.",
        "The 2026-09-28 diagnostic raised a possible mismatch; this follow-up narrows that concern using full source context.",
        "The block stays outside the contract-compatible high-only baseline because of management context.",
        "It is admitted to the proposed management-context sensitivity if the structural and exact procedural screens pass.",
        "", "## Recorded Decision", "",
    ]
    selected = [b for b in blocks if b["block_id"] == CASE_BLOCK]
    if selected:
        b = selected[0]
        lines.extend([json.dumps({k: b[k] for k in (
            "block_id", "question_sequence_ids", "ceo_answer_sequence_ids", "context_sequence_ids",
            "primary_block_candidate", "management_context_block_candidate", "inclusion_status",
            "inclusion_reason",
        )}, indent=2), ""])
    else:
        lines.append("The diagnostic case is absent from this run scope (for example a synthetic test).")
    lines += ["## Source Trace (Licensed Text; Local Only)", ""]
    for row in rows:
        if 10 <= int(row["sequence_id"]) <= 24:
            lines += [f"### Sequence {row['sequence_id']}: {row['text_name']}", "", row["text_contents"], ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(path, result, output):
    lines = [
        "# Proposed Analysis Sample", "", "As of 2026-10-02. Availability before LLM scoring; subject to measurement validation.",
        "", "## What This Table Answers", "",
        "For every candidate call: who is the externally confirmed CEO, which PRE source turns belong to them,",
        "how many source-screened Q&A blocks are available, and whether each CEO episode/turnover has enough paired calls.",
        "No score, sentiment, factual truth, or answer responsiveness is used to select calls.",
        "", "## Counts", "",
        f"Source calls: {result['candidate_calls']:,}; speaker-confirmed calls: {result['speaker_confirmed_calls']:,}.",
        f"All {result['source_turns_reconciled']:,} source turns and {result['qa_blocks_reconstructed']:,} extracted blocks reconcile.",
        "", "| Proposed lane | Paired calls | CEO episodes | CEO turnovers | Unique calls in supported turnovers |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for lane, counts in result["lanes"].items():
        lines.append(f"| {lane} | {counts['paired_calls']:,} | {counts['supported_ceo_episodes']:,} | {counts['supported_turnovers']:,} | {counts['calls_in_supported_turnovers']:,} |")
    lines += [
        "", "## Inclusion Rules", "",
        "1. Require the existing strict ExecuComp/shared-PRE-Q&A speaker gate.",
        "2. Preserve each anchored CEO PRE turn and its ordered source IDs. Reuse the existing 150-word splitter with a 50-word tail-merge parameter. Short whole turns are retained; 50 is not a new minimum disclosure threshold.",
        "3. PRE units with parent source gaps/fragments stay in review. Remove only empty or exact procedural-only patterns; mixed and unclassified interpretable material stays whole.",
        "4. Primary Q&A uses high-tier blocks with only benign short-answer, acknowledgement, or observed closing-boundary flags. Exact procedural-only patterns are excluded; other nonprocedural material remains eligible with content_class=uncertain.",
        "5. The management_context sensitivity adds usable-with-flag blocks only when the remaining flags are benign and every other management speaker is named, labelled as management, and has the same issuer label. CFO speech remains outside the CEO target.",
        "6. The qa_coverage sensitivity additionally requires two primary blocks, 100 primary CEO Q&A words and 25% coverage of all anchored CEO Q&A words.",
        f"7. Each CEO episode needs {result['parameters']['minimum_paired_calls']} paired calls in {result['parameters']['minimum_distinct_quarters']} distinct quarters. Both predecessor and successor must meet that rule.",
        "8. Generic, cautious, indirect, noncommittal and off-topic CEO answers are not excluded for those characteristics. No text-similarity filter is used.",
        "", "These are proposed pre-score lanes. The legacy v1 contract and historical labels are unchanged. Semantic content classification and model validation remain distinct tasks.",
        "", "## Attrition Reasons", "",
    ]
    for reason, count in sorted(result["call_reasons"].items()):
        lines.append(f"- {reason}: {count:,}")
    lines += ["", "## Block Decisions", ""]
    for reason, count in sorted(result["qa_block_decisions"].items()):
        lines.append(f"- {reason}: {count:,}")
    lines += ["", "## CEO Word Coverage", ""]
    for field, count in result["word_ledger"].items():
        lines.append(f"- {field}: {count:,}")
    lines += [
        "", "## Files", "",
        "- call_level_sample.csv: one metadata row per candidate event; sample flags, coverage and reasons.",
        "- episode_sample.csv / turnover_sample.csv: support after the PRE/Q&A availability rules.",
        "- ceo_presentations.csv: full anchored CEO PRE text with source IDs (licensed; local only).",
        "- ceo_pre_units.csv: every PRE chunk, text, source ID and proposed/review decision.",
        "- qa_block_decisions.csv: one metadata decision per extracted block. Join block_id to the original ceo_qa_blocks.csv for full question, CEO answer and management context.",
        "- flagged_exchange_followup.md: complete relevant source trace for the previously flagged case.",
        "- sample_summary.json: source checksums, word ledger, counts and output checksums.",
        "", "## Next", "",
        "Use the proposed sample and its sensitivity comparison to document sample construction.",
        "Complete the existing fresh human/model measurement evaluation once, then freeze the scoring procedure before a production run.",
        "Recompute final call/episode/turnover support after valid scores and scored-word coverage are available.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.fetch_root, args.output_dir)
    print(json.dumps({"status": result["status"], "lanes": result["lanes"],
                      "output": str(args.output_dir)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
