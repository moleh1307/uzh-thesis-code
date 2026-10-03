#!/usr/bin/env python3
"""Apply the strict ExecuComp-to-CCTS speaker gate to candidate CEO episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from build_ccts_ceo_presentation_representation import (
    SPECIAL_CEO_RE,
    is_current_ceo,
    label,
    speaker_key,
)
from validate_ccts_execucomp_speaker_identity import best


ACCEPTED_NAME_RELATIONS = {
    "exact_full_name",
    "same_first_last_middle_difference",
    "initial_prefix_plus_given_name",
    "initial_prefix_plus_recognized_nickname",
    "recognized_nickname_last_name",
    "recognized_transcription_variant",
}

EVENT_FIELDS = [
    "event_id", "gvkey", "expected_execid", "tenure_episode", "expected_ceo_name",
    "event_date", "calendar_quarter", "turnover_ids", "turn_rows", "pre_turn_rows",
    "qa_turn_rows", "other_section_turn_rows", "pre_ceo_candidate_count",
    "qa_ceo_candidate_count", "shared_ceo_candidate_count", "pre_ceo_speakers",
    "qa_ceo_speakers", "shared_ceo_speakers", "within_call_identity_status",
    "within_call_identity_ready", "identity_flags", "matched_pre_speakers",
    "matched_qa_speakers", "matched_shared_speakers", "pre_name_match_quality",
    "qa_name_match_quality", "external_name_match_quality",
    "external_speaker_validation_status", "event_speaker_gate_pass",
]

EPISODE_FIELDS = [
    "gvkey", "expected_execid", "tenure_episode", "expected_ceo_name", "turnover_ids",
    "source_candidate_calls", "source_candidate_distinct_quarters", "fetched_events",
    "events_with_pre_and_qa", "confirmed_calls", "confirmed_distinct_quarters",
    "minimum_confirmed_calls", "minimum_confirmed_quarters", "episode_analysis_gate_pass",
    "speaker_status_counts_json",
]

TURNOVER_FIELDS = [
    "turnover_id", "gvkey", "transition_gap_days", "old_execid", "old_tenure_episode",
    "old_ceo_name", "old_manifest_calls", "old_confirmed_calls",
    "old_confirmed_distinct_quarters", "old_episode_gate_pass", "new_execid",
    "new_tenure_episode", "new_ceo_name", "new_manifest_calls", "new_confirmed_calls",
    "new_confirmed_distinct_quarters", "new_episode_gate_pass", "turnover_gate_status",
    "turnover_analysis_gate_pass",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def episode_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    try:
        execid = str(int(row["execid"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid ExecuComp execid: {row.get('execid')!r}") from exc
    return row["gvkey"].strip(), execid, row["tenure_episode"].strip()


def quarter_from_date(date_text: str) -> str:
    if len(date_text) < 7 or date_text[4] != "-":
        raise ValueError(f"invalid event date: {date_text!r}")
    year, month = int(date_text[:4]), int(date_text[5:7])
    if month < 1 or month > 12:
        raise ValueError(f"invalid event date: {date_text!r}")
    return f"{year}Q{((month - 1) // 3) + 1}"


def make_candidate_map(rows: Iterable[Mapping[str, str]]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for row in rows:
        if not is_current_ceo(row):
            continue
        key = speaker_key(row)
        if key:
            candidates[key] = label(row)
    return candidates


def classify_event(
    source: Mapping[str, str],
    expected_name: str,
    pre_candidates: Mapping[str, str],
    qa_candidates: Mapping[str, str],
    turn_counts: Mapping[str, int],
) -> dict[str, Any]:
    shared_keys = sorted(set(pre_candidates) & set(qa_candidates))
    shared_labels = [qa_candidates[key] for key in shared_keys]
    flags: list[str] = []
    if not shared_keys:
        flags.append("NO_SHARED_PRE_QA_CEO_LABEL")
    if len(pre_candidates) > 1:
        flags.append("MULTIPLE_PRE_CEO_CANDIDATES")
    if len(shared_keys) > 1:
        flags.append("MULTIPLE_SHARED_CEO_CANDIDATES")
    if any(SPECIAL_CEO_RE.search(value) for value in shared_labels):
        flags.append("SPECIAL_CEO_TITLE")
    identity_ready = int(len(shared_keys) == 1 and not flags)
    if identity_ready:
        identity_status = "exact_single_shared_speaker_key"
    elif len(shared_keys) > 1:
        identity_status = "multiple_shared_speaker_keys"
    elif len(pre_candidates) > 1:
        identity_status = "review_side_multiple_ceo_candidates"
    elif "SPECIAL_CEO_TITLE" in flags:
        identity_status = "review_special_ceo_title"
    else:
        identity_status = "review_or_no_shared_speaker_key"

    pre_quality, pre_matches = best(expected_name, list(pre_candidates.values()))
    qa_quality, qa_matches = best(expected_name, list(qa_candidates.values()))
    shared_quality, shared_matches = best(expected_name, shared_labels)
    if identity_ready and shared_quality in ACCEPTED_NAME_RELATIONS:
        external_status = "confirmed_external_ceo_shared_pre_qa"
        quality = shared_quality
    elif identity_ready and shared_quality == "same_initial_last_name_review":
        external_status = "review_external_ceo_initial_last_name"
        quality = shared_quality
    elif identity_ready:
        external_status = "mismatch_external_ceo_vs_shared_speaker"
        quality = shared_quality
    elif pre_quality in ACCEPTED_NAME_RELATIONS and qa_quality in ACCEPTED_NAME_RELATIONS:
        external_status = "review_external_ceo_present_both_sections_not_strict_shared"
        quality = "both_sections_name_match"
    elif pre_quality in ACCEPTED_NAME_RELATIONS or qa_quality in ACCEPTED_NAME_RELATIONS:
        external_status = "review_external_ceo_present_one_section"
        quality = pre_quality if pre_quality in ACCEPTED_NAME_RELATIONS else qa_quality
    else:
        external_status = "not_confirmed_external_ceo_not_title_labelled"
        quality = shared_quality if shared_quality != "no_match" else (
            pre_quality if pre_quality != "no_match" else qa_quality
        )

    date_text = source["event_date"]
    return {
        "event_id": source["event_id"],
        "gvkey": source["gvkey"],
        "expected_execid": source["execid"],
        "tenure_episode": source["tenure_episode"],
        "expected_ceo_name": expected_name,
        "event_date": date_text,
        "calendar_quarter": source["calendar_quarter"],
        "turnover_ids": source["turnover_ids"],
        "turn_rows": turn_counts.get("all", 0),
        "pre_turn_rows": turn_counts.get("PRE", 0),
        "qa_turn_rows": turn_counts.get("Q&A", 0),
        "other_section_turn_rows": turn_counts.get("other", 0),
        "pre_ceo_candidate_count": len(pre_candidates),
        "qa_ceo_candidate_count": len(qa_candidates),
        "shared_ceo_candidate_count": len(shared_keys),
        "pre_ceo_speakers": "; ".join(pre_candidates.values()),
        "qa_ceo_speakers": "; ".join(qa_candidates.values()),
        "shared_ceo_speakers": "; ".join(shared_labels),
        "within_call_identity_status": identity_status,
        "within_call_identity_ready": identity_ready,
        "identity_flags": ";".join(sorted(set(flags))),
        "matched_pre_speakers": "; ".join(pre_matches),
        "matched_qa_speakers": "; ".join(qa_matches),
        "matched_shared_speakers": "; ".join(shared_matches),
        "pre_name_match_quality": pre_quality,
        "qa_name_match_quality": qa_quality,
        "external_name_match_quality": quality,
        "external_speaker_validation_status": external_status,
        "event_speaker_gate_pass": int(external_status == "confirmed_external_ceo_shared_pre_qa"),
    }


def _key_from_assignment(row: Mapping[str, str]) -> tuple[str, str, str]:
    return episode_key(row)


def validate_metadata_inputs(
    assignments: list[dict[str, str]],
    episodes: list[dict[str, str]],
    turnover_map: list[dict[str, str]],
    turnover_pairs: list[dict[str, str]],
    normalized_rows: list[dict[str, str]],
    sequence_rows: list[dict[str, str]],
    fetch_audit_path: Path,
    dedup_summary_path: Path,
    resolution_audit_path: Path,
) -> dict[str, Any]:
    assignment_by_event: dict[str, dict[str, str]] = {}
    for row in assignments:
        event_id = row["event_id"].strip()
        if not event_id or event_id in assignment_by_event:
            raise ValueError(f"blank or duplicate candidate event_id: {event_id!r}")
        if quarter_from_date(row["event_date"]) != row["calendar_quarter"]:
            raise ValueError(f"candidate date/quarter mismatch for event {event_id}")
        assignment_by_event[event_id] = row

    normalized_by_episode: dict[tuple[str, str, str], str] = {}
    for row in normalized_rows:
        key = episode_key(row)
        if key in normalized_by_episode:
            raise ValueError(f"duplicate normalized ExecuComp CEO episode: {key}")
        normalized_by_episode[key] = row["person_name"].strip()

    sequences_by_event: dict[str, dict[str, str]] = {}
    for row in sequence_rows:
        event_id = row["event_id"]
        if event_id in sequences_by_event:
            raise ValueError(f"duplicate event_id in exact call sequences: {event_id}")
        sequences_by_event[event_id] = row

    for event_id, row in assignment_by_event.items():
        key = _key_from_assignment(row)
        expected_name = normalized_by_episode.get(key)
        sequence = sequences_by_event.get(event_id)
        if expected_name is None or sequence is None:
            raise ValueError(f"missing ExecuComp episode or call-sequence record for {event_id}")
        if episode_key(sequence) != key:
            raise ValueError(f"ExecuComp episode key mismatch for event {event_id}")
        if sequence["ceo_name"].strip() != expected_name:
            raise ValueError(f"normalized CEO name differs from exact panel for event {event_id}")
        if sequence["event_date"] != row["event_date"]:
            raise ValueError(f"event date differs from exact panel for event {event_id}")
        if sequence["calendar_quarter"] != row["calendar_quarter"]:
            raise ValueError(f"event quarter differs from exact panel for event {event_id}")

    episode_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in episodes:
        key = episode_key(row)
        if key in episode_by_key:
            raise ValueError(f"duplicate candidate CEO episode: {key}")
        episode_by_key[key] = row
    assignment_episode_counts = Counter(_key_from_assignment(row) for row in assignments)
    assignment_episode_quarters: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in assignments:
        assignment_episode_quarters[_key_from_assignment(row)].add(row["calendar_quarter"])
    if set(episode_by_key) != set(assignment_episode_counts):
        raise ValueError("candidate episode keys differ from candidate event assignments")
    for key, row in episode_by_key.items():
        if int(row["calls_in_source_panel"]) != assignment_episode_counts[key]:
            raise ValueError(f"candidate call count mismatch for CEO episode {key}")
        if int(row["distinct_quarters_in_source_panel"]) != len(assignment_episode_quarters[key]):
            raise ValueError(f"candidate quarter count mismatch for CEO episode {key}")

    episode_turnovers: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    seen_turnover_sides: set[tuple[str, str]] = set()
    for row in turnover_map:
        key = episode_key(row)
        pair_key = (row["turnover_id"], row["side"])
        if pair_key in seen_turnover_sides:
            raise ValueError(f"duplicate turnover side mapping: {pair_key}")
        seen_turnover_sides.add(pair_key)
        episode_turnovers[key].append(pair_key)
    if set(episode_turnovers) != set(episode_by_key):
        raise ValueError("candidate turnover episode map differs from candidate episode manifest")

    for row in assignments:
        key = _key_from_assignment(row)
        mapped = sorted(turnover_id for turnover_id, _ in episode_turnovers[key])
        recorded = sorted(item for item in row["turnover_ids"].split(";") if item)
        if mapped != recorded:
            raise ValueError(f"turnover-ID association mismatch for event {row['event_id']}")

    pair_by_id: dict[str, dict[str, str]] = {}
    for row in turnover_pairs:
        turnover_id = row["turnover_id"]
        if turnover_id in pair_by_id:
            raise ValueError(f"duplicate candidate turnover pair: {turnover_id}")
        pair_by_id[turnover_id] = row
    mapped_turnovers = {row["turnover_id"] for row in turnover_map}
    if mapped_turnovers != set(pair_by_id):
        raise ValueError("candidate turnover pairs differ from episode map")
    for turnover_id, pair in pair_by_id.items():
        sides = {
            side: key for key, linked in episode_turnovers.items()
            for linked_turnover, side in linked if linked_turnover == turnover_id
        }
        expected_old = (pair["gvkey"], str(int(pair["old_execid"])), pair["old_tenure_episode"])
        expected_new = (pair["gvkey"], str(int(pair["new_execid"])), pair["new_tenure_episode"])
        if sides.get("old_episode") != expected_old or sides.get("new_episode") != expected_new:
            raise ValueError(f"old/new episode mapping mismatch for {turnover_id}")

    with dedup_summary_path.open(encoding="utf-8") as handle:
        dedup_summary = json.load(handle)
    if dedup_summary.get("status") != "complete":
        raise ValueError("derived-turn deduplication summary is not complete")
    if int(dedup_summary["input_events"]) != len(assignments):
        raise ValueError("deduplication event count differs from candidate event manifest")
    if int(dedup_summary["output_rows"]) != int(dedup_summary["unique_event_sequence_keys"]):
        raise ValueError("derived turns summary reports duplicate event/sequence keys")
    if dedup_summary.get("raw_input_modified") is not False:
        raise ValueError("deduplication summary does not confirm raw input remained unchanged")

    fetch_rows = read_rows(fetch_audit_path)
    fetch_by_event = {row["event_id"]: row for row in fetch_rows}
    if len(fetch_by_event) != len(fetch_rows) or set(fetch_by_event) != set(assignment_by_event):
        raise ValueError("CCTS fetch audit event IDs differ from candidate assignments")
    fetch_review_ids = set()
    if any(row["fetch_status"] not in {"ok", "review"} for row in fetch_rows):
        raise ValueError("CCTS fetch audit contains an unsupported status")
    for row in fetch_rows:
        if row["fetch_status"] == "review":
            if row["fetch_notes"] != "duplicate_sequence_id":
                raise ValueError(f"unresolved fetch review for event {row['event_id']}")
            fetch_review_ids.add(row["event_id"])
    resolved_review_ids = set(dedup_summary.get("review_event_ids_resolved", []))
    if fetch_review_ids != resolved_review_ids:
        raise ValueError("fetch-review events do not equal the duplicate-resolution summary")

    with resolution_audit_path.open(newline="", encoding="utf-8-sig") as handle:
        resolution_count = sum(1 for _ in csv.DictReader(handle))
    if resolution_count != int(dedup_summary["resolution_audit_rows"]):
        raise ValueError("duplicate-resolution audit row count differs from its summary")

    return {
        "assignment_by_event": assignment_by_event,
        "normalized_by_episode": normalized_by_episode,
        "episode_by_key": episode_by_key,
        "episode_turnovers": episode_turnovers,
        "pair_by_id": pair_by_id,
        "fetch_by_event": fetch_by_event,
        "dedup_summary": dedup_summary,
        "fetch_review_events_resolved": len(fetch_review_ids),
        "resolution_audit_rows": resolution_count,
    }


def build_episode_and_turnover_gates(
    event_rows: list[dict[str, Any]],
    episodes: list[dict[str, str]],
    turnover_map: list[dict[str, str]],
    turnover_pairs: list[dict[str, str]],
    min_calls: int,
    min_quarters: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events_by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in event_rows:
        events_by_episode[(event["gvkey"], str(int(event["expected_execid"])), event["tenure_episode"])].append(event)

    turnover_ids_by_episode: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in turnover_map:
        turnover_ids_by_episode[episode_key(row)].append(row["turnover_id"])
    episode_gates: dict[tuple[str, str, str], dict[str, Any]] = {}
    episode_rows: list[dict[str, Any]] = []
    for source in episodes:
        key = episode_key(source)
        rows = events_by_episode[key]
        if len(rows) != int(source["calls_in_source_panel"]):
            raise ValueError(f"fetched event count mismatch for CEO episode {key}")
        confirmed = [row for row in rows if int(row["event_speaker_gate_pass"]) == 1]
        quarters = {row["calendar_quarter"] for row in confirmed}
        status_counts = Counter(row["external_speaker_validation_status"] for row in rows)
        passed = len(confirmed) >= min_calls and len(quarters) >= min_quarters
        gate = {
            "confirmed_calls": len(confirmed),
            "confirmed_distinct_quarters": len(quarters),
            "episode_analysis_gate_pass": int(passed),
        }
        episode_gates[key] = gate
        episode_rows.append({
            "gvkey": source["gvkey"], "expected_execid": source["execid"],
            "tenure_episode": source["tenure_episode"], "expected_ceo_name": rows[0]["expected_ceo_name"],
            "turnover_ids": ";".join(sorted(set(turnover_ids_by_episode[key]))),
            "source_candidate_calls": source["calls_in_source_panel"],
            "source_candidate_distinct_quarters": source["distinct_quarters_in_source_panel"],
            "fetched_events": len(rows),
            "events_with_pre_and_qa": sum(int(row["pre_turn_rows"] > 0 and row["qa_turn_rows"] > 0) for row in rows),
            "confirmed_calls": len(confirmed), "confirmed_distinct_quarters": len(quarters),
            "minimum_confirmed_calls": min_calls, "minimum_confirmed_quarters": min_quarters,
            "episode_analysis_gate_pass": int(passed),
            "speaker_status_counts_json": json.dumps(dict(sorted(status_counts.items())), separators=(",", ":")),
        })

    sides_by_turnover: dict[str, dict[str, tuple[str, str, str]]] = defaultdict(dict)
    for row in turnover_map:
        sides_by_turnover[row["turnover_id"]][row["side"]] = episode_key(row)
    turnover_rows: list[dict[str, Any]] = []
    pair_lookup = {row["turnover_id"]: row for row in turnover_pairs}
    if set(sides_by_turnover) != set(pair_lookup):
        raise ValueError("turnover episode map and pair list differ")
    for turnover_id, pair in sorted(pair_lookup.items()):
        sides = sides_by_turnover[turnover_id]
        if set(sides) != {"old_episode", "new_episode"}:
            raise ValueError(f"turnover lacks exactly one old and one new side: {turnover_id}")
        old_key, new_key = sides["old_episode"], sides["new_episode"]
        old_gate, new_gate = episode_gates[old_key], episode_gates[new_key]
        old_pass, new_pass = int(old_gate["episode_analysis_gate_pass"]), int(new_gate["episode_analysis_gate_pass"])
        if old_pass and new_pass:
            status = "pass_both_episodes"
        elif not old_pass and not new_pass:
            status = "fail_both_episode_gates"
        elif not old_pass:
            status = "fail_old_episode_gate"
        else:
            status = "fail_new_episode_gate"
        turnover_rows.append({
            "turnover_id": turnover_id, "gvkey": pair["gvkey"],
            "transition_gap_days": pair["transition_gap_days"],
            "old_execid": pair["old_execid"], "old_tenure_episode": pair["old_tenure_episode"],
            "old_ceo_name": events_by_episode[old_key][0]["expected_ceo_name"],
            "old_manifest_calls": len(events_by_episode[old_key]),
            "old_confirmed_calls": old_gate["confirmed_calls"],
            "old_confirmed_distinct_quarters": old_gate["confirmed_distinct_quarters"],
            "old_episode_gate_pass": old_pass,
            "new_execid": pair["new_execid"], "new_tenure_episode": pair["new_tenure_episode"],
            "new_ceo_name": events_by_episode[new_key][0]["expected_ceo_name"],
            "new_manifest_calls": len(events_by_episode[new_key]),
            "new_confirmed_calls": new_gate["confirmed_calls"],
            "new_confirmed_distinct_quarters": new_gate["confirmed_distinct_quarters"],
            "new_episode_gate_pass": new_pass, "turnover_gate_status": status,
            "turnover_analysis_gate_pass": int(old_pass and new_pass),
        })
    return episode_rows, turnover_rows


def write_csv(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_turns(
    turns_path: Path,
    event_ids: list[str],
    assignment_by_event: Mapping[str, dict[str, str]],
    normalized_by_episode: Mapping[tuple[str, str, str], str],
    limit_events: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    target_ids = event_ids[:limit_events] if limit_events else event_ids
    event_rows: list[dict[str, Any]] = []
    current_id = ""
    current_pre: list[dict[str, str]] = []
    current_qa: list[dict[str, str]] = []
    current_pre_rows = current_qa_rows = current_other_rows = 0
    current_turn_rows = 0
    current_sequences: set[str] = set()
    closed_ids: set[str] = set()
    rows_scanned = 0
    started = time.monotonic()

    def finish_event() -> None:
        if not current_id:
            return
        if current_id in closed_ids:
            raise ValueError(f"event is not contiguous in derived turn table: {current_id}")
        closed_ids.add(current_id)
        expected_index = len(event_rows)
        if expected_index >= len(target_ids) or current_id != target_ids[expected_index]:
            expected = target_ids[expected_index] if expected_index < len(target_ids) else "<end>"
            raise ValueError(f"turn event order/coverage mismatch: expected {expected}, found {current_id}")
        source = assignment_by_event[current_id]
        key = _key_from_assignment(source)
        event_rows.append(classify_event(
            source,
            normalized_by_episode[key],
            make_candidate_map(current_pre),
            make_candidate_map(current_qa),
            {
                "all": current_turn_rows, "PRE": current_pre_rows, "Q&A": current_qa_rows,
                "other": current_other_rows,
            },
        ))

    with turns_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"event_id", "sequence_id", "analysis_text_type", "text_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"derived turn CSV missing columns: {sorted(missing)}")
        for row in reader:
            rows_scanned += 1
            event_id = row["event_id"].strip()
            if event_id not in assignment_by_event:
                raise ValueError(f"unexpected event in derived turns: {event_id}")
            if current_id != event_id:
                if current_id:
                    finish_event()
                    if limit_events and len(event_rows) == limit_events:
                        break
                current_id = event_id
                current_pre, current_qa = [], []
                current_pre_rows = current_qa_rows = current_other_rows = 0
                current_turn_rows = 0
                current_sequences = set()
            sequence_id = row["sequence_id"].strip()
            if not sequence_id or sequence_id in current_sequences:
                raise ValueError(f"blank or duplicate sequence_id for event {event_id}")
            current_sequences.add(sequence_id)
            current_turn_rows += 1
            section = row["analysis_text_type"].strip()
            if section == "PRE":
                current_pre_rows += 1
                current_pre.append(row)
            elif section == "Q&A":
                current_qa_rows += 1
                current_qa.append(row)
            else:
                current_other_rows += 1
            if rows_scanned % 250_000 == 0:
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    f"turn rows {rows_scanned:,}; events {len(event_rows):,}/{len(target_ids):,}; "
                    f"{rows_scanned / elapsed:,.0f} rows/sec",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            finish_event()

    if len(event_rows) != len(target_ids):
        raise ValueError(f"derived turn coverage incomplete: {len(event_rows)}/{len(target_ids)} events")
    return event_rows, rows_scanned


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    assignments = read_rows(args.event_assignments)
    episodes = read_rows(args.episode_manifest)
    turnover_map = read_rows(args.turnover_episode_map)
    turnover_pairs = read_rows(args.turnover_pairs)
    normalized_rows = read_rows(args.execucomp_normalized)
    sequence_rows = read_rows(args.call_sequences)
    context = validate_metadata_inputs(
        assignments, episodes, turnover_map, turnover_pairs, normalized_rows, sequence_rows,
        args.fetch_audit, args.dedup_summary, args.resolution_audit,
    )
    assignment_by_event = context["assignment_by_event"]
    event_ids = [row["event_id"] for row in assignments]
    limit = args.limit_events
    if limit is not None and (limit < 1 or limit > len(event_ids)):
        raise ValueError(f"--limit-events must be between 1 and {len(event_ids)}")
    event_rows, turns_scanned = scan_turns(
        args.turns,
        event_ids,
        assignment_by_event,
        context["normalized_by_episode"],
        limit,
    )

    limited = limit is not None
    if not limited and turns_scanned != int(context["dedup_summary"]["output_rows"]):
        raise ValueError(
            f"turn-row count differs from deduplication summary: "
            f"{turns_scanned}/{context['dedup_summary']['output_rows']}"
        )
    if limited:
        episode_rows: list[dict[str, Any]] = []
        turnover_rows: list[dict[str, Any]] = []
    else:
        episode_rows, turnover_rows = build_episode_and_turnover_gates(
            event_rows, episodes, turnover_map, turnover_pairs,
            args.minimum_confirmed_calls, args.minimum_confirmed_quarters,
        )
    turns_sha256 = None if limited else sha256_file(args.turns)
    if not limited and turns_sha256 != context["dedup_summary"].get("output_sha256"):
        raise ValueError("derived turn table SHA-256 differs from its deduplication summary")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    event_path = args.output_dir / "event_speaker_gate.csv"
    write_csv(event_path, EVENT_FIELDS, event_rows)
    artifacts = {"event_speaker_gate_csv": str(event_path)}
    if not limited:
        episode_path = args.output_dir / "episode_speaker_gate.csv"
        turnover_path = args.output_dir / "turnover_speaker_gate.csv"
        write_csv(episode_path, EPISODE_FIELDS, episode_rows)
        write_csv(turnover_path, TURNOVER_FIELDS, turnover_rows)
        artifacts.update({
            "episode_speaker_gate_csv": str(episode_path),
            "turnover_speaker_gate_csv": str(turnover_path),
        })

    event_status_counts = Counter(row["external_speaker_validation_status"] for row in event_rows)
    summary: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "smoke_complete" if limited else "complete",
        "limited_scope": limited,
        "event_scope": len(event_rows),
        "candidate_events_total": len(assignments),
        "turn_rows_scanned": turns_scanned,
        "event_gate_pass": sum(int(row["event_speaker_gate_pass"]) for row in event_rows),
        "event_status_counts": dict(sorted(event_status_counts.items())),
        "minimum_confirmed_calls": args.minimum_confirmed_calls if not limited else None,
        "minimum_confirmed_quarters": args.minimum_confirmed_quarters if not limited else None,
        "fetch_review_events_resolved": context["fetch_review_events_resolved"],
        "duplicate_sequence_rows_collapsed": context["dedup_summary"]["duplicate_rows_collapsed"],
        "episode_gate_pass": sum(int(row["episode_analysis_gate_pass"]) for row in episode_rows) if not limited else None,
        "episode_gate_fail": sum(not int(row["episode_analysis_gate_pass"]) for row in episode_rows) if not limited else None,
        "turnover_gate_pass": sum(int(row["turnover_analysis_gate_pass"]) for row in turnover_rows) if not limited else None,
        "turnover_gate_fail": sum(not int(row["turnover_analysis_gate_pass"]) for row in turnover_rows) if not limited else None,
        "provenance": {
            "turns_csv": str(args.turns),
            "turns_sha256_verified": turns_sha256,
            "event_assignments_csv": str(args.event_assignments),
            "episode_manifest_csv": str(args.episode_manifest),
            "turnover_episode_map_csv": str(args.turnover_episode_map),
            "turnover_pairs_csv": str(args.turnover_pairs),
            "execucomp_normalized_csv": str(args.execucomp_normalized),
            "call_sequences_csv": str(args.call_sequences),
            "fetch_audit_csv": str(args.fetch_audit),
            "dedup_summary_json": str(args.dedup_summary),
        },
        "artifacts": artifacts,
        "method": {
            "within_call_identity": "one strict shared current CEO-labelled speaker key in PRE and Q&A; multiple PRE candidates, multiple shared candidates, interim/co-CEO, and acting-CEO titles are review",
            "external_identity": "normalized ExecuComp CEO name must match the shared title-labelled CCTS speaker under the existing controlled exact, middle-name, documented nickname, and surname-transcription rules",
            "episode_gate": "at least the configured number of confirmed calls in at least the configured number of distinct calendar quarters",
            "turnover_gate": "both old and new CEO episodes pass the episode gate",
            "transcript_text_exported": False,
        },
        "caveats": [
            "This gate confirms a controlled cross-source name match plus within-call PRE/Q&A speaker consistency; it is not independent proof of legal identity.",
            "Speaker identity and Q&A-block quality are separate. No Q&A blocks were extracted or filtered in this step.",
            "A smoke run is a prefix sample for code validation only and has no episode- or turnover-level inference.",
        ],
        "gate_version": "v1.2_scoped_acting_ceo_review_rule",
    }
    summary_path = args.output_dir / "speaker_gate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_lines = [
        "# CCTS-ExecuComp Speaker Gate", "",
        f"- Status: `{summary['status']}`",
        f"- Events evaluated: {len(event_rows):,} / {len(assignments):,}",
        f"- Strict external speaker matches: {summary['event_gate_pass']:,}",
        "- Transcript text exported: no", "",
        "## Event Status", "",
    ]
    report_lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(event_status_counts.items()))
    if not limited:
        report_lines.extend([
            "", "## Episode and Turnover Gates", "",
            f"- CEO episodes passing: {summary['episode_gate_pass']:,} / {len(episode_rows):,}",
            f"- Turnovers passing both episodes: {summary['turnover_gate_pass']:,} / {len(turnover_rows):,}",
            f"- Threshold: at least {args.minimum_confirmed_calls} confirmed calls in {args.minimum_confirmed_quarters} distinct quarters.",
        ])
    report_lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in summary["caveats"]], ""])
    (args.output_dir / "speaker_gate_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True, type=Path)
    parser.add_argument("--event-assignments", required=True, type=Path)
    parser.add_argument("--episode-manifest", required=True, type=Path)
    parser.add_argument("--turnover-episode-map", required=True, type=Path)
    parser.add_argument("--turnover-pairs", required=True, type=Path)
    parser.add_argument("--execucomp-normalized", required=True, type=Path)
    parser.add_argument("--call-sequences", required=True, type=Path)
    parser.add_argument("--fetch-audit", required=True, type=Path)
    parser.add_argument("--dedup-summary", required=True, type=Path)
    parser.add_argument("--resolution-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-confirmed-calls", type=int, default=5)
    parser.add_argument("--minimum-confirmed-quarters", type=int, default=5)
    parser.add_argument("--limit-events", type=int)
    args = parser.parse_args()
    if args.minimum_confirmed_calls < 1 or args.minimum_confirmed_quarters < 1:
        parser.error("episode thresholds must be positive")
    try:
        run(args)
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        raise SystemExit(f"speaker gate failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
