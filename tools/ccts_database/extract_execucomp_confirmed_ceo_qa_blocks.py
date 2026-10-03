#!/usr/bin/env python3
"""Stream CEO Q&A candidate blocks from calls passing the strict identity gate.

The gate's one shared CEO label anchors extraction. The script preserves all
candidate blocks and review flags; it does not assign final content eligibility
or score text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Any, Iterable, Mapping, Sequence

from extract_ccts_ceo_qa_blocks import (
    BLOCK_COLUMNS,
    EVENT_COLUMNS,
    clean,
    extract_event,
    is_analyst,
    is_ceo,
    join_turns,
    speaker_key,
)


from ceo_title_evidence import GATE_VERSION, single_anchor

EXPECTED_GATE_VERSION = GATE_VERSION
EXPECTED_EXTERNAL_STATUS = "confirmed_external_ceo_shared_pre_qa"
SCRIPT_VERSION = "external_execucomp_speaker_gate_anchor_v1_20260928"
GATE_FIELDS = [
    "gvkey",
    "expected_execid",
    "tenure_episode",
    "expected_ceo_name",
    "event_date",
    "calendar_quarter",
    "turnover_ids",
    "validated_ceo_speaker",
    "external_name_match_quality",
    "external_speaker_validation_status",
    "speaker_gate_version",
]
BLOCK_FIELDS = [field for field in BLOCK_COLUMNS if field != "applicability_status"] + [
    "candidate_content_status",
    *GATE_FIELDS,
]
EVENT_RENAMES = {
    "eligible_ceo_qa_blocks": "candidate_ceo_qa_blocks",
    "high_quality_blocks": "extractor_high_tier_blocks",
}
EVENT_FIELDS = [EVENT_RENAMES.get(field, field) for field in EVENT_COLUMNS] + GATE_FIELDS
EPISODE_COUNT_FIELDS = [
    "confirmed_event_count",
    "candidate_block_count",
    "extractor_high_tier_block_count",
    "usable_with_flag_block_count",
    "review_block_count",
]
TURNOVER_COUNT_FIELDS = [
    "old_confirmed_event_count",
    "old_candidate_block_count",
    "old_high_tier_block_count",
    "old_usable_with_flag_block_count",
    "old_review_block_count",
    "new_confirmed_event_count",
    "new_candidate_block_count",
    "new_high_tier_block_count",
    "new_usable_with_flag_block_count",
    "new_review_block_count",
    "both_sides_have_candidate_blocks",
]
TEXT_FIELDS = {
    "analyst_question",
    "ceo_answer",
    "answer_context",
    "preceding_question_context",
}


def canonical_event_id(value: str, source: str) -> str:
    try:
        event_id = str(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event_id in {source}: {value!r}") from exc
    if int(event_id) <= 0:
        raise ValueError(f"nonpositive event_id in {source}: {value!r}")
    return event_id


def episode_key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return row["gvkey"].strip(), str(int(row["expected_execid"])), row["tenure_episode"].strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_speaker_gate(
    event_gate_path: Path,
    episode_gate_path: Path,
    turnover_gate_path: Path,
    gate_summary_path: Path,
    turns_path: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any], str]:
    with gate_summary_path.open("r", encoding="utf-8") as handle:
        gate_summary = json.load(handle)
    if gate_summary.get("status") != "complete":
        raise ValueError("speaker gate summary is not complete")
    if gate_summary.get("gate_version") != EXPECTED_GATE_VERSION:
        raise ValueError(
            f"expected gate {EXPECTED_GATE_VERSION!r}, found {gate_summary.get('gate_version')!r}"
        )

    turns_sha256 = sha256_file(turns_path)
    expected_sha256 = gate_summary.get("provenance", {}).get("turns_sha256_verified")
    if not expected_sha256 or turns_sha256 != expected_sha256:
        raise ValueError("analysis turn CSV hash does not match the verified speaker-gate input")

    gate_rows: dict[str, dict[str, str]] = {}
    required = {
        "event_id", "turn_rows", "pre_turn_rows", "qa_turn_rows",
        "shared_ceo_candidate_count", "shared_ceo_speakers", "matched_shared_speakers",
        "within_call_identity_ready", "external_name_match_quality",
        "external_speaker_validation_status", "event_speaker_gate_pass",
        "gvkey", "expected_execid", "tenure_episode", "expected_ceo_name",
        "event_date", "calendar_quarter", "turnover_ids",
    }
    with event_gate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"event gate is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            event_id = canonical_event_id(row["event_id"], f"{event_gate_path}:{line_number}")
            if event_id in gate_rows:
                raise ValueError(f"duplicate event in speaker gate: {event_id}")
            try:
                pass_value = int(row["event_speaker_gate_pass"])
                turn_count = int(row["turn_rows"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid gate counts for event {event_id}") from exc
            if pass_value not in (0, 1) or turn_count < 0:
                raise ValueError(f"invalid gate status/count for event {event_id}")
            row["event_id"] = event_id
            row["_gate_pass"] = str(pass_value)
            row["_turn_rows"] = str(turn_count)
            if pass_value:
                if row["external_speaker_validation_status"] != EXPECTED_EXTERNAL_STATUS:
                    raise ValueError(f"pass flag/status conflict for event {event_id}")
                if row["within_call_identity_ready"] != "1" or row["shared_ceo_candidate_count"] != "1":
                    raise ValueError(f"passing event {event_id} lacks one strict shared identity")
                matched = row["matched_shared_speakers"].strip()
                shared = row["shared_ceo_speakers"].strip()
                try:
                    matched_key = single_anchor(matched, speaker_key)
                    shared_key = single_anchor(shared, speaker_key)
                except ValueError as exc:
                    raise ValueError(f"passing event {event_id} has an inconsistent CEO anchor") from exc
                if not matched or not shared or not matched_key or matched_key != shared_key:
                    raise ValueError(f"passing event {event_id} has an inconsistent CEO anchor")
            gate_rows[event_id] = row

    expected_events = int(gate_summary.get("candidate_events_total", -1))
    expected_passes = int(gate_summary.get("event_gate_pass", -1))
    if len(gate_rows) != expected_events:
        raise ValueError(f"event gate rows={len(gate_rows)}; summary says {expected_events}")
    actual_passes = sum(row["_gate_pass"] == "1" for row in gate_rows.values())
    if actual_passes != expected_passes:
        raise ValueError(f"event gate pass rows={actual_passes}; summary says {expected_passes}")

    episodes = read_csv_rows(episode_gate_path)
    turnovers = read_csv_rows(turnover_gate_path)
    if len(episodes) != int(gate_summary.get("episode_gate_pass", -1)) + int(gate_summary.get("episode_gate_fail", -1)):
        raise ValueError("episode gate row count does not match summary")
    if len(turnovers) != int(gate_summary.get("turnover_gate_pass", -1)) + int(gate_summary.get("turnover_gate_fail", -1)):
        raise ValueError("turnover gate row count does not match summary")
    return gate_rows, episodes, turnovers, gate_summary, turns_sha256


def gate_metadata(row: Mapping[str, str], gate_version: str) -> dict[str, str]:
    return {
        "gvkey": row["gvkey"],
        "expected_execid": row["expected_execid"],
        "tenure_episode": row["tenure_episode"],
        "expected_ceo_name": row["expected_ceo_name"],
        "event_date": row["event_date"],
        "calendar_quarter": row["calendar_quarter"],
        "turnover_ids": row["turnover_ids"],
        "validated_ceo_speaker": row["matched_shared_speakers"],
        "external_name_match_quality": row["external_name_match_quality"],
        "external_speaker_validation_status": row["external_speaker_validation_status"],
        "speaker_gate_version": gate_version,
    }


def decode_sequence_ids(block: Mapping[str, Any], field: str, event_id: str) -> list[str]:
    try:
        values = json.loads(str(block[field]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {field} for event {event_id}") from exc
    if not isinstance(values, list) or any(not str(value).isdigit() for value in values):
        raise ValueError(f"invalid sequence list in {field} for event {event_id}")
    result = [str(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate sequence ID in {field} for event {event_id}")
    return result


def validate_blocks(
    event_id: str,
    rows: Sequence[Mapping[str, str]],
    blocks: Sequence[dict[str, Any]],
    anchor_key: str,
) -> None:
    source = {str(row["sequence_id"]): row for row in rows}
    if len(source) != len(rows):
        raise ValueError(f"duplicate sequence ID in extracted event {event_id}")
    for block in blocks:
        if str(block["event_id"]) != event_id:
            raise ValueError(f"block {block['block_id']} has the wrong event ID")
        ids_by_field = {
            field: decode_sequence_ids(block, field, event_id)
            for field in ("question_sequence_ids", "ceo_answer_sequence_ids", "context_sequence_ids")
        }
        for field, ids in ids_by_field.items():
            if any(sequence_id not in source for sequence_id in ids):
                raise ValueError(f"block {block['block_id']} references an absent source turn")
            if any(source[sequence_id].get("analysis_text_type") != "Q&A" for sequence_id in ids):
                raise ValueError(f"block {block['block_id']} contains a non-Q&A source turn")
        question_rows = [source[value] for value in ids_by_field["question_sequence_ids"]]
        answer_rows = [source[value] for value in ids_by_field["ceo_answer_sequence_ids"]]
        context_rows = [source[value] for value in ids_by_field["context_sequence_ids"]]
        if any(not is_analyst(row) for row in question_rows):
            raise ValueError(f"block {block['block_id']} contains a non-analyst question turn")
        if any(not is_ceo(row) or speaker_key(row) != anchor_key for row in answer_rows):
            raise ValueError(f"block {block['block_id']} contains a non-anchored CEO answer turn")
        if not set(ids_by_field["ceo_answer_sequence_ids"]).issubset(ids_by_field["context_sequence_ids"]):
            raise ValueError(f"block {block['block_id']} answer turns are absent from answer context")
        if block["analyst_question"] != join_turns(question_rows):
            raise ValueError(f"block {block['block_id']} question does not reconstruct from source")
        if block["ceo_answer"] != join_turns(answer_rows):
            raise ValueError(f"block {block['block_id']} CEO answer does not reconstruct from source")
        if block["answer_context"] != join_turns(context_rows):
            raise ValueError(f"block {block['block_id']} answer context does not reconstruct from source")


def write_csv_row(writer: csv.DictWriter, row: Mapping[str, Any]) -> None:
    writer.writerow({
        field: (row.get(field, "") or "") if field in TEXT_FIELDS else clean(row.get(field, ""))
        for field in writer.fieldnames or []
    })


def spread_sample(rows: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    by_event: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda value: (int(value["event_id"]), str(value["block_id"]))):
        by_event[str(row["event_id"])].append(row)
    groups = list(by_event.values())
    if limit <= 0 or not groups:
        return []
    if limit <= len(groups):
        indexes = [round(i * (len(groups) - 1) / max(1, limit - 1)) for i in range(limit)]
        return [groups[index][0] for index in indexes]
    selected = [group[0] for group in groups]
    remaining = [row for group in groups for row in group[1:]]
    remaining.sort(key=lambda value: (int(value["event_id"]), str(value["block_id"])))
    extra = limit - len(selected)
    if remaining and extra:
        indexes = [round(i * (len(remaining) - 1) / max(1, extra - 1)) for i in range(min(extra, len(remaining)))]
        selected.extend(remaining[index] for index in indexes)
    return selected[:limit]


def select_manual_sample(index: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    by_tier: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in index:
        by_tier[str(row["quality_tier"])].append(row)
    quotas = {
        "review": round(limit * 0.40),
        "usable_with_flag": round(limit * 0.30),
        "high": limit - round(limit * 0.40) - round(limit * 0.30),
    }
    selected: list[Mapping[str, Any]] = []
    for tier in ("review", "usable_with_flag", "high"):
        selected.extend(spread_sample(by_tier.get(tier, []), min(quotas[tier], len(by_tier.get(tier, [])))))
    remaining = max(0, min(limit, len(index)) - len(selected))
    if remaining:
        selected_ids = {str(row["block_id"]) for row in selected}
        fallback = [row for row in index if str(row["block_id"]) not in selected_ids]
        selected.extend(spread_sample(fallback, remaining))
    return selected[:limit]


def write_manual_audit_sample(
    blocks_path: Path,
    sample_index: Sequence[Mapping[str, Any]],
    sample_csv_path: Path,
    sample_md_path: Path,
) -> int:
    selected_ids = {str(row["block_id"]) for row in sample_index}
    wanted_order = {str(row["block_id"]): index for index, row in enumerate(sample_index)}
    found: dict[str, dict[str, str]] = {}
    with blocks_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        with sample_csv_path.open("w", encoding="utf-8", newline="") as sample_handle:
            writer = csv.DictWriter(sample_handle, fieldnames=reader.fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                block_id = row.get("block_id", "")
                if block_id not in selected_ids:
                    continue
                write_csv_row(writer, row)
                found[block_id] = row
    missing = selected_ids - set(found)
    if missing:
        raise ValueError(f"manual audit sample block IDs missing from block CSV: {sorted(missing)[:5]}")
    ordered = sorted(found.values(), key=lambda row: wanted_order[row["block_id"]])
    lines = [
        "# CEO Q&A Extraction Audit Sample",
        "",
        "Deterministic, quality-tier-stratified diagnostic sample; not a random precision estimate or independent human validation.",
        "Read each analyst question, CEO-only answer, and full management-answer context. Quality tiers are extractor diagnostics, not final content eligibility.",
        "",
    ]
    for index, block in enumerate(ordered, start=1):
        lines.extend([
            f"## Sample {index}: `{block['block_id']}`",
            "",
            f"- Event: `{block['event_id']}` | {block.get('company_name', '')} | {block.get('year', '')}",
            f"- ExecuComp CEO: {block.get('expected_ceo_name', '')} | CCTS speaker: {block.get('validated_ceo_speaker', '')}",
            f"- Episode: `{block.get('gvkey', '')}/{block.get('expected_execid', '')}/{block.get('tenure_episode', '')}` | quarter `{block.get('calendar_quarter', '')}`",
            f"- Tier: `{block.get('quality_tier', '')}` | flags: `{block.get('block_flags', '') or 'none'}`",
            "",
            "### Analyst Question",
            "",
            block["analyst_question"],
            "",
            "### CEO-Only Answer",
            "",
            block["ceo_answer"],
            "",
            "### Full Answer Context",
            "",
            block["answer_context"],
            "",
        ])
    sample_md_path.write_text("\n".join(lines), encoding="utf-8")
    return len(found)


def add_count(target: Counter[str], key: str, amount: int = 1) -> None:
    target[key] += amount


def run_extraction(
    turns_path: Path,
    event_gate_path: Path,
    episode_gate_path: Path,
    turnover_gate_path: Path,
    gate_summary_path: Path,
    output_dir: Path,
    manual_audit_blocks: int = 75,
) -> dict[str, Any]:
    paths = [turns_path, event_gate_path, episode_gate_path, turnover_gate_path, gate_summary_path]
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"required inputs missing: {missing}")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if manual_audit_blocks < 0:
        raise ValueError("manual audit sample size cannot be negative")

    gate_rows, episode_rows, turnover_rows, gate_summary, turns_sha256 = load_speaker_gate(
        event_gate_path, episode_gate_path, turnover_gate_path, gate_summary_path, turns_path
    )
    accepted = {event_id: row for event_id, row in gate_rows.items() if row["_gate_pass"] == "1"}
    gate_version = str(gate_summary["gate_version"])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        os.chmod(temp_dir, 0o700)
        blocks_path = temp_dir / "ceo_qa_blocks.csv"
        events_path = temp_dir / "ceo_qa_event_audit.csv"
        episode_coverage_path = temp_dir / "episode_block_coverage.csv"
        turnover_coverage_path = temp_dir / "turnover_block_coverage.csv"

        seen_events: set[str] = set()
        current_event: str | None = None
        current_rows: list[dict[str, str]] = []
        current_row_count = 0
        total_turn_rows = 0
        processed_events = 0
        processed_gate_events = 0
        started_at = time.monotonic()
        event_status_counts: Counter[str] = Counter()
        quality_counts: Counter[str] = Counter()
        content_status_counts: Counter[str] = Counter()
        episode_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        event_block_counts: dict[str, Counter[str]] = defaultdict(Counter)
        sample_index: list[dict[str, str]] = []
        block_count = 0
        duplicate_answer_sequence_count = 0

        def finish_event(event_id: str | None, rows: list[dict[str, str]], row_count: int) -> None:
            nonlocal processed_events, processed_gate_events, block_count, duplicate_answer_sequence_count
            if event_id is None:
                return
            if event_id not in gate_rows:
                raise ValueError(f"turn input contains event absent from speaker gate: {event_id}")
            if event_id in seen_events:
                raise ValueError(f"event rows are noncontiguous: {event_id}")
            seen_events.add(event_id)
            gate = gate_rows[event_id]
            expected_rows = int(gate["_turn_rows"])
            if row_count != expected_rows:
                raise ValueError(f"turn count mismatch for event {event_id}: {row_count} != {expected_rows}")
            processed_events += 1
            if processed_events % 2000 == 0:
                elapsed = max(0.001, time.monotonic() - started_at)
                print(
                    f"progress {processed_events:,}/{len(gate_rows):,} events "
                    f"({processed_gate_events:,} strict-pass; {processed_events / elapsed:.1f} events/sec)",
                    file=sys.stderr,
                    flush=True,
                )
            if gate["_gate_pass"] != "1":
                return
            if len(rows) != row_count:
                raise ValueError(f"passing event {event_id} was not buffered in full")
            section_counts = Counter(row.get("analysis_text_type", "") for row in rows)
            if section_counts["PRE"] != int(gate["pre_turn_rows"]) or section_counts["Q&A"] != int(gate["qa_turn_rows"]):
                raise ValueError(f"section row counts differ from speaker gate for event {event_id}")
            anchor_label = gate["matched_shared_speakers"].strip()
            anchor_key = single_anchor(anchor_label, speaker_key)
            if not anchor_key:
                raise ValueError(f"empty CEO anchor key for passing event {event_id}")
            event_row, event_blocks = extract_event(rows, validated_anchor_key=anchor_key)
            validate_blocks(event_id, rows, event_blocks, anchor_key)

            sequence_owners: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for block in event_blocks:
                for sequence_id in decode_sequence_ids(block, "ceo_answer_sequence_ids", event_id):
                    sequence_owners[sequence_id].append(block)
            duplicated = {sequence_id for sequence_id, owners in sequence_owners.items() if len(owners) > 1}
            if duplicated:
                duplicate_answer_sequence_count += len(duplicated)
                for block in event_blocks:
                    block_ids = set(decode_sequence_ids(block, "ceo_answer_sequence_ids", event_id))
                    if block_ids & duplicated:
                        flags = {flag for flag in str(block["block_flags"]).split(";") if flag}
                        flags.add("DUPLICATE_CEO_ANSWER_SEQUENCE_REVIEW")
                        block["block_flags"] = ";".join(sorted(flags))
                        block["quality_tier"] = "review"
                        block["applicability_status"] = "review"

            metadata = gate_metadata(gate, gate_version)
            for block in event_blocks:
                block["candidate_content_status"] = block.pop("applicability_status", "")
                block.update(metadata)
                tier = str(block["quality_tier"])
                content_status = str(block["candidate_content_status"])
                add_count(quality_counts, tier)
                add_count(content_status_counts, content_status)
                add_count(event_block_counts[event_id], tier)
                add_count(episode_counts[episode_key(gate)], tier)
                sample_index.append({
                    "block_id": str(block["block_id"]),
                    "event_id": event_id,
                    "quality_tier": tier,
                })
            event_row["candidate_ceo_qa_blocks"] = event_row.pop("eligible_ceo_qa_blocks")
            event_row["extractor_high_tier_blocks"] = event_row.pop("high_quality_blocks")
            event_row.update(metadata)
            event_status_counts[str(event_row["event_audit_status"])] += 1
            write_csv_row(event_writer, event_row)
            for block in event_blocks:
                write_csv_row(block_writer, block)
            processed_gate_events += 1
            block_count += len(event_blocks)

        with blocks_path.open("w", encoding="utf-8", newline="") as blocks_handle, events_path.open(
            "w", encoding="utf-8", newline=""
        ) as events_handle:
            block_writer = csv.DictWriter(blocks_handle, fieldnames=BLOCK_FIELDS, extrasaction="ignore")
            event_writer = csv.DictWriter(events_handle, fieldnames=EVENT_FIELDS, extrasaction="ignore")
            block_writer.writeheader()
            event_writer.writeheader()
            with turns_path.open("r", encoding="utf-8-sig", newline="") as turns_handle:
                reader = csv.DictReader(turns_handle)
                required_turn_fields = {"event_id", "sequence_id", "analysis_text_type", "text_name", "text_contents"}
                missing = required_turn_fields - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"turn input is missing columns: {sorted(missing)}")
                for line_number, raw in enumerate(reader, start=2):
                    if None in raw or any(raw.get(field) is None for field in (reader.fieldnames or [])):
                        raise ValueError(f"malformed CSV record at logical row {line_number}")
                    event_id = canonical_event_id(raw["event_id"], f"{turns_path}:{line_number}")
                    row = dict(raw)
                    if current_event is None:
                        current_event = event_id
                    elif event_id != current_event:
                        finish_event(current_event, current_rows, current_row_count)
                        current_event = event_id
                        current_rows = []
                        current_row_count = 0
                    if event_id not in gate_rows:
                        raise ValueError(f"turn input contains event absent from speaker gate: {event_id}")
                    if gate_rows[event_id]["_gate_pass"] == "1":
                        current_rows.append(row)
                    current_row_count += 1
                    total_turn_rows += 1
                finish_event(current_event, current_rows, current_row_count)

        if seen_events != set(gate_rows):
            missing = sorted(set(gate_rows) - seen_events, key=int)
            raise ValueError(f"event coverage mismatch; missing {len(missing)} events, first={missing[:10]}")
        if total_turn_rows != int(gate_summary.get("turn_rows_scanned", -1)):
            raise ValueError("turn row total differs from speaker-gate summary")
        if processed_gate_events != len(accepted):
            raise ValueError(f"processed gate events={processed_gate_events}; expected={len(accepted)}")

        episode_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
        for row in episode_rows:
            key = episode_key(row)
            if key in episode_by_key:
                raise ValueError(f"duplicate episode key in gate: {key}")
            episode_by_key[key] = row
        for key, counts in episode_counts.items():
            if key not in episode_by_key:
                raise ValueError(f"extracted event episode absent from episode gate: {key}")

        confirmed_events_by_episode: Counter[tuple[str, str, str]] = Counter()
        for gate in accepted.values():
            confirmed_events_by_episode[episode_key(gate)] += 1

        episode_columns = list(episode_rows[0].keys()) + EPISODE_COUNT_FIELDS
        with episode_coverage_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=episode_columns, extrasaction="ignore")
            writer.writeheader()
            for row in episode_rows:
                key = episode_key(row)
                counts = episode_counts.get(key, Counter())
                enriched = dict(row)
                enriched["confirmed_event_count"] = confirmed_events_by_episode[key]
                enriched["candidate_block_count"] = sum(counts.values())
                enriched["extractor_high_tier_block_count"] = counts["high"]
                enriched["usable_with_flag_block_count"] = counts["usable_with_flag"]
                enriched["review_block_count"] = counts["review"]
                writer.writerow(enriched)

        turnover_columns = list(turnover_rows[0].keys()) + TURNOVER_COUNT_FIELDS
        passing_turnover_episode_keys: set[tuple[str, str, str]] = set()
        with turnover_coverage_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=turnover_columns, extrasaction="ignore")
            writer.writeheader()
            for row in turnover_rows:
                gvkey = row["gvkey"].strip()
                old_key = (gvkey, str(int(row["old_execid"])), row["old_tenure_episode"].strip())
                new_key = (gvkey, str(int(row["new_execid"])), row["new_tenure_episode"].strip())
                if row.get("turnover_analysis_gate_pass") == "1":
                    passing_turnover_episode_keys.update((old_key, new_key))
                old_counts = episode_counts.get(old_key, Counter())
                new_counts = episode_counts.get(new_key, Counter())
                old_events = confirmed_events_by_episode[old_key]
                new_events = confirmed_events_by_episode[new_key]
                enriched = dict(row)
                enriched.update({
                    "old_confirmed_event_count": old_events,
                    "old_candidate_block_count": sum(old_counts.values()),
                    "old_high_tier_block_count": old_counts["high"],
                    "old_usable_with_flag_block_count": old_counts["usable_with_flag"],
                    "old_review_block_count": old_counts["review"],
                    "new_confirmed_event_count": new_events,
                    "new_candidate_block_count": sum(new_counts.values()),
                    "new_high_tier_block_count": new_counts["high"],
                    "new_usable_with_flag_block_count": new_counts["usable_with_flag"],
                    "new_review_block_count": new_counts["review"],
                    "both_sides_have_candidate_blocks": int(bool(sum(old_counts.values())) and bool(sum(new_counts.values()))),
                })
                writer.writerow(enriched)

        selected_sample = select_manual_sample(sample_index, manual_audit_blocks)
        manual_csv = temp_dir / "manual_audit_sample.csv"
        manual_md = temp_dir / "manual_audit_sample.md"
        sample_written = write_manual_audit_sample(blocks_path, selected_sample, manual_csv, manual_md)

        episode_pass_rows = [row for row in episode_rows if row.get("episode_analysis_gate_pass") == "1"]
        turnover_pass_rows = [row for row in turnover_rows if row.get("turnover_analysis_gate_pass") == "1"]
        passing_turnover_blocks = sum(sum(episode_counts.get(key, Counter()).values()) for key in passing_turnover_episode_keys)
        all_blocks = sum(quality_counts.values())
        if all_blocks != block_count:
            raise ValueError("block quality totals do not reconcile with output block count")

        summary: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "complete",
            "script_version": SCRIPT_VERSION,
            "scope": {
                "candidate_events": len(gate_rows),
                "strict_event_gate_pass": len(accepted),
                "events_extracted": processed_gate_events,
                "all_input_events_reconciled": processed_events,
                "turn_rows_scanned": total_turn_rows,
            },
            "blocks": {
                "candidate_blocks_total": block_count,
                "quality_tier_counts": dict(quality_counts),
                "candidate_content_status_counts": dict(content_status_counts),
                "events_with_candidate_blocks": sum(1 for row in accepted if event_block_counts.get(row)),
                "events_without_candidate_blocks": sum(1 for event_id in accepted if not event_block_counts.get(event_id)),
                "duplicate_answer_sequence_ids_reviewed": duplicate_answer_sequence_count,
                "manual_audit_sample_rows": sample_written,
                "manual_audit_sample_quality_tier_counts": dict(Counter(str(row["quality_tier"]) for row in selected_sample)),
            },
            "episode_coverage": {
                "episode_gate_pass": len(episode_pass_rows),
                "episode_gate_pass_with_candidate_blocks": sum(
                    1 for row in episode_pass_rows if sum(episode_counts.get(episode_key(row), Counter()).values()) > 0
                ),
                "candidate_blocks_in_episode_gate_passing_episodes": sum(
                    sum(episode_counts.get(episode_key(row), Counter()).values()) for row in episode_pass_rows
                ),
            },
            "turnover_coverage": {
                "turnover_gate_pass": len(turnover_pass_rows),
                "turnover_gate_pass_with_candidate_blocks_on_both_sides": sum(
                    1 for row in turnover_pass_rows
                    if sum(episode_counts.get((row["gvkey"].strip(), str(int(row["old_execid"])), row["old_tenure_episode"].strip()), Counter()).values()) > 0
                    and sum(episode_counts.get((row["gvkey"].strip(), str(int(row["new_execid"])), row["new_tenure_episode"].strip()), Counter()).values()) > 0
                ),
                "unique_candidate_blocks_in_episodes_appearing_in_passing_turnovers": passing_turnover_blocks,
            },
            "provenance": {
                "turns_csv": str(turns_path.resolve()),
                "turns_sha256": turns_sha256,
                "event_gate_csv": str(event_gate_path.resolve()),
                "episode_gate_csv": str(episode_gate_path.resolve()),
                "turnover_gate_csv": str(turnover_gate_path.resolve()),
                "gate_summary_json": str(gate_summary_path.resolve()),
                "gate_version": gate_version,
                "external_identity_rule": "only event_speaker_gate_pass=1 and exactly one shared CCTS CEO speaker key; extraction is anchored to the gate's matched_shared_speakers label",
            },
            "artifacts": {},
            "caveats": [
                "All candidate blocks are retained with extractor quality flags; these flags are not final content eligibility or scoring decisions.",
                "The analyst question is context. The CEO-only answer is the Q&A measurement target under the current specification.",
                "The manual sample is deterministic and quality-tier stratified; it is a diagnostic review, not an independent human validation or population precision estimate.",
                "No LLM or API was used. Transcript text remains in this local output directory.",
                "Raw turns and the speaker-gate outputs were not modified.",
            ],
        }

        final_artifact_names = [
            "ceo_qa_blocks.csv",
            "ceo_qa_event_audit.csv",
            "episode_block_coverage.csv",
            "turnover_block_coverage.csv",
            "manual_audit_sample.csv",
            "manual_audit_sample.md",
        ]
        summary["artifacts"] = {name: str(output_dir / name) for name in final_artifact_names}
        for name in final_artifact_names:
            summary.setdefault("artifact_sha256", {})[name] = sha256_file(temp_dir / name)

        report = [
            "# Strict-Gate CEO Q&A Block Extraction",
            "",
            f"- Date (UTC): `{summary['created_at_utc']}`",
            f"- Strict speaker-gate calls: {len(accepted):,} / {len(gate_rows):,}",
            f"- Calls extracted: {processed_gate_events:,}",
            f"- Candidate Q&A blocks: {block_count:,}",
            f"- Calls with / without candidate blocks: {summary['blocks']['events_with_candidate_blocks']:,} / {summary['blocks']['events_without_candidate_blocks']:,}",
            f"- Candidate block tiers: `{json.dumps(dict(quality_counts), sort_keys=True)}`",
            f"- Episode-gate pass with ≥1 candidate block: {summary['episode_coverage']['episode_gate_pass_with_candidate_blocks']:,} / {len(episode_pass_rows):,}",
            f"- Turnover-gate pass with candidate blocks on both CEO sides: {summary['turnover_coverage']['turnover_gate_pass_with_candidate_blocks_on_both_sides']:,} / {len(turnover_pass_rows):,}",
            f"- Manual audit sample: {sample_written:,} blocks; quality tiers `{json.dumps(summary['blocks']['manual_audit_sample_quality_tier_counts'], sort_keys=True)}`",
            "",
            "## Interpretation",
            "",
            "These are extracted candidate blocks, not yet final eligible scoring units. The event gate confirms a controlled ExecuComp-name match and a shared title-labelled CEO key in PRE and Q&A; the extractor anchors CEO answers to that key. Extractor `high` means structurally unflagged under its current rules, not substantively validated. The content-class decision remains separate.",
            "",
            "The analyst question is retained as context; only the CEO answer is the intended Q&A measurement text. No LLM scoring or API call was made.",
            "",
            "## Audit Notes",
            "",
            f"- Source sequence reconstruction passed for every extracted question, CEO answer, and answer context.",
            f"- Duplicate CEO-answer sequence IDs across blocks: {duplicate_answer_sequence_count:,}; affected blocks are marked for review.",
            f"- Input turn SHA-256 verified against the speaker-gate summary: `{turns_sha256}`.",
            "- The manual audit sample is deterministic and stratified across extractor quality tiers; it is not a probability sample or independent human validation.",
            "",
            "## Artifacts",
            "",
        ]
        report.extend(f"- `{name}`" for name in final_artifact_names)
        report.extend(["", "## Caveats", ""])
        report.extend(f"- {item}" for item in summary["caveats"])
        (temp_dir / "block_extraction_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        summary["artifacts"]["block_extraction_report.md"] = str(output_dir / "block_extraction_report.md")
        summary["artifact_sha256"]["block_extraction_report.md"] = sha256_file(temp_dir / "block_extraction_report.md")
        (temp_dir / "block_extraction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temp_dir / "VERSION_NOTES.md").write_text(
            f"# {SCRIPT_VERSION}\n\n"
            "First full strict-speaker-gate-anchored CEO Q&A candidate extraction. "
            "All 32,190 source event row counts were reconciled; only gate-pass calls were extracted. "
            "This version does not make final content-eligibility decisions and does not score text.\n\n"
            f"Source turn SHA-256: `{turns_sha256}`\n"
            f"Speaker gate: `{gate_version}`\n",
            encoding="utf-8",
        )
        os.replace(temp_dir, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True, type=Path)
    parser.add_argument("--event-gate", required=True, type=Path)
    parser.add_argument("--episode-gate", required=True, type=Path)
    parser.add_argument("--turnover-gate", required=True, type=Path)
    parser.add_argument("--gate-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manual-audit-blocks", type=int, default=75)
    args = parser.parse_args(argv)
    try:
        result = run_extraction(
            args.turns,
            args.event_gate,
            args.episode_gate,
            args.turnover_gate,
            args.gate_summary,
            args.output_dir,
            args.manual_audit_blocks,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
