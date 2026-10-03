#!/usr/bin/env python3
"""Build a deterministic, source-clean PRE/Q&A evaluation manifest.

The output contains selection metadata, counts, and source hashes only. It
never writes transcript text, human labels, prompts, or model outputs. This
tool is intentionally for a fresh evaluation manifest, not model training,
prompt selection, or production-universe scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PERIOD_BINS = ("2002_2007", "2008_2013", "2014_2019", "2020_2025")
SIDE_ORDER = ("old_episode", "new_episode")
STRATA = tuple(f"{period}|{side}" for period in PERIOD_BINS for side in SIDE_ORDER)

CALL_FIELDS = (
    "evaluation_split",
    "split_rank",
    "selection_stratum",
    "period_bin",
    "event_id",
    "start_date",
    "calendar_quarter",
    "year",
    "turnover_id",
    "turnover_side",
    "expected_execid",
    "expected_ceo_name",
    "company_id",
    "company_name",
    "clean_pre_units_available",
    "clean_qa_units_available",
    "pre_units_selected",
    "qa_units_selected",
    "selected_unit_count",
    "source_clean_pre_word_count",
    "source_clean_qa_target_word_count",
    "source_clean_qa_question_word_count",
    "event_source_gate",
)

UNIT_FIELDS = (
    "evaluation_split",
    "split_rank",
    "unit_rank_within_call",
    "sampling_bucket",
    "unit_type",
    "source_unit_id",
    "block_id",
    "event_id",
    "start_date",
    "calendar_quarter",
    "period_bin",
    "turnover_id",
    "turnover_side",
    "expected_execid",
    "company_id",
    "company_name",
    "question_word_count",
    "target_word_count",
    "source_content_sha256",
    "eligibility_source_marker_free",
)

EXCLUSION_FIELDS = (
    "source_path",
    "rows_read",
    "unique_event_ids",
    "event_ids_new_to_union",
)

PROHIBITED_OUTPUT_FIELDS = {
    "ceo_presentation_segment",
    "analyst_question",
    "ceo_answer",
    "answer_context",
    "prompt_input_json",
    "response",
    "human_label",
    "model_output",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(
    rows: Sequence[Mapping[str, str]], required: Iterable[str], path: Path
) -> None:
    if not rows:
        raise SystemExit(f"CSV is empty: {path}")
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise SystemExit(f"missing columns in {path}: {', '.join(missing)}")


def unique_index(
    rows: Sequence[dict[str, str]], key_field: str, path: Path
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row[key_field]
        if not key:
            raise SystemExit(f"blank {key_field} in {path}")
        if key in index:
            raise SystemExit(f"duplicate {key_field}={key!r} in {path}")
        index[key] = row
    return index


def integer(value: str, field: str, path: Path, *, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid integer {value!r} for {field} in {path}") from exc
    if parsed < minimum:
        raise SystemExit(f"{field} must be >= {minimum} in {path}; got {parsed}")
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(seed: int, lane: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{lane}|{value}".encode("utf-8")).hexdigest()


def year_from_date(value: str, path: Path) -> int:
    match = re.match(r"^(\d{4})-", value)
    if not match:
        raise SystemExit(f"cannot parse year from start_date {value!r} in {path}")
    return int(match.group(1))


def period_bin(year: int) -> str:
    if 2002 <= year <= 2007:
        return "2002_2007"
    if 2008 <= year <= 2013:
        return "2008_2013"
    if 2014 <= year <= 2019:
        return "2014_2019"
    if 2020 <= year <= 2025:
        return "2020_2025"
    raise SystemExit(f"year {year} is outside the evaluation frame 2002-2025")


def event_id_from_row(row: Mapping[str, str], path: Path) -> str:
    event_id = row.get("event_id", "").strip()
    if event_id:
        if re.fullmatch(r"\d+", event_id):
            return event_id
        raise SystemExit(f"invalid event_id {event_id!r} in exclusion file {path}")

    for field in ("source_custom_id", "custom_id", "block_id"):
        value = row.get(field, "").strip()
        if not value:
            continue
        match = re.search(r"(?:ccts|pre|qa)_(\d+)", value)
        if match:
            return match.group(1)
    raise SystemExit(f"cannot determine event_id for a row in exclusion file {path}")


def event_ids_from_exclusion_file(path: Path) -> tuple[set[str], int]:
    rows = read_csv(path)
    if not rows:
        raise SystemExit(f"exclusion file is empty: {path}")
    return {event_id_from_row(row, path) for row in rows}, len(rows)


def sampling_bucket(unit_type: str, target_word_count: int) -> str:
    if unit_type == "pre":
        return "pre_short_lt100" if target_word_count < 100 else "pre_long_ge100"
    if target_word_count < 50:
        return "qa_short_lt50"
    if target_word_count <= 200:
        return "qa_medium_50_200"
    return "qa_long_gt200"


def allocate_targets(
    counts: Mapping[str, int], total: int, minimum_per_stratum: int
) -> dict[str, int]:
    minimum_total = minimum_per_stratum * len(STRATA)
    if total < minimum_total:
        raise SystemExit(
            f"evaluation-calls={total} is below the required minimum {minimum_total}"
        )
    missing = [stratum for stratum in STRATA if counts.get(stratum, 0) < minimum_per_stratum]
    if missing:
        raise SystemExit(
            "not enough eligible calls for the per-stratum minimum: " + ", ".join(missing)
        )
    targets = {stratum: minimum_per_stratum for stratum in STRATA}
    remaining = total - minimum_total
    order = sorted(STRATA, key=lambda item: (-counts[item], item))
    index = 0
    while remaining:
        targets[order[index % len(order)]] += 1
        remaining -= 1
        index += 1
    return targets


def select_calls(
    candidates: Sequence[dict[str, str]],
    targets: Mapping[str, int],
    seed: int,
    max_calls_per_execid: int,
    max_calls_per_company: int,
) -> list[dict[str, str]]:
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_stratum[row["selection_stratum"]].append(row)
    for rows in by_stratum.values():
        rows.sort(key=lambda row: stable_key(seed, "fresh_evaluation_call", row["event_id"]))

    selected: list[dict[str, str]] = []
    selected_events: set[str] = set()
    execid_counts: Counter[str] = Counter()
    company_counts: Counter[str] = Counter()
    for stratum in STRATA:
        target = targets[stratum]
        chosen = 0
        for row in by_stratum[stratum]:
            if row["event_id"] in selected_events:
                continue
            if execid_counts[row["expected_execid"]] >= max_calls_per_execid:
                continue
            if company_counts[row["company_id"]] >= max_calls_per_company:
                continue
            selected.append(row)
            selected_events.add(row["event_id"])
            execid_counts[row["expected_execid"]] += 1
            company_counts[row["company_id"]] += 1
            chosen += 1
            if chosen == target:
                break
        if chosen != target:
            raise SystemExit(
                f"could not select {target} calls for {stratum}; selected {chosen}; "
                f"available candidates={len(by_stratum[stratum])}; "
                f"max_calls_per_execid={max_calls_per_execid}; "
                f"max_calls_per_company={max_calls_per_company}"
            )
    if len(selected) != sum(targets.values()):
        raise SystemExit("selection size does not match requested target")
    return selected


def choose_units(
    rows: Sequence[dict[str, str]],
    unit_type: str,
    desired_buckets: Sequence[str],
    seed: int,
    event_id: str,
    count: int,
) -> list[dict[str, str]]:
    by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        bucket = sampling_bucket(unit_type, int(row["target_word_count"]))
        by_bucket[bucket].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(
            key=lambda row: stable_key(seed, f"fresh_evaluation_unit|{event_id}", row["source_unit_id"])
        )

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    for bucket in desired_buckets:
        for row in by_bucket[bucket]:
            if row["source_unit_id"] in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row["source_unit_id"])
            break
        if len(selected) == count:
            break
    if len(selected) < count:
        remaining = sorted(
            (row for row in rows if row["source_unit_id"] not in selected_ids),
            key=lambda row: stable_key(
                seed, f"fresh_evaluation_unit_fill|{event_id}", row["source_unit_id"]
            ),
        )
        selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise SystemExit(
            f"could not select {count} {unit_type} units for event {event_id}; found {len(rows)}"
        )
    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def report(summary: Mapping[str, object]) -> str:
    selection = summary["selection"]
    frame = summary["frame"]
    counts = summary["counts"]
    lines = [
        "# Untouched PRE/Q&A Evaluation Manifest",
        "",
        "## Status",
        "",
        "- A fresh, deterministic source-clean evaluation manifest was created.",
        "- No transcript text, human labels, prompts, model outputs, or scores were written.",
        "- This is a new `current_source_marker_free` evaluation lane, not a reuse of the old locked validation labels.",
        "- It is not a model-promotion, prompt-selection, or full-universe-scoring result.",
        "",
        "## Selection",
        "",
        f"- Seed: `{selection['seed']}`",
        f"- Calls: `{selection['evaluation_calls']}`",
        f"- Units per call: `{selection['pre_units_per_call']} PRE + {selection['qa_units_per_call']} Q&A`",
        f"- Event-disjoint from every supplied prior-sample exclusion: `{selection['event_disjoint_from_prior_samples']}`",
        f"- Maximum calls per CEO: `{selection['max_calls_per_execid']}` (observed `{selection['max_observed_calls_per_execid']}`)",
        f"- Maximum calls per company: `{selection['max_calls_per_company']}` (observed `{selection['max_observed_calls_per_company']}`)",
        "",
        "## Source-clean eligibility",
        "",
        "- PRE: current unit exactly matches the 150-word/50-word-minimum lineage contract, and both parent-turn and unit-level source-marker fields are empty.",
        "- Q&A: source tier is `high`, diagnostic lane is `automatic_candidate_not_approved`, and the source reason field is empty.",
        "- Call: the existing source frame reports both strict identity readiness and an event speaker gate pass.",
        "- This definition supports a controlled evaluation frame only; it does not claim that the full production universe is already approved.",
        "",
        "## Frame",
        "",
        f"- Source-frame events with strict identity and speaker gates: `{frame['source_ready_events']}`",
        f"- Prior-sample exclusion union: `{frame['excluded_prior_event_ids']}` event IDs",
        f"- Candidate calls with at least the requested clean PRE/Q&A units after exclusions: `{frame['unit_ready_candidate_calls']}`",
        "",
        "| Stratum | Available | Selected |",
        "|---|---:|---:|",
    ]
    for stratum in STRATA:
        lines.append(
            f"| {stratum} | {frame['available_by_stratum'][stratum]} | "
            f"{counts['selected_by_stratum'][stratum]} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for label, path in summary["artifacts"].items():
        lines.append(f"- {label}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def validate_output_headers(path: Path, expected_fields: Sequence[str]) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    if tuple(header) != tuple(expected_fields):
        raise SystemExit(f"unexpected output header in {path}")
    bad = PROHIBITED_OUTPUT_FIELDS & set(header)
    if bad:
        raise SystemExit(f"prohibited text-bearing output fields in {path}: {sorted(bad)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-units", required=True, type=Path)
    parser.add_argument("--pre-lineage", required=True, type=Path)
    parser.add_argument("--pre-representations", required=True, type=Path)
    parser.add_argument("--qa-blocks", required=True, type=Path)
    parser.add_argument("--qa-lanes", required=True, type=Path)
    parser.add_argument("--exclude-file", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--evaluation-calls", type=int, default=20)
    parser.add_argument("--pre-units-per-call", type=int, default=2)
    parser.add_argument("--qa-units-per-call", type=int, default=3)
    parser.add_argument("--minimum-calls-per-stratum", type=int, default=2)
    parser.add_argument("--max-calls-per-execid", type=int, default=2)
    parser.add_argument("--max-calls-per-company", type=int, default=3)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    for field in (
        "evaluation_calls",
        "pre_units_per_call",
        "qa_units_per_call",
        "minimum_calls_per_stratum",
        "max_calls_per_execid",
        "max_calls_per_company",
    ):
        if getattr(args, field) < 1:
            raise SystemExit(f"{field} must be positive")

    pre_rows = read_csv(args.pre_units)
    lineage_rows = read_csv(args.pre_lineage)
    representation_rows = read_csv(args.pre_representations)
    qa_rows = read_csv(args.qa_blocks)
    qa_lane_rows = read_csv(args.qa_lanes)
    require_columns(
        pre_rows,
        (
            "custom_id",
            "event_id",
            "source_sequence_id",
            "chunk_index",
            "unit_word_count",
            "ceo_presentation_segment",
        ),
        args.pre_units,
    )
    require_columns(
        lineage_rows,
        (
            "custom_id",
            "event_id",
            "sequence_id",
            "chunk_index",
            "unit_word_count",
            "source_turn_reasons",
            "unit_reasons",
            "segment_sha256",
        ),
        args.pre_lineage,
    )
    require_columns(
        representation_rows,
        (
            "event_id",
            "start_date",
            "calendar_quarter",
            "year",
            "turnover_id",
            "turnover_side",
            "expected_execid",
            "expected_ceo_name",
            "company_id",
            "company_name",
            "within_call_strict_ready",
            "strict_identity_ready",
            "event_speaker_gate_pass",
        ),
        args.pre_representations,
    )
    require_columns(
        qa_rows,
        (
            "block_id",
            "event_id",
            "quality_tier",
            "question_word_count",
            "ceo_answer_word_count",
            "analyst_question",
            "ceo_answer",
        ),
        args.qa_blocks,
    )
    require_columns(
        qa_lane_rows,
        ("block_id", "event_id", "source_quality_tier", "diagnostic_lane", "reasons"),
        args.qa_lanes,
    )

    current_pre = unique_index(pre_rows, "custom_id", args.pre_units)
    representations = unique_index(representation_rows, "event_id", args.pre_representations)
    qa_blocks = unique_index(qa_rows, "block_id", args.qa_blocks)
    qa_lanes = unique_index(qa_lane_rows, "block_id", args.qa_lanes)
    if set(qa_blocks) != set(qa_lanes):
        raise SystemExit("qa-block and qa-lane block_id coverage differs")

    pre_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    pre_filter_counts: Counter[str] = Counter()
    for lineage in lineage_rows:
        unit_id = lineage["custom_id"]
        current = current_pre.get(unit_id)
        if current is None:
            pre_filter_counts["missing_current_unit"] += 1
            continue
        source_marker_free = not lineage["source_turn_reasons"].strip() and not lineage[
            "unit_reasons"
        ].strip()
        if not source_marker_free:
            pre_filter_counts["source_marker_or_unit_marker"] += 1
            continue
        metadata_match = (
            lineage["event_id"] == current["event_id"]
            and lineage["sequence_id"] == current["source_sequence_id"]
            and lineage["chunk_index"] == current["chunk_index"]
            and lineage["unit_word_count"] == current["unit_word_count"]
        )
        content_hash = sha256_text(current["ceo_presentation_segment"])
        if not metadata_match or content_hash != lineage["segment_sha256"]:
            pre_filter_counts["current_contract50_mismatch"] += 1
            continue
        pre_by_event[lineage["event_id"]].append(
            {
                "unit_type": "pre",
                "source_unit_id": unit_id,
                "block_id": "",
                "event_id": lineage["event_id"],
                "question_word_count": "",
                "target_word_count": lineage["unit_word_count"],
                "source_content_sha256": content_hash,
                "eligibility_source_marker_free": "1",
            }
        )
        pre_filter_counts["clean_contract50_consistent"] += 1

    qa_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    qa_filter_counts: Counter[str] = Counter()
    for block_id, lane in qa_lanes.items():
        block = qa_blocks[block_id]
        if lane["event_id"] != block["event_id"]:
            raise SystemExit(f"event_id mismatch for Q&A block {block_id}")
        clean_lane = (
            lane["source_quality_tier"] == "high"
            and lane["diagnostic_lane"] == "automatic_candidate_not_approved"
            and not lane["reasons"].strip()
        )
        if not clean_lane:
            qa_filter_counts["not_clean_automatic_high"] += 1
            continue
        if block["quality_tier"] != "high":
            qa_filter_counts["quality_tier_mismatch"] += 1
            continue
        target_word_count = integer(
            block["ceo_answer_word_count"], "ceo_answer_word_count", args.qa_blocks, minimum=1
        )
        question_word_count = integer(
            block["question_word_count"], "question_word_count", args.qa_blocks, minimum=0
        )
        qa_by_event[block["event_id"]].append(
            {
                "unit_type": "qa",
                "source_unit_id": block_id,
                "block_id": block_id,
                "event_id": block["event_id"],
                "question_word_count": str(question_word_count),
                "target_word_count": str(target_word_count),
                "source_content_sha256": sha256_text(
                    block["analyst_question"] + "\n" + block["ceo_answer"]
                ),
                "eligibility_source_marker_free": "1",
            }
        )
        qa_filter_counts["clean_automatic_high"] += 1

    exclusion_rows: list[dict[str, object]] = []
    excluded_event_ids: set[str] = set()
    for path in args.exclude_file:
        event_ids, row_count = event_ids_from_exclusion_file(path)
        new_ids = event_ids - excluded_event_ids
        exclusion_rows.append(
            {
                "source_path": str(path.resolve()),
                "rows_read": row_count,
                "unique_event_ids": len(event_ids),
                "event_ids_new_to_union": len(new_ids),
            }
        )
        excluded_event_ids.update(event_ids)

    candidate_counts: Counter[str] = Counter()
    candidates: list[dict[str, str]] = []
    for event_id, representation in representations.items():
        event_gate_pass = (
            representation["within_call_strict_ready"] == "1"
            and representation["strict_identity_ready"] == "1"
            and representation["event_speaker_gate_pass"] == "1"
        )
        if not event_gate_pass:
            candidate_counts["event_source_gate_not_ready"] += 1
            continue
        candidate_counts["event_source_gate_ready"] += 1
        if event_id in excluded_event_ids:
            candidate_counts["excluded_prior_event"] += 1
            continue
        pre_available = len(pre_by_event.get(event_id, []))
        qa_available = len(qa_by_event.get(event_id, []))
        if pre_available < args.pre_units_per_call or qa_available < args.qa_units_per_call:
            candidate_counts["insufficient_clean_units"] += 1
            continue
        year = year_from_date(representation["start_date"], args.pre_representations)
        if year != integer(representation["year"], "year", args.pre_representations, minimum=1):
            raise SystemExit(f"conflicting start_date/year metadata for event {event_id}")
        side = representation["turnover_side"]
        if side not in SIDE_ORDER:
            raise SystemExit(f"unexpected turnover_side {side!r} for event {event_id}")
        candidate = dict(representation)
        candidate.update(
            {
                "year": str(year),
                "period_bin": period_bin(year),
                "selection_stratum": f"{period_bin(year)}|{side}",
                "clean_pre_units_available": str(pre_available),
                "clean_qa_units_available": str(qa_available),
            }
        )
        candidates.append(candidate)
        candidate_counts["unit_ready_candidate"] += 1

    available_by_stratum = Counter(row["selection_stratum"] for row in candidates)
    for stratum in STRATA:
        available_by_stratum.setdefault(stratum, 0)
    targets = allocate_targets(
        available_by_stratum, args.evaluation_calls, args.minimum_calls_per_stratum
    )
    selected = select_calls(
        candidates,
        targets,
        args.seed,
        args.max_calls_per_execid,
        args.max_calls_per_company,
    )
    selected.sort(
        key=lambda row: (
            STRATA.index(row["selection_stratum"]),
            stable_key(args.seed, "fresh_evaluation_rank", row["event_id"]),
        )
    )

    call_rows: list[dict[str, object]] = []
    unit_rows: list[dict[str, object]] = []
    for split_rank, call in enumerate(selected, start=1):
        event_id = call["event_id"]
        selected_pre = choose_units(
            pre_by_event[event_id],
            "pre",
            ("pre_short_lt100", "pre_long_ge100"),
            args.seed,
            event_id,
            args.pre_units_per_call,
        )
        selected_qa = choose_units(
            qa_by_event[event_id],
            "qa",
            ("qa_short_lt50", "qa_medium_50_200", "qa_long_gt200"),
            args.seed,
            event_id,
            args.qa_units_per_call,
        )
        call_rows.append(
            {
                "evaluation_split": "fresh_evaluation",
                "split_rank": split_rank,
                "selection_stratum": call["selection_stratum"],
                "period_bin": call["period_bin"],
                "event_id": event_id,
                "start_date": call["start_date"],
                "calendar_quarter": call["calendar_quarter"],
                "year": call["year"],
                "turnover_id": call["turnover_id"],
                "turnover_side": call["turnover_side"],
                "expected_execid": call["expected_execid"],
                "expected_ceo_name": call["expected_ceo_name"],
                "company_id": call["company_id"],
                "company_name": call["company_name"],
                "clean_pre_units_available": call["clean_pre_units_available"],
                "clean_qa_units_available": call["clean_qa_units_available"],
                "pre_units_selected": len(selected_pre),
                "qa_units_selected": len(selected_qa),
                "selected_unit_count": len(selected_pre) + len(selected_qa),
                "source_clean_pre_word_count": sum(
                    int(row["target_word_count"]) for row in pre_by_event[event_id]
                ),
                "source_clean_qa_target_word_count": sum(
                    int(row["target_word_count"]) for row in qa_by_event[event_id]
                ),
                "source_clean_qa_question_word_count": sum(
                    int(row["question_word_count"]) for row in qa_by_event[event_id]
                ),
                "event_source_gate": "within_call_strict_ready;strict_identity_ready;event_speaker_gate_pass",
            }
        )
        selected_units = [("pre", row) for row in selected_pre] + [("qa", row) for row in selected_qa]
        selected_units.sort(
            key=lambda item: stable_key(
                args.seed, f"fresh_evaluation_unit_rank|{event_id}", item[1]["source_unit_id"]
            )
        )
        for unit_rank, (unit_type, unit) in enumerate(selected_units, start=1):
            unit_rows.append(
                {
                    "evaluation_split": "fresh_evaluation",
                    "split_rank": split_rank,
                    "unit_rank_within_call": unit_rank,
                    "sampling_bucket": sampling_bucket(unit_type, int(unit["target_word_count"])),
                    "unit_type": unit_type,
                    "source_unit_id": unit["source_unit_id"],
                    "block_id": unit["block_id"],
                    "event_id": event_id,
                    "start_date": call["start_date"],
                    "calendar_quarter": call["calendar_quarter"],
                    "period_bin": call["period_bin"],
                    "turnover_id": call["turnover_id"],
                    "turnover_side": call["turnover_side"],
                    "expected_execid": call["expected_execid"],
                    "company_id": call["company_id"],
                    "company_name": call["company_name"],
                    "question_word_count": unit["question_word_count"],
                    "target_word_count": unit["target_word_count"],
                    "source_content_sha256": unit["source_content_sha256"],
                    "eligibility_source_marker_free": unit["eligibility_source_marker_free"],
                }
            )

    if len(call_rows) != args.evaluation_calls:
        raise SystemExit("selected call count mismatch")
    if len(unit_rows) != args.evaluation_calls * (
        args.pre_units_per_call + args.qa_units_per_call
    ):
        raise SystemExit("selected unit count mismatch")
    if len({row["event_id"] for row in call_rows}) != len(call_rows):
        raise SystemExit("event-disjoint selection invariant failed")
    if {row["event_id"] for row in call_rows} & excluded_event_ids:
        raise SystemExit("selected event overlaps a prior-sample exclusion")
    if len({row["source_unit_id"] for row in unit_rows}) != len(unit_rows):
        raise SystemExit("duplicate selected source unit")

    selected_by_stratum = Counter(row["selection_stratum"] for row in call_rows)
    execid_counts = Counter(row["expected_execid"] for row in call_rows)
    company_counts = Counter(row["company_id"] for row in call_rows)
    created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_paths = {
        "pre_units_csv": args.pre_units,
        "pre_lineage_csv": args.pre_lineage,
        "pre_representations_csv": args.pre_representations,
        "qa_blocks_csv": args.qa_blocks,
        "qa_lanes_csv": args.qa_lanes,
    }

    stage_dir = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    try:
        calls_path = stage_dir / "evaluation_call_manifest.csv"
        units_path = stage_dir / "evaluation_unit_manifest.csv"
        exclusions_path = stage_dir / "prior_sample_exclusion_audit.csv"
        write_csv(calls_path, call_rows, CALL_FIELDS)
        write_csv(units_path, unit_rows, UNIT_FIELDS)
        write_csv(exclusions_path, exclusion_rows, EXCLUSION_FIELDS)
        validate_output_headers(calls_path, CALL_FIELDS)
        validate_output_headers(units_path, UNIT_FIELDS)
        validate_output_headers(exclusions_path, EXCLUSION_FIELDS)

        summary: dict[str, object] = {
            "created_at_utc": created_at,
            "status": "fresh_current_source_marker_free_evaluation_manifest_unscored",
            "scope": {
                "purpose": "event-disjoint evaluation manifest only",
                "contains_transcript_text": False,
                "contains_human_labels": False,
                "contains_prompts": False,
                "contains_model_outputs": False,
                "historical_label_reuse": False,
                "model_run": False,
                "production_universe_approval": False,
            },
            "selection": {
                "seed": args.seed,
                "evaluation_calls": args.evaluation_calls,
                "pre_units_per_call": args.pre_units_per_call,
                "qa_units_per_call": args.qa_units_per_call,
                "minimum_calls_per_stratum": args.minimum_calls_per_stratum,
                "max_calls_per_execid": args.max_calls_per_execid,
                "max_calls_per_company": args.max_calls_per_company,
                "event_disjoint_from_prior_samples": True,
                "max_observed_calls_per_execid": max(execid_counts.values()),
                "max_observed_calls_per_company": max(company_counts.values()),
                "period_bins": list(PERIOD_BINS),
                "turnover_sides": list(SIDE_ORDER),
                "target_calls_by_stratum": dict(targets),
            },
            "eligibility_rules": {
                "pre": "marker-free parent and unit lineage plus exact current contract50 text/hash/metadata match",
                "qa": "high source tier, automatic_candidate_not_approved lane, empty reason code, and high block tier",
                "call": "within_call_strict_ready=1, strict_identity_ready=1, event_speaker_gate_pass=1",
            },
            "frame": {
                "source_ready_events": candidate_counts["event_source_gate_ready"],
                "excluded_prior_event_ids": len(excluded_event_ids),
                "excluded_prior_events_in_source_frame": candidate_counts["excluded_prior_event"],
                "unit_ready_candidate_calls": candidate_counts["unit_ready_candidate"],
                "insufficient_clean_units_after_exclusion": candidate_counts["insufficient_clean_units"],
                "available_by_stratum": {stratum: available_by_stratum[stratum] for stratum in STRATA},
            },
            "counts": {
                "pre_filter": dict(pre_filter_counts),
                "qa_filter": dict(qa_filter_counts),
                "candidate_filter": dict(candidate_counts),
                "selected_calls": len(call_rows),
                "selected_units": len(unit_rows),
                "selected_pre_units": sum(row["unit_type"] == "pre" for row in unit_rows),
                "selected_qa_units": sum(row["unit_type"] == "qa" for row in unit_rows),
                "selected_by_stratum": {stratum: selected_by_stratum[stratum] for stratum in STRATA},
            },
            "sources": {
                label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for label, path in source_paths.items()
            },
            "exclusion_sources": exclusion_rows,
            "artifacts": {
                "evaluation_call_manifest_csv": str((args.output_dir / calls_path.name).resolve()),
                "evaluation_unit_manifest_csv": str((args.output_dir / units_path.name).resolve()),
                "prior_sample_exclusion_audit_csv": str((args.output_dir / exclusions_path.name).resolve()),
                "summary_json": str((args.output_dir / "evaluation_manifest_summary.json").resolve()),
                "report_md": str((args.output_dir / "evaluation_manifest_report.md").resolve()),
            },
        }
        summary_path = stage_dir / "evaluation_manifest_summary.json"
        report_path = stage_dir / "evaluation_manifest_report.md"
        write_json(summary_path, summary)
        report_path.write_text(report(summary), encoding="utf-8")
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir.rename(args.output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
