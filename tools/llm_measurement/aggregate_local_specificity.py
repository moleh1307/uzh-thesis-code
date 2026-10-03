#!/usr/bin/env python3
"""Aggregate local specificity scores at call level and crosswalk human labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Optional, Sequence


COMMON_MIN_WORDS = 50
COMMON_MAX_WORDS = 200
CALL_METADATA_COLUMNS = (
    "anchor_rank",
    "anchor_pair_rank",
    "turnover_id",
    "turnover_side",
    "calendar_quarter",
    "source_phase",
    "audit_stratum",
    "tenure_episode",
    "call_index_in_episode",
    "calls_in_episode",
    "distinct_quarters_in_episode",
    "event_id",
    "start_date",
    "year",
    "company_id",
    "company_name",
    "company_ticker",
    "event_title",
    "speaker_validation_status",
    "primary_specificity_call_eligible",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_unique(rows: Iterable[Mapping[str, Any]], key: str, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value:
            raise SystemExit(f"{label} has an empty {key}")
        if value in result:
            raise SystemExit(f"{label} has duplicate {key}: {value}")
        result[value] = row
    return result


def validate_model_score(value: object) -> tuple[bool, Optional[int], Optional[int], str]:
    if not isinstance(value, dict):
        return False, None, None, "parsed output is not an object"
    if set(value) != {"ok", "specificity"}:
        return False, None, None, "parsed output keys are not exactly ok,specificity"
    ok = value.get("ok")
    specificity = value.get("specificity")
    if isinstance(ok, bool) or not isinstance(ok, int) or ok not in {0, 1}:
        return False, None, None, "ok is not integer 0 or 1"
    if isinstance(specificity, bool) or not isinstance(specificity, int) or not 0 <= specificity <= 5:
        return False, ok, None, "specificity is not integer 0 through 5"
    if ok == 0 and specificity != 0:
        return False, ok, specificity, "ok=0 must have specificity=0"
    if ok == 1 and specificity == 0:
        return False, ok, specificity, "ok=1 must have specificity 1 through 5"
    return True, ok, specificity, ""


def parse_int(value: object, field: str, custom_id: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{custom_id} has non-integer {field}: {value!r}") from exc


def safe_mean(values: Sequence[float | int]) -> Optional[float]:
    return float(mean(values)) if values else None


def safe_median(values: Sequence[float | int]) -> Optional[float]:
    return float(median(values)) if values else None


def weighted_mean(scores_and_words: Sequence[tuple[int, int]]) -> Optional[float]:
    denominator = sum(words for _, words in scores_and_words)
    if denominator <= 0:
        return None
    return sum(score * words for score, words in scores_and_words) / denominator


def summarize_units(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [unit for unit in units if unit.get("scored")]
    common = [
        unit
        for unit in scored
        if COMMON_MIN_WORDS <= int(unit["unit_word_count"]) <= COMMON_MAX_WORDS
    ]

    def scores(rows: Sequence[Mapping[str, Any]]) -> list[int]:
        return [int(row["model_specificity"]) for row in rows]

    def score_words(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
        return [(int(row["model_specificity"]), int(row["unit_word_count"])) for row in rows]

    return {
        "unit_count_total": len(units),
        "unit_count_scored": len(scored),
        "unit_count_unscored": len(units) - len(scored),
        "unit_count_ok0": sum(unit.get("model_ok") == 0 for unit in units),
        "unit_count_invalid": sum(bool(unit.get("model_validation_error")) for unit in units),
        "word_count_total": sum(int(unit["unit_word_count"]) for unit in units),
        "word_count_scored": sum(int(unit["unit_word_count"]) for unit in scored),
        "word_count_common_50_200_scored": sum(int(unit["unit_word_count"]) for unit in common),
        "common_50_200_unit_count_scored": len(common),
        "word_weighted_mean": weighted_mean(score_words(scored)),
        "equal_unit_mean": safe_mean(scores(scored)),
        "unit_median": safe_median(scores(scored)),
        "common_50_200_word_weighted_mean": weighted_mean(score_words(common)),
        "common_50_200_equal_unit_mean": safe_mean(scores(common)),
        "common_50_200_unit_median": safe_median(scores(common)),
    }


def summarize_human_units(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [
        unit
        for unit in units
        if str(unit.get("human_specificity", "")).strip() not in {"", "0"}
    ]
    common = [
        unit
        for unit in scored
        if COMMON_MIN_WORDS <= int(unit["unit_word_count"]) <= COMMON_MAX_WORDS
    ]

    def scores(rows: Sequence[Mapping[str, Any]]) -> list[int]:
        return [int(row["human_specificity"]) for row in rows]

    def score_words(rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
        return [(int(row["human_specificity"]), int(row["unit_word_count"])) for row in rows]

    return {
        "unit_count_total": len(units),
        "unit_count_scored": len(scored),
        "unit_count_unscored": len(units) - len(scored),
        "word_count_total": sum(int(unit["unit_word_count"]) for unit in units),
        "word_count_scored": sum(int(unit["unit_word_count"]) for unit in scored),
        "word_count_common_50_200_scored": sum(int(unit["unit_word_count"]) for unit in common),
        "common_50_200_unit_count_scored": len(common),
        "word_weighted_mean": weighted_mean(score_words(scored)),
        "equal_unit_mean": safe_mean(scores(scored)),
        "unit_median": safe_median(scores(scored)),
        "common_50_200_word_weighted_mean": weighted_mean(score_words(common)),
        "common_50_200_equal_unit_mean": safe_mean(scores(common)),
        "common_50_200_unit_median": safe_median(scores(common)),
    }


def difference(left: object, right: object) -> Optional[float]:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def pearson(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
    if len(left) < 2 or len(right) < 2:
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if not left_ss or not right_ss:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def ranks(values: Sequence[int]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in range(cursor, end):
            result[ordered[index][0]] = average
        cursor = end
    return result


def weighted_kappa(left: Sequence[int], right: Sequence[int], quadratic: bool) -> Optional[float]:
    if not left:
        return None
    observed: Counter[tuple[int, int]] = Counter(zip(left, right))
    left_counts: Counter[int] = Counter(left)
    right_counts: Counter[int] = Counter(right)
    total = len(left)
    observed_weighted = 0.0
    expected_weighted = 0.0
    for i in range(1, 6):
        for j in range(1, 6):
            distance = abs(i - j) / 4
            weight = distance**2 if quadratic else distance
            observed_weighted += weight * observed[(i, j)] / total
            expected_weighted += weight * left_counts[i] * right_counts[j] / (total * total)
    if not expected_weighted:
        return None
    return 1 - observed_weighted / expected_weighted


def agreement_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    comparable_rows = [row for row in rows if row.get("model_scored")]
    human = [int(row["human_specificity"]) for row in comparable_rows]
    model = [int(row["model_specificity"]) for row in comparable_rows]
    absolute = [abs(x - y) for x, y in zip(human, model)]
    return {
        "n": len(comparable_rows),
        "n_exact_overlap": len(rows),
        "n_model_unscored": len(rows) - len(comparable_rows),
        "human_mean": safe_mean(human),
        "model_mean": safe_mean(model),
        "exact_agreement": sum(value == 0 for value in absolute) / len(absolute) if absolute else None,
        "within_one_agreement": sum(value <= 1 for value in absolute) / len(absolute) if absolute else None,
        "two_or_more_rate": sum(value >= 2 for value in absolute) / len(absolute) if absolute else None,
        "mean_absolute_error": safe_mean(absolute),
        "spearman": pearson(ranks(human), ranks(model)) if rows else None,
        "linear_weighted_kappa": weighted_kappa(human, model, quadratic=False),
        "quadratic_weighted_kappa": weighted_kappa(human, model, quadratic=True),
        "human_distribution": {str(score): human.count(score) for score in range(1, 6)},
        "model_distribution": {str(score): model.count(score) for score in range(1, 6)},
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_unit_rows(
    input_rows: Mapping[str, Mapping[str, str]], output_rows: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for custom_id, source in input_rows.items():
        output = output_rows[custom_id]
        valid, model_ok, model_specificity, validation_error = validate_model_score(output.get("parsed"))
        output_error = output.get("validation_error")
        model_validation_error = str(output_error or validation_error or "")
        if output.get("status") != "completed" and not model_validation_error:
            model_validation_error = f"status={output.get('status', '')}"
        scored = bool(
            output.get("status") == "completed"
            and not model_validation_error
            and valid
            and model_ok == 1
            and model_specificity is not None
        )
        result.append(
            {
                "custom_id": custom_id,
                "unit_type": source["unit_type"],
                "event_id": source["event_id"],
                "start_date": source["start_date"],
                "unit_word_count": parse_int(source["unit_word_count"], "unit_word_count", custom_id),
                "question_word_count": source.get("question_word_count", ""),
                "source_quality_tier": source.get("source_quality_tier", ""),
                "source_block_flags": source.get("source_block_flags", ""),
                "status": output.get("status", ""),
                "model_ok": model_ok if model_ok is not None else "",
                "model_specificity": model_specificity if model_specificity is not None else "",
                "scored": scored,
                "model_validation_error": model_validation_error,
                "input_tokens": output.get("input_tokens", ""),
                "output_tokens": output.get("output_tokens", ""),
                "elapsed_seconds": output.get("elapsed_seconds", ""),
            }
        )
    return result


def build_call_rows(
    unit_rows: Sequence[Mapping[str, Any]],
    call_metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in unit_rows:
        grouped[str(row["event_id"])].append(row)
    result: list[dict[str, Any]] = []
    for event_id in sorted(grouped, key=lambda value: (grouped[value][0]["start_date"], value)):
        units = grouped[event_id]
        by_type = {
            unit_type: summarize_units([unit for unit in units if unit["unit_type"] == unit_type])
            for unit_type in ("pre", "qa")
        }
        row: dict[str, Any] = {column: call_metadata.get(event_id, {}).get(column, "") for column in CALL_METADATA_COLUMNS}
        row.update(
            {
                "event_id": event_id,
                "start_date": units[0]["start_date"],
                "pre_scored_call_eligible": bool(
                    by_type["pre"]["unit_count_scored"] > 0 and by_type["qa"]["unit_count_scored"] > 0
                ),
            }
        )
        for unit_type, summary in by_type.items():
            prefix = unit_type
            for key, value in summary.items():
                row[f"{prefix}_{key}"] = value
        for measure in (
            "word_weighted_mean",
            "equal_unit_mean",
            "unit_median",
            "common_50_200_word_weighted_mean",
            "common_50_200_equal_unit_mean",
            "common_50_200_unit_median",
        ):
            row[f"pre_minus_qa_{measure}"] = difference(
                row[f"pre_{measure}"], row[f"qa_{measure}"]
            )
        result.append(row)
    return result


def build_human_call_rows(
    human_crosswalk: Sequence[Mapping[str, Any]],
    call_metadata: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in human_crosswalk:
        grouped[str(row["event_id"])].append(row)
    result: list[dict[str, Any]] = []
    for event_id in sorted(grouped, key=lambda value: (call_metadata.get(value, {}).get("start_date", ""), value)):
        units = grouped[event_id]
        by_type = {
            unit_type: summarize_human_units(
                [unit for unit in units if unit["unit_type"] == unit_type]
            )
            for unit_type in ("pre", "qa")
        }
        row: dict[str, Any] = {
            column: call_metadata.get(event_id, {}).get(column, "")
            for column in CALL_METADATA_COLUMNS
        }
        row.update(
            {
                "event_id": event_id,
                "start_date": call_metadata.get(event_id, {}).get(
                    "start_date", units[0].get("start_date", "")
                ),
                "human_scored_call_eligible": bool(
                    by_type["pre"]["unit_count_scored"] > 0
                    and by_type["qa"]["unit_count_scored"] > 0
                ),
            }
        )
        for unit_type, summary in by_type.items():
            for key, value in summary.items():
                row[f"{unit_type}_{key}"] = value
        for measure in (
            "word_weighted_mean",
            "equal_unit_mean",
            "unit_median",
            "common_50_200_word_weighted_mean",
            "common_50_200_equal_unit_mean",
            "common_50_200_unit_median",
        ):
            row[f"pre_minus_qa_{measure}"] = difference(
                row[f"pre_{measure}"], row[f"qa_{measure}"]
            )
        result.append(row)
    return result


def call_level_comparison(
    model_call_rows: Sequence[Mapping[str, Any]],
    human_call_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model_by_event = index_unique(model_call_rows, "event_id", "model call results")
    human_by_event = index_unique(human_call_rows, "event_id", "human call results")
    event_ids = sorted(set(model_by_event) & set(human_by_event))

    def compare(model_field: str, human_field: str) -> dict[str, Any]:
        pairs = [
            (float(model_by_event[event_id][model_field]), float(human_by_event[event_id][human_field]))
            for event_id in event_ids
            if model_by_event[event_id].get(model_field) not in {None, ""}
            and human_by_event[event_id].get(human_field) not in {None, ""}
        ]
        model_values = [pair[0] for pair in pairs]
        human_values = [pair[1] for pair in pairs]
        differences = [model - human for model, human in pairs]
        return {
            "n_calls": len(pairs),
            "model_call_mean": safe_mean(model_values),
            "human_call_mean": safe_mean(human_values),
            "model_minus_human_call_mean": safe_mean(differences),
            "mean_absolute_difference": safe_mean([abs(value) for value in differences]),
            "pearson": pearson(model_values, human_values),
        }

    fields = {
        "PRE CEO-word-weighted": (
            "pre_word_weighted_mean",
            "pre_word_weighted_mean",
        ),
        "Q&A CEO-word-weighted": (
            "qa_word_weighted_mean",
            "qa_word_weighted_mean",
        ),
        "PRE-Q&A CEO-word-weighted": (
            "pre_minus_qa_word_weighted_mean",
            "pre_minus_qa_word_weighted_mean",
        ),
        "PRE equal-unit mean": ("pre_equal_unit_mean", "pre_equal_unit_mean"),
        "Q&A equal-unit mean": ("qa_equal_unit_mean", "qa_equal_unit_mean"),
        "PRE-Q&A equal-unit mean": (
            "pre_minus_qa_equal_unit_mean",
            "pre_minus_qa_equal_unit_mean",
        ),
    }
    return {
        label: compare(model_field, human_field)
        for label, (model_field, human_field) in fields.items()
    }


def build_human_crosswalk(
    unit_rows: Sequence[Mapping[str, Any]],
    human_coded: Mapping[str, Mapping[str, str]],
    human_key: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    model_by_id = index_unique(unit_rows, "custom_id", "unit results")
    result: list[dict[str, Any]] = []
    for custom_id in sorted(set(model_by_id) & set(human_key)):
        key_row = human_key[custom_id]
        audit_id = str(key_row["audit_id"])
        if audit_id not in human_coded:
            raise SystemExit(f"human key points to missing audit_id: {audit_id}")
        human_row = human_coded[audit_id]
        if human_row.get("human_ok") != "1" or not str(human_row.get("human_specificity", "")):
            raise SystemExit(f"human overlap row is not a valid frozen score: {audit_id}")
        source = model_by_id[custom_id]
        human_score = parse_int(human_row["human_specificity"], "human_specificity", audit_id)
        if human_score not in range(1, 6):
            raise SystemExit(f"human score outside 1-5: {audit_id}")
        model_score = source["model_specificity"] if source["scored"] else ""
        result.append(
            {
                "audit_id": audit_id,
                "custom_id": custom_id,
                "audit_split": human_row.get("audit_split", ""),
                "unit_type": source["unit_type"],
                "event_id": source["event_id"],
                "unit_word_count": source["unit_word_count"],
                "human_specificity": human_score,
                "model_ok": source["model_ok"],
                "model_specificity": model_score,
                "model_scored": bool(source["scored"]),
                "signed_model_minus_human": (
                    int(model_score) - human_score if source["scored"] else ""
                ),
                "absolute_difference": abs(int(model_score) - human_score) if source["scored"] else "",
            }
        )
    return result


def aggregate_call_metrics(call_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "calls": len(call_rows),
        "calls_with_scored_pre": sum(row.get("pre_unit_count_scored", 0) > 0 for row in call_rows),
        "calls_with_scored_qa": sum(row.get("qa_unit_count_scored", 0) > 0 for row in call_rows),
        "calls_with_both_scored": sum(bool(row.get("pre_scored_call_eligible")) for row in call_rows),
    }
    measures = (
        "word_weighted_mean",
        "equal_unit_mean",
        "unit_median",
        "common_50_200_word_weighted_mean",
        "common_50_200_equal_unit_mean",
        "common_50_200_unit_median",
    )
    for measure in measures:
        pre_values = [float(row[f"pre_{measure}"]) for row in call_rows if row.get(f"pre_{measure}") is not None]
        qa_values = [float(row[f"qa_{measure}"]) for row in call_rows if row.get(f"qa_{measure}") is not None]
        diff_values = [
            float(row[f"pre_minus_qa_{measure}"])
            for row in call_rows
            if row.get(f"pre_minus_qa_{measure}") is not None
        ]
        summary[measure] = {
            "pre_call_mean": safe_mean(pre_values),
            "qa_call_mean": safe_mean(qa_values),
            "pre_minus_qa_call_mean": safe_mean(diff_values),
            "pre_call_median": safe_median(pre_values),
            "qa_call_median": safe_median(qa_values),
            "pre_minus_qa_call_median": safe_median(diff_values),
            "n_pre": len(pre_values),
            "n_qa": len(qa_values),
            "n_difference": len(diff_values),
        }
    return summary


def fmt(value: object, digits: int = 4) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: object) -> str:
    if value is None:
        return "NA"
    return f"{100 * float(value):.1f}%"


def build_report(
    *,
    timestamp: str,
    unit_rows: Sequence[Mapping[str, Any]],
    call_rows: Sequence[Mapping[str, Any]],
    call_summary: Mapping[str, Any],
    human_crosswalk: Sequence[Mapping[str, Any]],
    human_metrics: Mapping[str, Any],
    human_call_rows: Sequence[Mapping[str, Any]],
    human_call_comparison: Mapping[str, Any],
    output_dir: Path,
) -> str:
    unit_counts = Counter(row["unit_type"] for row in unit_rows)
    score_counts = Counter(
        int(row["model_specificity"])
        for row in unit_rows
        if row["scored"]
    )
    overall_human_metrics = human_metrics.get("all exact-overlap units", {})
    lines = [
        "# Local Specificity Call-Level Diagnostic",
        "",
        f"- Created UTC: {timestamp}",
        "- Status: completed diagnostic aggregation",
        "- Scope: the frozen ten-call turnover-focused anchor; this is not a representative full-universe estimate.",
        "- Unit scores are preserved separately and `ok=0` units are excluded from score means without imputation.",
        "",
        "## Integrity",
        "",
        f"- Unit rows: {len(unit_rows):,} ({unit_counts['pre']:,} PRE, {unit_counts['qa']:,} Q&A)",
        f"- Strict-valid completed rows: {sum(row['status'] == 'completed' and not row['model_validation_error'] for row in unit_rows):,}/{len(unit_rows):,}",
        f"- Scored `ok=1` rows: {sum(row['scored'] for row in unit_rows):,}/{len(unit_rows):,}",
        f"- `ok=0` rows retained for audit: {sum(row['model_ok'] == 0 for row in unit_rows):,}",
        f"- Invalid/error rows: {sum(bool(row['model_validation_error']) for row in unit_rows):,}",
        f"- Score distribution among scored rows: {dict(sorted(score_counts.items()))}",
        "",
        "## Call-Level Results",
        "",
        f"- Calls: {call_summary['calls']:,}",
        f"- Calls with scored PRE: {call_summary['calls_with_scored_pre']:,}",
        f"- Calls with scored Q&A: {call_summary['calls_with_scored_qa']:,}",
        f"- Calls with both scored segments: {call_summary['calls_with_both_scored']:,}",
        "",
        "The primary aggregation is CEO-word-weighted. Equal-unit means, medians, and common 50-200-word support are reported as sensitivities.",
        "",
        "| Measure | PRE call mean | Q&A call mean | PRE-Q&A mean | N difference |",
        "|---|---:|---:|---:|---:|",
    ]
    measure_labels = {
        "word_weighted_mean": "CEO-word-weighted",
        "equal_unit_mean": "Equal-unit mean",
        "unit_median": "Unit median",
        "common_50_200_word_weighted_mean": "Common 50-200 word weighted",
        "common_50_200_equal_unit_mean": "Common 50-200 equal-unit",
        "common_50_200_unit_median": "Common 50-200 median",
    }
    for measure, label in measure_labels.items():
        item = call_summary[measure]
        lines.append(
            f"| {label} | {fmt(item['pre_call_mean'])} | {fmt(item['qa_call_mean'])} | "
            f"{fmt(item['pre_minus_qa_call_mean'])} | {item['n_difference']} |"
        )
    lines += [
        "",
        "## Human Crosswalk",
        "",
        f"- Exact custom-ID overlap with the matched human audit: {len(human_crosswalk):,} units.",
        f"- Comparable human/model rows: {overall_human_metrics.get('n', 0):,}; model-unscored overlap rows retained separately: {overall_human_metrics.get('n_model_unscored', 0):,}.",
        "- This is a unit-level diagnostic only. It does not by itself establish production validity.",
        "",
        "| Subset | N | Exact | Within 1 | >=2 | MAE | Spearman | Linear kappa | Quadratic kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in human_metrics.items():
        lines.append(
            f"| {label} | {item['n']} | {fmt_pct(item['exact_agreement'])} | "
            f"{fmt_pct(item['within_one_agreement'])} | {fmt_pct(item['two_or_more_rate'])} | "
            f"{fmt(item['mean_absolute_error'])} | {fmt(item['spearman'])} | "
            f"{fmt(item['linear_weighted_kappa'])} | {fmt(item['quadratic_weighted_kappa'])} |"
        )
    if human_call_comparison:
        lines += [
            "",
            "## Matched Human/Model Call-Level Comparison",
            "",
            f"- Matched calls with human labels: {len(human_call_rows):,}.",
            "- Call-level values use the same CEO-word-weighted and equal-unit aggregation rules separately for each scorer.",
            "- The two model-unscored PRE units are retained in the human call-level aggregate and excluded from the model aggregate without imputation.",
            "",
            "| Measure | N calls | Model mean | Human mean | Model-Human | Mean absolute difference | Pearson |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, item in human_call_comparison.items():
            lines.append(
                f"| {label} | {item['n_calls']} | {fmt(item['model_call_mean'])} | "
                f"{fmt(item['human_call_mean'])} | {fmt(item['model_minus_human_call_mean'])} | "
                f"{fmt(item['mean_absolute_difference'])} | {fmt(item['pearson'])} |"
            )
    lines += [
        "",
        "## Interpretation and Limits",
        "",
        "- The local run is technically complete and reproducible under its frozen runtime settings; this report does not establish substantive construct validity.",
        "- The model marked two PRE observations `ok=0`, while the human coder marked both as scoreable. They remain visible in the unit table; the model-unscored rows are excluded from model means without imputation and require unit-gate adjudication.",
        "- The ten calls were selected as a turnover-focused calibration anchor, so their PRE-Q&A difference is descriptive and must not be generalized to the CCTS universe.",
        "- The human crosswalk contains exact custom-ID matches from the same 277-unit audit package. Model-unscored rows remain visible and are excluded from agreement metrics without imputation.",
        "- The next methodological gate is to review this diagnostic against the frozen human lane and decide whether the unit gate or prompt needs a new contract version before a larger local run.",
        "",
        "## Artifacts",
        "",
        f"- Unit results: `{output_dir / 'specificity_unit_results.csv'}`",
        f"- Call-level results: `{output_dir / 'specificity_call_level.csv'}`",
        f"- Human crosswalk: `{output_dir / 'specificity_human_crosswalk.csv'}`",
        f"- Human call-level results: `{output_dir / 'specificity_human_call_level.csv'}`",
        f"- Machine-readable summary: `{output_dir / 'specificity_call_level_summary.json'}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--call-manifest", type=Path, default=None)
    parser.add_argument("--human-coded", type=Path, default=None)
    parser.add_argument("--human-key", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output exists; pass --force to replace: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_rows = index_unique(read_csv_rows(args.input_manifest), "custom_id", "input manifest")
    output_rows = index_unique(read_jsonl(args.output_jsonl), "custom_id", "model output")
    if set(input_rows) != set(output_rows):
        missing = sorted(set(input_rows) - set(output_rows))
        extra = sorted(set(output_rows) - set(input_rows))
        raise SystemExit(f"input/output custom_id mismatch; missing={missing[:5]}, extra={extra[:5]}")

    unit_rows = build_unit_rows(input_rows, output_rows)
    call_metadata: dict[str, Mapping[str, str]] = {}
    if args.call_manifest:
        raw_call_rows = read_csv_rows(args.call_manifest)
        call_metadata = index_unique(raw_call_rows, "event_id", "call manifest")
        unit_events = {str(row["event_id"]) for row in unit_rows}
        missing_events = sorted(unit_events - set(call_metadata))
        if missing_events:
            raise SystemExit(f"call manifest is missing unit events: {missing_events[:5]}")
        call_metadata = {event_id: row for event_id, row in call_metadata.items() if event_id in unit_events}
    call_rows = build_call_rows(unit_rows, call_metadata)
    call_summary = aggregate_call_metrics(call_rows)

    human_crosswalk: list[dict[str, Any]] = []
    human_metrics: dict[str, Any] = {}
    human_call_rows: list[dict[str, Any]] = []
    human_call_comparison: dict[str, Any] = {}
    if (args.human_coded is None) != (args.human_key is None):
        raise SystemExit("provide both --human-coded and --human-key, or neither")
    if args.human_coded and args.human_key:
        human_coded = index_unique(read_csv_rows(args.human_coded), "audit_id", "human coded file")
        human_key = index_unique(read_csv_rows(args.human_key), "source_custom_id", "human key")
        human_crosswalk = build_human_crosswalk(unit_rows, human_coded, human_key)
        human_metrics["all exact-overlap units"] = agreement_metrics(human_crosswalk)
        for unit_type in ("pre", "qa"):
            subset = [row for row in human_crosswalk if row["unit_type"] == unit_type]
            if subset:
                human_metrics[unit_type.upper()] = agreement_metrics(subset)
        for split in ("development", "heldout"):
            subset = [row for row in human_crosswalk if row["audit_split"] == split]
            if subset:
                human_metrics[f"{split} exact-overlap units"] = agreement_metrics(subset)
        human_call_rows = build_human_call_rows(human_crosswalk, call_metadata)
        human_call_comparison = call_level_comparison(call_rows, human_call_rows)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    unit_fields = [
        "custom_id", "unit_type", "event_id", "start_date", "unit_word_count", "question_word_count",
        "source_quality_tier", "source_block_flags", "status", "model_ok", "model_specificity", "scored",
        "model_validation_error", "input_tokens", "output_tokens", "elapsed_seconds",
    ]
    write_csv(args.output_dir / "specificity_unit_results.csv", unit_rows, unit_fields)

    call_fields = list(CALL_METADATA_COLUMNS) + [
        "event_id", "start_date", "pre_scored_call_eligible",
    ]
    for unit_type in ("pre", "qa"):
        for key in summarize_units([]):
            call_fields.append(f"{unit_type}_{key}")
    for measure in (
        "word_weighted_mean", "equal_unit_mean", "unit_median",
        "common_50_200_word_weighted_mean", "common_50_200_equal_unit_mean", "common_50_200_unit_median",
    ):
        call_fields.append(f"pre_minus_qa_{measure}")
    call_fields = list(dict.fromkeys(call_fields))
    write_csv(args.output_dir / "specificity_call_level.csv", call_rows, call_fields)

    human_fields = [
        "audit_id", "custom_id", "audit_split", "unit_type", "event_id", "unit_word_count",
        "human_specificity", "model_ok", "model_specificity", "model_scored",
        "signed_model_minus_human", "absolute_difference",
    ]
    write_csv(args.output_dir / "specificity_human_crosswalk.csv", human_crosswalk, human_fields)

    human_call_fields = list(CALL_METADATA_COLUMNS) + [
        "event_id", "start_date", "human_scored_call_eligible",
    ]
    for unit_type in ("pre", "qa"):
        for key in summarize_human_units([]):
            human_call_fields.append(f"{unit_type}_{key}")
    for measure in (
        "word_weighted_mean", "equal_unit_mean", "unit_median",
        "common_50_200_word_weighted_mean", "common_50_200_equal_unit_mean", "common_50_200_unit_median",
    ):
        human_call_fields.append(f"pre_minus_qa_{measure}")
    human_call_fields = list(dict.fromkeys(human_call_fields))
    write_csv(args.output_dir / "specificity_human_call_level.csv", human_call_rows, human_call_fields)

    input_paths: dict[str, Path] = {
        "input_manifest": args.input_manifest,
        "output_jsonl": args.output_jsonl,
        "run_manifest": args.run_manifest,
    }
    if args.call_manifest:
        input_paths["call_manifest"] = args.call_manifest
    if args.human_coded:
        input_paths["human_coded"] = args.human_coded
    if args.human_key:
        input_paths["human_key"] = args.human_key

    summary = {
        "created_at_utc": timestamp,
        "status": "completed_diagnostic_aggregation",
        "scope": "ten-call turnover-focused local specificity anchor",
        "inputs": {key: str(path.resolve()) for key, path in input_paths.items()},
        "input_sha256": {key: sha256_path(path) for key, path in input_paths.items()},
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_path(Path(__file__).resolve()),
        "unit_counts": dict(Counter(row["unit_type"] for row in unit_rows)),
        "model_score_distribution_ok1": {
            str(score): sum(row["scored"] and row["model_specificity"] == score for row in unit_rows)
            for score in range(1, 6)
        },
        "unit_ok0_custom_ids": [row["custom_id"] for row in unit_rows if row["model_ok"] == 0],
        "invalid_or_error_custom_ids": [row["custom_id"] for row in unit_rows if row["model_validation_error"]],
        "call_level": call_summary,
        "human_crosswalk": {
            "exact_custom_id_overlap": len(human_crosswalk),
            "metrics": human_metrics,
            "call_level_comparison_attempted": bool(human_call_comparison),
            "matched_call_count": len(human_call_rows),
            "call_level_comparison": human_call_comparison,
        },
        "aggregation_contract": {
            "primary": "CEO-word-weighted mean of scored units within each call and segment",
            "sensitivities": [
                "equal-unit mean",
                "unit median",
                "common 50-200 word support",
            ],
            "missing_score_rule": "ok=0, invalid, or error units are retained but excluded; no imputation",
            "common_support_words": [COMMON_MIN_WORDS, COMMON_MAX_WORDS],
        },
        "artifacts": {
            "unit_results_csv": str((args.output_dir / "specificity_unit_results.csv").resolve()),
            "call_level_csv": str((args.output_dir / "specificity_call_level.csv").resolve()),
            "human_crosswalk_csv": str((args.output_dir / "specificity_human_crosswalk.csv").resolve()),
            "human_call_level_csv": str((args.output_dir / "specificity_human_call_level.csv").resolve()),
        },
    }
    summary_path = args.output_dir / "specificity_call_level_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_path = args.output_dir / "specificity_call_level_report.md"
    report_path.write_text(
        build_report(
            timestamp=timestamp,
            unit_rows=unit_rows,
            call_rows=call_rows,
            call_summary=call_summary,
            human_crosswalk=human_crosswalk,
            human_metrics=human_metrics,
            human_call_rows=human_call_rows,
            human_call_comparison=human_call_comparison,
            output_dir=args.output_dir,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
