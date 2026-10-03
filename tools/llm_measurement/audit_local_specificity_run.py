#!/usr/bin/env python3
"""Audit local specificity output and an optional exact-repeat run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Any

from specificity_validation import strict_json, validate_result, verify_provenance, reconcile_attempts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(strict_json(line))
            except ValueError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["custom_id"]: row for row in rows}
    if len(result) != len(rows):
        raise SystemExit(f"duplicate custom_id in manifest: {path}")
    return result


def validate_score(value: object) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "parsed output is not an object"
    if set(value) != {"ok", "specificity"}:
        return False, "parsed output keys are not exactly ok,specificity"
    ok = value["ok"]
    specificity = value["specificity"]
    if isinstance(ok, bool) or not isinstance(ok, int) or ok not in {0, 1}:
        return False, "ok is not integer 0 or 1"
    if isinstance(specificity, bool) or not isinstance(specificity, int) or not 0 <= specificity <= 5:
        return False, "specificity is not integer 0 through 5"
    if ok == 0 and specificity != 0:
        return False, "ok=0 must have specificity=0"
    if ok == 1 and specificity == 0:
        return False, "ok=1 must have specificity 1 through 5"
    return True, ""


def index_results(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            raise SystemExit(f"{label} has empty custom_id")
        if custom_id in result:
            raise SystemExit(f"{label} has duplicate custom_id: {custom_id}")
        result[custom_id] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--repeat-output-jsonl", type=Path, default=None)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--repeat-run-manifest", type=Path, default=None)
    parser.add_argument("--model-hash-file", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = index_results(read_jsonl(args.input_jsonl), "input")
    output_attempts = read_jsonl(args.output_jsonl)
    repeat_attempts = read_jsonl(args.repeat_output_jsonl) if args.repeat_output_jsonl else None
    try:
        outputs = reconcile_attempts(output_attempts)
        repeats = reconcile_attempts(repeat_attempts) if repeat_attempts is not None else None
    except ValueError as exc:
        raise SystemExit(f"invalid attempt history: {exc}") from exc
    manifest = read_csv(args.manifest_csv)
    expected_ids = set(inputs)
    if not inputs:
        raise SystemExit("empty input cannot count as a passed audit")
    if set(manifest) != expected_ids:
        raise SystemExit("manifest custom_id set does not match input")
    try:
        provenance = verify_provenance(args.input_jsonl, args.output_jsonl, args.run_manifest,
                                       args.manifest_csv, list(inputs.values()), output_attempts)
        repeat_provenance = (verify_provenance(args.input_jsonl, args.repeat_output_jsonl,
                             args.repeat_run_manifest, args.manifest_csv,
                             list(inputs.values()), repeat_attempts) if repeats is not None else None)
        if repeat_provenance and repeat_provenance["run_binding_sha256"] != provenance["run_binding_sha256"]:
            raise ValueError("repeat run execution identity differs from original run")
    except (ValueError, OSError) as exc:
        raise SystemExit(f"provenance rejected: {exc}") from exc
    coverage = {"missing_output_ids": sorted(expected_ids - outputs.keys()),
                "unexpected_output_ids": sorted(outputs.keys() - expected_ids),
                "missing_repeat_ids": sorted(expected_ids - repeats.keys()) if repeats is not None else [],
                "unexpected_repeat_ids": sorted(repeats.keys() - expected_ids) if repeats is not None else []}

    audit_rows: list[dict[str, object]] = []
    strict_valid = 0
    row_errors = 0
    repeat_strict_valid = None if repeats is None else 0
    repeat_row_errors = None if repeats is None else 0
    repeat_valid_pairs = None if repeats is None else 0
    repeat_raw_pairs = None if repeats is None else 0
    repeat_score_pairs = None if repeats is None else 0
    repeat_exact_output = None if repeats is None else 0
    repeat_exact_score = None if repeats is None else 0
    for custom_id in inputs:
        original = outputs.get(custom_id)
        output = original or {}
        repeat = repeats.get(custom_id) if repeats is not None else None
        parsed = output.get("parsed")
        output_valid, validation_error = validate_result(original)
        strict_valid += int(output_valid)
        row_errors += int(output.get("status") == "error")
        repeat_valid, repeat_error = validate_result(repeat)
        comparable = output_valid and repeat_valid
        raw_comparable = (comparable and isinstance(output.get("raw_output"), str)
                          and isinstance(repeat.get("raw_output"), str))
        score_comparable = comparable and parsed["ok"] == 1 and repeat["parsed"]["ok"] == 1
        same_raw = raw_comparable and output["raw_output"] == repeat["raw_output"]
        same_score = score_comparable and parsed["specificity"] == repeat["parsed"]["specificity"]
        if repeats is not None:
            repeat_strict_valid += int(repeat_valid)
            repeat_row_errors += int(repeat is not None and repeat.get("status") == "error")
            repeat_valid_pairs += int(comparable)
            repeat_raw_pairs += int(raw_comparable)
            repeat_score_pairs += int(score_comparable)
            repeat_exact_output += int(same_raw)
            repeat_exact_score += int(same_score)
        source = manifest[custom_id]
        audit_rows.append({
            "custom_id": custom_id,
            "unit_type": source.get("unit_type", ""),
            "event_id": source.get("event_id", ""),
            "start_date": source.get("start_date", ""),
            "unit_word_count": source.get("unit_word_count", ""),
            "question_word_count": source.get("question_word_count", ""),
            "status": output.get("status", "missing"),
            "final_attempt_number": output.get("attempt_number", 1) if original else "",
            "retry_reason": output.get("retry_reason", ""),
            "technical_valid": int(output_valid),
            "ok": parsed.get("ok") if isinstance(parsed, dict) else "",
            "specificity": parsed.get("specificity") if isinstance(parsed, dict) else "",
            "validation_error": output.get("validation_error") or validation_error,
            "input_tokens": output.get("input_tokens", ""),
            "output_tokens": output.get("output_tokens", ""),
            "elapsed_seconds": output.get("elapsed_seconds", ""),
            "repeat_status": repeat.get("status", "") if repeat else "missing" if repeats is not None else "",
            "repeat_technical_valid": int(repeat_valid) if repeats is not None else "",
            "repeat_validation_error": repeat_error if repeats is not None else "",
            "repeat_ok": repeat["parsed"]["ok"] if repeat_valid else "",
            "repeat_valid_pair": int(comparable) if repeats is not None else "",
            "repeat_raw_comparable": int(raw_comparable) if repeats is not None else "",
            "repeat_score_comparable": int(score_comparable) if repeats is not None else "",
            "repeat_exact_output": int(same_raw) if raw_comparable else "",
            "repeat_exact_score": int(same_score) if score_comparable else "",
        })

    ok0_rows = [row for row in audit_rows if row["technical_valid"] and row["ok"] == 0]
    repeat_ok0_ids = [row["custom_id"] for row in audit_rows if row["repeat_ok"] == 0]
    score_distribution = Counter(
        int(row["specificity"])
        for row in audit_rows
        if row["technical_valid"] and row["ok"] == 1
    )
    type_counts = Counter(row["unit_type"] for row in audit_rows)
    valid_by_type = Counter(
        row["unit_type"] for row in audit_rows if row["technical_valid"] and row["ok"] == 1
    )
    input_tokens = sum(int(row["input_tokens"]) for row in audit_rows if row["input_tokens"] not in {"", None})
    output_tokens = sum(int(row["output_tokens"]) for row in audit_rows if row["output_tokens"] not in {"", None})

    fields = list(audit_rows[0]) if audit_rows else []
    audit_csv = args.output_dir / "dry_run_audit.csv"
    with audit_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    technical_pass = (strict_valid == len(inputs) and not any(coverage.values())
                      and (repeats is None or repeat_strict_valid == len(inputs)))
    status = ("failed_technical_validation" if not technical_pass else
              "passed_with_manual_edge_case" if ok0_rows or repeat_ok0_ids else "passed")
    report_lines = [
        "# Local Specificity Run Audit",
        "",
        f"- Audit timestamp UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Status: **{status}**",
        f"- Requested units: {len(inputs):,}",
        f"- Strict-valid outputs: {strict_valid:,}/{len(inputs):,}",
        f"- Row errors: {row_errors:,}",
        f"- Missing output IDs: {len(coverage['missing_output_ids']):,}",
        f"- Unexpected output IDs: {len(coverage['unexpected_output_ids']):,}",
        f"- Input tokens: {input_tokens:,}",
        f"- Output tokens: {output_tokens:,}",
        "",
        "## Unit Counts",
        "",
        *(f"- {unit_type}: {count:,} total; {valid_by_type[unit_type]:,} scored `ok=1`" for unit_type, count in sorted(type_counts.items())),
        "",
        "## Score Distribution",
        "",
        *(f"- specificity {score}: {score_distribution[score]:,}" for score in range(1, 6)),
        "",
    ]
    if repeats is not None:
        report_lines.extend([
            "## Repeat Diagnostics", "",
            f"- Requested pairs: {len(inputs):,}; technically valid pairs: {repeat_valid_pairs:,}.",
            f"- Repeat strict-valid outputs: {repeat_strict_valid:,}/{len(inputs):,}; row errors: {repeat_row_errors:,}.",
            f"- Missing repeat IDs: {len(coverage['missing_repeat_ids']):,}; unexpected repeat IDs: {len(coverage['unexpected_repeat_ids']):,}.",
            f"- Exact raw output: {repeat_exact_output:,}/{repeat_raw_pairs:,} comparable pairs; {len(inputs) - repeat_raw_pairs:,} excluded.",
            f"- Exact numerical score: {repeat_exact_score:,}/{repeat_score_pairs:,} comparable ok=1 pairs; {len(inputs) - repeat_score_pairs:,} excluded.",
            "- Failed/missing pairs are excluded, not counted as agreements; valid abstentions are not numerical scores.",
            "",
        ])
    report_lines.extend(["## Manual Edge-Case Queue", ""])
    if ok0_rows:
        report_lines.extend(
            f"- `{row['custom_id']}` ({row['unit_type']}, event {row['event_id']}, {row['unit_word_count']} words): model returned `ok=0, specificity=0`; inspect the exact input in the request JSONL before production scaling."
            for row in ok0_rows
        )
    else:
        report_lines.append("- None.")
    if repeat_ok0_ids:
        report_lines.append("- Repeat-run valid abstentions: " + ", ".join(f"`{key}`" for key in repeat_ok0_ids) + ".")
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- Technical pass requires all expected IDs and schema-valid completed outputs in both supplied runs; it does not certify substantive scoring quality or satisfy a scientific repeatability threshold.",
        "- Whole-raw JSON, token/EOS evidence, parsed equality, schema and exact input/output/unit-metadata run bindings were independently checked. Incompatible historical artifacts are refused rather than upgraded.",
        "- `ok=0` is retained as a manual edge case. It is not silently converted to a 1-5 score or dropped from the source package.",
        "- This audit contains no transcript text; the exact model inputs remain in the local request JSONL.",
    ])
    report_path = args.output_dir / "dry_run_audit_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    comparison = {
        "status": status,
        "provenance": provenance,
        "repeat_provenance": repeat_provenance,
        "input_requests": len(inputs),
        "strict_valid_outputs": strict_valid,
        "row_errors": row_errors,
        "coverage": coverage,
        "invalid_output_ids": [row["custom_id"] for row in audit_rows if not row["technical_valid"]],
        "repeat_invalid_output_ids": [row["custom_id"] for row in audit_rows
                                      if repeats is not None and not row["repeat_technical_valid"]],
        "repeat_strict_valid_outputs": repeat_strict_valid,
        "repeat_row_errors": repeat_row_errors,
        "repeat_valid_pairs": repeat_valid_pairs,
        "repeat_raw_comparable_pairs": repeat_raw_pairs,
        "repeat_score_comparable_pairs": repeat_score_pairs,
        "repeat_excluded_pairs": len(inputs) - repeat_valid_pairs if repeats is not None else None,
        "repeat_raw_excluded_pairs": len(inputs) - repeat_raw_pairs if repeats is not None else None,
        "repeat_score_excluded_pairs": len(inputs) - repeat_score_pairs if repeats is not None else None,
        "repeat_exact_raw_outputs": repeat_exact_output,
        "repeat_exact_parsed_scores": repeat_exact_score,
        "unit_counts": dict(type_counts),
        "score_distribution_ok1": {str(score): score_distribution[score] for score in range(1, 6)},
        "ok0_custom_ids": [row["custom_id"] for row in ok0_rows],
        "repeat_ok0_custom_ids": repeat_ok0_ids,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "artifacts": {
            "audit_csv": str(audit_csv),
            "report_md": str(report_path),
            "input_jsonl": str(args.input_jsonl.resolve()),
            "output_jsonl": str(args.output_jsonl.resolve()),
            "repeat_output_jsonl": str(args.repeat_output_jsonl.resolve()) if args.repeat_output_jsonl else None,
            "run_manifest": str(args.run_manifest.resolve()),
            "repeat_run_manifest": str(args.repeat_run_manifest.resolve()) if args.repeat_run_manifest else None,
            "model_hash_file": str(args.model_hash_file.resolve()) if args.model_hash_file else None,
        },
    }
    comparison_path = args.output_dir / "dry_run_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2))
    return 0 if technical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
