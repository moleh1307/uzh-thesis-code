#!/usr/bin/env python3
"""Build a text-free CCTS event master table for thesis planning.

Input is the exact event-level PRE/Q&A census. Output is one row per earnings
call with metadata, identifier coverage, raw section eligibility, and a
preliminary firm-matching status. It does not export transcript full text.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv
from artifact_publication import fresh_artifact_directory

configure_csv()
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set


DEFAULT_CENSUS = Path(
    "data/02_ccts_database/01_event_level_census/latest/event_level_section_census.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "data/02_ccts_database/04_event_master"
)

SECURITY_IDENTIFIER_FIELDS = ["isin", "cusip", "sedol"]
EXTERNAL_IDENTIFIER_FIELDS = ["isin", "cusip", "sedol", "company_ric", "company_ticker"]
ALL_IDENTIFIER_FIELDS = ["company_id", *EXTERNAL_IDENTIFIER_FIELDS]

BASE_COLUMNS = [
    "event_id",
    "start_date",
    "year",
    "event_type_name",
    "event_title",
    "title_company_guess",
    "company_id",
    "company_name",
    "company_ticker",
    "isin",
    "cusip",
    "sedol",
    "company_ric",
    "file_name",
    "event_story_id",
    "story_type",
    "version",
    "total_text_rows",
    "pre_rows",
    "qa_rows",
    "other_text_rows",
    "distinct_text_types",
    "text_types",
    "has_text",
    "has_pre",
    "has_qa",
    "has_both_pre_and_qa",
    "pre_only",
    "qa_only",
    "no_text_rows",
    "participant_rows",
]

OPTIONAL_SIGNAL_COLUMNS = [
    "possible_qa_dialogue_encoded_as_pre",
    "pre_text_name_ceo_signal_rows",
    "qa_text_name_ceo_signal_rows",
    "pre_text_name_cfo_signal_rows",
    "qa_text_name_cfo_signal_rows",
    "pre_text_name_analyst_signal_rows",
    "qa_text_name_analyst_signal_rows",
    "pre_text_name_operator_signal_rows",
    "qa_text_name_operator_signal_rows",
    "participant_type_rows",
    "participant_name_rows",
    "participant_ceo_signal_rows",
    "participant_cfo_signal_rows",
    "participant_analyst_signal_rows",
    "participant_operator_signal_rows",
]

DERIVED_COLUMNS = [
    "section_input_status",
    "presentation_qa_difference_status",
    "text_export_status",
    "section_repair_needed_before_difference",
    "section_repair_priority",
    "company_name_tokens",
    "title_token_matches",
    "title_token_match_ratio",
    "company_tokens_not_in_event_title",
    "external_identifier_count",
    "security_identifier_count",
    "has_security_identifier",
    "has_external_identifier",
    "has_internal_company_id",
    "available_identifiers",
    "firm_match_status_prelim",
    "firm_match_risk_prelim",
    "recommended_match_key",
    "review_reason",
    "manager_layer_status",
]

COMMON_COMPANY_TOKENS = {
    "a",
    "ab",
    "ag",
    "and",
    "asa",
    "bhd",
    "bm",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "group",
    "holdings",
    "inc",
    "incorporated",
    "kk",
    "limited",
    "llc",
    "ltd",
    "nv",
    "oyj",
    "plc",
    "sa",
    "se",
    "spa",
    "the",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def safe_int(value: Any) -> int:
    value = clean_text(value)
    if not value:
        return 0
    return int(float(value))


def normalize_token_text(value: str) -> str:
    value = html.unescape(value or "").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return clean_text(value)


def tokenize_company(value: str) -> List[str]:
    tokens = []
    for token in normalize_token_text(value).split():
        if len(token) <= 1:
            continue
        if token in COMMON_COMPANY_TOKENS:
            continue
        tokens.append(token)
    return tokens


def token_set(value: str) -> Set[str]:
    return set(tokenize_company(value))


def ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return ""
    return f"{numerator / denominator:.3f}"


def estimate_title_company_phrase(title: str) -> str:
    value = html.unescape(clean_text(title))
    if not value:
        return ""
    value = re.sub(r"^\s*[-–—:\t]+", "", value)
    value = re.sub(r"^\s*(Q[1-4]|H[12]|FY)\s+\d{4}\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*(first|second|third|fourth)\s+quarter\s+\d{4}\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*(half year|full year|interim|annual)\s+\d{4}\s+", "", value, flags=re.I)
    value = re.sub(r"^\s*\d{4}\s+(first|second|third|fourth)\s+quarter\s+", "", value, flags=re.I)
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value)
    endings = [
        r"earnings conference call",
        r"international earnings conference call",
        r"earnings call and analyst meeting",
        r"earnings call",
        r"earnings presentation and webcast",
        r"earnings presentation & webcast",
        r"earnings presentation",
        r"earnings press conference",
        r"results briefing",
        r"analyst meeting",
        r"conference call",
        r"presentation and webcast",
        r"presentation",
        r"webcast",
    ]
    for ending in endings:
        value = re.sub(rf"\s+{ending}\s*$", "", value, flags=re.I)
    value = re.sub(r"[’']s\s+fiscal\s+year\s+\d{4}.*$", "", value, flags=re.I)
    value = re.sub(r"\s+fiscal\s+year\s+\d{4}.*$", "", value, flags=re.I)
    value = re.sub(r"\s+(first|second|third|fourth)\s+quarter\s+\d{4}\s*$", "", value, flags=re.I)
    value = re.sub(r"\s+q[1-4]\s+\d{4}\s*$", "", value, flags=re.I)
    value = re.sub(r"\s+earnings\s+results\s*$", "", value, flags=re.I)
    value = re.sub(r"\s+[-–—]\s+.*$", "", value)
    return clean_text(value.strip(" ,.-:;"))


def available_identifiers(row: Mapping[str, str]) -> List[str]:
    return [field for field in ALL_IDENTIFIER_FIELDS if clean_text(row.get(field, ""))]


def section_status(row: Mapping[str, str]) -> Dict[str, str]:
    has_both = safe_int(row.get("has_both_pre_and_qa")) == 1
    pre_only = safe_int(row.get("pre_only")) == 1
    qa_only = safe_int(row.get("qa_only")) == 1
    no_text = safe_int(row.get("no_text_rows")) == 1
    pre_rows = safe_int(row.get("pre_rows"))
    qa_rows = safe_int(row.get("qa_rows"))
    total_rows = safe_int(row.get("total_text_rows"))

    if has_both:
        return {
            "section_input_status": "raw_both_pre_and_qa",
            "presentation_qa_difference_status": "candidate_raw_sections",
            "text_export_status": "candidate_text_export",
            "section_repair_needed_before_difference": "0",
            "section_repair_priority": "none",
        }
    if pre_only:
        priority = "high" if pre_rows >= 3 else "medium"
        return {
            "section_input_status": "raw_pre_only",
            "presentation_qa_difference_status": "needs_hidden_qna_repair_probe",
            "text_export_status": "candidate_after_repair_if_qna_found",
            "section_repair_needed_before_difference": "1",
            "section_repair_priority": priority,
        }
    if qa_only:
        return {
            "section_input_status": "raw_qa_only",
            "presentation_qa_difference_status": "exclude_no_presentation",
            "text_export_status": "text_only_no_difference",
            "section_repair_needed_before_difference": "0",
            "section_repair_priority": "none",
        }
    if no_text or total_rows == 0:
        return {
            "section_input_status": "no_text",
            "presentation_qa_difference_status": "exclude_no_text",
            "text_export_status": "exclude_no_text",
            "section_repair_needed_before_difference": "0",
            "section_repair_priority": "none",
        }
    return {
        "section_input_status": "other_or_unusual_text_type",
        "presentation_qa_difference_status": "review_section_type",
        "text_export_status": "review_before_text_export",
        "section_repair_needed_before_difference": "1" if qa_rows == 0 else "0",
        "section_repair_priority": "medium",
    }


def firm_status(row: Mapping[str, str]) -> Dict[str, str]:
    tokens = tokenize_company(row.get("company_name", ""))
    title_tokens = token_set(row.get("event_title", ""))
    matches = sorted(set(tokens) & title_tokens)
    token_mismatch = bool(tokens and not matches)

    security_count = sum(1 for field in SECURITY_IDENTIFIER_FIELDS if clean_text(row.get(field, "")))
    external_count = sum(1 for field in EXTERNAL_IDENTIFIER_FIELDS if clean_text(row.get(field, "")))
    has_security = security_count > 0
    has_external = external_count > 0
    has_internal = bool(clean_text(row.get("company_id", "")))

    if has_security:
        key = "security_identifier_plus_event_date"
    elif clean_text(row.get("company_ric", "")):
        key = "company_ric_plus_event_date"
    elif clean_text(row.get("company_ticker", "")):
        key = "ticker_plus_event_date_plus_historical_name_review"
    elif has_internal:
        key = "ccts_company_id_bridge_required"
    else:
        key = "manual_name_date_review_required"

    if token_mismatch:
        if has_security:
            status = "review_title_name_conflict_identifier_backed"
            risk = "medium"
        elif has_external:
            status = "review_title_name_conflict_external_id_limited"
            risk = "medium_high"
        elif has_internal:
            status = "high_risk_title_name_conflict_internal_id_only"
            risk = "high"
        else:
            status = "high_risk_title_name_conflict_no_identifier"
            risk = "high"
    else:
        if has_security:
            status = "ok_identifier_backed"
            risk = "low"
        elif has_external:
            status = "ok_external_id_limited"
            risk = "medium_low"
        elif has_internal:
            status = "review_internal_id_only"
            risk = "medium_high"
        else:
            status = "review_missing_identifier"
            risk = "high"

    reasons = []
    if token_mismatch:
        reasons.append("company_tokens_not_in_event_title")
    if not has_security:
        reasons.append("no_isin_cusip_sedol")
    if not has_external:
        reasons.append("no_external_identifier")

    return {
        "company_name_tokens": ";".join(tokens),
        "title_token_matches": ";".join(matches),
        "title_token_match_ratio": ratio(len(matches), len(set(tokens))),
        "company_tokens_not_in_event_title": "1" if token_mismatch else "0",
        "external_identifier_count": str(external_count),
        "security_identifier_count": str(security_count),
        "has_security_identifier": "1" if has_security else "0",
        "has_external_identifier": "1" if has_external else "0",
        "has_internal_company_id": "1" if has_internal else "0",
        "available_identifiers": ";".join(available_identifiers(row)),
        "firm_match_status_prelim": status,
        "firm_match_risk_prelim": risk,
        "recommended_match_key": key,
        "review_reason": ";".join(reasons),
    }


def manager_status(row: Mapping[str, str], input_columns: Set[str]) -> str:
    if "participant_ceo_signal_rows" in input_columns:
        ceo = safe_int(row.get("participant_ceo_signal_rows"))
        if ceo > 0:
            return "participant_ceo_signal_available"
        if safe_int(row.get("participant_rows")) > 0:
            return "participants_available_no_ceo_signal"
        return "no_participant_rows"
    if safe_int(row.get("participant_rows")) > 0:
        return "participant_rows_only_role_signals_not_computed"
    return "no_participant_rows"


def iter_master_rows(census_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with census_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit(f"Missing header in {census_path}")
        input_columns = set(reader.fieldnames)
        optional = [column for column in OPTIONAL_SIGNAL_COLUMNS if column in input_columns]
        output_columns = [*BASE_COLUMNS, *optional, *DERIVED_COLUMNS]
        rows: List[Dict[str, str]] = []
        for row in reader:
            out = {column: clean_text(row.get(column, "")) for column in BASE_COLUMNS}
            for column in optional:
                out[column] = clean_text(row.get(column, ""))
            out["title_company_guess"] = estimate_title_company_phrase(row.get("event_title", ""))
            out.update(section_status(row))
            out.update(firm_status(row))
            out["manager_layer_status"] = manager_status(row, input_columns)
            rows.append(out)
    return rows, output_columns


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_by_year(path: Path, rows: Sequence[Mapping[str, str]]) -> int:
    buckets: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        year = row["year"]
        buckets[year]["events"] += 1
        buckets[year][row["presentation_qa_difference_status"]] += 1
        buckets[year][row["firm_match_risk_prelim"]] += 1
        buckets[year][row["manager_layer_status"]] += 1

    columns = [
        "year",
        "events",
        "candidate_raw_sections",
        "needs_hidden_qna_repair_probe",
        "exclude_no_presentation",
        "exclude_no_text",
        "firm_low_risk",
        "firm_medium_low_risk",
        "firm_medium_risk",
        "firm_medium_high_risk",
        "firm_high_risk",
        "participant_rows_only_role_signals_not_computed",
        "participant_ceo_signal_available",
        "participants_available_no_ceo_signal",
        "no_participant_rows",
    ]
    out_rows = []
    for year in sorted(buckets, key=lambda value: int(value)):
        bucket = buckets[year]
        out_rows.append(
            {
                "year": year,
                "events": bucket["events"],
                "candidate_raw_sections": bucket["candidate_raw_sections"],
                "needs_hidden_qna_repair_probe": bucket["needs_hidden_qna_repair_probe"],
                "exclude_no_presentation": bucket["exclude_no_presentation"],
                "exclude_no_text": bucket["exclude_no_text"],
                "firm_low_risk": bucket["low"],
                "firm_medium_low_risk": bucket["medium_low"],
                "firm_medium_risk": bucket["medium"],
                "firm_medium_high_risk": bucket["medium_high"],
                "firm_high_risk": bucket["high"],
                "participant_rows_only_role_signals_not_computed": bucket[
                    "participant_rows_only_role_signals_not_computed"
                ],
                "participant_ceo_signal_available": bucket["participant_ceo_signal_available"],
                "participants_available_no_ceo_signal": bucket["participants_available_no_ceo_signal"],
                "no_participant_rows": bucket["no_participant_rows"],
            }
        )
    return write_csv(path, out_rows, columns)


def make_latest_symlink(target: Path, latest: Path) -> None:
    target = target.resolve(strict=True)
    if not target.is_dir():
        raise ValueError(f"latest target is not a directory: {target}")
    if latest.exists() and not latest.is_symlink():
        raise SystemExit(f"Refusing to replace non-symlink path: {latest}")
    latest.parent.mkdir(parents=True, exist_ok=True)
    temporary = latest.with_name(f".{latest.name}-{uuid.uuid4().hex}.tmp")
    try:
        temporary.symlink_to(target, target_is_directory=True)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)


def summary_text(payload: Mapping[str, Any], output_dir: Path) -> str:
    status_counts = payload["presentation_qa_difference_status_counts"]
    firm_counts = payload["firm_match_status_prelim_counts"]
    risk_counts = payload["firm_match_risk_prelim_counts"]
    manager_counts = payload["manager_layer_status_counts"]
    return "\n".join(
        [
            "# CCTS Text-Free Event Master",
            "",
            f"Created UTC: `{payload['created_utc']}`",
            "",
            "## Scope",
            "",
            f"- Census input: `{payload['census_input']}`",
            f"- Events: {payload['events']:,}",
            "- Transcript full text exported: no",
            f"- Master CSV: `{output_dir / 'ccts_event_master.csv'}`",
            f"- By-year CSV: `{output_dir / 'ccts_event_master_by_year.csv'}`",
            "",
            "## PRE/Q&A Difference Eligibility",
            "",
            f"- Raw PRE+Q&A candidates: {status_counts.get('candidate_raw_sections', 0):,}",
            f"- PRE-only events needing hidden-Q&A repair probe: {status_counts.get('needs_hidden_qna_repair_probe', 0):,}",
            f"- Q&A-only events excluded from difference design: {status_counts.get('exclude_no_presentation', 0):,}",
            f"- No-text events excluded: {status_counts.get('exclude_no_text', 0):,}",
            "",
            "## Firm Matching",
            "",
            f"- Has ISIN/CUSIP/SEDOL: {payload['has_security_identifier']:,}",
            f"- Has any external identifier/ticker: {payload['has_external_identifier']:,}",
            f"- Has CCTS company_id: {payload['has_internal_company_id']:,}",
            "",
            "Firm status counts:",
            *[f"- `{key}`: {value:,}" for key, value in sorted(firm_counts.items())],
            "",
            "Firm risk counts:",
            *[f"- `{key}`: {value:,}" for key, value in sorted(risk_counts.items())],
            "",
            "## Manager Layer",
            "",
            *[f"- `{key}`: {value:,}" for key, value in sorted(manager_counts.items())],
            "",
            "## Interpretation",
            "",
            "- This is a planning/master table, not the final measurement table.",
            "- Use this to decide the full export scope before moving text.",
            "- `firm_match_status_prelim` is title-only and text-free; after text extraction, opening-text checks can refine it.",
            "- `participant_rows_only_role_signals_not_computed` means the current census was section-only. For full manager signals, run the optional DB command below.",
            "",
            "## Optional Long DB Command",
            "",
            "To compute full participant/text-name CEO/CFO/analyst/operator signal counts without exporting transcript text, run:",
            "",
            "```bash",
            "python3 ./tools/ccts_database/build_ccts_event_level_census.py \\",
            "  --output-dir data/02_ccts_database/01_event_level_census/v1_full_signals_manual_run \\",
            "  --batch-size 500 \\",
            "  --statement-timeout-ms 300000",
            "```",
            "",
            "That command can take much longer than this local master build and requires UZH VPN/database access.",
            "",
            "## Next Recommended Step",
            "",
            "If this master table looks right, promote the event extractor to a full text export for raw PRE+Q&A candidates plus a targeted PRE-only repair pass.",
            "",
        ]
    )


def write_master_package(census: Path, output_dir: Path, stage: Path, created: str) -> Dict[str, Any]:
    rows, columns = iter_master_rows(census)
    master_csv = output_dir / "ccts_event_master.csv"
    by_year_csv = output_dir / "ccts_event_master_by_year.csv"
    summary_json = output_dir / "ccts_event_master_summary.json"
    summary_md = output_dir / "ccts_event_master_summary.md"

    write_csv(stage / master_csv.name, rows, columns)
    write_by_year(stage / by_year_csv.name, rows)

    payload: Dict[str, Any] = {
        "created_utc": created,
        "census_input": str(census),
        "output_dir": str(output_dir),
        "events": len(rows),
        "presentation_qa_difference_status_counts": dict(
            Counter(row["presentation_qa_difference_status"] for row in rows)
        ),
        "section_input_status_counts": dict(Counter(row["section_input_status"] for row in rows)),
        "firm_match_status_prelim_counts": dict(
            Counter(row["firm_match_status_prelim"] for row in rows)
        ),
        "firm_match_risk_prelim_counts": dict(
            Counter(row["firm_match_risk_prelim"] for row in rows)
        ),
        "manager_layer_status_counts": dict(Counter(row["manager_layer_status"] for row in rows)),
        "has_security_identifier": sum(row["has_security_identifier"] == "1" for row in rows),
        "has_external_identifier": sum(row["has_external_identifier"] == "1" for row in rows),
        "has_internal_company_id": sum(row["has_internal_company_id"] == "1" for row in rows),
        "artifacts": {
            "master_csv": str(master_csv),
            "by_year_csv": str(by_year_csv),
            "summary_json": str(summary_json),
            "summary_md": str(summary_md),
        },
    }
    summary_json = stage / summary_json.name
    summary_md = stage / summary_md.name
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text(summary_text(payload, output_dir), encoding="utf-8")
    return payload


def build_master(census: Path, output_root: Path, run_name: str | None = None) -> Dict[str, Any]:
    if not census.is_file():
        raise SystemExit(f"Missing census input: {census}")
    created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = run_name or f"v1_{created}"
    if Path(run_name).name != run_name or run_name in (".", "..", "latest"):
        raise ValueError("run name must be one new directory name, not latest")
    output_root = output_root.resolve()
    output_dir = output_root / run_name
    latest = output_root / "latest"
    if latest.exists() and not latest.is_symlink():
        raise SystemExit(f"Refusing to replace non-symlink path: {latest}")
    with fresh_artifact_directory(output_dir) as stage:
        payload = write_master_package(census, output_dir, stage, created)
    make_latest_symlink(output_dir, latest)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_master(args.census, args.output_root, args.run_name)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
