#!/usr/bin/env python3
"""Build an event-level CCTS to ExecuComp CEO-tenure coverage panel.

The bridge is deliberately conservative: CCTS and ExecuComp issuers are joined
only through normalized CUSIP8. CEO coverage is then evaluated at the CCTS
event date. CCTS company_id and company names are retained for audit but never
used as issuer join keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_FIELDS = [
    "event_id", "event_date", "year", "event_title", "ccts_company_id",
    "ccts_company_name", "ccts_ticker", "ccts_isin", "ccts_cusip",
    "ccts_cusip8", "has_both_pre_and_qa", "ccts_firm_match_status_prelim",
    "ccts_firm_match_risk_prelim", "issuer_match_status", "gvkey",
    "execucomp_company_name", "ceo_coverage_status", "ceo_candidate_count",
    "execid", "tenure_episode", "ceo_name", "ceo_start_date", "ceo_end_date",
    "tenure_date_quality", "start_date_source", "end_date_source",
    "normalization_status", "match_confidence", "review_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccts-census", type=Path, required=True)
    parser.add_argument("--execucomp-tenures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-pre-qa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only events with both raw PRE and Q&A (default: true).",
    )
    parser.add_argument(
        "--firm-match-status",
        action="append",
        default=[],
        help="Keep only this CCTS firm_match_status_prelim; may be repeated.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def cusip8(value: object) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", clean(value).upper())
    return normalized[:8] if len(normalized) >= 8 else ""


def event_date(value: str) -> str:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", clean(value))
    return match.group(1) if match else ""


def true_value(value: object) -> bool:
    return clean(value).lower() in {"1", "true", "t", "yes", "y"}


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "ccts_execucomp_event_coverage.csv"
    if panel_path.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace: {panel_path}")

    tenure_by_cusip: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.execucomp_tenures.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = cusip8(row.get("cusip"))
            if key:
                tenure_by_cusip[key].append(row)

    panel: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    year_counts: dict[str, Counter[str]] = defaultdict(Counter)
    issuer_gvkeys: dict[str, set[str]] = defaultdict(set)

    with args.ccts_census.open(newline="", encoding="utf-8-sig") as handle:
        for source in csv.DictReader(handle):
            has_pre_qa = true_value(source.get("has_both_pre_and_qa"))
            if args.require_pre_qa and not has_pre_qa:
                continue
            firm_status = clean(source.get("firm_match_status_prelim"))
            if args.firm_match_status and firm_status not in set(args.firm_match_status):
                continue

            date = event_date(source.get("start_date", ""))
            year = date[:4]
            key = cusip8(source.get("cusip"))
            candidates = tenure_by_cusip.get(key, []) if key else []
            if not key:
                issuer_status = "no_ccts_cusip"
            elif not candidates:
                issuer_status = "no_execucomp_cusip8_match"
            else:
                gvkeys = {clean(row.get("gvkey")) for row in candidates}
                issuer_status = "exact_cusip8_single_gvkey" if len(gvkeys) == 1 else "ambiguous_cusip8_multiple_gvkeys"
                issuer_gvkeys[key].update(gvkeys)

            covering = [
                row for row in candidates
                if date and clean(row.get("ceo_start_date")) <= date <= clean(row.get("ceo_end_date"))
            ]
            if not candidates:
                coverage_status = "not_testable_no_issuer_match"
            elif not date:
                coverage_status = "review_missing_event_date"
            elif not covering:
                coverage_status = "issuer_match_no_ceo_date_coverage"
            elif len(covering) > 1:
                coverage_status = "review_multiple_ceos_cover_event"
            else:
                quality = clean(covering[0].get("normalization_status"))
                coverage_status = (
                    "covered_exact_ceo_tenure"
                    if quality == "exact_tenure_dates"
                    else "covered_review_ceo_tenure"
                )

            chosen = covering[0] if len(covering) == 1 else {}
            quality = clean(chosen.get("normalization_status"))
            if coverage_status == "covered_exact_ceo_tenure":
                confidence = "high"
                reason = "exact CUSIP8 issuer match and exact CEO tenure covers event date"
            elif coverage_status == "covered_review_ceo_tenure":
                confidence = "medium"
                reason = "exact CUSIP8 issuer match; CEO tenure includes annual-bound or source-date review"
            elif coverage_status == "review_multiple_ceos_cover_event":
                confidence = "review"
                reason = "multiple normalized CEO episodes cover the same event date"
            else:
                confidence = "unmatched"
                reason = coverage_status.replace("_", " ")

            out = {
                "event_id": clean(source.get("event_id")),
                "event_date": date,
                "year": year,
                "event_title": clean(source.get("event_title")),
                "ccts_company_id": clean(source.get("company_id")),
                "ccts_company_name": clean(source.get("company_name")),
                "ccts_ticker": clean(source.get("company_ticker")),
                "ccts_isin": clean(source.get("isin")),
                "ccts_cusip": clean(source.get("cusip")),
                "ccts_cusip8": key,
                "has_both_pre_and_qa": "1" if has_pre_qa else "0",
                "ccts_firm_match_status_prelim": firm_status,
                "ccts_firm_match_risk_prelim": clean(source.get("firm_match_risk_prelim")),
                "issuer_match_status": issuer_status,
                "gvkey": clean(chosen.get("gvkey")) or (clean(candidates[0].get("gvkey")) if candidates and issuer_status == "exact_cusip8_single_gvkey" else ""),
                "execucomp_company_name": clean(chosen.get("company_name")) or (clean(candidates[0].get("company_name")) if candidates else ""),
                "ceo_coverage_status": coverage_status,
                "ceo_candidate_count": str(len(covering)),
                "execid": clean(chosen.get("execid")),
                "tenure_episode": clean(chosen.get("tenure_episode")),
                "ceo_name": clean(chosen.get("person_name")),
                "ceo_start_date": clean(chosen.get("ceo_start_date")),
                "ceo_end_date": clean(chosen.get("ceo_end_date")),
                "tenure_date_quality": quality,
                "start_date_source": clean(chosen.get("start_date_source")),
                "end_date_source": clean(chosen.get("end_date_source")),
                "normalization_status": quality,
                "match_confidence": confidence,
                "review_reason": reason,
            }
            panel.append(out)
            status_counts[coverage_status] += 1
            year_counts[year]["events"] += 1
            year_counts[year][coverage_status] += 1

    by_year = []
    coverage_keys = sorted(status_counts)
    for year in sorted(year_counts):
        total = year_counts[year]["events"]
        row = {"year": year, "events": str(total)}
        for key in coverage_keys:
            row[key] = str(year_counts[year][key])
        covered = year_counts[year]["covered_exact_ceo_tenure"] + year_counts[year]["covered_review_ceo_tenure"]
        row["covered_any"] = str(covered)
        row["covered_any_pct"] = f"{100 * covered / total:.4f}" if total else "0.0000"
        by_year.append(row)

    write_csv(panel_path, panel, OUTPUT_FIELDS)
    exact_path = args.output_dir / "ccts_execucomp_exact_ceo_coverage.csv"
    review_path = args.output_dir / "ccts_execucomp_match_review.csv"
    exact_rows = [row for row in panel if row["ceo_coverage_status"] == "covered_exact_ceo_tenure"]
    review_rows = [
        row for row in panel
        if row["issuer_match_status"] == "exact_cusip8_single_gvkey"
        and row["ceo_coverage_status"] != "covered_exact_ceo_tenure"
    ]
    write_csv(exact_path, exact_rows, OUTPUT_FIELDS)
    write_csv(review_path, review_rows, OUTPUT_FIELDS)
    by_year_fields = ["year", "events", *coverage_keys, "covered_any", "covered_any_pct"]
    write_csv(args.output_dir / "ccts_execucomp_coverage_by_year.csv", by_year, by_year_fields)

    exact = status_counts["covered_exact_ceo_tenure"]
    review = status_counts["covered_review_ceo_tenure"]
    exact_gvkeys = {row["gvkey"] for row in exact_rows if row["gvkey"]}
    exact_ceo_episodes = {
        (row["gvkey"], row["execid"], row["tenure_episode"])
        for row in exact_rows
    }
    exact_dates = [row["event_date"] for row in exact_rows if row["event_date"]]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "inputs": {
            "ccts_census": str(args.ccts_census),
            "execucomp_tenures": str(args.execucomp_tenures),
        },
        "definition": {
            "ccts_scope": "raw PRE+Q&A events" if args.require_pre_qa else "all supplied CCTS events",
            "ccts_firm_match_status_filter": args.firm_match_status,
            "issuer_join": "normalized exact CUSIP8 only",
            "ceo_join": "event date inclusively within normalized CEO episode",
            "company_id_used_for_join": False,
            "company_name_used_for_join": False,
        },
        "counts": {
            "events": len(panel),
            "events_with_ccts_cusip8": sum(bool(row["ccts_cusip8"]) for row in panel),
            "exact_cusip8_issuer_matches": sum(row["issuer_match_status"] == "exact_cusip8_single_gvkey" for row in panel),
            "covered_exact_ceo_tenure": exact,
            "covered_review_ceo_tenure": review,
            "covered_any_ceo_tenure": exact + review,
            "exact_analysis_rows": len(exact_rows),
            "issuer_matched_review_rows": len(review_rows),
            "exact_analysis_distinct_gvkeys": len(exact_gvkeys),
            "exact_analysis_distinct_ceo_episodes": len(exact_ceo_episodes),
            "exact_analysis_event_date_min": min(exact_dates) if exact_dates else None,
            "exact_analysis_event_date_max": max(exact_dates) if exact_dates else None,
            "coverage_status": dict(sorted(status_counts.items())),
            "distinct_matched_cusip8": len(issuer_gvkeys),
            "distinct_matched_gvkeys": len({g for values in issuer_gvkeys.values() for g in values}),
        },
        "percentages": {
            "pct_exact_issuer_match_all_events": round(100 * sum(row["issuer_match_status"] == "exact_cusip8_single_gvkey" for row in panel) / len(panel), 4) if panel else 0,
            "pct_any_ceo_coverage_all_events": round(100 * (exact + review) / len(panel), 4) if panel else 0,
            "pct_exact_ceo_coverage_all_events": round(100 * exact / len(panel), 4) if panel else 0,
        },
        "artifacts": {
            "event_panel_csv": str(panel_path),
            "exact_analysis_csv": str(exact_path),
            "issuer_matched_review_csv": str(review_path),
            "by_year_csv": str(args.output_dir / "ccts_execucomp_coverage_by_year.csv"),
        },
        "caveats": [
            "ExecuComp coverage begins in 1993 and is not a universe-wide issuer database.",
            "CUSIP changes over time can cause valid firms to remain unmatched.",
            "Annual-bound and invalid-source-order CEO episodes remain review evidence, not exact tenure evidence.",
            "No fuzzy company-name or CCTS company_id join is performed.",
        ],
    }
    summary_path = args.output_dir / "ccts_execucomp_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# CCTS-ExecuComp Coverage Report", "",
        "## Definition", "",
        "- CCTS scope: events with raw PRE and Q&A.",
        f"- CCTS firm-match filter: {', '.join(args.firm_match_status) if args.firm_match_status else 'none'}.",
        "- Issuer bridge: normalized exact CUSIP8 only.",
        "- CEO coverage: event date falls inclusively within one normalized CEO episode.",
        "- CCTS `company_id` and company names are not join keys.", "",
        "## Results", "",
        f"- CCTS events: {len(panel):,}",
        f"- Events with CCTS CUSIP8: {summary['counts']['events_with_ccts_cusip8']:,}",
        f"- Exact issuer matches: {summary['counts']['exact_cusip8_issuer_matches']:,}",
        f"- Exact CEO-tenure coverage: {exact:,}",
        f"- Exact-analysis firms: {len(exact_gvkeys):,}",
        f"- Exact-analysis CEO episodes: {len(exact_ceo_episodes):,}",
        f"- Exact-analysis event dates: {min(exact_dates) if exact_dates else 'n/a'} to {max(exact_dates) if exact_dates else 'n/a'}",
        f"- Review CEO-tenure coverage: {review:,}",
        f"- Issuer-matched rows held outside the exact analysis set: {len(review_rows):,}",
        f"- Any CEO-tenure coverage: {exact + review:,} ({summary['percentages']['pct_any_ceo_coverage_all_events']:.4f}%)", "",
        "## Coverage Status", "",
    ]
    report.extend(f"- `{key}`: {value:,}" for key, value in sorted(status_counts.items()))
    report += ["", "## Caveats", ""] + [f"- {item}" for item in summary["caveats"]]
    (args.output_dir / "ccts_execucomp_coverage_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
