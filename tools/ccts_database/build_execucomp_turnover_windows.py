#!/usr/bin/env python3
"""Build longitudinal CEO-call sequences and externally defined turnover windows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path


SEQUENCE_FIELDS = [
    "gvkey", "execucomp_company_name", "execid", "tenure_episode", "ceo_name",
    "ceo_start_date", "ceo_end_date", "event_id", "event_date", "year",
    "calendar_quarter", "event_title", "ccts_cusip8", "call_index_in_episode",
    "calls_in_episode", "distinct_quarters_in_episode", "days_since_ceo_start",
    "days_until_ceo_end", "days_from_previous_call", "days_to_next_call",
    "has_at_least_5_calls", "has_at_least_5_distinct_quarters",
]

TURNOVER_FIELDS = [
    "turnover_id", "gvkey", "company_name", "old_execid", "old_tenure_episode",
    "old_ceo_name", "old_ceo_start_date", "old_ceo_end_date", "new_execid",
    "old_tenure_quality",
    "new_tenure_episode", "new_ceo_name", "new_ceo_start_date", "new_ceo_end_date",
    "new_tenure_quality",
    "roster_transition_gap_days", "roster_transition_status", "old_calls",
    "old_distinct_quarters", "new_calls", "new_distinct_quarters", "old_last_event_id",
    "old_last_call_date", "new_first_event_id", "new_first_call_date",
    "call_gap_days", "within_120_call_gap", "five_calls_each_side",
    "five_quarters_each_side", "anchor_turnover_ready", "turnover_window_status",
    "audit_notes",
]
AUDIT_FIELDS = ["audit_rank", "audit_stratum", *TURNOVER_FIELDS]


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-event-panel", type=Path, required=True)
    parser.add_argument("--execucomp-tenures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-call-gap-days", type=int, default=120)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def d(value: str) -> date:
    return date.fromisoformat(value)


def quarter(value: str) -> str:
    dt = d(value)
    return f"{dt.year}Q{((dt.month - 1) // 3) + 1}"


def episode_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["gvkey"], row["execid"], row["tenure_episode"]


def main() -> int:
    cfg = args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_path = cfg.output_dir / "ceo_call_sequences.csv"
    if sequence_path.exists() and not cfg.force:
        raise SystemExit(f"output exists; pass --force to replace: {sequence_path}")

    events = read_csv(cfg.exact_event_panel)
    tenures = read_csv(cfg.execucomp_tenures)
    calls_by_episode: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in events:
        calls_by_episode[episode_key(row)].append(row)

    sequence_rows: list[dict[str, object]] = []
    for key, calls in sorted(calls_by_episode.items()):
        calls.sort(key=lambda row: (row["event_date"], row["event_id"]))
        quarters = {quarter(row["event_date"]) for row in calls}
        for index, row in enumerate(calls):
            current = d(row["event_date"])
            previous = d(calls[index - 1]["event_date"]) if index else None
            following = d(calls[index + 1]["event_date"]) if index + 1 < len(calls) else None
            sequence_rows.append({
                **row,
                "calendar_quarter": quarter(row["event_date"]),
                "call_index_in_episode": index + 1,
                "calls_in_episode": len(calls),
                "distinct_quarters_in_episode": len(quarters),
                "days_since_ceo_start": (current - d(row["ceo_start_date"])).days,
                "days_until_ceo_end": (d(row["ceo_end_date"]) - current).days,
                "days_from_previous_call": (current - previous).days if previous else "",
                "days_to_next_call": (following - current).days if following else "",
                "has_at_least_5_calls": int(len(calls) >= 5),
                "has_at_least_5_distinct_quarters": int(len(quarters) >= 5),
            })

    tenure_by_firm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in tenures:
        tenure_by_firm[row["gvkey"]].append(row)
    for rows in tenure_by_firm.values():
        rows.sort(key=lambda row: (row["ceo_start_date"], row["ceo_end_date"], row["execid"], row["tenure_episode"]))

    turnover_rows: list[dict[str, object]] = []
    rank = 0
    for gvkey, rows in sorted(tenure_by_firm.items()):
        for old, new in zip(rows, rows[1:]):
            if old["execid"] == new["execid"]:
                continue
            rank += 1
            old_key, new_key = episode_key(old), episode_key(new)
            old_calls = sorted(calls_by_episode.get(old_key, []), key=lambda row: (row["event_date"], row["event_id"]))
            new_calls = sorted(calls_by_episode.get(new_key, []), key=lambda row: (row["event_date"], row["event_id"]))
            old_exact = old["normalization_status"] == "exact_tenure_dates"
            new_exact = new["normalization_status"] == "exact_tenure_dates"
            roster_gap = (d(new["ceo_start_date"]) - d(old["ceo_end_date"])).days
            if not old_exact or not new_exact:
                roster_status = "review_nonexact_adjacent_tenure"
            elif roster_gap < 0:
                roster_status = "review_overlapping_exact_tenures"
            else:
                roster_status = "exact_adjacent_ceo_change"

            old_last, new_first = (old_calls[-1] if old_calls else None), (new_calls[0] if new_calls else None)
            call_gap = (d(new_first["event_date"]) - d(old_last["event_date"])).days if old_last and new_first else None
            old_quarters = len({quarter(row["event_date"]) for row in old_calls})
            new_quarters = len({quarter(row["event_date"]) for row in new_calls})
            within = call_gap is not None and 0 <= call_gap <= cfg.max_call_gap_days
            five_calls = len(old_calls) >= 5 and len(new_calls) >= 5
            five_quarters = old_quarters >= 5 and new_quarters >= 5
            ready = roster_status == "exact_adjacent_ceo_change" and within and five_quarters

            if roster_status != "exact_adjacent_ceo_change":
                window_status = roster_status
            elif not old_calls and not new_calls:
                window_status = "exact_turnover_no_calls_either_side"
            elif not old_calls:
                window_status = "exact_turnover_no_old_ceo_call"
            elif not new_calls:
                window_status = "exact_turnover_no_new_ceo_call"
            elif not within:
                window_status = "exact_turnover_call_gap_exceeds_120_days"
            elif not five_quarters:
                window_status = "exact_turnover_within_120_insufficient_5q_each_side"
            else:
                window_status = "anchor_turnover_ready"

            notes = []
            if roster_gap == 0:
                notes.append("old CEO end date equals new CEO start date")
            if old_calls and d(old_calls[-1]["event_date"]) > d(old["ceo_end_date"]):
                notes.append("old call falls after old tenure end")
            if new_calls and d(new_calls[0]["event_date"]) < d(new["ceo_start_date"]):
                notes.append("new call falls before new tenure start")
            turnover_rows.append({
                "turnover_id": f"execucomp_turnover_{rank:05d}", "gvkey": gvkey,
                "company_name": new["company_name"] or old["company_name"],
                "old_execid": old["execid"], "old_tenure_episode": old["tenure_episode"],
                "old_ceo_name": old["person_name"], "old_ceo_start_date": old["ceo_start_date"],
                "old_ceo_end_date": old["ceo_end_date"], "old_tenure_quality": old["normalization_status"],
                "new_execid": new["execid"],
                "new_tenure_episode": new["tenure_episode"], "new_ceo_name": new["person_name"],
                "new_ceo_start_date": new["ceo_start_date"], "new_ceo_end_date": new["ceo_end_date"],
                "new_tenure_quality": new["normalization_status"],
                "roster_transition_gap_days": roster_gap, "roster_transition_status": roster_status,
                "old_calls": len(old_calls), "old_distinct_quarters": old_quarters,
                "new_calls": len(new_calls), "new_distinct_quarters": new_quarters,
                "old_last_event_id": old_last["event_id"] if old_last else "",
                "old_last_call_date": old_last["event_date"] if old_last else "",
                "new_first_event_id": new_first["event_id"] if new_first else "",
                "new_first_call_date": new_first["event_date"] if new_first else "",
                "call_gap_days": call_gap if call_gap is not None else "",
                "within_120_call_gap": int(within), "five_calls_each_side": int(five_calls),
                "five_quarters_each_side": int(five_quarters), "anchor_turnover_ready": int(ready),
                "turnover_window_status": window_status, "audit_notes": "; ".join(notes),
            })

    exact_turnovers = [row for row in turnover_rows if row["roster_transition_status"] == "exact_adjacent_ceo_change"]
    observed_turnovers = [row for row in exact_turnovers if row["old_calls"] and row["new_calls"]]
    anchor_rows = [row for row in turnover_rows if row["anchor_turnover_ready"] == 1]
    review_rows = [row for row in turnover_rows if row["anchor_turnover_ready"] != 1]

    by_status: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in turnover_rows:
        by_status[str(row["turnover_window_status"])].append(row)
    for rows in by_status.values():
        rows.sort(key=lambda row: (str(row["gvkey"]), str(row["turnover_id"])))
    audit_sample: list[dict[str, object]] = []
    statuses = sorted(by_status)
    cursor = 0
    while len(audit_sample) < cfg.audit_size and statuses:
        status = statuses[cursor % len(statuses)]
        rows = by_status[status]
        if rows:
            picked = rows.pop(0)
            audit_sample.append({
                "audit_rank": len(audit_sample) + 1,
                "audit_stratum": status,
                **picked,
            })
        statuses = [item for item in statuses if by_status[item]]
        cursor += 1

    write_csv(sequence_path, sequence_rows, SEQUENCE_FIELDS)
    write_csv(cfg.output_dir / "ceo_turnover_windows.csv", turnover_rows, TURNOVER_FIELDS)
    write_csv(cfg.output_dir / "ceo_turnover_anchor_ready.csv", anchor_rows, TURNOVER_FIELDS)
    write_csv(cfg.output_dir / "ceo_turnover_review.csv", review_rows, TURNOVER_FIELDS)
    write_csv(cfg.output_dir / "ceo_turnover_audit_sample.csv", audit_sample, AUDIT_FIELDS)

    status_counts = Counter(row["turnover_window_status"] for row in turnover_rows)
    call_counts = Counter(int(row["calls_in_episode"]) for row in sequence_rows)
    episode_summaries = {
        (row["gvkey"], row["execid"], row["tenure_episode"]): int(row["calls_in_episode"])
        for row in sequence_rows
    }
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "inputs": {"exact_event_panel": str(cfg.exact_event_panel), "execucomp_tenures": str(cfg.execucomp_tenures)},
        "definitions": {
            "turnover": "adjacent different-execid episodes in the complete normalized ExecuComp roster",
            "exact_turnover": "both adjacent episodes have exact_tenure_dates and do not overlap",
            "anchor_ready": f"exact turnover, old-last/new-first observed call gap 0-{cfg.max_call_gap_days} days, and at least five distinct call quarters on each side",
        },
        "counts": {
            "sequence_events": len(sequence_rows), "sequence_ceo_episodes": len(episode_summaries),
            "episodes_with_at_least_5_calls": sum(value >= 5 for value in episode_summaries.values()),
            "roster_different_ceo_adjacencies": len(turnover_rows),
            "exact_adjacent_turnovers": len(exact_turnovers),
            "exact_turnovers_with_calls_both_sides": len(observed_turnovers),
            "within_120_days_with_calls_both_sides": sum(row["within_120_call_gap"] == 1 for row in observed_turnovers),
            "anchor_turnover_ready": len(anchor_rows), "turnover_status": dict(sorted(status_counts.items())),
            "audit_sample_rows": len(audit_sample),
            "audit_sample_by_status": dict(sorted(Counter(row["audit_stratum"] for row in audit_sample).items())),
        },
        "artifacts": {
            "sequences_csv": str(sequence_path),
            "turnover_windows_csv": str(cfg.output_dir / "ceo_turnover_windows.csv"),
            "anchor_ready_csv": str(cfg.output_dir / "ceo_turnover_anchor_ready.csv"),
            "review_csv": str(cfg.output_dir / "ceo_turnover_review.csv"),
            "audit_sample_csv": str(cfg.output_dir / "ceo_turnover_audit_sample.csv"),
        },
        "caveats": [
            "Five calls and five quarters are anchor-paper benchmarks, not yet final thesis restrictions.",
            "Turnovers interrupted by a nonexact normalized episode are not promoted to exact adjacency.",
            "This stage validates metadata continuity only; transcript speaker identity remains a separate audit gate.",
        ],
    }
    (cfg.output_dir / "ceo_longitudinal_turnover_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# CEO Longitudinal and Turnover Build", "", "## Definitions", "",
        f"- Turnover: {summary['definitions']['turnover']}.",
        f"- Anchor-ready: {summary['definitions']['anchor_ready']}.", "", "## Counts", "",
    ]
    lines.extend(f"- {key.replace('_', ' ').capitalize()}: {value:,}" for key, value in summary["counts"].items() if not isinstance(value, dict))
    lines += ["", "## Turnover Status", ""]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(status_counts.items()))
    lines += ["", "## Caveats", ""] + [f"- {item}" for item in summary["caveats"]]
    (cfg.output_dir / "ceo_longitudinal_turnover_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
