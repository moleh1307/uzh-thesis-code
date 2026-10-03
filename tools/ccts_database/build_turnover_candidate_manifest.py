#!/usr/bin/env python3
"""Build a local, text-free CCTS fetch list for metadata-ready CEO transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SSD_ROOT = Path("data")
LONGITUDINAL_ROOT = (
    SSD_ROOT / "03_external_data/execucomp/longitudinal/20260711_exact_panel_turnover_v1"
)
DEFAULT_OUTPUT = (
    SSD_ROOT
    / "03_external_data/execucomp/longitudinal/20260927_turnover_candidate_manifest_v1"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pair_path = LONGITUDINAL_ROOT / "ceo_turnover_anchor_ready.csv"
    calls_path = LONGITUDINAL_ROOT / "ceo_call_sequences.csv"
    for path in (pair_path, calls_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = read_csv(pair_path)
    pairs = [row for row in pair_rows if row.get("anchor_turnover_ready") == "1"]
    if len({row["turnover_id"] for row in pairs}) != len(pairs):
        raise ValueError("Duplicate turnover_id in anchor-ready panel.")
    if any(
        row.get("five_calls_each_side") != "1"
        or row.get("five_quarters_each_side") != "1"
        for row in pairs
    ):
        raise ValueError("Anchor-ready rows do not all pass the 5-call/5-quarter screen.")

    episode_map: dict[tuple[str, str, str], dict[str, object]] = {}
    turnover_episode_rows: list[dict[str, object]] = []
    for pair in pairs:
        gvkey = pair["gvkey"].strip()
        for side in ("old", "new"):
            execid = pair[f"{side}_execid"].strip()
            episode = pair[f"{side}_tenure_episode"].strip()
            key = (gvkey, execid, episode)
            item = episode_map.setdefault(
                key,
                {
                    "gvkey": gvkey,
                    "execid": execid,
                    "tenure_episode": episode,
                    "turnover_ids": set(),
                    "roles": set(),
                },
            )
            item["turnover_ids"].add(pair["turnover_id"])
            item["roles"].add(side)
            turnover_episode_rows.append(
                {
                    "turnover_id": pair["turnover_id"],
                    "gvkey": gvkey,
                    "side": f"{side}_episode",
                    "execid": execid,
                    "tenure_episode": episode,
                    "transition_gap_days": pair["roster_transition_gap_days"],
                    "old_endpoint_event_id": pair["old_last_event_id"],
                    "old_endpoint_date": pair["old_last_call_date"],
                    "new_endpoint_event_id": pair["new_first_event_id"],
                    "new_endpoint_date": pair["new_first_call_date"],
                    "episode_calls_metadata": pair[f"{side}_calls"],
                    "episode_quarters_metadata": pair[f"{side}_distinct_quarters"],
                }
            )

    target_episode_keys = set(episode_map)
    episode_calls: dict[tuple[str, str, str], dict[str, object]] = {}
    event_assignments: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    event_details: dict[str, dict[str, str]] = {}
    with calls_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["gvkey"].strip(),
                row["execid"].strip(),
                row["tenure_episode"].strip(),
            )
            if key not in target_episode_keys:
                continue
            event_id = row["event_id"].strip()
            if not event_id:
                raise ValueError(f"Missing event_id in target episode {key}.")
            record = episode_calls.setdefault(key, {"events": {}, "quarters": set()})
            previous = record["events"].get(event_id)
            safe_row = {
                "event_id": event_id,
                "event_date": row["event_date"].strip(),
                "year": row["year"].strip(),
                "calendar_quarter": row["calendar_quarter"].strip(),
                "ccts_cusip8": row["ccts_cusip8"].strip(),
                "call_index_in_episode": row["call_index_in_episode"].strip(),
            }
            if previous is not None:
                reason = "conflicting" if previous != safe_row else "duplicate"
                raise ValueError(f"{reason.title()} event row {event_id} within episode {key}.")
            record["events"][event_id] = safe_row
            record["quarters"].add(row["calendar_quarter"].strip())
            event_assignments[event_id].add(key)
            if event_id in event_details and event_details[event_id] != safe_row:
                raise ValueError(f"Conflicting source metadata for event {event_id}.")
            event_details[event_id] = safe_row

    missing_episodes = sorted(target_episode_keys - set(episode_calls))
    if missing_episodes:
        raise ValueError(f"{len(missing_episodes)} candidate CEO episodes have no calls.")

    for pair in pairs:
        gvkey = pair["gvkey"].strip()
        for side in ("old", "new"):
            key = (
                gvkey,
                pair[f"{side}_execid"].strip(),
                pair[f"{side}_tenure_episode"].strip(),
            )
            actual = episode_calls[key]
            expected_calls = int(pair[f"{side}_calls"])
            expected_quarters = int(pair[f"{side}_distinct_quarters"])
            if len(actual["events"]) != expected_calls or len(actual["quarters"]) != expected_quarters:
                raise ValueError(
                    f"Pair/sequence counts disagree for {pair['turnover_id']} {side}: "
                    f"calls={len(actual['events'])}/{expected_calls}, "
                    f"quarters={len(actual['quarters'])}/{expected_quarters}"
                )

    episode_output: list[dict[str, object]] = []
    for key in sorted(target_episode_keys):
        gvkey, execid, episode = key
        source = episode_calls[key]
        events = source["events"]
        quarters = source["quarters"]
        if len(events) < 5 or len(quarters) < 5:
            raise ValueError(
                f"Candidate episode misses 5-call/5-quarter gate: {key}; "
                f"calls={len(events)}, quarters={len(quarters)}"
            )
        pair_meta = episode_map[key]
        episode_output.append(
            {
                "gvkey": gvkey,
                "execid": execid,
                "tenure_episode": episode,
                "calls_in_source_panel": len(events),
                "distinct_quarters_in_source_panel": len(quarters),
                "candidate_turnover_count": len(pair_meta["turnover_ids"]),
                "candidate_sides": ";".join(sorted(pair_meta["roles"])),
                "turnover_ids": ";".join(sorted(pair_meta["turnover_ids"])),
            }
        )
    assignment_output: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for event_id in sorted(event_assignments):
        keys = sorted(event_assignments[event_id])
        event = event_details[event_id]
        event_rows.append(
            {
                **event,
                "expected_ceo_episode_assignments": len(keys),
                "expected_execid_assignments": len({key[1] for key in keys}),
            }
        )
        for gvkey, execid, episode in keys:
            pair_meta = episode_map[(gvkey, execid, episode)]
            assignment_output.append(
                {
                    **event,
                    "gvkey": gvkey,
                    "execid": execid,
                    "tenure_episode": episode,
                    "candidate_turnover_count": len(pair_meta["turnover_ids"]),
                    "turnover_ids": ";".join(sorted(pair_meta["turnover_ids"])),
                }
            )
    event_rows.sort(key=lambda row: (row["event_date"], row["event_id"]))
    assignment_output.sort(
        key=lambda row: (row["event_date"], row["event_id"], row["gvkey"], row["execid"], row["tenure_episode"])
    )
    fetch_manifest = [{"event_id": row["event_id"]} for row in event_rows]
    write_csv(args.output_dir / "candidate_turnover_pairs.csv", [
        {
            "turnover_id": row["turnover_id"],
            "gvkey": row["gvkey"],
            "old_execid": row["old_execid"],
            "old_tenure_episode": row["old_tenure_episode"],
            "new_execid": row["new_execid"],
            "new_tenure_episode": row["new_tenure_episode"],
            "transition_gap_days": row["roster_transition_gap_days"],
            "old_calls": row["old_calls"],
            "old_distinct_quarters": row["old_distinct_quarters"],
            "new_calls": row["new_calls"],
            "new_distinct_quarters": row["new_distinct_quarters"],
            "old_endpoint_event_id": row["old_last_event_id"],
            "old_endpoint_date": row["old_last_call_date"],
            "new_endpoint_event_id": row["new_first_event_id"],
            "new_endpoint_date": row["new_first_call_date"],
        }
        for row in pairs
    ])
    write_csv(args.output_dir / "candidate_episode_manifest.csv", episode_output)
    write_csv(args.output_dir / "candidate_event_assignments.csv", assignment_output)
    write_csv(args.output_dir / "candidate_event_fetch_manifest.csv", fetch_manifest)
    write_csv(args.output_dir / "candidate_turnover_episode_map.csv", turnover_episode_rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "metadata_only_candidate_frame_not_speaker_validated",
        "candidate_turnover_pairs": len(pairs),
        "candidate_pair_episode_assignments": len(turnover_episode_rows),
        "unique_candidate_ceo_firm_episodes": len(episode_output),
        "unique_candidate_event_ids_to_fetch": len(event_rows),
        "candidate_event_episode_assignments": len(assignment_output),
        "event_ids_assigned_to_multiple_execids": sum(
            int(row["expected_execid_assignments"]) > 1 for row in event_rows
        ),
        "every_episode_passes_minimum_5_calls_5_quarters": True,
        "source": {
            "anchor_ready_pairs": {
                "path": str(pair_path),
                "sha256": sha256(pair_path),
            },
            "exact_panel_call_sequences": {
                "path": str(calls_path),
                "sha256": sha256(calls_path),
            },
        },
        "privacy_boundary": "No transcript text, CEO names, or company names are exported.",
        "next_gate": "Fetch only these event IDs through the authorized CCTS connection, then apply the current strict speaker/section gate and repaired Q&A-block extractor before scoring.",
    }
    (args.output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report = f"""# Metadata-Ready CEO Transition Candidate Frame

**Built UTC:** {summary['created_at_utc']}  
**Status:** metadata-only; not transcript-speaker-validated and not an analysis sample

## Contents

- {len(pairs):,} exact-panel CEO transition candidates that pass the five-call/five-quarter screen on both sides.
- {len(episode_output):,} unique CEO-firm-tenure episodes.
- {len(event_rows):,} unique CCTS event IDs across those episodes.
- One-row-per-event fetch list: `candidate_event_fetch_manifest.csv` (`event_id` column is directly usable by the CCTS turn fetcher).
- Candidate pair and CEO-episode maps preserve how events connect to transitions.
- {summary['event_ids_assigned_to_multiple_execids']:,} event IDs map to multiple candidate ExecuComp IDs; all assignments are retained for review rather than guessed.
- Event outputs contain identifiers, dates, and quarter fields only; no event titles or transcript text are exported.

## Next Gate

This package only prepares the event list. It does not connect to the database or export transcript text. Next, fetch the listed CCTS events through the authorized UZH/VPN connection, then apply the strict external-CEO/shared-PRE-Q&A identity gate and the current repaired Q&A-block extractor. Exclude or flag any identity or extraction ambiguity; only then can this expanded frame be used for measurement.

Source checksums and counts: `manifest_summary.json`. No LLM was called.
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "candidate_turnover_pairs": len(pairs),
        "unique_ceo_firm_episodes": len(episode_output),
        "unique_event_ids_to_fetch": len(event_rows),
        "speaker_validated": False,
        "transcripts_fetched": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
