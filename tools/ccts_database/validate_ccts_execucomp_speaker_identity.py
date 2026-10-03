#!/usr/bin/env python3
"""Compare CCTS title-labelled PRE/Q&A speakers with ExecuComp CEO names."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FIELDS = [
    "selection_rank", "turnover_rank", "turnover_id", "turnover_side", "audit_stratum",
    "event_id", "event_date", "company_name", "expected_execid", "expected_ceo_name",
    "within_call_identity_status", "within_call_strict_ready", "pre_ceo_speakers",
    "qa_ceo_speakers", "shared_ceo_speakers", "matched_pre_speakers", "matched_qa_speakers",
    "matched_shared_speakers", "external_name_match_quality", "external_speaker_validation_status",
    "validation_notes",
]
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"}
PREFIXES = {"dr", "mr", "mrs", "ms"}
NICKNAME_PAIRS = {
    frozenset(pair)
    for pair in [
        ("ron", "ronald"), ("bob", "robert"), ("rob", "robert"),
        ("pete", "peter"), ("matt", "matthew"), ("jim", "james"),
        ("steve", "steven"), ("fred", "frederick"), ("joe", "joseph"),
        ("monty", "montgomery"), ("mike", "michael"),
        ("jack", "john"), ("al", "allan"), ("al", "alvin"),
        ("doug", "douglas"), ("don", "donald"), ("ed", "edward"),
        ("ray", "raymond"), ("hank", "william"),
        ("bill", "william"), ("dick", "richard"), ("jerry", "gerald"),
        ("jeff", "jeffery"), ("jeff", "jeffrey"), ("jeffery", "jeffrey"),
        ("rick", "richard"), ("steve", "stephen"), ("steven", "stephen"),
    ]
}
SURNAME_TRANSCRIPTION_PAIRS = {
    frozenset(pair)
    for pair in [("macmillan", "macmilllan"), ("taiclet", "taicle")]
}


def clean(value: object) -> str:
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


def bare_name(value: str) -> list[str]:
    # CCTS normally uses "Name, Company - Title", but some old labels omit
    # company and retain only "Name - Title". Strip the role before parsing.
    left = re.split(r"\s+-\s+", clean(value), maxsplit=1)[0]
    left = left.split(",", 1)[0]
    # Treat O'Neill and ONeill (and similar transcript punctuation variants)
    # as the same surname without introducing general fuzzy matching.
    left = re.sub(r"['’]", "", left)
    tokens = re.findall(r"[a-z]+", left.lower())
    tokens = [token for token in tokens if token not in SUFFIXES]
    while tokens and tokens[0] in PREFIXES:
        tokens.pop(0)
    return tokens


def relation(expected: str, labelled: str) -> str:
    left, right = bare_name(expected), bare_name(labelled)
    if len(left) < 2 or len(right) < 2:
        return "no_match"
    surname_variant = frozenset((left[-1], right[-1])) in SURNAME_TRANSCRIPTION_PAIRS
    if left[-1] != right[-1] and not surname_variant:
        return "no_match"
    if surname_variant and (left[0] == right[0] or frozenset((left[0], right[0])) in NICKNAME_PAIRS):
        return "recognized_transcription_variant"
    if left == right:
        return "exact_full_name"
    if left[0] == right[0]:
        return "same_first_last_middle_difference"
    if len(left[0]) == 1 and len(left) >= 3 and left[1] == right[0]:
        return "initial_prefix_plus_given_name"
    if len(left[0]) == 1 and len(left) >= 3 and frozenset((left[1], right[0])) in NICKNAME_PAIRS:
        return "initial_prefix_plus_recognized_nickname"
    if frozenset((left[0], right[0])) in NICKNAME_PAIRS:
        return "recognized_nickname_last_name"
    if left[0][0] == right[0][0]:
        return "same_initial_last_name_review"
    return "same_last_name_different_first_name"


def labels(value: str) -> list[str]:
    return [item.strip() for item in clean(value).split(";") if item.strip()]


def best(expected: str, values: list[str]) -> tuple[str, list[str]]:
    ranked = [(relation(expected, value), value) for value in values]
    order = {"exact_full_name": 0, "same_first_last_middle_difference": 1, "initial_prefix_plus_given_name": 2, "initial_prefix_plus_recognized_nickname": 3, "recognized_nickname_last_name": 4, "recognized_transcription_variant": 5, "same_initial_last_name_review": 6, "same_last_name_different_first_name": 7, "no_match": 8}
    ranked.sort(key=lambda item: order[item[0]])
    if not ranked:
        return "no_match", []
    quality = ranked[0][0]
    return quality, [value for kind, value in ranked if kind == quality]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--representation-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "ccts_execucomp_speaker_validation.csv"
    if output.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force to replace: {output}")

    with args.representation_events.open(newline="", encoding="utf-8-sig") as handle:
        representation = {row["event_id"]: row for row in csv.DictReader(handle)}
    with args.event_manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest = list(csv.DictReader(handle))

    results = []
    for source in manifest:
        row = representation.get(source["event_id"])
        if row is None:
            raise SystemExit(f"representation missing event {source['event_id']}")
        expected = source["expected_ceo_name"]
        pre, qa, shared = labels(row["pre_ceo_speakers"]), labels(row["qa_ceo_speakers"]), labels(row["shared_ceo_speakers"])
        pre_quality, pre_matches = best(expected, pre)
        qa_quality, qa_matches = best(expected, qa)
        shared_quality, shared_matches = best(expected, shared)
        strict = row["strict_identity_ready"] == "1"

        accepted = {"exact_full_name", "same_first_last_middle_difference", "initial_prefix_plus_given_name", "initial_prefix_plus_recognized_nickname", "recognized_nickname_last_name", "recognized_transcription_variant"}
        if strict and shared_quality in accepted:
            status = "confirmed_external_ceo_shared_pre_qa"
            quality = shared_quality
            notes = "expected external CEO matches the one shared title-labelled PRE/Q&A speaker"
        elif strict and shared_quality == "same_initial_last_name_review":
            status = "review_external_ceo_initial_last_name"
            quality = shared_quality
            notes = "shared PRE/Q&A speaker has matching surname and first initial only"
        elif strict:
            status = "mismatch_external_ceo_vs_shared_speaker"
            quality = shared_quality
            notes = "a single CCTS CEO speaker is shared across PRE/Q&A but does not match the expected external CEO"
        elif pre_quality in accepted and qa_quality in accepted:
            status = "review_external_ceo_present_both_sections_not_strict_shared"
            quality = "both_sections_name_match"
            notes = "expected CEO appears title-labelled in both sections but CCTS speaker-key rule is not strict"
        elif pre_quality in accepted or qa_quality in accepted:
            status = "review_external_ceo_present_one_section"
            quality = pre_quality if pre_quality != "no_match" else qa_quality
            notes = "expected CEO appears title-labelled on only one section"
        else:
            status = "not_confirmed_external_ceo_not_title_labelled"
            quality = shared_quality if shared_quality != "no_match" else (pre_quality if pre_quality != "no_match" else qa_quality)
            notes = "no sufficiently close expected-CEO name match among title-labelled CCTS CEO candidates"

        results.append({
            "selection_rank": source["selection_rank"], "turnover_rank": source["turnover_rank"],
            "turnover_id": source["turnover_id"], "turnover_side": source["turnover_side"],
            "audit_stratum": source["audit_stratum"], "event_id": source["event_id"],
            "event_date": source["start_date"][:10], "company_name": source["company_name"],
            "expected_execid": source["expected_execid"], "expected_ceo_name": expected,
            "within_call_identity_status": row["identity_status"], "within_call_strict_ready": row["strict_identity_ready"],
            "pre_ceo_speakers": row["pre_ceo_speakers"], "qa_ceo_speakers": row["qa_ceo_speakers"],
            "shared_ceo_speakers": row["shared_ceo_speakers"], "matched_pre_speakers": "; ".join(pre_matches),
            "matched_qa_speakers": "; ".join(qa_matches), "matched_shared_speakers": "; ".join(shared_matches),
            "external_name_match_quality": quality, "external_speaker_validation_status": status,
            "validation_notes": notes,
        })

    results.sort(key=lambda row: int(row["selection_rank"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    counts = Counter(row["external_speaker_validation_status"] for row in results)
    pair_statuses = []
    by_turnover = {}
    for row in results:
        by_turnover.setdefault(row["turnover_id"], []).append(row)
    for turnover_id, rows in sorted(by_turnover.items()):
        statuses = {row["external_speaker_validation_status"] for row in rows}
        pair_statuses.append({
            "turnover_id": turnover_id,
            "events": len(rows),
            "pair_validation_status": "confirmed_both_sides" if statuses == {"confirmed_external_ceo_shared_pre_qa"} else "review_or_mismatch",
        })
    pair_path = args.output_dir / "ccts_execucomp_turnover_pair_validation.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["turnover_id", "events", "pair_validation_status"])
        writer.writeheader()
        writer.writerows(pair_statuses)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "events": len(results), "turnovers": len(pair_statuses),
        "event_status_counts": dict(sorted(counts.items())),
        "confirmed_both_sides_turnovers": sum(row["pair_validation_status"] == "confirmed_both_sides" for row in pair_statuses),
        "event_validation_csv": str(output), "turnover_pair_validation_csv": str(pair_path),
        "caveat": "Name matching permits only the documented nickname pairs in the script; other legal-name changes and initials remain review.",
    }
    (args.output_dir / "ccts_execucomp_speaker_validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# CCTS-ExecuComp Speaker Validation", "", f"- Events: {len(results):,}", f"- Turnovers: {len(pair_statuses):,}", f"- Both sides confirmed: {summary['confirmed_both_sides_turnovers']:,}", "", "## Event Status", ""]
    lines.extend(f"- `{key}`: {value:,}" for key, value in sorted(counts.items()))
    lines += ["", "## Caveat", "", f"- {summary['caveat']}"]
    (args.output_dir / "ccts_execucomp_speaker_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
