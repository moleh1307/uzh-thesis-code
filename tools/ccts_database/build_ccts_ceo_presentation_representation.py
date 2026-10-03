#!/usr/bin/env python3
"""Build call-level CEO PRE representations and PRE/Q&A identity checks.

The input is a small, canonical CCTS turn sample. The output preserves both
the CEO-only PRE text and CEO-only Q&A text for the same labelled speaker;
it does not score language or infer managerial tenure across calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Any, Iterable, Mapping, Sequence

from ceo_title_evidence import SPECIAL_CEO_RE, candidate_evidence, evidence_labels, shared_evidence


CEO_RE = re.compile(r"\b(?:ceo|chief\s+executive\s+officer)\b", re.IGNORECASE)
FORMER_CEO_RE = re.compile(r"\b(?:former|retired|ex[-\s])\b", re.IGNORECASE)
CEO_ASSISTANT_RE = re.compile(r"\b(?:ceo\s+assistant|assistant\s+to\s+(?:the\s+)?ceo)\b", re.IGNORECASE)

EVENT_COLUMNS = [
    "event_id", "sample_rank", "start_date", "year", "company_name", "company_ticker", "event_title",
    "pre_ceo_speakers", "qa_ceo_speakers", "shared_ceo_speakers", "shared_ceo_candidate_count",
    "identity_status", "identity_flags", "source_block_event_status", "pre_ceo_turn_count",
    "qa_ceo_turn_count", "pre_ceo_word_count", "qa_ceo_word_count", "strict_identity_ready",
    "ceo_presentation_text", "ceo_qa_text",
    "pre_sequence_ids", "qa_sequence_ids", "pre_raw_sequence_ids", "representation_version",
]
TURN_COLUMNS = [
    "event_id", "sample_rank", "year", "company_name", "sequence_id", "section", "text_name",
    "speaker_identity_key", "text_contents",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def word_count(value: str) -> int:
    return len(clean(value).split())


def label(row: Mapping[str, str]) -> str:
    return re.sub(r"\s*\[\d+\]\s*$", "", clean(row.get("text_name", ""))).strip()


def speaker_key(row: Mapping[str, str]) -> str:
    # The left side of standard CCTS labels is "Name, Company" and remains
    # stable across PRE/Q&A while retaining more information than name alone.
    return normalize(re.split(r"\s+-\s+", label(row), maxsplit=1)[0])


def is_current_ceo(row: Mapping[str, str]) -> bool:
    speaker = label(row)
    if not CEO_RE.search(speaker) or FORMER_CEO_RE.search(speaker) or CEO_ASSISTANT_RE.search(speaker):
        return False
    return True


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # These fields carry semantic turn boundaries, not display whitespace.
            text_fields = {"ceo_presentation_text", "ceo_qa_text", "text_contents"}
            writer.writerow({
                field: (row.get(field, "") or "") if field in text_fields
                else clean(row.get(field, "")) for field in fields
            })


def candidate_map(rows: Sequence[Mapping[str, str]]) -> dict[str, tuple[str, ...]]:
    return candidate_evidence(rows, is_current_ceo, speaker_key, label)


def choose_manual_rows(rows: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    ordered = sorted(rows, key=lambda row: (str(row["year"]), str(row["event_id"])))
    ready = [row for row in ordered if str(row["strict_identity_ready"]) == "1"]
    review = [row for row in ordered if str(row["strict_identity_ready"]) != "1"]
    selected = review[: min(len(review), max(0, round(limit * 0.25)))]
    remaining = limit - len(selected)
    if ready and remaining:
        indexes = [round(i * (len(ready) - 1) / max(1, remaining - 1)) for i in range(min(remaining, len(ready)))]
        selected.extend(ready[index] for index in indexes)
    return selected[:limit]


def write_manual_audit(path: Path, rows: Sequence[Mapping[str, Any]], limit: int) -> None:
    selected = choose_manual_rows(rows, limit)
    lines = [
        "# CCTS CEO Presentation Identity Audit", "",
        "Each record compares the title-labelled CEO identity and CEO-only text in PRE versus Q&A for one event.",
        "",
    ]
    for index, row in enumerate(selected, start=1):
        lines.extend([
            f"## Event {index}: `{row['event_id']}`", "",
            f"- Company: {row['company_name']} ({row['year']})",
            f"- Identity status: `{row['identity_status']}` | Flags: `{row['identity_flags'] or 'none'}`",
            f"- PRE CEO: {row['pre_ceo_speakers'] or 'none'}",
            f"- Q&A CEO: {row['qa_ceo_speakers'] or 'none'}", "",
            "### CEO Presentation Text", "", str(row["ceo_presentation_text"]), "",
            "### CEO Q&A Text", "", str(row["ceo_qa_text"]), "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True, type=Path)
    parser.add_argument("--block-event-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manual-audit-events", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    turns_by_event: dict[str, list[dict[str, str]]] = defaultdict(list)
    for turn in read_csv(args.turns):
        turns_by_event[turn["event_id"]].append(turn)
    for turns in turns_by_event.values():
        turns.sort(key=lambda row: int(row.get("sequence_id") or 0))
    block_audit = {row["event_id"]: row for row in read_csv(args.block_event_audit)}

    event_rows: list[dict[str, Any]] = []
    output_turns: list[dict[str, Any]] = []
    for event_id in sorted(turns_by_event, key=int):
        turns = turns_by_event[event_id]
        first = turns[0]
        pre_turns = [turn for turn in turns if turn.get("analysis_text_type") == "PRE"]
        qa_turns = [turn for turn in turns if turn.get("analysis_text_type") == "Q&A"]
        pre_candidates = candidate_map(pre_turns)
        qa_candidates = candidate_map(qa_turns)
        shared_keys = sorted(set(pre_candidates) & set(qa_candidates))
        shared_labels = shared_evidence(pre_candidates, qa_candidates, shared_keys)
        pre_ceo_turns = [turn for turn in pre_turns if speaker_key(turn) in shared_keys]
        qa_ceo_turns = [turn for turn in qa_turns if speaker_key(turn) in shared_keys]
        flags: list[str] = []
        if not shared_keys:
            flags.append("NO_SHARED_PRE_QA_CEO_LABEL")
        if len(pre_candidates) > 1:
            flags.append("MULTIPLE_PRE_CEO_CANDIDATES")
        if len(shared_keys) > 1:
            flags.append("MULTIPLE_SHARED_CEO_CANDIDATES")
        if any(SPECIAL_CEO_RE.search(text) for text in evidence_labels(pre_candidates) + evidence_labels(qa_candidates)):
            flags.append("SPECIAL_CEO_TITLE")
        source_status = block_audit.get(event_id, {}).get("event_audit_status", "missing_block_audit")
        if source_status != "usable":
            flags.append(f"BLOCK_AUDIT_{source_status.upper()}")
        strict_ready = int(len(shared_keys) == 1 and not flags)
        if strict_ready:
            identity_status = "exact_single_shared_speaker_key"
        elif len(shared_keys) > 1:
            identity_status = "multiple_shared_speaker_keys"
        elif len(pre_candidates) > 1:
            identity_status = "review_side_multiple_ceo_candidates"
        else:
            identity_status = "review_or_no_shared_speaker_key"
        pre_text = "\n\n".join(clean(turn.get("text_contents", "")) for turn in pre_ceo_turns if clean(turn.get("text_contents", "")))
        qa_text = "\n\n".join(clean(turn.get("text_contents", "")) for turn in qa_ceo_turns if clean(turn.get("text_contents", "")))
        row = {
            "event_id": event_id, "sample_rank": first.get("sample_rank", ""), "start_date": first.get("start_date", ""),
            "year": first.get("year", ""), "company_name": first.get("company_name", ""),
            "company_ticker": first.get("company_ticker", ""), "event_title": first.get("event_title", ""),
            "pre_ceo_speakers": "; ".join(evidence_labels(pre_candidates)), "qa_ceo_speakers": "; ".join(evidence_labels(qa_candidates)),
            "shared_ceo_speakers": "; ".join(shared_labels), "shared_ceo_candidate_count": len(shared_keys),
            "identity_status": identity_status, "identity_flags": ";".join(sorted(set(flags))),
            "source_block_event_status": source_status, "pre_ceo_turn_count": len(pre_ceo_turns),
            "qa_ceo_turn_count": len(qa_ceo_turns), "pre_ceo_word_count": word_count(pre_text),
            "qa_ceo_word_count": word_count(qa_text), "strict_identity_ready": strict_ready,
            "ceo_presentation_text": pre_text, "ceo_qa_text": qa_text,
            "pre_sequence_ids": json.dumps([turn["sequence_id"] for turn in pre_ceo_turns if clean(turn.get("text_contents", ""))]),
            "qa_sequence_ids": json.dumps([turn["sequence_id"] for turn in qa_ceo_turns if clean(turn.get("text_contents", ""))]),
            "pre_raw_sequence_ids": json.dumps([turn.get("raw_sequence_id", "") for turn in pre_ceo_turns if clean(turn.get("text_contents", ""))]),
            "representation_version": "turn_preserving_20260906",
        }
        event_rows.append(row)
        for section, selected_turns in [("PRE", pre_ceo_turns), ("Q&A", qa_ceo_turns)]:
            for turn in selected_turns:
                output_turns.append({
                    "event_id": event_id, "sample_rank": first.get("sample_rank", ""), "year": first.get("year", ""),
                    "company_name": first.get("company_name", ""), "sequence_id": turn.get("sequence_id", ""),
                    "section": section, "text_name": label(turn), "speaker_identity_key": speaker_key(turn),
                    "text_contents": turn.get("text_contents", ""),
                })

    write_csv(args.output_dir / "ceo_presentation_event_table.csv", EVENT_COLUMNS, event_rows)
    write_csv(args.output_dir / "ceo_presentation_turns.csv", TURN_COLUMNS, output_turns)
    write_manual_audit(args.output_dir / "manual_presentation_identity_audit.md", event_rows, args.manual_audit_events)
    status_counts = Counter(row["identity_status"] for row in event_rows)
    ready_count = sum(int(row["strict_identity_ready"]) for row in event_rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "events": len(event_rows),
        "strict_identity_ready_events": ready_count, "identity_status_counts": dict(status_counts),
        "event_table_csv": str(args.output_dir / "ceo_presentation_event_table.csv"),
        "turns_csv": str(args.output_dir / "ceo_presentation_turns.csv"),
        "manual_audit_md": str(args.output_dir / "manual_presentation_identity_audit.md"),
        "caveat": "Exact speaker-key agreement within CCTS is a within-call identity check only. It is not a cross-call CEO tenure identifier.",
    }
    (args.output_dir / "ceo_presentation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# CCTS CEO Presentation Representation", "", f"- Events: {len(event_rows):,}", f"- Strict within-call PRE/Q&A identity matches: {ready_count:,}", ""]
    lines.extend(f"- `{key}`: {value:,}" for key, value in status_counts.most_common())
    lines.extend(["", "## Caveat", "", summary["caveat"]])
    (args.output_dir / "ceo_presentation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
