#!/usr/bin/env python3
"""Build a provenance-preserving CCTS turn table with unique event/sequence keys."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Any


REQUIRED_TURN_FIELDS = {
    "event_id",
    "sequence_id",
    "source_event_company_name",
    "text_contents",
}
ADDED_TURN_FIELDS = {
    "source_event_company_names_json",
    "deduplication_status",
}
RESOLUTION_FIELDS = [
    "event_id",
    "sequence_id",
    "raw_rows",
    "retained_rows",
    "collapsed_rows",
    "source_event_company_names_json",
    "identical_except_source_company_name",
    "text_char_count",
    "text_sha256",
]


def canonical_event_id(value: str, source: str) -> str:
    try:
        event_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid event_id in {source}: {value!r}") from exc
    if event_id <= 0:
        raise ValueError(f"Nonpositive event_id in {source}: {event_id}")
    return str(event_id)


def load_fetch_audit(path: Path) -> tuple[dict[str, int], set[str]]:
    expected_rows: dict[str, int] = {}
    review_ids: set[str] = set()
    required = {"event_id", "fetched_rows", "fetch_status", "fetch_notes"}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"Fetch audit is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            event_id = canonical_event_id(row["event_id"], f"{path}:{line_number}")
            if event_id in expected_rows:
                raise ValueError(f"Duplicate event_id in fetch audit: {event_id}")
            try:
                fetched_rows = int(row["fetched_rows"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid fetched_rows for event {event_id} in fetch audit"
                ) from exc
            if fetched_rows < 0:
                raise ValueError(f"Negative fetched_rows for event {event_id}")
            expected_rows[event_id] = fetched_rows

            status = (row["fetch_status"] or "").strip()
            notes = {
                note.strip()
                for note in (row["fetch_notes"] or "").split(";")
                if note.strip()
            }
            if status == "ok":
                if notes:
                    raise ValueError(
                        f"Event {event_id} is marked ok but has audit notes: {sorted(notes)}"
                    )
            elif status == "review":
                if notes != {"duplicate_sequence_id"}:
                    raise ValueError(
                        f"Event {event_id} has a review reason this script will not resolve: "
                        f"{sorted(notes)}"
                    )
                review_ids.add(event_id)
            else:
                raise ValueError(
                    f"Unexpected fetch_status {status!r} for event {event_id}"
                )

    if not expected_rows:
        raise ValueError("Fetch audit contains no events")
    return expected_rows, review_ids


def json_names(names: list[str]) -> str:
    return json.dumps(names, ensure_ascii=False, separators=(",", ":"))


def make_event_rows(
    event_id: str,
    rows: list[dict[str, str]],
    input_fields: list[str],
    review_ids: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        sequence_id = (row["sequence_id"] or "").strip()
        if not sequence_id:
            raise ValueError(f"Blank sequence_id in event {event_id}")
        grouped.setdefault(sequence_id, []).append(row)

    output_rows: list[dict[str, str]] = []
    resolutions: list[dict[str, str]] = []
    found_duplicate = False

    for sequence_id, duplicates in grouped.items():
        if len(duplicates) == 1:
            row = dict(duplicates[0])
            name = row["source_event_company_name"] or ""
            row["source_event_company_names_json"] = json_names([name] if name else [])
            row["deduplication_status"] = "unchanged"
            output_rows.append(row)
            continue

        if event_id not in review_ids:
            raise ValueError(
                f"Unflagged duplicate event/sequence key: {event_id}/{sequence_id}"
            )
        if len(duplicates) != 2:
            raise ValueError(
                f"Expected exactly two source rows for {event_id}/{sequence_id}; "
                f"found {len(duplicates)}"
            )

        names = [row["source_event_company_name"] or "" for row in duplicates]
        unique_names = sorted(set(names))
        if len(unique_names) != 2 or any(not name.strip() for name in unique_names):
            raise ValueError(
                f"Duplicate {event_id}/{sequence_id} does not contain two distinct "
                "nonblank company names"
            )

        comparison_fields = [
            field for field in input_fields if field != "source_event_company_name"
        ]
        if any(
            duplicates[0][field] != duplicates[1][field]
            for field in comparison_fields
        ):
            differing = [
                field
                for field in comparison_fields
                if duplicates[0][field] != duplicates[1][field]
            ]
            raise ValueError(
                f"Conflicting duplicate {event_id}/{sequence_id}; differing fields: "
                f"{differing}"
            )

        canonical = dict(duplicates[0])
        canonical["source_event_company_name"] = unique_names[0]
        canonical["source_event_company_names_json"] = json_names(unique_names)
        canonical["deduplication_status"] = "collapsed_exact_name_alias_duplicate"
        output_rows.append(canonical)

        text = canonical["text_contents"] or ""
        resolutions.append(
            {
                "event_id": event_id,
                "sequence_id": sequence_id,
                "raw_rows": str(len(duplicates)),
                "retained_rows": "1",
                "collapsed_rows": str(len(duplicates) - 1),
                "source_event_company_names_json": json_names(unique_names),
                "identical_except_source_company_name": "true",
                "text_char_count": str(len(text)),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
        found_duplicate = True

    return output_rows, resolutions, found_duplicate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_derived_table(input_csv: Path, audit_csv: Path, output_dir: Path) -> dict[str, Any]:
    input_csv = input_csv.resolve()
    audit_csv = audit_csv.resolve()
    output_dir = output_dir.resolve()
    if input_csv == audit_csv or output_dir in {input_csv.parent, audit_csv.parent}:
        raise ValueError("Output directory must be separate from the raw input directory")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    if not input_csv.is_file() or not audit_csv.is_file():
        raise FileNotFoundError("Both input CSV and fetch-audit CSV must exist")

    expected_rows, review_ids = load_fetch_audit(audit_csv)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    turns_name = "ccts_turns_analysis_v1.csv"
    resolution_name = "duplicate_sequence_resolution_audit.csv"
    summary_name = "deduplication_summary.json"
    temp_turns = temp_dir / turns_name
    temp_resolutions = temp_dir / resolution_name
    input_stat = input_csv.stat()

    input_rows = 0
    output_rows_count = 0
    collapsed_rows = 0
    duplicate_groups = 0
    event_count = 0
    max_rows_per_event = 0
    seen_event_ids: set[str] = set()
    found_review_ids: set[str] = set()
    current_event: str | None = None
    current_rows: list[dict[str, str]] = []

    try:
        with input_csv.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            input_fields = list(reader.fieldnames or [])
            missing = REQUIRED_TURN_FIELDS - set(input_fields)
            if missing:
                raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
            if ADDED_TURN_FIELDS & set(input_fields):
                raise ValueError(
                    f"Input already contains derived columns: {sorted(ADDED_TURN_FIELDS & set(input_fields))}"
                )
            output_fields = input_fields + sorted(ADDED_TURN_FIELDS)

            with temp_turns.open("w", encoding="utf-8", newline="") as turns_handle, temp_resolutions.open(
                "w", encoding="utf-8", newline=""
            ) as resolution_handle:
                turns_writer = csv.DictWriter(
                    turns_handle, fieldnames=output_fields, extrasaction="raise"
                )
                resolution_writer = csv.DictWriter(
                    resolution_handle, fieldnames=RESOLUTION_FIELDS, extrasaction="raise"
                )
                turns_writer.writeheader()
                resolution_writer.writeheader()

                def flush_event() -> None:
                    nonlocal output_rows_count, collapsed_rows, duplicate_groups
                    nonlocal event_count, max_rows_per_event
                    if current_event is None:
                        return
                    if current_event not in expected_rows:
                        raise ValueError(f"Event missing from fetch audit: {current_event}")
                    if len(current_rows) != expected_rows[current_event]:
                        raise ValueError(
                            f"Row-count mismatch for event {current_event}: "
                            f"input={len(current_rows)}, audit={expected_rows[current_event]}"
                        )
                    transformed, resolutions, has_duplicates = make_event_rows(
                        current_event, current_rows, input_fields, review_ids
                    )
                    for row in transformed:
                        turns_writer.writerow(row)
                    for row in resolutions:
                        resolution_writer.writerow(row)
                    output_rows_count += len(transformed)
                    collapsed_rows += len(current_rows) - len(transformed)
                    duplicate_groups += len(resolutions)
                    event_count += 1
                    max_rows_per_event = max(max_rows_per_event, len(current_rows))
                    if has_duplicates:
                        found_review_ids.add(current_event)
                    seen_event_ids.add(current_event)

                for line_number, raw_row in enumerate(reader, start=2):
                    if None in raw_row or any(raw_row.get(field) is None for field in input_fields):
                        raise ValueError(f"Malformed CSV record at logical row {line_number}")
                    row = {field: raw_row[field] for field in input_fields}
                    event_id = canonical_event_id(row["event_id"], f"{input_csv}:{line_number}")
                    if not (row["sequence_id"] or "").strip():
                        raise ValueError(f"Blank sequence_id at logical row {line_number}")

                    if current_event is None:
                        current_event = event_id
                    elif event_id != current_event:
                        flush_event()
                        if event_id in seen_event_ids:
                            raise ValueError(
                                f"Event rows are not contiguous; event {event_id} reappears"
                            )
                        current_event = event_id
                        current_rows = []
                    current_rows.append(row)
                    input_rows += 1

                flush_event()

        if seen_event_ids != set(expected_rows):
            missing_events = sorted(set(expected_rows) - seen_event_ids, key=int)
            extra_events = sorted(seen_event_ids - set(expected_rows), key=int)
            raise ValueError(
                f"Event coverage mismatch; missing={missing_events[:10]}, "
                f"extra={extra_events[:10]}"
            )
        if found_review_ids != review_ids:
            raise ValueError(
                "Duplicate-key events do not match fetch-audit review events; "
                f"audit={sorted(review_ids, key=int)}, found={sorted(found_review_ids, key=int)}"
            )
        if input_rows - output_rows_count != collapsed_rows:
            raise ValueError("Internal row-count reconciliation failed")

        final_stat = input_csv.stat()
        if (input_stat.st_size, input_stat.st_mtime_ns) != (
            final_stat.st_size,
            final_stat.st_mtime_ns,
        ):
            raise ValueError("Raw input CSV changed while the derived table was being built")

        summary: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "status": "complete",
            "input_csv": str(input_csv),
            "input_sha256": sha256_file(input_csv),
            "fetch_audit_csv": str(audit_csv),
            "fetch_audit_sha256": sha256_file(audit_csv),
            "input_rows": input_rows,
            "input_events": event_count,
            "max_rows_per_event": max_rows_per_event,
            "output_csv": str(output_dir / turns_name),
            "output_sha256": sha256_file(temp_turns),
            "output_rows": output_rows_count,
            "unique_event_sequence_keys": output_rows_count,
            "duplicate_sequence_groups_collapsed": duplicate_groups,
            "duplicate_rows_collapsed": collapsed_rows,
            "review_event_ids_resolved": sorted(found_review_ids, key=int),
            "resolution_audit_csv": str(output_dir / resolution_name),
            "resolution_audit_rows": duplicate_groups,
            "rule": (
                "Collapse exactly two rows only when every input field except "
                "source_event_company_name is identical; choose the lexicographically "
                "smallest name as the representative and retain all names in "
                "source_event_company_names_json."
            ),
            "raw_input_modified": False,
        }
        summary_path = temp_dir / summary_name
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        os.replace(temp_dir, output_dir)
        return summary
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an analysis-only CCTS turn table, collapsing only exact duplicate "
            "sequence rows flagged by the fetch audit. The raw CSV is never modified."
        )
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--fetch-audit-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_derived_table(
            args.input_csv, args.fetch_audit_csv, args.output_dir
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
