#!/usr/bin/env python3
"""Build an exact staged event-level CCTS PRE/Q&A census.

The output is one row per standard earnings-call event in the practical-dated
sample. It does not export transcript full text. It only exports metadata,
identifier fields, section row counts, and diagnostic speaker/title signals.

Design:
- Pull candidate events from transcript_metadata.
- Query transcript_textraw in event_id batches using the event_id indexes.
- Query transcript_participants in event_id batches.
- Write incremental count files so interrupted runs can resume.
- Join metadata + counts into a thesis-ready event-level census table.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence, Tuple

import psycopg2


DEFAULT_CREDENTIALS = Path(
    "secrets/ccts_database_credentials.md"
)
DEFAULT_OUTPUT_ROOT = Path("data/02_ccts_database/01_event_level_census")
STANDARD_EARNING_TYPE = "Earning Conference Call/Presentation"

METADATA_COLUMNS = [
    "event_id",
    "start_date",
    "year",
    "event_type_name",
    "event_title",
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
]

TEXT_COUNT_COLUMNS = [
    "event_id",
    "total_text_rows",
    "pre_rows",
    "qa_rows",
    "other_text_rows",
    "distinct_text_types",
    "text_types",
    "pre_text_name_ceo_signal_rows",
    "qa_text_name_ceo_signal_rows",
    "pre_text_name_cfo_signal_rows",
    "qa_text_name_cfo_signal_rows",
    "pre_text_name_analyst_signal_rows",
    "qa_text_name_analyst_signal_rows",
    "pre_text_name_operator_signal_rows",
    "qa_text_name_operator_signal_rows",
]

TEXT_SECTION_COUNT_COLUMNS = [
    "event_id",
    "total_text_rows",
    "pre_rows",
    "qa_rows",
    "other_text_rows",
    "distinct_text_types",
    "text_types",
]

PARTICIPANT_COUNT_COLUMNS = [
    "event_id",
    "participant_rows",
    "participant_type_rows",
    "participant_name_rows",
    "participant_ceo_signal_rows",
    "participant_cfo_signal_rows",
    "participant_analyst_signal_rows",
    "participant_operator_signal_rows",
]

PARTICIPANT_SECTION_COUNT_COLUMNS = [
    "event_id",
    "participant_rows",
]

FINAL_COLUMNS = [
    *METADATA_COLUMNS,
    *[c for c in TEXT_COUNT_COLUMNS if c != "event_id"],
    "has_text",
    "has_pre",
    "has_qa",
    "has_both_pre_and_qa",
    "pre_only",
    "qa_only",
    "no_text_rows",
    "possible_qa_dialogue_encoded_as_pre",
    *[c for c in PARTICIPANT_COUNT_COLUMNS if c != "event_id"],
]

FINAL_SECTION_COLUMNS = [
    *METADATA_COLUMNS,
    *[c for c in TEXT_SECTION_COUNT_COLUMNS if c != "event_id"],
    "has_text",
    "has_pre",
    "has_qa",
    "has_both_pre_and_qa",
    "pre_only",
    "qa_only",
    "no_text_rows",
    "participant_rows",
]


def read_credentials(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    fields = {
        "host": r"'dbhost'\s*:\s*'([^']+)'",
        "dbname": r"'dbname'\s*:\s*'([^']+)'",
        "user": r"'dbuser'\s*:\s*'([^']+)'",
        "password": r"'dbpass'\s*:\s*'([^']+)'",
    }
    values: Dict[str, str] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, text)
        if not match:
            raise SystemExit(f"Could not find {key} in credential note: {path}")
        values[key] = match.group(1)
    return values


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([clean_cell(value) for value in row])
            count += 1
    return count


def append_csv(path: Path, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    count = 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow(columns)
        for row in rows:
            writer.writerow([clean_cell(value) for value in row])
            count += 1
    return count


def clean_cell(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def read_processed_event_ids(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    processed: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("event_id"):
                processed.add(int(row["event_id"]))
    return processed


def read_count_file(path: Path, columns: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event_id = int(row["event_id"])
            parsed: Dict[str, Any] = {}
            for col in columns:
                value = row.get(col, "")
                if col == "event_id":
                    continue
                if col in {"text_types"}:
                    parsed[col] = value
                else:
                    parsed[col] = int(value or 0)
            out[event_id] = parsed
    return out


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def read_event_id_filter(
    path: Path | None,
    filter_column: str | None,
    filter_value: str,
) -> set[int] | None:
    if path is None:
        return None
    if not path.exists():
        raise SystemExit(f"Event-id filter CSV does not exist: {path}")
    event_ids: set[int] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "event_id" not in reader.fieldnames:
            raise SystemExit(f"Event-id filter CSV must contain event_id column: {path}")
        if filter_column and filter_column not in reader.fieldnames:
            raise SystemExit(f"Filter column {filter_column!r} not found in {path}")
        for row in reader:
            if filter_column:
                value = row.get(filter_column, "")
                if filter_value == "__truthy__":
                    if not truthy(value):
                        continue
                elif str(value).strip() != filter_value:
                    continue
            event_id = row.get("event_id", "")
            if event_id:
                event_ids.add(int(event_id))
    return event_ids


def read_filtered_metadata_rows_from_csv(
    path: Path | None,
    filter_column: str | None,
    filter_value: str,
    limit_events: int | None,
) -> List[Tuple[Any, ...]] | None:
    if path is None or not path.exists():
        return None
    rows: List[Tuple[Any, ...]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return None
        if any(column not in reader.fieldnames for column in METADATA_COLUMNS):
            return None
        if filter_column and filter_column not in reader.fieldnames:
            return None
        for row in reader:
            if filter_column:
                value = row.get(filter_column, "")
                if filter_value == "__truthy__":
                    if not truthy(value):
                        continue
                elif str(value).strip() != filter_value:
                    continue
            rows.append(tuple(row.get(column, "") for column in METADATA_COLUMNS))
            if limit_events and len(rows) >= limit_events:
                break
    rows.sort(key=lambda row: (row[1], int(row[0])))
    return rows


def chunks(items: Sequence[int], size: int) -> Iterator[List[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def signal_sql(name_expr: str) -> str:
    lowered = f"LOWER(COALESCE({name_expr}, ''))"
    return (
        f"POSITION('chief executive officer' IN {lowered}) > 0 OR "
        f"POSITION('president and ceo' IN {lowered}) > 0 OR "
        f"POSITION('chairman and ceo' IN {lowered}) > 0 OR "
        f"{lowered} ~ '(^|[^a-z])ceo([^a-z]|$)'"
    )


def cfo_signal_sql(name_expr: str) -> str:
    lowered = f"LOWER(COALESCE({name_expr}, ''))"
    return (
        f"POSITION('chief financial officer' IN {lowered}) > 0 OR "
        f"{lowered} ~ '(^|[^a-z])cfo([^a-z]|$)'"
    )


def analyst_signal_sql(name_expr: str, participant_type_expr: str | None = None) -> str:
    lowered = f"LOWER(COALESCE({name_expr}, ''))"
    clauses = [f"POSITION('analyst' IN {lowered}) > 0"]
    if participant_type_expr:
        clauses.append(f"POSITION('analyst' IN LOWER(COALESCE({participant_type_expr}, ''))) > 0")
    return " OR ".join(clauses)


def operator_signal_sql(name_expr: str, participant_type_expr: str | None = None) -> str:
    lowered = f"LOWER(COALESCE({name_expr}, ''))"
    clauses = [f"POSITION('operator' IN {lowered}) > 0"]
    if participant_type_expr:
        clauses.append(f"POSITION('operator' IN LOWER(COALESCE({participant_type_expr}, ''))) > 0")
    return " OR ".join(clauses)


def normalize_signal_text(value: Any) -> str:
    return f" {str(value or '').lower()} "


def ceo_signal_text(value: Any) -> bool:
    text = normalize_signal_text(value)
    return (
        "chief executive officer" in text
        or "president and ceo" in text
        or "chairman and ceo" in text
        or re.search(r"(^|[^a-z])ceo([^a-z]|$)", text) is not None
    )


def cfo_signal_text(value: Any) -> bool:
    text = normalize_signal_text(value)
    return "chief financial officer" in text or re.search(r"(^|[^a-z])cfo([^a-z]|$)", text) is not None


def analyst_signal_text(name_value: Any, type_value: Any = "") -> bool:
    return "analyst" in normalize_signal_text(name_value) or "analyst" in normalize_signal_text(type_value)


def operator_signal_text(name_value: Any, type_value: Any = "") -> bool:
    return "operator" in normalize_signal_text(name_value) or "operator" in normalize_signal_text(type_value)


def fetch_metadata(cur: Any, min_year: int, limit_events: int | None) -> List[Tuple[Any, ...]]:
    sql = f"""
        SELECT
            event_id,
            start_date::text AS start_date,
            EXTRACT(YEAR FROM start_date)::int AS year,
            event_type_name,
            event_title,
            company_id,
            company_name,
            company_ticker,
            isin,
            cusip,
            sedol,
            company_ric,
            file_name,
            event_story_id,
            story_type,
            version
        FROM transcript_metadata
        WHERE event_type_name = %s
          AND EXTRACT(YEAR FROM start_date)::int >= %s
        ORDER BY start_date, event_id
    """
    params: List[Any] = [STANDARD_EARNING_TYPE, min_year]
    if limit_events:
        sql += " LIMIT %s"
        params.append(limit_events)
    cur.execute(sql, params)
    return list(cur.fetchall())


def fetch_metadata_for_event_ids(
    cur: Any,
    min_year: int,
    event_ids: Sequence[int],
    limit_events: int | None,
    batch_size: int,
) -> List[Tuple[Any, ...]]:
    """Fetch metadata only for a pre-filtered event universe."""
    out: List[Tuple[Any, ...]] = []
    total = len(event_ids)
    query_batch_size = max(batch_size, 5000)
    for batch_number, batch in enumerate(chunks(list(event_ids), query_batch_size), start=1):
        cur.execute(
            """
            SELECT
                event_id,
                start_date::text AS start_date,
                EXTRACT(YEAR FROM start_date)::int AS year,
                event_type_name,
                event_title,
                company_id,
                company_name,
                company_ticker,
                isin,
                cusip,
                sedol,
                company_ric,
                file_name,
                event_story_id,
                story_type,
                version
            FROM transcript_metadata
            WHERE event_id = ANY(%s)
              AND event_type_name = %s
              AND EXTRACT(YEAR FROM start_date)::int >= %s;
            """,
            (list(batch), STANDARD_EARNING_TYPE, min_year),
        )
        out.extend(cur.fetchall())
        if batch_number == 1 or len(out) % max(query_batch_size * 5, 25000) < query_batch_size or len(out) >= total:
            print(f"metadata filtered fetch: {min(len(out), total):,}/{total:,} event_ids checked", flush=True)

    out.sort(key=lambda row: (row[1], row[0]))
    if limit_events:
        out = out[:limit_events]
    return out


def query_text_counts(cur: Any, event_ids: Sequence[int]) -> Mapping[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {
        int(event_id): {col: 0 for col in TEXT_COUNT_COLUMNS[1:] if col != "text_types"} | {"text_types": ""}
        for event_id in event_ids
    }
    cur.execute(
        """
        WITH counts AS (
            SELECT
                event_id,
                COUNT(*) AS total_text_rows,
                COUNT(*) FILTER (WHERE text_type = 'PRE') AS pre_rows,
                COUNT(*) FILTER (WHERE text_type = 'Q&A') AS qa_rows,
                COUNT(*) FILTER (WHERE COALESCE(text_type, '') NOT IN ('PRE', 'Q&A')) AS other_text_rows
            FROM transcript_textraw
            WHERE event_id = ANY(%s)
            GROUP BY event_id
        )
        SELECT
            event_id,
            total_text_rows,
            pre_rows,
            qa_rows,
            other_text_rows,
            ((pre_rows > 0)::int + (qa_rows > 0)::int + (other_text_rows > 0)::int) AS distinct_text_types,
            CONCAT_WS(
                ';',
                CASE WHEN pre_rows > 0 THEN 'PRE' END,
                CASE WHEN qa_rows > 0 THEN 'Q&A' END,
                CASE WHEN other_text_rows > 0 THEN 'OTHER' END
            ) AS text_types
        FROM counts;
        """,
        (list(event_ids),),
    )
    for row in cur.fetchall():
        event_id = int(row[0])
        out[event_id].update(dict(zip(TEXT_SECTION_COUNT_COLUMNS[1:], row[1:])))

    signal_specs = [
        ("ceo", signal_sql("text_name")),
        ("cfo", cfo_signal_sql("text_name")),
        ("analyst", analyst_signal_sql("text_name")),
        ("operator", operator_signal_sql("text_name")),
    ]
    for label, expression in signal_specs:
        cur.execute(
            f"""
            SELECT
                event_id,
                COUNT(*) FILTER (WHERE text_type = 'PRE' AND ({expression})) AS pre_signal_rows,
                COUNT(*) FILTER (WHERE text_type = 'Q&A' AND ({expression})) AS qa_signal_rows
            FROM transcript_textraw
            WHERE event_id = ANY(%s)
            GROUP BY event_id;
            """,
            (list(event_ids),),
        )
        for row in cur.fetchall():
            event_id = int(row[0])
            out[event_id][f"pre_text_name_{label}_signal_rows"] = int(row[1] or 0)
            out[event_id][f"qa_text_name_{label}_signal_rows"] = int(row[2] or 0)
    return out


def query_text_section_counts(cur: Any, event_ids: Sequence[int]) -> Mapping[int, Dict[str, Any]]:
    cur.execute(
        """
        WITH counts AS (
            SELECT
                event_id,
                COUNT(*) AS total_text_rows,
                COUNT(*) FILTER (WHERE text_type = 'PRE') AS pre_rows,
                COUNT(*) FILTER (WHERE text_type = 'Q&A') AS qa_rows,
                COUNT(*) FILTER (WHERE COALESCE(text_type, '') NOT IN ('PRE', 'Q&A')) AS other_text_rows
            FROM transcript_textraw
            WHERE event_id = ANY(%s)
            GROUP BY event_id
        )
        SELECT
            event_id,
            total_text_rows,
            pre_rows,
            qa_rows,
            other_text_rows,
            ((pre_rows > 0)::int + (qa_rows > 0)::int + (other_text_rows > 0)::int) AS distinct_text_types,
            CONCAT_WS(
                ';',
                CASE WHEN pre_rows > 0 THEN 'PRE' END,
                CASE WHEN qa_rows > 0 THEN 'Q&A' END,
                CASE WHEN other_text_rows > 0 THEN 'OTHER' END
            ) AS text_types
        FROM counts;
        """,
        (list(event_ids),),
    )
    out: Dict[int, Dict[str, Any]] = {}
    for row in cur.fetchall():
        out[int(row[0])] = dict(zip(TEXT_SECTION_COUNT_COLUMNS[1:], row[1:]))
    return out


def query_participant_counts(cur: Any, event_ids: Sequence[int]) -> Mapping[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {
        int(event_id): {col: 0 for col in PARTICIPANT_COUNT_COLUMNS[1:]} for event_id in event_ids
    }
    cur.execute(
        """
        SELECT
            event_id,
            COUNT(*) AS participant_rows,
            COUNT(*) FILTER (WHERE NULLIF(TRIM(COALESCE(participant_type, '')), '') IS NOT NULL) AS participant_type_rows,
            COUNT(*) FILTER (WHERE NULLIF(TRIM(COALESCE(participant_name, '')), '') IS NOT NULL) AS participant_name_rows
        FROM transcript_participants
        WHERE event_id = ANY(%s)
        GROUP BY event_id;
        """,
        (list(event_ids),),
    )
    for row in cur.fetchall():
        event_id = int(row[0])
        out[event_id].update(dict(zip(PARTICIPANT_COUNT_COLUMNS[1:4], row[1:])))

    signal_specs = [
        ("ceo", f"{signal_sql('participant_name')} OR {signal_sql('participant_type')}"),
        ("cfo", f"{cfo_signal_sql('participant_name')} OR {cfo_signal_sql('participant_type')}"),
        ("analyst", analyst_signal_sql("participant_name", "participant_type")),
        ("operator", operator_signal_sql("participant_name", "participant_type")),
    ]
    for label, expression in signal_specs:
        cur.execute(
            f"""
            SELECT
                event_id,
                COUNT(*) FILTER (WHERE {expression}) AS signal_rows
            FROM transcript_participants
            WHERE event_id = ANY(%s)
            GROUP BY event_id;
            """,
            (list(event_ids),),
        )
        for row in cur.fetchall():
            event_id = int(row[0])
            out[event_id][f"participant_{label}_signal_rows"] = int(row[1] or 0)
    return out


def query_participant_section_counts(cur: Any, event_ids: Sequence[int]) -> Mapping[int, Dict[str, Any]]:
    cur.execute(
        """
        SELECT event_id, COUNT(*) AS participant_rows
        FROM transcript_participants
        WHERE event_id = ANY(%s)
        GROUP BY event_id;
        """,
        (list(event_ids),),
    )
    out: Dict[int, Dict[str, Any]] = {}
    for row in cur.fetchall():
        out[int(row[0])] = dict(zip(PARTICIPANT_SECTION_COUNT_COLUMNS[1:], row[1:]))
    return out


def count_rows_for_events(
    *,
    cur: Any,
    event_ids: Sequence[int],
    output_path: Path,
    columns: Sequence[str],
    query_fn: Any,
    batch_size: int,
    label: str,
    progress_every: int,
) -> None:
    processed = read_processed_event_ids(output_path)
    remaining = [event_id for event_id in event_ids if event_id not in processed]
    total = len(event_ids)
    if not remaining:
        print(f"{label}: already complete ({len(processed):,}/{total:,})", flush=True)
        return

    started = time.time()
    written = 0
    for batch_number, batch in enumerate(chunks(remaining, batch_size), start=1):
        batch_start_done = len(processed) + written
        if batch_number == 1 or batch_start_done % progress_every < batch_size:
            print(
                f"{label}: starting batch {batch_number:,} "
                f"({batch_start_done:,}/{total:,} events already written; batch_size={len(batch):,})",
                flush=True,
            )
        batch_counts = query_fn(cur, batch)
        rows: List[List[Any]] = []
        for event_id in batch:
            values = batch_counts.get(event_id, {})
            rows.append([event_id] + [values.get(col, 0 if col != "text_types" else "") for col in columns[1:]])
        append_csv(output_path, columns, rows)
        written += len(rows)
        done = len(processed) + written
        if batch_number == 1 or done % progress_every < batch_size or done == total:
            elapsed = time.time() - started
            rate = written / elapsed if elapsed else 0
            print(
                f"{label}: {done:,}/{total:,} events counted "
                f"({rate:.1f} events/sec in this run)",
                flush=True,
            )


def safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def build_final_table(
    metadata_rows: Sequence[Sequence[Any]],
    text_counts: Mapping[int, Mapping[str, Any]],
    participant_counts: Mapping[int, Mapping[str, Any]],
    output_path: Path,
    section_only: bool,
) -> Counter:
    summary = Counter()
    by_year: Dict[int, Counter] = defaultdict(Counter)
    final_columns = FINAL_SECTION_COLUMNS if section_only else FINAL_COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(final_columns)
        for meta in metadata_rows:
            meta_dict = dict(zip(METADATA_COLUMNS, meta))
            event_id = int(meta_dict["event_id"])
            year = int(meta_dict["year"])
            text = text_counts.get(event_id, {})
            parts = participant_counts.get(event_id, {})

            total_text_rows = safe_int(text.get("total_text_rows"))
            pre_rows = safe_int(text.get("pre_rows"))
            qa_rows = safe_int(text.get("qa_rows"))
            has_text = int(total_text_rows > 0)
            has_pre = int(pre_rows > 0)
            has_qa = int(qa_rows > 0)
            has_both = int(has_pre and has_qa)
            pre_only = int(has_pre and not has_qa)
            qa_only = int(has_qa and not has_pre)
            no_text_rows = int(total_text_rows == 0)
            possible_pre_encoded = 0
            if not section_only:
                possible_pre_encoded = int(
                    pre_only
                    and (
                        safe_int(text.get("pre_text_name_analyst_signal_rows")) > 0
                        or safe_int(text.get("pre_text_name_operator_signal_rows")) > 1
                    )
                )

            summary["events"] += 1
            summary["has_text"] += has_text
            summary["has_pre"] += has_pre
            summary["has_qa"] += has_qa
            summary["has_both_pre_and_qa"] += has_both
            summary["pre_only"] += pre_only
            summary["qa_only"] += qa_only
            summary["no_text_rows"] += no_text_rows
            summary["possible_qa_dialogue_encoded_as_pre"] += possible_pre_encoded
            if not section_only:
                summary["text_name_ceo_signal_events"] += int(
                    safe_int(text.get("pre_text_name_ceo_signal_rows"))
                    + safe_int(text.get("qa_text_name_ceo_signal_rows"))
                    > 0
                )
                summary["text_name_cfo_signal_events"] += int(
                    safe_int(text.get("pre_text_name_cfo_signal_rows"))
                    + safe_int(text.get("qa_text_name_cfo_signal_rows"))
                    > 0
                )
                summary["text_name_analyst_signal_events"] += int(
                    safe_int(text.get("pre_text_name_analyst_signal_rows"))
                    + safe_int(text.get("qa_text_name_analyst_signal_rows"))
                    > 0
                )

            bucket = by_year[year]
            bucket["events"] += 1
            bucket["has_text"] += has_text
            bucket["has_pre"] += has_pre
            bucket["has_qa"] += has_qa
            bucket["has_both_pre_and_qa"] += has_both
            bucket["pre_only"] += pre_only
            bucket["qa_only"] += qa_only
            bucket["no_text_rows"] += no_text_rows
            bucket["possible_qa_dialogue_encoded_as_pre"] += possible_pre_encoded

            if section_only:
                writer.writerow(
                    [clean_cell(value) for value in [
                        *[meta_dict[col] for col in METADATA_COLUMNS],
                        *[text.get(col, 0 if col != "text_types" else "") for col in TEXT_SECTION_COUNT_COLUMNS[1:]],
                        has_text,
                        has_pre,
                        has_qa,
                        has_both,
                        pre_only,
                        qa_only,
                        no_text_rows,
                        parts.get("participant_rows", 0),
                    ]]
                )
            else:
                writer.writerow(
                    [clean_cell(value) for value in [
                        *[meta_dict[col] for col in METADATA_COLUMNS],
                        *[text.get(col, 0 if col != "text_types" else "") for col in TEXT_COUNT_COLUMNS[1:]],
                        has_text,
                        has_pre,
                        has_qa,
                        has_both,
                        pre_only,
                        qa_only,
                        no_text_rows,
                        possible_pre_encoded,
                        *[parts.get(col, 0) for col in PARTICIPANT_COUNT_COLUMNS[1:]],
                    ]]
                )
    summary["_by_year"] = by_year  # type: ignore[assignment]
    return summary


def write_by_year(path: Path, by_year: Mapping[int, Counter]) -> None:
    rows: List[List[Any]] = []
    for year, bucket in sorted(by_year.items()):
        events = max(1, bucket["events"])
        rows.append(
            [
                year,
                bucket["events"],
                bucket["has_text"],
                bucket["has_pre"],
                bucket["has_qa"],
                bucket["has_both_pre_and_qa"],
                bucket["pre_only"],
                bucket["qa_only"],
                bucket["no_text_rows"],
                bucket["possible_qa_dialogue_encoded_as_pre"],
                round(100 * bucket["has_both_pre_and_qa"] / events, 2),
            ]
        )
    write_csv(
        path,
        [
            "year",
            "events",
            "has_text",
            "has_pre",
            "has_qa",
            "has_both_pre_and_qa",
            "pre_only",
            "qa_only",
            "no_text_rows",
            "possible_qa_dialogue_encoded_as_pre",
            "pct_has_both_pre_and_qa",
        ],
        rows,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-year", type=int, default=2001)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit-events", type=int)
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument(
        "--sslmode",
        default="disable",
        choices=["disable", "allow", "prefer", "require", "verify-ca", "verify-full"],
        help="PostgreSQL SSL mode. The CCTS server currently works with disable/allow; prefer/require can hang on the VPN path.",
    )
    parser.add_argument("--force", action="store_true", help="Remove count outputs in output-dir before running.")
    parser.add_argument(
        "--event-id-csv",
        type=Path,
        help="Optional CSV containing event_id values to include. Useful for running signals only on a pre-filtered event universe.",
    )
    parser.add_argument(
        "--event-id-filter-column",
        help="Optional column in --event-id-csv used to filter included event_id rows.",
    )
    parser.add_argument(
        "--event-id-filter-value",
        default="__truthy__",
        help="Value required in --event-id-filter-column. Default treats 1/true/yes as included.",
    )
    parser.add_argument(
        "--section-only",
        action="store_true",
        help="Only compute metadata, PRE/Q&A row counts, section flags, and participant row counts. Skip full role/title signal counts.",
    )
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_ROOT / f"v1_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata_standard_earnings_practical.csv"
    text_counts_path = output_dir / "text_section_counts_by_event.csv"
    participant_counts_path = output_dir / "participant_counts_by_event.csv"
    final_path = output_dir / "event_level_section_census.csv"
    by_year_path = output_dir / "event_level_section_census_by_year.csv"
    summary_path = output_dir / "event_level_census_summary.json"
    report_path = output_dir / "event_level_census_summary.md"

    if args.force:
        for path in [metadata_path, text_counts_path, participant_counts_path, final_path, by_year_path, summary_path, report_path]:
            if path.exists():
                path.unlink()

    creds = read_credentials(args.credentials)
    event_id_filter = read_event_id_filter(
        args.event_id_csv,
        args.event_id_filter_column,
        args.event_id_filter_value,
    )
    local_metadata_rows = read_filtered_metadata_rows_from_csv(
        args.event_id_csv,
        args.event_id_filter_column,
        args.event_id_filter_value,
        args.limit_events,
    )
    if event_id_filter is not None:
        print(
            f"loaded event-id filter: {len(event_id_filter):,} event_ids from {args.event_id_csv}",
            flush=True,
        )
    if local_metadata_rows is not None:
        print(
            f"loaded metadata locally from event-id CSV: {len(local_metadata_rows):,} events",
            flush=True,
        )
    print(
        f"connecting to CCTS database {creds['dbname']} at {creds['host']} "
        f"with connect_timeout={args.connect_timeout}s",
        flush=True,
    )
    conn = psycopg2.connect(
        host=creds["host"],
        port=5432,
        dbname=creds["dbname"],
        user=creds["user"],
        password=creds["password"],
        connect_timeout=args.connect_timeout,
        sslmode=args.sslmode,
        application_name="uzh_thesis_ccts_event_level_census",
    )

    try:
        with conn.cursor() as cur:
            print(f"connected. setting statement_timeout={args.statement_timeout_ms}ms", flush=True)
            cur.execute("SET statement_timeout = %s;", (args.statement_timeout_ms,))

            print(
                f"fetching metadata for standard earnings events from year >= {args.min_year}"
                + (f" with limit {args.limit_events}" if args.limit_events else ""),
                flush=True,
            )
            if local_metadata_rows is not None:
                metadata_rows = local_metadata_rows
            elif event_id_filter is not None:
                metadata_rows = fetch_metadata_for_event_ids(
                    cur,
                    args.min_year,
                    sorted(event_id_filter),
                    args.limit_events,
                    args.batch_size,
                )
            else:
                metadata_rows = fetch_metadata(cur, args.min_year, args.limit_events)
            write_csv(metadata_path, METADATA_COLUMNS, metadata_rows)
            event_ids = [int(row[0]) for row in metadata_rows]
            print(f"metadata: {len(event_ids):,} practical standard earnings events", flush=True)

            total_standard = None
            excluded_before_min_year = None
            cur.execute(
                "SELECT COUNT(*) FROM transcript_metadata WHERE event_type_name = %s;",
                (STANDARD_EARNING_TYPE,),
            )
            total_standard = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*)
                FROM transcript_metadata
                WHERE event_type_name = %s
                  AND EXTRACT(YEAR FROM start_date)::int < %s;
                """,
                (STANDARD_EARNING_TYPE, args.min_year),
            )
            excluded_before_min_year = int(cur.fetchone()[0])

            count_rows_for_events(
                cur=cur,
                event_ids=event_ids,
                output_path=text_counts_path,
                columns=TEXT_SECTION_COUNT_COLUMNS if args.section_only else TEXT_COUNT_COLUMNS,
                query_fn=query_text_section_counts if args.section_only else query_text_counts,
                batch_size=args.batch_size,
                label="text counts",
                progress_every=max(args.batch_size * 20, 5000),
            )

            count_rows_for_events(
                cur=cur,
                event_ids=event_ids,
                output_path=participant_counts_path,
                columns=PARTICIPANT_SECTION_COUNT_COLUMNS if args.section_only else PARTICIPANT_COUNT_COLUMNS,
                query_fn=query_participant_section_counts if args.section_only else query_participant_counts,
                # Participant title/name signal expressions are expensive on the
                # remote CCTS instance. Honor the requested batch size so a
                # slow first query cannot make an entire long run look stalled.
                batch_size=args.batch_size,
                label="participant counts",
                progress_every=max(args.batch_size * 20, 5000),
            )
    finally:
        conn.close()

    text_columns = TEXT_SECTION_COUNT_COLUMNS if args.section_only else TEXT_COUNT_COLUMNS
    participant_columns = PARTICIPANT_SECTION_COUNT_COLUMNS if args.section_only else PARTICIPANT_COUNT_COLUMNS
    text_counts = read_count_file(text_counts_path, text_columns)
    participant_counts = read_count_file(participant_counts_path, participant_columns)
    summary_counter = build_final_table(
        metadata_rows, text_counts, participant_counts, final_path, section_only=args.section_only
    )
    by_year = summary_counter.pop("_by_year")  # type: ignore[assignment]
    write_by_year(by_year_path, by_year)  # type: ignore[arg-type]

    events = max(1, int(summary_counter["events"]))
    summary: Dict[str, Any] = {
        "created_at_utc": timestamp,
        "output_dir": str(output_dir),
        "database": creds["dbname"],
        "user": creds["user"],
        "password_written": False,
        "standard_earning_type": STANDARD_EARNING_TYPE,
        "scope": {
            "min_year": args.min_year,
            "limit_events": args.limit_events,
            "total_standard_earning_rows_in_metadata": total_standard,
            "excluded_standard_rows_before_min_year": excluded_before_min_year,
            "events_in_census": int(summary_counter["events"]),
            "section_only": bool(args.section_only),
        },
        "counts": {k: int(v) for k, v in summary_counter.items()},
        "percentages": {
            "pct_has_text": round(100 * summary_counter["has_text"] / events, 2),
            "pct_has_pre": round(100 * summary_counter["has_pre"] / events, 2),
            "pct_has_qa": round(100 * summary_counter["has_qa"] / events, 2),
            "pct_has_both_pre_and_qa": round(100 * summary_counter["has_both_pre_and_qa"] / events, 2),
            "pct_possible_qa_dialogue_encoded_as_pre": round(
                100 * summary_counter["possible_qa_dialogue_encoded_as_pre"] / events, 2
            ),
        },
        "artifacts": {
            "metadata_standard_earnings_practical_csv": str(metadata_path),
            "text_section_counts_by_event_csv": str(text_counts_path),
            "participant_counts_by_event_csv": str(participant_counts_path),
            "event_level_section_census_csv": str(final_path),
            "event_level_section_census_by_year_csv": str(by_year_path),
            "event_level_census_summary_json": str(summary_path),
            "event_level_census_summary_md": str(report_path),
        },
        "caveats": [
            "This table is exact for the selected practical-dated standard earnings-call metadata scope, not for placeholder year rows before min_year.",
            "Counts are row counts from transcript_textraw, not word/token counts.",
            "possible_qa_dialogue_encoded_as_pre is a diagnostic flag, not a final section classifier.",
            "No transcript full text is exported.",
        ],
    }
    if args.section_only:
        summary["caveats"].append(
            "Role/title signal counts were intentionally skipped in section-only mode; compute them later on a targeted validated sample."
        )
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_path.write_text(
        "\n".join(
            [
                "# CCTS Event-Level PRE/Q&A Census v1",
                "",
                f"Created UTC: {timestamp}",
                "",
                "## Scope",
                "",
                f"- Event type: `{STANDARD_EARNING_TYPE}`",
                f"- Included years: `{args.min_year}` and later",
                f"- Events in census: {summary['scope']['events_in_census']:,}",
                f"- Standard metadata rows excluded before {args.min_year}: {summary['scope']['excluded_standard_rows_before_min_year']:,}",
                "- No transcript full text exported.",
                "",
                "## Section Counts",
                "",
                f"- Events with any text rows: {summary['counts']['has_text']:,} ({summary['percentages']['pct_has_text']}%)",
                f"- Events with PRE rows: {summary['counts']['has_pre']:,} ({summary['percentages']['pct_has_pre']}%)",
                f"- Events with Q&A rows: {summary['counts']['has_qa']:,} ({summary['percentages']['pct_has_qa']}%)",
                f"- Events with both PRE and Q&A rows: {summary['counts']['has_both_pre_and_qa']:,} ({summary['percentages']['pct_has_both_pre_and_qa']}%)",
                f"- PRE-only events: {summary['counts']['pre_only']:,}",
                f"- Q&A-only events: {summary['counts']['qa_only']:,}",
                f"- Events with no text rows: {summary['counts']['no_text_rows']:,}",
                f"- Diagnostic possible Q&A encoded as PRE: {summary['counts']['possible_qa_dialogue_encoded_as_pre']:,} ({summary['percentages']['pct_possible_qa_dialogue_encoded_as_pre']}%)",
                "",
                "## Role/Title Signal Coverage",
                "",
                "- Skipped in section-only mode." if args.section_only else f"- Events with CEO signal in `text_name`: {summary['counts']['text_name_ceo_signal_events']:,}",
                "" if args.section_only else f"- Events with CFO signal in `text_name`: {summary['counts']['text_name_cfo_signal_events']:,}",
                "" if args.section_only else f"- Events with analyst signal in `text_name`: {summary['counts']['text_name_analyst_signal_events']:,}",
                "",
                "## Artifacts",
                "",
                *[f"- `{Path(path).name}`: `{path}`" for path in summary["artifacts"].values()],
                "",
                "## Caveats",
                "",
                *[f"- {caveat}" for caveat in summary["caveats"]],
                "",
                "## Next Recommended Step",
                "",
                "Use `event_level_section_census.csv` as the event universe, then build a small validated extraction sample that reconstructs presentation and Q&A text for flagged and clean events.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
