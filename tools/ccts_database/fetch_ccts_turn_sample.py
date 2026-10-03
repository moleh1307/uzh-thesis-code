#!/usr/bin/env python3
"""Fetch CCTS transcript turns in small, resumable event batches.

Each batch is checkpointed as a pair of CSV shards plus a checksum marker. The
combined CSVs are rebuilt from valid checkpoints, so an interrupted run can be
restarted without duplicating transcript rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Any, Iterable, Mapping, Sequence

import psycopg2

DEFAULT_CREDENTIALS = Path("secrets/ccts_database_credentials.md")
TURN_COLUMNS = [
    "sample_rank", "event_id", "start_date", "year", "company_name", "company_ticker", "event_title",
    "sequence_id", "raw_sequence_id", "source_event_company_name", "text_type", "analysis_text_type",
    "text_name", "text_contents",
]
AUDIT_COLUMNS = [
    "event_id", "sample_rank", "year", "company_name", "expected_pre_rows", "expected_qa_rows",
    "database_rows", "database_min_sequence_id", "database_max_sequence_id", "fetched_rows",
    "fetched_pre_rows", "fetched_qa_rows", "fetch_status", "fetch_notes",
]
CHECKPOINT_SCHEMA = 2


class CheckpointBindingError(RuntimeError):
    """A shard belongs to a different source or extractor contract."""


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def as_int(value: Any) -> int:
    return int(value or 0)


def optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def read_credentials(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "host": r"'dbhost'\s*:\s*'([^']+)'", "dbname": r"'dbname'\s*:\s*'([^']+)'",
        "user": r"'dbuser'\s*:\s*'([^']+)'", "password": r"'dbpass'\s*:\s*'([^']+)'",
    }
    values: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Could not find {key} in credential note: {path}")
        values[key] = match.group(1)
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean(row.get(column, "")) for column in columns})
    os.replace(temporary, path)


def chunks(items: Sequence[Mapping[str, str]], size: int) -> list[list[Mapping[str, str]]]:
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def normalize_manifest(rows: list[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    selected = rows[:limit] if limit is not None else rows
    seen: set[int] = set()
    normalized: list[dict[str, str]] = []
    for rank, row in enumerate(selected, start=1):
        raw_id = (row.get("event_id") or "").strip()
        if not raw_id:
            raise ValueError(f"Manifest row {rank + 1} has no event_id.")
        event_id = int(raw_id)
        if event_id <= 0:
            raise ValueError(f"Manifest row {rank + 1} has nonpositive event_id {event_id}.")
        if event_id in seen:
            raise ValueError(f"Manifest contains duplicate event_id {event_id}.")
        seen.add(event_id)
        item = dict(row)
        item["event_id"] = str(event_id)
        item.setdefault("selection_rank", str(rank))
        normalized.append(item)
    if not normalized:
        raise ValueError("The selected event manifest is empty.")
    return normalized


def fetch_batch(
    cur: Any, events: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_ids = [as_int(event["event_id"]) for event in events]
    event_by_id = {as_int(event["event_id"]): event for event in events}

    cur.execute(
        """
        SELECT event_id, COUNT(*), MIN(sequence_id), MAX(sequence_id)
        FROM transcript_textraw
        WHERE event_id = ANY(%s)
        GROUP BY event_id
        ORDER BY event_id
        """,
        (event_ids,),
    )
    db_counts = {
        int(row[0]): {"rows": int(row[1]), "min_seq": row[2], "max_seq": row[3]}
        for row in cur.fetchall()
    }

    cur.execute(
        """
        SELECT event_id, sequence_id, company_name, text_type, text_name, text_contents
        FROM transcript_textraw
        WHERE event_id = ANY(%s)
        ORDER BY event_id, sequence_id
        """,
        (event_ids,),
    )
    records_by_event: dict[int, list[dict[str, Any]]] = {event_id: [] for event_id in event_ids}
    for row in cur.fetchall():
        event_id = int(row[0])
        event = event_by_id.get(event_id)
        if event is None:
            raise ValueError(f"Database returned unrequested event_id {event_id}.")
        text_type = clean(row[3])
        records_by_event[event_id].append(
            {
                "sample_rank": event.get("selection_rank", ""),
                "event_id": event_id,
                "start_date": event.get("start_date", ""),
                "year": event.get("year", ""),
                "company_name": event.get("company_name", ""),
                "company_ticker": event.get("company_ticker", ""),
                "event_title": event.get("event_title", ""),
                "sequence_id": row[1],
                "raw_sequence_id": row[1],
                "source_event_company_name": row[2],
                "text_type": text_type,
                "analysis_text_type": text_type if text_type in {"PRE", "Q&A"} else "OTHER",
                "text_name": row[4],
                "text_contents": row[5] or "",
            }
        )

    audits: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for event in events:
        event_id = as_int(event["event_id"])
        records = records_by_event[event_id]
        db = db_counts.get(event_id, {"rows": 0, "min_seq": None, "max_seq": None})
        pre_rows = sum(row["analysis_text_type"] == "PRE" for row in records)
        qa_rows = sum(row["analysis_text_type"] == "Q&A" for row in records)
        expected_pre = optional_int(event.get("pre_rows"))
        expected_qa = optional_int(event.get("qa_rows"))
        problems: list[str] = []
        if len(records) != db["rows"]:
            problems.append("fetched_row_count_mismatch")
        if db["rows"] == 0:
            problems.append("no_database_text_rows")
        if expected_pre is not None and pre_rows != expected_pre:
            problems.append("pre_count_mismatch")
        if expected_qa is not None and qa_rows != expected_qa:
            problems.append("qa_count_mismatch")
        sequence_ids = [str(row["sequence_id"]) for row in records]
        if len(sequence_ids) != len(set(sequence_ids)):
            problems.append("duplicate_sequence_id")
        turns.extend(records)
        audits.append(
            {
                "event_id": event_id,
                "sample_rank": event.get("selection_rank", ""),
                "year": event.get("year", ""),
                "company_name": event.get("company_name", ""),
                "expected_pre_rows": expected_pre if expected_pre is not None else "",
                "expected_qa_rows": expected_qa if expected_qa is not None else "",
                "database_rows": db["rows"],
                "database_min_sequence_id": db["min_seq"] if db["min_seq"] is not None else "",
                "database_max_sequence_id": db["max_seq"] if db["max_seq"] is not None else "",
                "fetched_rows": len(records),
                "fetched_pre_rows": pre_rows,
                "fetched_qa_rows": qa_rows,
                "fetch_status": "ok" if not problems else "review",
                "fetch_notes": ";".join(problems),
            }
        )
    return turns, audits


def checkpoint_key(event_ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()[:20]
    return f"batch_{digest}"


def checkpoint_paths(checkpoint_dir: Path, key: str) -> tuple[Path, Path, Path]:
    stem = key
    return (
        checkpoint_dir / f"{stem}_turns.csv",
        checkpoint_dir / f"{stem}_audit.csv",
        checkpoint_dir / f"{stem}.done.json",
    )


def read_valid_checkpoint(
    checkpoint_dir: Path, key: str, manifest_ids: set[str], run_binding_sha256: str | None = None
) -> dict[str, Any] | None:
    turns_path, audit_path, marker_path = checkpoint_paths(checkpoint_dir, key)
    if not (turns_path.is_file() and audit_path.is_file() and marker_path.is_file()):
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if not isinstance(marker, dict):
            return None
        if run_binding_sha256 is not None and marker.get("run_binding_sha256") != run_binding_sha256:
            raise CheckpointBindingError("Checkpoint source/extractor binding mismatch; use a fresh output directory.")
        event_ids = marker.get("event_ids")
        if (
            marker.get("schema") != CHECKPOINT_SCHEMA
            or marker.get("checkpoint_key") != key
            or not isinstance(event_ids, list)
            or not event_ids
            or any(event_id not in manifest_ids for event_id in event_ids)
        ):
            return None
        if marker.get("turns_sha256") != sha256_file(turns_path):
            return None
        if marker.get("audit_sha256") != sha256_file(audit_path):
            return None
        audit_rows = read_csv(audit_path)
        if [row.get("event_id", "") for row in audit_rows] != event_ids:
            return None
        if len(audit_rows) != len(event_ids):
            return None
        marker["key"] = key
        return marker
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def load_valid_checkpoints(
    checkpoint_dir: Path, event_ids: Sequence[str], run_binding_sha256: str | None = None
) -> list[dict[str, Any]]:
    manifest_ids = set(event_ids)
    checkpoints: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    for marker_path in sorted(checkpoint_dir.glob("batch_*.done.json")):
        key = marker_path.name.removesuffix(".done.json")
        marker = read_valid_checkpoint(checkpoint_dir, key, manifest_ids, run_binding_sha256)
        if marker is None:
            continue
        overlap = completed_ids.intersection(marker["event_ids"])
        if overlap:
            raise ValueError(
                "Checkpoint shards overlap on event IDs: " + ", ".join(sorted(overlap)[:10])
            )
        completed_ids.update(marker["event_ids"])
        checkpoints.append(marker)
    positions = {event_id: index for index, event_id in enumerate(event_ids)}
    checkpoints.sort(key=lambda marker: min(positions[event_id] for event_id in marker["event_ids"]))
    return checkpoints


def save_checkpoint(
    checkpoint_dir: Path,
    key: str,
    event_ids: Sequence[str],
    turns: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    run_binding_sha256: str | None = None,
) -> None:
    turns_path, audit_path, marker_path = checkpoint_paths(checkpoint_dir, key)
    atomic_write_csv(turns_path, TURN_COLUMNS, turns)
    atomic_write_csv(audit_path, AUDIT_COLUMNS, audits)
    marker = {
        "schema": CHECKPOINT_SCHEMA,
        "run_binding_sha256": run_binding_sha256,
        "checkpoint_key": key,
        "event_ids": list(event_ids),
        "turn_rows": len(turns),
        "audit_rows": len(audits),
        "turns_sha256": sha256_file(turns_path),
        "audit_sha256": sha256_file(audit_path),
    }
    atomic_write_text(marker_path, json.dumps(marker, indent=2) + "\n")


def materialize_csv(
    output_path: Path,
    columns: Sequence[str],
    checkpoint_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    kind: str,
    event_order: Sequence[str] | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    row_count = 0
    order = list(event_order) if event_order is not None else [event for shard in checkpoints for event in shard["event_ids"]]
    positions = {event: rank for rank, event in enumerate(order)}
    if len(positions) != len(order):
        raise ValueError("duplicate manifest event order")
    owned = set()
    for checkpoint in checkpoints:
        ids = checkpoint["event_ids"]
        if len(set(ids)) != len(ids) or owned.intersection(ids) or not set(ids) <= positions.keys():
            raise ValueError("duplicate or unexpected checkpoint event IDs")
        owned.update(ids)
    try:
        # Disk-backed ordering keeps memory and open files bounded even for fragmented shards.
        with tempfile.TemporaryDirectory(prefix=".fetch-order-", dir=output_path.parent) as stage:
            db = sqlite3.connect(str(Path(stage) / "rows.sqlite"))
            try:
                db.execute("CREATE TABLE rows (rank INTEGER, ordinal INTEGER, payload TEXT)")
                for checkpoint in checkpoints:
                    turns_path, audit_path, _ = checkpoint_paths(checkpoint_dir, checkpoint["key"])
                    shard_path = turns_path if kind == "turns" else audit_path
                    allowed = set(checkpoint["event_ids"])
                    audit_seen = set()
                    with shard_path.open("r", encoding="utf-8", newline="") as shard_handle:
                        reader = csv.DictReader(shard_handle)
                        if reader.fieldnames != list(columns):
                            raise ValueError("checkpoint CSV schema mismatch")
                        for ordinal, row in enumerate(reader):
                            event = row.get("event_id")
                            if event not in allowed or None in row or any(value is None for value in row.values()):
                                raise ValueError("unexpected or incomplete checkpoint row")
                            if kind == "audit" and event in audit_seen:
                                raise ValueError("duplicate audit event")
                            audit_seen.add(event)
                            db.execute("INSERT INTO rows VALUES (?, ?, ?)",
                                       (positions[event], ordinal, json.dumps(row, ensure_ascii=False)))
                    if kind == "audit" and audit_seen != allowed:
                        raise ValueError("incomplete checkpoint audit coverage")
                db.commit()
                with temporary.open("w", encoding="utf-8", newline="") as out_handle:
                    writer = csv.DictWriter(out_handle, fieldnames=columns, extrasaction="raise")
                    writer.writeheader()
                    for (payload,) in db.execute("SELECT payload FROM rows ORDER BY rank, ordinal"):
                        writer.writerow(json.loads(payload))
                        row_count += 1
                os.replace(temporary, output_path)
            finally:
                db.close()
    finally:
        temporary.unlink(missing_ok=True)
    return row_count


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def prepare_run_config(
    output_dir: Path,
    manifest_path: Path,
    full_manifest: Sequence[Mapping[str, str]],
    source_identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if set(source_identity) != {"host", "dbname", "user", "port", "sslmode"}:
        raise ValueError("Source identity must contain only host, dbname, user, port, and sslmode.")
    checkpoint_dir = output_dir / "checkpoints"
    config_path = output_dir / "fetch_run_config.json"
    event_digest = hashlib.sha256(
        "\n".join(str(row["event_id"]) for row in full_manifest).encode("utf-8")
    ).hexdigest()
    config = {
        "schema": CHECKPOINT_SCHEMA,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_events": len(full_manifest),
        "manifest_event_ids_sha256": event_digest,
        "source_identity": dict(source_identity),
        "extraction_identity": {
            "runner_sha256": sha256_file(Path(__file__)),
            "csv_contract_sha256": sha256_file(Path(__file__).resolve().parents[1] / "csv_contract.py"),
            "turn_columns": TURN_COLUMNS,
            "audit_columns": AUDIT_COLUMNS,
        },
    }
    config["run_binding_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError(
                "This output directory has a different manifest, source, extractor, or historical unbound contract. "
                "Resume with the same contract or use a fresh output directory."
            )
    else:
        legacy_outputs = [
            output_dir / "ccts_turns.csv",
            output_dir / "ccts_turn_fetch_audit.csv",
            checkpoint_dir,
        ]
        if any(path.exists() and (not path.is_dir() or any(path.iterdir())) for path in legacy_outputs):
            raise FileExistsError(
                "Output directory already contains uncheckpointed fetch data; use a fresh output directory."
            )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, config)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, config


def write_outputs(
    output_dir: Path,
    manifest_path: Path,
    full_manifest: Sequence[Mapping[str, str]],
    checkpoint_dir: Path,
    scope_event_ids: Sequence[str],
    run_binding_sha256: str,
) -> dict[str, Any]:
    manifest_ids = [str(event["event_id"]) for event in full_manifest]
    checkpoints = load_valid_checkpoints(checkpoint_dir, manifest_ids, run_binding_sha256)
    turns_path = output_dir / "ccts_turns.csv"
    audit_path = output_dir / "ccts_turn_fetch_audit.csv"
    turn_count = materialize_csv(turns_path, TURN_COLUMNS, checkpoint_dir, checkpoints, "turns", manifest_ids)
    materialize_csv(audit_path, AUDIT_COLUMNS, checkpoint_dir, checkpoints, "audit", manifest_ids)
    audits = read_csv(audit_path)
    scope_ids = set(scope_event_ids)
    scope_audits = [row for row in audits if row.get("event_id", "") in scope_ids]
    total_events = len(scope_ids)
    complete_events = len(scope_audits)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "complete" if complete_events == total_events else "partial",
        "manifest": str(manifest_path),
        "manifest_events_total": len(full_manifest),
        "events_expected": total_events,
        "events_complete": complete_events,
        "events_in_output": len(audits),
        "limited_scope": total_events < len(full_manifest),
        "ok_events": sum(row.get("fetch_status") == "ok" for row in scope_audits),
        "review_events": sum(row.get("fetch_status") != "ok" for row in scope_audits),
        "turn_rows": turn_count,
        "turns_csv": str(turns_path),
        "audit_csv": str(audit_path),
        "checkpoint_dir": str(checkpoint_dir),
        "run_binding_sha256": run_binding_sha256,
    }
    atomic_write_json(output_dir / "ccts_turn_fetch_summary.json", summary)
    atomic_write_json(
        output_dir / "ccts_turn_fetch_progress.json",
        {
            "status": summary["status"],
            "manifest_events_total": len(full_manifest),
            "events_complete": complete_events,
            "events_expected": total_events,
            "events_in_output": len(audits),
            "limited_scope": total_events < len(full_manifest),
            "percent_complete": round(100 * complete_events / total_events, 3) if total_events else 0,
            "updated_at_utc": summary["created_at_utc"],
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--credentials", default=DEFAULT_CREDENTIALS, type=Path)
    parser.add_argument("--connect-timeout", type=int, default=30)
    parser.add_argument("--statement-timeout-ms", type=int, default=30000)
    parser.add_argument("--sslmode", default="disable")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Small event batch size; completed events are skipped if this is reduced on resume.",
    )
    parser.add_argument("--limit-events", type=int, help="Process only the first N manifest events (smoke testing).")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit_events is not None and args.limit_events < 1:
        parser.error("--limit-events must be at least 1")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    full_manifest = normalize_manifest(read_csv(args.manifest), None)
    manifest = full_manifest[:args.limit_events] if args.limit_events is not None else full_manifest
    creds = read_credentials(args.credentials)
    source_identity = {key: creds[key] for key in ("host", "dbname", "user")}
    source_identity.update(port=args.port, sslmode=args.sslmode)
    checkpoint_dir, config = prepare_run_config(args.output_dir, args.manifest, full_manifest, source_identity)
    binding = config["run_binding_sha256"]
    event_ids = [str(event["event_id"]) for event in full_manifest]
    existing_checkpoints = load_valid_checkpoints(checkpoint_dir, event_ids, binding)
    completed_ids = {
        event_id
        for checkpoint in existing_checkpoints
        for event_id in checkpoint["event_ids"]
    }
    remaining = [event for event in manifest if str(event["event_id"]) not in completed_ids]
    pending = chunks(remaining, args.batch_size)
    scope_ids = [str(event["event_id"]) for event in manifest]
    already_complete = sum(event_id in completed_ids for event_id in scope_ids)
    print(
        f"events: {already_complete:,}/{len(manifest):,} checkpointed; "
        f"{len(remaining):,} remaining; batch_size={args.batch_size}",
        flush=True,
    )

    conn = None
    started = time.time()
    run_events = 0
    try:
        if pending:
            print(
                f"connecting to CCTS database with connect_timeout={args.connect_timeout}s",
                flush=True,
            )
            conn = psycopg2.connect(
                host=creds["host"],
                port=args.port,
                dbname=creds["dbname"],
                user=creds["user"],
                password=creds["password"],
                connect_timeout=args.connect_timeout,
                sslmode=args.sslmode,
                application_name="uzh_thesis_ccts_turn_batch_fetch",
            )
            with conn.cursor() as cur:
                print(f"connected. setting statement_timeout={args.statement_timeout_ms}ms", flush=True)
                cur.execute("SET statement_timeout = %s", (args.statement_timeout_ms,))
                last_reported = already_complete
                for run_batch_number, batch in enumerate(pending, start=1):
                    event_ids = [str(event["event_id"]) for event in batch]
                    key = checkpoint_key(event_ids)
                    print(
                        f"starting remaining batch {run_batch_number:,}/{len(pending):,} "
                        f"({len(batch)} events; completed={already_complete + run_events:,}/{len(manifest):,})",
                        flush=True,
                    )
                    turns, audits = fetch_batch(cur, batch)
                    save_checkpoint(checkpoint_dir, key, event_ids, turns, audits, binding)
                    run_events += len(batch)
                    done = already_complete + run_events
                    elapsed = max(time.time() - started, 0.001)
                    rate = run_events / elapsed
                    eta = format_duration((len(manifest) - done) / rate) if rate else "unknown"
                    report_progress = (
                        run_batch_number == 1
                        or done - last_reported >= max(args.batch_size * 10, 100)
                        or done == len(manifest)
                    )
                    if report_progress:
                        print(
                            f"progress {done:,}/{len(manifest):,} ({100 * done / len(manifest):.1f}%; "
                            f"{rate:.2f} events/sec; eta {eta})",
                            flush=True,
                        )
                        last_reported = done
    finally:
        if conn is not None:
            conn.close()
        summary = write_outputs(
            args.output_dir,
            args.manifest,
            full_manifest,
            checkpoint_dir,
            scope_ids,
            binding,
        )

    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
