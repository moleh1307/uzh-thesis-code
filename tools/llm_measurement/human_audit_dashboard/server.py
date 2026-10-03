#!/usr/bin/env python3
"""Serve the local blinded specificity-coding dashboard with atomic persistence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


AUDIT_COLUMNS = [
    "audit_id",
    "audit_split",
    "unit_type",
    "ceo_presentation_segment",
    "analyst_question",
    "ceo_answer",
    "human_ok",
    "human_specificity",
    "human_notes",
]
CONTENT_CLASS_AUDIT_COLUMNS = [
    "audit_id",
    "audit_split",
    "unit_type",
    "ceo_presentation_segment",
    "analyst_question",
    "ceo_answer",
    "human_content_class",
    "human_ok",
    "human_specificity",
    "human_notes",
]
RECODE_COLUMNS = [
    "recode_id",
    "unit_type",
    "ceo_presentation_segment",
    "analyst_question",
    "ceo_answer",
    "human_ok",
    "human_specificity",
    "human_notes",
]
PUBLIC_COLUMNS = [
    "audit_id",
    "unit_type",
    "ceo_presentation_segment",
    "analyst_question",
    "ceo_answer",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str], str, str, bool]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if fieldnames == AUDIT_COLUMNS:
            output_columns = AUDIT_COLUMNS
            id_column = "audit_id"
            export_filename = "specificity_human_audit_coded.csv"
            content_class_required = False
        elif fieldnames == CONTENT_CLASS_AUDIT_COLUMNS:
            output_columns = CONTENT_CLASS_AUDIT_COLUMNS
            id_column = "audit_id"
            export_filename = "fresh_evaluation_human_coded.csv"
            content_class_required = True
        elif fieldnames == RECODE_COLUMNS:
            output_columns = RECODE_COLUMNS
            id_column = "recode_id"
            export_filename = "specificity_human_recode60_coded.csv"
            content_class_required = False
        else:
            raise SystemExit(
                f"unexpected blinded CSV columns: {reader.fieldnames}; expected either "
                f"{AUDIT_COLUMNS}, {CONTENT_CLASS_AUDIT_COLUMNS}, or {RECODE_COLUMNS}"
            )
        rows = list(reader)
    if not rows:
        raise SystemExit("blinded CSV has no rows")
    ids = [row[id_column] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate {id_column} in blinded CSV")
    if any(row["unit_type"] not in {"pre", "qa"} for row in rows):
        raise SystemExit("unexpected unit_type in blinded CSV")
    if id_column == "recode_id":
        for row in rows:
            row["audit_id"] = row["recode_id"]
    return rows, output_columns, id_column, export_filename, content_class_required


def atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PersistenceConflict(ValueError):
    """Stop before writing when annotation files disagree or change externally."""


class AuditStore:
    def __init__(self, input_path: Path, progress_path: Path, coded_csv_path: Path,
                 *, label_source: str = "dashboard"):
        self.input_path = input_path.resolve()
        self.progress_path = progress_path.resolve()
        self.coded_csv_path = coded_csv_path.resolve()
        if len({self.input_path, self.progress_path, self.coded_csv_path}) != 3:
            raise PersistenceConflict("input, progress and coded CSV must be distinct files")
        if label_source not in {"dashboard", "coded-csv"}:
            raise ValueError("label_source must be dashboard or coded-csv")
        self.label_source = label_source
        (
            self.rows,
            self.output_columns,
            self.id_column,
            self.export_filename,
            self.content_class_required,
        ) = read_rows(self.input_path)
        self.rows_by_id = {row["audit_id"]: row for row in self.rows}
        self.source_sha256 = sha256(self.input_path)
        self.lock = threading.Lock()
        self.labels: dict[str, dict[str, Any]] = {}
        self.created_at = utc_now()
        self.updated_at = self.created_at
        self._load_progress()
        if self.coded_csv_path.exists():
            csv_labels = self._read_coded_labels()
            if label_source == "coded-csv":
                if self.progress_path.exists() and self._label_values(csv_labels) != self._label_values(self.labels):
                    backup = self.progress_path.with_name(
                        self.progress_path.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    )
                    atomic_text_write(backup, self.progress_path.read_text(encoding="utf-8"))
                self.labels = csv_labels
                self.updated_at = utc_now()
                self._write_progress()
            elif self._label_values(csv_labels) != self._label_values(self.labels):
                raise PersistenceConflict(
                    "coded CSV and progress labels disagree (or CSV labels have no progress file); "
                    "both files are unchanged. Review them, then use --label-source coded-csv "
                    "only if the CSV is the intended source"
                )
        elif label_source == "coded-csv":
            raise PersistenceConflict("--label-source coded-csv requires an existing coded CSV")
        if not self.coded_csv_path.exists():
            self._write_coded_csv()
        self.file_hashes = self._file_hashes()

    def _load_progress(self) -> None:
        if not self.progress_path.exists():
            return
        data = json.loads(self.progress_path.read_text(encoding="utf-8"))
        if data.get("source_sha256") != self.source_sha256:
            raise SystemExit(
                "progress file belongs to a different blinded CSV; move it aside or use the matching input"
            )
        labels = data.get("labels", {})
        if not isinstance(labels, dict) or any(key not in self.rows_by_id for key in labels):
            raise SystemExit("progress file contains invalid audit IDs")
        for audit_id, label in labels.items():
            if not isinstance(label, dict) or "clear" in label:
                raise PersistenceConflict("progress contains an invalid label")
            try:
                _, validated = self.validate_label(dict(label, audit_id=audit_id))
            except ValueError as exc:
                raise PersistenceConflict(f"invalid progress label for {audit_id}: {exc}") from exc
            if validated is None:
                raise PersistenceConflict("progress contains an invalid cleared label")
            validated["saved_at"] = label.get("saved_at", validated["saved_at"])
            self.labels[audit_id] = validated
        self.created_at = str(data.get("created_at") or self.created_at)
        self.updated_at = str(data.get("updated_at") or self.updated_at)

    def _label_values(self, labels: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        fields = [field for field in self.output_columns if field.startswith("human_")]
        return {key: {field: label.get(field, "") for field in fields}
                for key, label in labels.items()}

    def _read_coded_labels(self) -> dict[str, dict[str, Any]]:
        rows, columns, _, _, _ = read_rows(self.coded_csv_path)
        if columns != self.output_columns or [row["audit_id"] for row in rows] != list(self.rows_by_id):
            raise PersistenceConflict("coded CSV schema or ordered IDs differ from blinded input")
        labels = {}
        for row, source in zip(rows, self.rows):
            if None in row or any(row.get(field) is None for field in columns):
                raise PersistenceConflict("coded CSV contains an incomplete or extra-field row")
            if any(row[field] != source[field] for field in columns if not field.startswith("human_")):
                raise PersistenceConflict("coded CSV source text/metadata differs from blinded input")
            fields = {field: row[field] for field in columns if field.startswith("human_")}
            if not any(fields.values()):
                continue
            try:
                payload = dict(fields, audit_id=row["audit_id"])
                payload["human_ok"] = int(fields["human_ok"]) if fields["human_ok"] else ""
                payload["human_specificity"] = int(fields["human_specificity"]) if fields["human_specificity"] else ""
                _, label = self.validate_label(payload)
            except ValueError as exc:
                raise PersistenceConflict(f"invalid coded CSV label for {row['audit_id']}: {exc}") from exc
            labels[row["audit_id"]] = label
        return labels

    def _file_hashes(self) -> dict[Path, str | None]:
        return {path: sha256(path) if path.exists() else None
                for path in (self.input_path, self.progress_path, self.coded_csv_path)}

    def _assert_files_unchanged(self) -> None:
        if self._file_hashes() != self.file_hashes:
            raise PersistenceConflict(
                "annotation files changed outside this dashboard; save/export stopped without "
                "overwriting them. Stop the dashboard and review/import the CSV explicitly"
            )

    def validate_label(self, payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        audit_id = str(payload.get("audit_id", "")).strip()
        if payload.get("clear") is True:
            return audit_id, None
        human_ok = payload.get("human_ok")
        specificity = payload.get("human_specificity")
        content_class = str(payload.get("human_content_class", "")).strip()
        notes = str(payload.get("human_notes", "")).strip()
        if len(notes) > 2000:
            raise ValueError("human_notes exceeds 2000 characters")
        source_missing = (self.content_class_required and content_class == "uncertain"
                          and human_ok in (None, "") and specificity in (None, ""))
        if source_missing:
            if not isinstance(payload.get("human_notes"), str) or not notes:
                raise ValueError("source-uninterpretable rows require a source-quality reason")
            return audit_id, {"human_ok": "", "human_specificity": "",
                              "human_content_class": "uncertain", "human_notes": notes,
                              "annotation_state": "source_uninterpretable", "saved_at": utc_now()}
        if type(human_ok) is not int or human_ok not in (0, 1):
            raise ValueError("human_ok must be 0 or 1")
        if type(specificity) is not int:
            raise ValueError("human_specificity must be an integer")
        if self.content_class_required:
            allowed_classes = {"substantive", "mixed", "procedural_only", "uncertain"}
            if content_class not in allowed_classes:
                raise ValueError(
                    "human_content_class must be substantive, mixed, procedural_only, or uncertain"
                )
            if content_class == "procedural_only":
                if human_ok != 0 or specificity != 0:
                    raise ValueError("procedural_only rows require human_ok=0 and specificity=0")
            elif human_ok != 1 or specificity not in range(1, 6):
                raise ValueError(
                    "substantive, mixed, and uncertain rows require human_ok=1 and specificity 1-5"
                )
        else:
            if human_ok == 1 and specificity not in range(1, 6):
                raise ValueError("scorable rows require specificity 1-5")
            if human_ok == 0 and specificity != 0:
                raise ValueError("unscorable rows require specificity 0")
        label: dict[str, Any] = {
            "human_ok": human_ok,
            "human_specificity": specificity,
            "human_notes": notes,
            "saved_at": utc_now(),
            "annotation_state": "scored" if human_ok == 1 else "procedural_only",
        }
        if self.content_class_required:
            label["human_content_class"] = content_class
        return audit_id, label

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        audit_id, label = self.validate_label(payload)
        if audit_id not in self.rows_by_id:
            raise ValueError("unknown audit_id")
        with self.lock:
            self._assert_files_unchanged()
            if label is None:
                self.labels.pop(audit_id, None)
            else:
                self.labels[audit_id] = label
            self.updated_at = utc_now()
            self._persist()
            self.file_hashes = self._file_hashes()
            return self.stats()

    def _persist(self) -> None:
        self._write_progress()
        self._write_coded_csv()

    def _write_progress(self) -> None:
        state = {
            "version": 1,
            "source_csv": str(self.input_path),
            "source_sha256": self.source_sha256,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "labels": self.labels,
            "label_source_at_startup": self.label_source,
        }
        atomic_text_write(self.progress_path, json.dumps(state, indent=2) + "\n")

    def export_csv_text(self) -> str:
        with self.lock:
            self._assert_files_unchanged()
            return self._coded_csv_text()

    def _coded_csv_text(self) -> str:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(
            output,
            fieldnames=self.output_columns,
            extrasaction="ignore",
        )
        writer.writeheader()
        for source in self.rows:
            row = dict(source)
            label = self.labels.get(source["audit_id"])
            if label:
                row["human_ok"] = label["human_ok"]
                row["human_specificity"] = label["human_specificity"]
                row["human_notes"] = label["human_notes"]
                if self.content_class_required:
                    row["human_content_class"] = label["human_content_class"]
            else:
                row["human_ok"] = ""
                row["human_specificity"] = ""
                row["human_notes"] = ""
                if self.content_class_required:
                    row["human_content_class"] = ""
            writer.writerow(row)
        return output.getvalue()

    def _write_coded_csv(self) -> None:
        atomic_text_write(self.coded_csv_path, self._coded_csv_text())

    def stats(self) -> dict[str, int]:
        reviewed = len(self.labels)
        return {
            "total": len(self.rows),
            "reviewed": reviewed,
            "scored": sum(label["human_ok"] == 1 for label in self.labels.values()),
            "source_missing": sum(label.get("annotation_state") == "source_uninterpretable" for label in self.labels.values()),
            "procedural_only": sum(label["human_ok"] == 0 for label in self.labels.values()),
            "remaining": len(self.rows) - reviewed,
            "pre_total": sum(row["unit_type"] == "pre" for row in self.rows),
            "qa_total": sum(row["unit_type"] == "qa" for row in self.rows),
            "pre_scored": sum(
                row["unit_type"] == "pre" and self.labels.get(row["audit_id"], {}).get("human_ok") == 1
                for row in self.rows
            ),
            "qa_scored": sum(
                row["unit_type"] == "qa" and self.labels.get(row["audit_id"], {}).get("human_ok") == 1
                for row in self.rows
            ),
            "pre_reviewed": sum(row["unit_type"] == "pre" and row["audit_id"] in self.labels for row in self.rows),
            "qa_reviewed": sum(row["unit_type"] == "qa" and row["audit_id"] in self.labels for row in self.rows),
        }

    def public_state(self) -> dict[str, Any]:
        public_rows = []
        for row in self.rows:
            item = {field: row[field] for field in PUBLIC_COLUMNS}
            item["label"] = self.labels.get(row["audit_id"])
            public_rows.append(item)
        return {
            "contract": (
                "specificity_v1_20260720_delayed_recode"
                if self.id_column == "recode_id"
                else (
                    "specificity_current_source_clean_evaluation_v1"
                    if self.content_class_required
                    else "specificity_v1_20260720"
                )
            ),
            "content_class_required": self.content_class_required,
            "rows": public_rows,
            "stats": self.stats(),
            "updated_at": self.updated_at,
        }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "UZHSpecificityAudit/1.0"

    @property
    def app(self) -> "DashboardServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_bytes(
        self,
        content: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_bytes(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json({"ok": True, "stats": self.app.store.stats()})
            return
        if path == "/api/state":
            self.send_json(self.app.store.public_state())
            return
        if path == "/api/export":
            try:
                content = self.app.store.export_csv_text().encode("utf-8")
            except PersistenceConflict as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
                return
            self.send_bytes(
                content,
                "text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": (
                        f'attachment; filename="{self.app.store.export_filename}"'
                    )
                },
            )
            return
        static = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path in static:
            filename, content_type = static[path]
            self.send_bytes((self.app.static_dir / filename).read_bytes(), content_type)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            stats = self.app.store.save(payload)
            self.send_json({"ok": True, "stats": stats})
        except PersistenceConflict as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], static_dir: Path, store: AuditStore):
        super().__init__(address, DashboardHandler)
        self.static_dir = static_dir
        self.store = store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--coded-csv", required=True, type=Path)
    parser.add_argument("--label-source", choices=("dashboard", "coded-csv"), default="dashboard",
                        help="default: require CSV/progress agreement; coded-csv: explicitly import CSV labels")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    static_dir = Path(__file__).resolve().parent
    try:
        store = AuditStore(args.input, args.progress, args.coded_csv, label_source=args.label_source)
    except PersistenceConflict as exc:
        raise SystemExit(str(exc)) from exc
    server = DashboardServer((args.host, args.port), static_dir, store)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "input": str(args.input.resolve()),
                "progress": str(args.progress.resolve()),
                "coded_csv": str(args.coded_csv.resolve()),
                "label_source": args.label_source,
                "mode": (
                    "delayed_recode"
                    if store.id_column == "recode_id"
                    else (
                        "fresh_evaluation_audit"
                        if store.content_class_required
                        else "initial_audit"
                    )
                ),
                "stats": store.stats(),
            },
            indent=2,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
