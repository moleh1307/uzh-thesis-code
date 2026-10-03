"""Schema-bound atomic census count batches; uncomputed counts never become zero."""
import csv
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_json(path, value):
    atomic_csv_or_json(path, json.dumps(value, sort_keys=True) + "\n")


def atomic_csv_or_json(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".census-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def integer(value):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError("count/ID must be a nonnegative integer")
    return int(value)


def validate_rows(rows, columns, allowed_ids):
    result = {}
    for row in rows:
        if set(row) != set(columns) or any(row[column] is None for column in columns):
            raise ValueError("incomplete or extra census count cells")
        try:
            event = integer(row["event_id"])
            if event <= 0 or (allowed_ids is not None and event not in allowed_ids) or event in result:
                raise ValueError("duplicate, nonpositive or out-of-scope census event ID")
            parsed = {column: (str(row[column]) if column == "text_types" else integer(row[column]))
                      for column in columns if column != "event_id"}
            if any(value < 0 for column, value in parsed.items() if column != "text_types"):
                raise ValueError("negative census count")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid census count row: {exc}") from exc
        result[event] = parsed
    return result


def read_counts(path, columns, allowed_ids):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError("census count schema/mode mismatch")
        return validate_rows(reader, columns, allowed_ids)


class CountCheckpoint:
    def __init__(self, output, columns, event_ids, source_binding, *, create=True):
        self.output, self.columns = Path(output), list(columns)
        self.event_ids = list(event_ids)
        if len(set(event_ids)) != len(event_ids) or any(event <= 0 for event in event_ids):
            raise ValueError("invalid census scope IDs")
        self.config_path = self.output.with_suffix(".checkpoint.json")
        self.shards = self.output.with_suffix(".checkpoints")
        self.contract = {"schema": 1, "columns": self.columns,
                         "event_order_sha256": digest(self.event_ids), "source_binding": source_binding}
        self.binding_hash = digest(self.contract)
        self.counts = {}
        if self.config_path.exists():
            if json.loads(self.config_path.read_text()) != self.contract:
                raise ValueError("census checkpoint mode/schema/scope/source/configuration mismatch")
        elif self.output.exists() or (self.shards.exists() and any(self.shards.iterdir())):
            raise ValueError("unbound historical census checkpoint; use a fresh output directory")
        for marker_path in sorted(self.shards.glob("*.done.json")):
            marker = json.loads(marker_path.read_text())
            key = marker_path.name.removesuffix(".done.json")
            csv_path = self.shards / (key + ".csv")
            if (marker.get("run_binding_sha256") != self.binding_hash or not csv_path.is_file()
                    or marker.get("csv_sha256") != file_hash(csv_path)):
                raise ValueError("census batch checkpoint identity/hash mismatch")
            batch = read_counts(csv_path, self.columns, set(self.event_ids))
            if list(batch) != marker.get("event_ids") or self.counts.keys() & batch.keys():
                raise ValueError("duplicate or inconsistent census batch coverage")
            self.counts.update(batch)
        if self.output.exists():
            saved = read_counts(self.output, self.columns, set(self.event_ids))
            if list(saved) != self.event_ids or saved != self.counts:
                raise ValueError("derived census counts differ from committed batch evidence")
        if create:
            self.shards.mkdir(parents=True, exist_ok=True)
            if not self.config_path.exists():
                atomic_json(self.config_path, self.contract)

    def commit(self, batch_ids, values):
        if not batch_ids or set(values) - set(batch_ids):
            raise ValueError("empty batch or query returned out-of-batch events")
        if self.counts.keys() & set(batch_ids):
            raise ValueError("census batch already committed")
        # A missing GROUP BY event is measured zero; a missing cell in a returned row is not.
        rows = []
        for event in batch_ids:
            data = values[event] if event in values else {
                column: "" if column == "text_types" else 0 for column in self.columns[1:]}
            if set(data) != set(self.columns[1:]):
                raise ValueError("query returned uncomputed/missing census signal cells")
            rows.append({"event_id": event, **data})
        parsed = validate_rows(rows, self.columns, set(self.event_ids))
        key = digest(batch_ids)
        csv_path = self.shards / (key + ".csv")
        self.write_csv(csv_path, rows)
        atomic_json(self.shards / (key + ".done.json"), {
            "run_binding_sha256": self.binding_hash, "event_ids": list(batch_ids),
            "csv_sha256": file_hash(csv_path)})
        self.counts.update(parsed)

    def write_csv(self, path, rows):
        fd, name = tempfile.mkstemp(prefix=".census-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.columns)
                writer.writeheader()
                writer.writerows(rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def materialize(self):
        if set(self.counts) != set(self.event_ids):
            raise ValueError("census stage incomplete; missing events cannot become zero")
        self.write_csv(self.output, ({"event_id": event, **self.counts[event]} for event in self.event_ids))


def prepare_run(path, binding, protected_paths):
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("binding") != binding:
            raise ValueError("census run mode/scope/source/configuration mismatch")
        return saved
    if any(item.exists() for item in protected_paths):
        raise ValueError("unbound historical census output; use a fresh output directory")
    saved = {"binding": binding, "metadata_sha256": None}
    atomic_json(path, saved)
    return saved


def publish_metadata(path, stage, config_path, config):
    config.update(metadata_sha256=file_hash(stage), metadata_publish_pending=True)
    atomic_json(config_path, config)
    recover_metadata(path, stage, config_path, config)


def recover_metadata(path, stage, config_path, config):
    if not config.get("metadata_publish_pending"):
        return
    expected = config["metadata_sha256"]
    if path.exists():
        if file_hash(path) != expected:
            raise ValueError("pending census metadata identity mismatch")
    elif stage.exists() and file_hash(stage) == expected:
        os.replace(stage, path)
    else:
        raise ValueError("pending census metadata is missing or damaged")
    config["metadata_publish_pending"] = False
    atomic_json(config_path, config)
