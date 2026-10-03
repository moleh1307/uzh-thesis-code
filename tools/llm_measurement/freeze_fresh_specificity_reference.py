#!/usr/bin/env python3
"""Archive the fresh 100 human reference without imputing source missingness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()


HUMAN_FIELDS = {"human_content_class", "human_ok", "human_specificity", "human_notes"}
MISSING_ID = "FRESHV1_0053"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_rows(data: bytes) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Missing or duplicate CSV headers")
    rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError("Malformed CSV row")
    return rows


def validate_reference(coded, blinded, requests, scores, classes, revisions, confirmation):
    expected_ids = [f"FRESHV1_{i:04d}" for i in range(1, 101)]
    for name, rows, field in (("coded", coded, "audit_id"),
                              ("blinded", blinded, "audit_id"),
                              ("requests", requests, "custom_id")):
        if [r[field] for r in rows] != expected_ids:
            raise ValueError(f"{name}: expected exactly 100 unique IDs in frozen order")
    if len(scores["scores_in_audit_id_order"]) != 100 or len(classes["classes_in_audit_id_order"]) != 100:
        raise ValueError("Original submissions must contain 100 entries")
    final_classes = dict(zip(expected_ids, classes["classes_in_audit_id_order"]))
    changes = revisions["revised_classes"]
    if not set(changes).issubset(final_classes):
        raise ValueError("Unknown revised ID")
    final_classes.update(changes)
    if confirmation["audit_id"] != "FRESHV1_0008" or confirmation["confirmed_class"] != "substantive":
        raise ValueError("Unexpected final class confirmation")
    final_classes[confirmation["audit_id"]] = confirmation["confirmed_class"]
    if Counter(r["unit_type"] for r in coded) != {"pre": 40, "qa": 60}:
        raise ValueError("Expected 40 PRE and 60 Q&A rows")
    ledger = []
    for i, (row, source, request) in enumerate(zip(coded, blinded, requests)):
        audit_id = row["audit_id"]
        if set(row) != set(source) or not HUMAN_FIELDS.issubset(row):
            raise ValueError(f"CSV schema mismatch: {audit_id}")
        if any(row[k] != source[k] for k in row if k not in HUMAN_FIELDS):
            raise ValueError(f"Source or identity field changed: {audit_id}")
        value = scores["scores_in_audit_id_order"][i]
        expected_score = "" if value is None else str(value)
        expected_ok = "" if value is None else ("0" if value == 0 else "1")
        if row["human_specificity"] != expected_score or row["human_ok"] != expected_ok:
            raise ValueError(f"Submitted score/status changed: {audit_id}")
        label = row["human_content_class"]
        if label != final_classes[audit_id]:
            raise ValueError(f"Class differs from authorized revision history: {audit_id}")
        if audit_id == MISSING_ID:
            if label != "uncertain" or expected_score or expected_ok or not row["human_notes"].strip():
                raise ValueError("0053 must retain uncertain, blank score/status and source note")
            state = "source_uninterpretable"
        elif label == "procedural_only":
            if audit_id != "FRESHV1_0014" or expected_score != "0" or expected_ok != "0":
                raise ValueError("0014 must retain procedural_only, status 0 and score 0")
            state = "procedural_only"
        elif label in {"substantive", "mixed"} and expected_ok == "1" and expected_score in {"1", "2", "3", "4", "5"}:
            state = "scorable"
        else:
            raise ValueError(f"Unexpected reference state: {audit_id}")
        target = json.loads(next(m["content"] for m in request["messages"] if m["role"] == "user"))
        expected_target = {"unit_type": row["unit_type"]}
        if row["unit_type"] == "pre":
            expected_target["ceo_presentation_segment"] = row["ceo_presentation_segment"]
        else:
            expected_target.update(analyst_question=row["analyst_question"], ceo_answer=row["ceo_answer"])
        if target != expected_target:
            raise ValueError(f"Frozen model input differs: {audit_id}")
        ledger.append({"audit_id": audit_id, "unit_type": row["unit_type"],
                       "reference_state": state, "human_content_class": label,
                       "human_ok": row["human_ok"], "human_specificity": row["human_specificity"],
                       "include_primary_eligibility": int(state != "source_uninterpretable"),
                       "include_primary_numeric": int(state == "scorable"),
                       "retain_model_output_diagnostic": 1})
    return ledger


def freeze_reference(coded_csv: Path, setup: Path, project: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen reference: {output}")
    sources = {
        "human_reference_frozen.csv": coded_csv,
        "blinded_input_snapshot.csv": setup / "blinded_input_snapshot.csv",
        "setup_freeze_manifest.json": setup / "freeze_manifest.json",
        "original_score_submission.json": project / "notes/2026-10-03-human-score-submission.json",
        "original_class_submission.json": project / "notes/2026-10-03-human-content-class-submission.json",
        "authorized_class_revisions.json": project / "notes/2026-10-03-human-content-class-revisions.json",
        "final_0008_confirmation.json": project / "notes/2026-10-03-human-content-class-0008-confirmation.json",
        "content_class_convention.md": project / "methods/llm_measurement/human-content-class-convention-20261003.md",
        "human_rating_instructions.md": coded_csv.parent / "README.md",
        "0053_source_trace.md": project / "reviews/20261003_fresh0053_source_trace.md",
        "freeze_reference_builder.py": Path(__file__).resolve(),
    }
    snapshots = {name: path.read_bytes() for name, path in sources.items()}
    setup_manifest = json.loads(snapshots["setup_freeze_manifest.json"])
    for name, expected in setup_manifest["artifact_sha256"].items():
        if digest((setup / name).read_bytes()) != expected:
            raise ValueError(f"Frozen setup artifact changed: {name}")
    requests_bytes = (setup / "specificity_requests_100.jsonl").read_bytes()
    requests = [json.loads(line) for line in requests_bytes.decode().splitlines() if line.strip()]
    coded = csv_rows(snapshots["human_reference_frozen.csv"])
    ledger = validate_reference(coded, csv_rows(snapshots["blinded_input_snapshot.csv"]), requests,
                                json.loads(snapshots["original_score_submission.json"]),
                                json.loads(snapshots["original_class_submission.json"]),
                                json.loads(snapshots["authorized_class_revisions.json"]),
                                json.loads(snapshots["final_0008_confirmation.json"]))
    manifest = {
        "status": "human_reference_frozen_with_documented_source_missingness",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "Record project-specific approval separately before use; this code does not grant authorization",
        "model_run": False, "remote_deployment": False,
        "setup_directory": str(setup), "setup_manifest_sha256": digest(snapshots["setup_freeze_manifest.json"]),
        "model_requests_sha256": digest(requests_bytes),
        "counts": {"all_ids": 100, "pre": 40, "qa": 60,
                   "classes": dict(Counter(r["human_content_class"] for r in coded)),
                   "reference_states": dict(Counter(r["reference_state"] for r in ledger)),
                   "primary_eligibility_reference": sum(r["include_primary_eligibility"] for r in ledger),
                   "primary_numeric_reference": sum(r["include_primary_numeric"] for r in ledger)},
        "reference_limits": "One human rater; discussion-assisted class revisions; not ground truth or independent human validation.",
        "missingness_policy": {
            MISSING_ID: "Retain blank status/score; exclude primary eligibility and numeric comparisons; retain source and any model output diagnostically. Not imputed 0 or replaced.",
            "FRESHV1_0014": "Retain procedural_only/ok=0/score=0; include eligibility comparison, exclude 1-5 numeric comparisons.",
            "numeric_model_missingness": "Compare numeric scores only where human and model are both scorable/valid. Report excluded IDs, eligibility disagreements and model failures separately; never hide them by restricting denominators.",
            "call_contrasts": "Use identical eligible units and original CEO-word weights for human/model comparisons, documenting exclusions and per-call coverage. This sample is not complete calls.",
        },
        "live_source_unchanged": True, "dashboard_progress_used": False,
        "privacy": "Local archival reference only. Do not deploy human labels, revision notes or review findings with model inputs.",
        "artifacts": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".reference-stage-", dir=output.parent))
    try:
        for name, data in snapshots.items():
            (stage / name).write_bytes(data)
        with (stage / "comparison_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
            writer.writeheader()
            writer.writerows(ledger)
        (stage / "README.md").write_text(
            "# Fixed Human Reference\n\n"
            "100 original IDs retained: 98 scorable, 0014 procedural-only/0, "
            "0053 source-uninterpretable with blank status and score.\n\n"
            "Use human_reference_frozen.csv, not the editable CSV or stale dashboard JSON. "
            "comparison_ledger.csv defines comparison support. Preserve all 100 model "
            "outputs diagnostically; 0053 is not a numerical zero.\n\n"
            "These are one person's ratings, with discussion-assisted class revisions, "
            "not infallible ground truth. No model was run. No new recoding or wait is required.\n\n"
            "The existing setup manifest remains unchanged as a historical snapshot. "
            "This reference manifest establishes completion of the label-freeze step only. "
            "Before execution, deploy and verify the exact frozen runner and inputs and "
            "complete checkpoint/GPU preflight. Do not upload this human reference to the scoring job.\n",
            encoding="utf-8")
        for path in sorted(stage.iterdir()):
            manifest["artifacts"][path.name] = {"sha256": digest(path.read_bytes()),
                "bytes": path.stat().st_size, "source": str(sources[path.name]) if path.name in sources else "generated"}
        for name, path in sources.items():
            if path.read_bytes() != snapshots[name]:
                raise ValueError(f"Source changed during archival: {path}")
        if digest(requests_bytes) != digest((setup / "specificity_requests_100.jsonl").read_bytes()):
            raise ValueError("Model requests changed during archival")
        (stage / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in stage.iterdir():
            path.chmod(0o444)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded-csv", type=Path, required=True)
    parser.add_argument("--setup-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_reference(args.coded_csv, args.setup_dir, args.project_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "status": manifest["status"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
