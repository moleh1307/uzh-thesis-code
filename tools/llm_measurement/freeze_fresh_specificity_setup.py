#!/usr/bin/env python3
"""Freeze the reviewed definition and existing blinded inputs; never run a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()

from prepare_fresh_evaluation_human_package import validate_source_checksums
from run_local_specificity import sha256_path, validate_request


def freeze(proposal: Path, package: Path, evaluation: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"freeze already exists; not replacing it: {output}")
    package_manifest = package / "fresh_evaluation_human_package_manifest.json"
    original = json.loads(package_manifest.read_text())
    for key, expected in original["artifacts_sha256"].items():
        path = Path(original["artifacts"][key])
        if sha256_path(path) != expected:
            raise ValueError(f"original package artifact differs: {key}")
    summary_path = evaluation / "evaluation_manifest_summary.json"
    summary = json.loads(summary_path.read_text())
    print("Checking the existing evaluation's source checksums...", flush=True)
    source_paths = validate_source_checksums(summary)
    blinded = package / "fresh_evaluation_human_blinded.csv"
    with blinded.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts = {kind: sum(r["unit_type"] == kind for r in rows) for kind in ("pre", "qa")}
    if len(rows) != 100 or len({r["audit_id"] for r in rows}) != 100 or counts != {"pre": 40, "qa": 60}:
        raise ValueError("evaluation is not the reviewed 100-unit scope")
    if any(r[field].strip() for r in rows for field in (
        "human_content_class", "human_ok", "human_specificity", "human_notes"
    )):
        raise ValueError("model input source is not the unchanged blank blinded file")
    prompt = (proposal / "specificity-system-prompt.txt").read_text()
    settings = json.loads((proposal / "proposed-settings.json").read_text())
    schema = json.loads((proposal / "specificity-production-schema.json").read_text())
    frozen_anchor = proposal.parent / "v1/specificity-system-prompt.txt"
    if any(p not in prompt for p in frozen_anchor.read_text().strip().split("\n\n")[:3]):
        raise ValueError("original scoring anchors or context rules changed")
    requests = []
    for row in rows:
        if row["unit_type"] == "pre":
            user = {"unit_type": "pre", "ceo_presentation_segment": row["ceo_presentation_segment"]}
        else:
            user = {"unit_type": "qa", "analyst_question": row["analyst_question"], "ceo_answer": row["ceo_answer"]}
        request = {"custom_id": row["audit_id"], "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
        ], "response_schema": schema}
        validate_request(request)
        requests.append(request)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fresh100-freeze-", dir=output.parent) as temporary:
        temp = Path(temporary) / "packet"
        temp.mkdir(mode=0o700)
        names = ("review-plan.md", "specificity-system-prompt.txt", "proposed-settings.json", "specificity-production-schema.json")
        for name in names:
            shutil.copyfile(proposal / name, temp / name)
        for name in ("run_local_specificity.py", "freeze_fresh_specificity_setup.py",
                     "specificity_validation.py", "local_execution_identity.py"):
            shutil.copyfile(Path(__file__).with_name(name), temp / name)
        shutil.copyfile(blinded, temp / "blinded_input_snapshot.csv")
        input_path = temp / "specificity_requests_100.jsonl"
        with input_path.open("x", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
        sources = [package_manifest, blinded, summary_path,
                   evaluation / "evaluation_unit_manifest.csv", evaluation / "evaluation_call_manifest.csv",
                   Path(original["artifacts"]["private_key_csv"]), *source_paths.values(),
                   *(proposal / name for name in names)]
        manifest = {
            "status": "definition_and_input_frozen_pending_human_labels_and_remote_preflight",
            "contract": "specificity_fresh100_complete_target_20261003",
            "client_date": "2026-10-03", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": "Record project-specific approval separately before use; this code does not grant authorization",
            "units": 100, "pre_units": 40, "qa_units": 60,
            "model": settings["model"], "decoding": settings["decoding"],
            "human_labels_frozen": False, "model_run": False, "remote_preflight_complete": False,
            "live_human_coded_csv": str(package / "fresh_evaluation_human_coded.csv"),
            "live_human_progress_json": str(package / "fresh_evaluation_human_progress.json"),
            "original_proposal_copies": "Preserved verbatim; this manifest records the new freeze status",
            "source_sha256": {str(path): sha256_path(path) for path in dict.fromkeys(sources)},
            "artifact_sha256": {path.name: sha256_path(path) for path in sorted(temp.iterdir())},
            "privacy": "Licensed text local only; private key hashed, not read or copied; no human labels in model input",
            "next": "Complete and freeze existing blinded labels, verify runtime/checkpoint, then run the frozen evaluation; no automatic execution",
        }
        (temp / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in temp.iterdir():
            path.chmod(0o600)
        os.replace(temp, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposal", "package", "evaluation", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    result = freeze(args.proposal.resolve(), args.package.resolve(), args.evaluation.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "units": result["units"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
