#!/usr/bin/env python3
"""Prepare a local blinded human-coding package from a fresh evaluation manifest.

The public blinded CSV deliberately contains licensed transcript text, so it
must remain local. It contains no source IDs, event IDs, company metadata,
human labels, or model outputs. The source join key is written separately
with owner-only permissions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from typing import Iterable, Mapping, Sequence


BLINDED_FIELDS = (
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
)

KEY_FIELDS = (
    "audit_id",
    "source_unit_id",
    "block_id",
    "event_id",
    "start_date",
    "calendar_quarter",
    "period_bin",
    "turnover_id",
    "turnover_side",
    "expected_execid",
    "company_id",
    "company_name",
    "unit_type",
    "sampling_bucket",
    "question_word_count",
    "target_word_count",
    "source_content_sha256",
    "source_integrity_lane",
)

MANIFEST_UNIT_FIELDS = (
    "evaluation_split",
    "split_rank",
    "unit_rank_within_call",
    "sampling_bucket",
    "unit_type",
    "source_unit_id",
    "block_id",
    "event_id",
    "start_date",
    "calendar_quarter",
    "period_bin",
    "turnover_id",
    "turnover_side",
    "expected_execid",
    "company_id",
    "company_name",
    "question_word_count",
    "target_word_count",
    "source_content_sha256",
    "eligibility_source_marker_free",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require_columns(
    rows: Sequence[Mapping[str, str]], required: Iterable[str], path: Path
) -> None:
    if not rows:
        raise SystemExit(f"CSV is empty: {path}")
    missing = sorted(set(required) - set(rows[0]))
    if missing:
        raise SystemExit(f"missing columns in {path}: {', '.join(missing)}")


def unique_index(
    rows: Sequence[dict[str, str]], key_field: str, path: Path
) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get(key_field, "").strip()
        if not key:
            raise SystemExit(f"blank {key_field} in {path}")
        if key in index:
            raise SystemExit(f"duplicate {key_field}={key!r} in {path}")
        index[key] = row
    return index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(seed: int, lane: str, value: str) -> str:
    return hashlib.sha256(f"{seed}|{lane}|{value}".encode("utf-8")).hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def validate_source_checksums(summary: Mapping[str, object]) -> dict[str, Path]:
    sources = summary.get("sources")
    if not isinstance(sources, dict):
        raise SystemExit("evaluation manifest summary has no sources object")
    required = {"pre_units_csv", "pre_lineage_csv", "qa_blocks_csv", "qa_lanes_csv"}
    missing = required - set(sources)
    if missing:
        raise SystemExit("evaluation manifest summary is missing sources: " + ", ".join(sorted(missing)))
    paths: dict[str, Path] = {}
    for label in required:
        item = sources[label]
        if not isinstance(item, dict):
            raise SystemExit(f"invalid source entry for {label}")
        path = Path(str(item.get("path", "")))
        expected = str(item.get("sha256", ""))
        if not path.is_file():
            raise SystemExit(f"source file is unavailable: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise SystemExit(
                f"source checksum changed for {label}: expected {expected}, observed {observed}"
            )
        paths[label] = path
    return paths


def assert_blinded_columns(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        header = tuple(next(csv.reader(handle)))
    if header != BLINDED_FIELDS:
        raise SystemExit(f"unexpected blinded CSV columns in {path}")
    forbidden = {
        "source_unit_id",
        "block_id",
        "event_id",
        "start_date",
        "turnover_id",
        "turnover_side",
        "expected_execid",
        "company_id",
        "company_name",
        "model_ok",
        "model_specificity",
    }
    leaked = forbidden & set(header)
    if leaked:
        raise SystemExit(f"blinded CSV leaks metadata fields: {sorted(leaked)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-manifest-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    manifest_dir = args.evaluation_manifest_dir.resolve()
    summary_path = manifest_dir / "evaluation_manifest_summary.json"
    units_path = manifest_dir / "evaluation_unit_manifest.csv"
    if not summary_path.is_file() or not units_path.is_file():
        raise SystemExit("evaluation manifest directory is missing its summary or unit manifest")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "fresh_current_source_marker_free_evaluation_manifest_unscored":
        raise SystemExit("evaluation manifest does not have the expected fresh-unscored status")
    sources = validate_source_checksums(summary)
    unit_rows = read_csv(units_path)
    require_columns(unit_rows, MANIFEST_UNIT_FIELDS, units_path)
    if not unit_rows:
        raise SystemExit("evaluation unit manifest is empty")
    if any(row["evaluation_split"] != "fresh_evaluation" for row in unit_rows):
        raise SystemExit("evaluation unit manifest contains a non-fresh split")
    if any(row["eligibility_source_marker_free"] != "1" for row in unit_rows):
        raise SystemExit("evaluation unit manifest contains a non-clean source marker lane")
    if len({row["source_unit_id"] for row in unit_rows}) != len(unit_rows):
        raise SystemExit("evaluation unit manifest contains duplicate source_unit_id values")

    pre_source_rows = read_csv(sources["pre_units_csv"])
    lineage_rows = read_csv(sources["pre_lineage_csv"])
    qa_source_rows = read_csv(sources["qa_blocks_csv"])
    qa_lane_rows = read_csv(sources["qa_lanes_csv"])
    require_columns(
        pre_source_rows,
        (
            "custom_id",
            "unit_type",
            "event_id",
            "source_sequence_id",
            "chunk_index",
            "unit_word_count",
            "ceo_presentation_segment",
        ),
        sources["pre_units_csv"],
    )
    require_columns(
        lineage_rows,
        (
            "custom_id",
            "event_id",
            "sequence_id",
            "chunk_index",
            "unit_word_count",
            "source_turn_reasons",
            "unit_reasons",
            "segment_sha256",
        ),
        sources["pre_lineage_csv"],
    )
    require_columns(
        qa_source_rows,
        (
            "block_id",
            "event_id",
            "quality_tier",
            "question_word_count",
            "ceo_answer_word_count",
            "analyst_question",
            "ceo_answer",
        ),
        sources["qa_blocks_csv"],
    )
    require_columns(
        qa_lane_rows,
        ("block_id", "event_id", "source_quality_tier", "diagnostic_lane", "reasons"),
        sources["qa_lanes_csv"],
    )
    pre_units = unique_index(pre_source_rows, "custom_id", sources["pre_units_csv"])
    pre_lineage = unique_index(lineage_rows, "custom_id", sources["pre_lineage_csv"])
    qa_blocks = unique_index(qa_source_rows, "block_id", sources["qa_blocks_csv"])
    qa_lanes = unique_index(qa_lane_rows, "block_id", sources["qa_lanes_csv"])

    public_rows: list[dict[str, object]] = []
    key_rows: list[dict[str, object]] = []
    ordered_units = sorted(
        unit_rows,
        key=lambda row: stable_key(args.seed, "fresh_evaluation_human_blind_order", row["source_unit_id"]),
    )
    for index, unit in enumerate(ordered_units, start=1):
        source_id = unit["source_unit_id"]
        unit_type = unit["unit_type"]
        if unit_type == "pre":
            source = pre_units.get(source_id)
            lineage = pre_lineage.get(source_id)
            if source is None or lineage is None:
                raise SystemExit(f"PRE source is unavailable for selected ID: {source_id}")
            if lineage["source_turn_reasons"].strip() or lineage["unit_reasons"].strip():
                raise SystemExit(f"PRE selected ID is not marker-free: {source_id}")
            if (
                source["unit_type"] != "pre"
                or source["event_id"] != unit["event_id"]
                or source["source_sequence_id"] != lineage["sequence_id"]
                or source["chunk_index"] != lineage["chunk_index"]
                or source["unit_word_count"] != unit["target_word_count"]
                or lineage["unit_word_count"] != unit["target_word_count"]
            ):
                raise SystemExit(f"PRE source metadata does not match selected manifest ID: {source_id}")
            target_text = source["ceo_presentation_segment"]
            if not target_text.strip():
                raise SystemExit(f"PRE selected ID has empty text: {source_id}")
            if (
                sha256_text(target_text) != lineage["segment_sha256"]
                or sha256_text(target_text) != unit["source_content_sha256"]
            ):
                raise SystemExit(f"PRE content hash mismatch for selected ID: {source_id}")
            presentation, question, answer = target_text, "", ""
        elif unit_type == "qa":
            source = qa_blocks.get(source_id)
            lane = qa_lanes.get(source_id)
            if source is None or lane is None:
                raise SystemExit(f"Q&A source is unavailable for selected ID: {source_id}")
            if (
                source["event_id"] != unit["event_id"]
                or lane["event_id"] != unit["event_id"]
                or lane["source_quality_tier"] != "high"
                or lane["diagnostic_lane"] != "automatic_candidate_not_approved"
                or lane["reasons"].strip()
                or source["quality_tier"] != "high"
                or source["ceo_answer_word_count"] != unit["target_word_count"]
                or source["question_word_count"] != unit["question_word_count"]
            ):
                raise SystemExit(f"Q&A selected ID is not in the clean automatic high lane: {source_id}")
            question, answer = source["analyst_question"], source["ceo_answer"]
            if not question.strip() or not answer.strip():
                raise SystemExit(f"Q&A selected ID has empty question or answer: {source_id}")
            if sha256_text(question + "\n" + answer) != unit["source_content_sha256"]:
                raise SystemExit(f"Q&A content hash mismatch for selected ID: {source_id}")
            presentation = ""
        else:
            raise SystemExit(f"unsupported unit_type in manifest: {unit_type!r}")

        audit_id = f"FRESHV1_{index:04d}"
        public_rows.append(
            {
                "audit_id": audit_id,
                "audit_split": "fresh_evaluation",
                "unit_type": unit_type,
                "ceo_presentation_segment": presentation,
                "analyst_question": question,
                "ceo_answer": answer,
                "human_content_class": "",
                "human_ok": "",
                "human_specificity": "",
                "human_notes": "",
            }
        )
        key_rows.append(
            {
                "audit_id": audit_id,
                "source_unit_id": source_id,
                "block_id": unit["block_id"],
                "event_id": unit["event_id"],
                "start_date": unit["start_date"],
                "calendar_quarter": unit["calendar_quarter"],
                "period_bin": unit["period_bin"],
                "turnover_id": unit["turnover_id"],
                "turnover_side": unit["turnover_side"],
                "expected_execid": unit["expected_execid"],
                "company_id": unit["company_id"],
                "company_name": unit["company_name"],
                "unit_type": unit_type,
                "sampling_bucket": unit["sampling_bucket"],
                "question_word_count": unit["question_word_count"],
                "target_word_count": unit["target_word_count"],
                "source_content_sha256": unit["source_content_sha256"],
                "source_integrity_lane": "current_source_marker_free",
            }
        )

    if len({row["audit_id"] for row in public_rows}) != len(public_rows):
        raise SystemExit("non-unique public audit IDs")
    if len({row["audit_id"] for row in key_rows}) != len(key_rows):
        raise SystemExit("non-unique private audit IDs")
    if {row["audit_id"] for row in public_rows} != {row["audit_id"] for row in key_rows}:
        raise SystemExit("public/private audit ID sets differ")

    stage_dir = args.output_dir.parent / f".{args.output_dir.name}.staging-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(stage_dir, 0o700)
    try:
        blinded_path = stage_dir / "fresh_evaluation_human_blinded.csv"
        key_path = stage_dir / "fresh_evaluation_human_key.csv"
        package_manifest_path = stage_dir / "fresh_evaluation_human_package_manifest.json"
        readme_path = stage_dir / "README.md"
        write_csv(blinded_path, public_rows, BLINDED_FIELDS)
        write_csv(key_path, key_rows, KEY_FIELDS)
        os.chmod(blinded_path, 0o600)
        os.chmod(key_path, 0o600)
        assert_blinded_columns(blinded_path)

        type_counts = Counter(str(row["unit_type"]) for row in public_rows)
        source_summary = {
            label: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for label, path in sources.items()
        }
        package_manifest: dict[str, object] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "fresh_current_source_clean_human_coding_package_ready_unscored",
            "scope": {
                "evaluation_manifest": str(manifest_dir),
                "model_outputs_in_package": False,
                "historical_human_labels_in_package": False,
                "transcript_text_local_only": True,
                "human_coding_complete": False,
            },
            "package": {
                "name": "Fresh current-source-clean PRE/Q&A human coding",
                "audit_split": "fresh_evaluation",
                "seed": args.seed,
                "dashboard_schema": "specificity_current_source_clean_evaluation_v1",
                "content_classes": [
                    "substantive",
                    "mixed",
                    "procedural_only",
                    "uncertain",
                ],
            },
            "counts": {
                "units": len(public_rows),
                "pre_units": type_counts["pre"],
                "qa_units": type_counts["qa"],
                "events": len({row["event_id"] for row in key_rows}),
                "companies": len({row["company_id"] for row in key_rows}),
                "expected_execids": len({row["expected_execid"] for row in key_rows}),
            },
            "integrity": {
                "public_audit_ids_unique": True,
                "public_private_audit_id_sets_equal": True,
                "all_selected_units_current_source_marker_free": True,
                "all_selected_units_reverified_against_current_source": True,
                "public_columns_exact": list(BLINDED_FIELDS),
                "private_key_columns_exact": list(KEY_FIELDS),
            },
            "blinding": {
                "browser_fields": [
                    "audit_id",
                    "unit_type",
                    "ceo_presentation_segment",
                    "analyst_question",
                    "ceo_answer",
                ],
                "hidden_metadata": [
                    "source_unit_id",
                    "block_id",
                    "event_id",
                    "start_date",
                    "turnover_id",
                    "turnover_side",
                    "expected_execid",
                    "company_id",
                    "company_name",
                ],
                "note": "Source text can identify a speaker or company incidentally; event and model metadata are excluded from the dashboard view.",
            },
            "provenance": {
                "evaluation_manifest_dir": str(manifest_dir),
                "evaluation_unit_manifest": str(units_path.resolve()),
                "evaluation_unit_manifest_sha256": sha256_file(units_path),
                "evaluation_manifest_summary": str(summary_path.resolve()),
                "evaluation_manifest_summary_sha256": sha256_file(summary_path),
                "source_files": source_summary,
                "builder_script": str(Path(__file__).resolve()),
                "builder_script_sha256": sha256_file(Path(__file__).resolve()),
            },
            "artifacts": {
                "blinded_csv": str((args.output_dir / blinded_path.name).resolve()),
                "private_key_csv": str((args.output_dir / key_path.name).resolve()),
                "package_manifest_json": str((args.output_dir / package_manifest_path.name).resolve()),
                "readme": str((args.output_dir / readme_path.name).resolve()),
            },
        }
        package_manifest["artifacts_sha256"] = {
            "blinded_csv": sha256_file(blinded_path),
            "private_key_csv": sha256_file(key_path),
        }
        package_manifest_path.write_text(
            json.dumps(package_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.chmod(package_manifest_path, 0o644)

        progress_path = args.output_dir / "fresh_evaluation_human_progress.json"
        coded_path = args.output_dir / "fresh_evaluation_human_coded.csv"
        readme = f"""# Fresh Source-Clean PRE/Q&A Human Coding

## Status

This is an unscored, blinded human-coding package for a fresh 20-call
evaluation lane. It contains {len(public_rows)} units: {type_counts['pre']} PRE
and {type_counts['qa']} Q&A. No model outputs or historical human labels are
included.

The blinded CSV contains licensed CCTS transcript text and must remain local.
Do not upload it to cloud tools or share it outside the licensed research
environment.

## Coding Rules

1. Choose a content class first.
   - `substantive`: interpretable CEO business, financial, operational,
     strategic, governance, management, market, risk, forecast, or commitment content.
   - `mixed`: substantive CEO content plus a closing, handoff, thanks, or other procedural language.
   - `procedural_only`: only routing, audio checks, thanks, handoffs, or closings with no substantive CEO content.
   - `uncertain`: not confidently classifiable from the supplied text.
2. For `substantive`, `mixed`, or `uncertain`, set a specificity score from 1 to 5.
3. For `procedural_only`, use **Unscorable**. This records `human_ok=0` and score 0.
4. For PRE, score the CEO presentation segment. For Q&A, read the analyst question only for context and score the CEO answer.
5. Score the complete CEO target as supplied. Do not remove closing language from a mixed unit.
6. Do not use outside knowledge, factual correctness, favorability, or confidence as a substitute for specificity.

## Dashboard Command

```bash
python3 ./tools/llm_measurement/human_audit_dashboard/server.py \\
  --input {args.output_dir / blinded_path.name} \\
  --progress {progress_path} \\
  --coded-csv {coded_path} \\
  --host 127.0.0.1 \\
  --port 8772
```

Open `http://127.0.0.1:8772` in a browser. Every saved label is persisted to
both the progress JSON and coded CSV.

## Files

- `fresh_evaluation_human_blinded.csv`: dashboard input; source metadata and model outputs are hidden.
- `fresh_evaluation_human_key.csv`: owner-only private source join key. Do not open while coding.
- `fresh_evaluation_human_package_manifest.json`: checksums and provenance.
- `fresh_evaluation_human_progress.json`: created by the dashboard.
- `fresh_evaluation_human_coded.csv`: created by the dashboard.
"""
        readme_path.write_text(readme, encoding="utf-8")
        os.chmod(readme_path, 0o644)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_dir.rename(args.output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(json.dumps(package_manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
