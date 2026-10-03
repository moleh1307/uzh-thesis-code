#!/usr/bin/env python3
"""Build local-only PRE and Q&A specificity scoring manifests under contract v1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv
from artifact_publication import fresh_artifact_directory

configure_csv()
from typing import Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ccts_database'))
from ccts_qa_episodes import procedure_kind, source_quality_reasons


SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")

PRE_FIELDS = ["custom_id", "unit_type", "event_id", "episode_rank", "turnover_id", "turnover_side",
              "expected_execid", "start_date", "calendar_quarter", "source_turn_index",
              "source_sequence_id", "source_raw_sequence_id", "chunk_index", "unit_word_count",
              "ceo_presentation_segment", "prompt_input_json"]
QA_FIELDS = ["custom_id", "unit_type", "block_id", "event_id", "episode_rank", "turnover_id",
             "turnover_side", "expected_execid", "start_date", "calendar_quarter",
             "question_word_count", "unit_word_count", "analyst_question", "ceo_answer", "prompt_input_json"]
COVERAGE_FIELDS = ["event_id", "episode_rank", "turnover_id", "turnover_side", "expected_execid",
                   "expected_ceo_name", "company_id", "company_name", "start_date", "calendar_quarter",
                   "pre_unit_count", "pre_scored_word_count", "source_pre_word_count", "high_qa_unit_count",
                   "high_qa_scored_word_count", "source_all_ceo_qa_word_count", "high_qa_word_coverage",
                   "primary_specificity_call_eligible", "qa_quality_screen_call_eligible"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def word_count(text: str) -> int:
    return len(clean(text).split())


def qa_non_substantive_reasons(text: str) -> list[str]:
    normalized = clean(text)
    words = word_count(normalized)
    reasons: list[str] = []
    if not normalized:
        return ["empty_ceo_answer"]
    procedural = procedure_kind(normalized)
    reasons.extend(source_quality_reasons(normalized))
    if procedural:
        reasons.append('procedural_' + procedural)
    if words < 25 and re.search(r"(?:--|—)\s*$", normalized):
        reasons.append("incomplete_fragment")
    return reasons


def qa_question_review_reasons(text: str) -> list[str]:
    if not clean(text):
        return ['empty_analyst_question']
    return ['question_' + reason for reason in source_quality_reasons(text)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hard_split(text: str, max_words: int) -> list[str]:
    words = clean(text).split()
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def split_long_turn(text: str, max_words: int, min_fragment_words: int) -> list[str]:
    normalized = clean(text)
    if not normalized:
        return []
    if word_count(normalized) <= max_words:
        return [normalized]

    sentences: list[str] = []
    for sentence in SENTENCE_BOUNDARY_RE.split(normalized):
        if word_count(sentence) > max_words:
            sentences.extend(hard_split(sentence, max_words))
        elif clean(sentence):
            sentences.append(clean(sentence))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = word_count(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))

    if len(chunks) > 1 and word_count(chunks[-1]) < min_fragment_words:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()
    return chunks


def build_pre_units(
    presentations: Sequence[dict[str, str]], max_words: int, min_fragment_words: int
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in presentations:
        turns = [clean(turn) for turn in event["ceo_presentation_text"].split("\n\n") if clean(turn)]
        expected_turns = event.get("pre_ceo_turn_count")
        if expected_turns not in (None, "") and len(turns) != int(expected_turns):
            raise ValueError(f"PRE turn boundary/count mismatch for {event['event_id']}; rebuild the presentation CSV from ordered raw turns")
        sequence_ids = json.loads(event.get("pre_sequence_ids") or "[]")
        raw_ids = json.loads(event.get("pre_raw_sequence_ids") or "[]")
        if sequence_ids and len(sequence_ids) != len(turns):
            raise ValueError(f"PRE sequence provenance mismatch for {event['event_id']}")
        if raw_ids and len(raw_ids) != len(turns):
            raise ValueError(f"PRE raw-sequence provenance mismatch for {event['event_id']}")
        for turn_index, turn in enumerate(turns, start=1):
            chunks = split_long_turn(turn, max_words, min_fragment_words)
            for chunk_index, chunk in enumerate(chunks, start=1):
                source_sequence = sequence_ids[turn_index - 1] if sequence_ids else ""
                turn_key = f"s{source_sequence}" if sequence_ids else f"t{turn_index}"
                unit_id = f"specificity_pre_{event['event_id']}_turnpreserved_{turn_key}_{chunk_index:03d}"
                prompt_input = {
                    "unit_type": "pre",
                    "ceo_presentation_segment": chunk,
                }
                rows.append({
                    "custom_id": unit_id,
                    "unit_type": "pre",
                    "event_id": event["event_id"],
                    "episode_rank": event["episode_rank"],
                    "turnover_id": event["turnover_id"],
                    "turnover_side": event["turnover_side"],
                    "expected_execid": event["expected_execid"],
                    "start_date": event["start_date"],
                    "calendar_quarter": event["calendar_quarter"],
                    "source_turn_index": turn_index,
                    "source_sequence_id": source_sequence,
                    "source_raw_sequence_id": raw_ids[turn_index - 1] if raw_ids else "",
                    "chunk_index": chunk_index,
                    "unit_word_count": word_count(chunk),
                    "ceo_presentation_segment": chunk,
                    "prompt_input_json": json.dumps(
                        prompt_input, ensure_ascii=False, separators=(",", ":")
                    ),
                })
    return rows


def build_qa_units(
    blocks: Sequence[dict[str, str]], event_by_id: dict[str, dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for block in blocks:
        if block["quality_tier"] != "high":
            raise SystemExit(f"non-high block in strict input: {block['block_id']}")
        event = event_by_id.get(block["event_id"])
        if event is None:
            raise SystemExit(f"Q&A block has no episode-eligible event: {block['block_id']}")
        question = clean(block["analyst_question"])
        answer = clean(block["ceo_answer"])
        exclusion_reasons = qa_non_substantive_reasons(answer)
        if not question:
            exclusion_reasons.append("empty_analyst_question")
        exclusion_reasons.extend(reason for reason in qa_question_review_reasons(question)
                                 if reason not in exclusion_reasons)
        if exclusion_reasons:
            exclusions.append({
                "block_id": block["block_id"],
                "event_id": block["event_id"],
                "question_word_count": word_count(question),
                "ceo_answer_word_count": word_count(answer),
                "exclusion_reasons": ";".join(exclusion_reasons),
                "analyst_question": question,
                "ceo_answer": answer,
            })
            continue
        prompt_input = {
            "unit_type": "qa",
            "analyst_question": question,
            "ceo_answer": answer,
        }
        rows.append({
            "custom_id": f"specificity_qa_{block['block_id']}",
            "unit_type": "qa",
            "block_id": block["block_id"],
            "event_id": block["event_id"],
            "episode_rank": event["episode_rank"],
            "turnover_id": event["turnover_id"],
            "turnover_side": event["turnover_side"],
            "expected_execid": event["expected_execid"],
            "start_date": event["start_date"],
            "calendar_quarter": event["calendar_quarter"],
            "question_word_count": word_count(question),
            "unit_word_count": word_count(answer),
            "analyst_question": question,
            "ceo_answer": answer,
            "prompt_input_json": json.dumps(
                prompt_input, ensure_ascii=False, separators=(",", ":")
            ),
        })
    return rows, exclusions


def build_call_coverage(
    presentations: Sequence[dict[str, str]],
    pre_units: Sequence[dict[str, object]],
    qa_units: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    pre_by_event: dict[str, list[dict[str, object]]] = defaultdict(list)
    qa_by_event: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in pre_units:
        pre_by_event[str(row["event_id"])].append(row)
    for row in qa_units:
        qa_by_event[str(row["event_id"])].append(row)

    rows: list[dict[str, object]] = []
    for event in presentations:
        event_id = event["event_id"]
        event_pre = pre_by_event[event_id]
        event_qa = qa_by_event[event_id]
        pre_words = sum(int(row["unit_word_count"]) for row in event_pre)
        qa_high_words = sum(int(row["unit_word_count"]) for row in event_qa)
        qa_all_words = int(event["qa_ceo_word_count"])
        coverage = qa_high_words / qa_all_words if qa_all_words else None
        primary = int(bool(event_pre) and bool(event_qa))
        quality = int(
            bool(event_pre)
            and len(event_qa) >= 2
            and qa_high_words >= 100
            and coverage is not None
            and coverage >= 0.25
        )
        rows.append({
            "event_id": event_id,
            "episode_rank": event["episode_rank"],
            "turnover_id": event["turnover_id"],
            "turnover_side": event["turnover_side"],
            "expected_execid": event["expected_execid"],
            "expected_ceo_name": event["expected_ceo_name"],
            "company_id": event["company_id"],
            "company_name": event["company_name"],
            "start_date": event["start_date"],
            "calendar_quarter": event["calendar_quarter"],
            "pre_unit_count": len(event_pre),
            "pre_scored_word_count": pre_words,
            "source_pre_word_count": event["pre_ceo_word_count"],
            "high_qa_unit_count": len(event_qa),
            "high_qa_scored_word_count": qa_high_words,
            "source_all_ceo_qa_word_count": qa_all_words,
            "high_qa_word_coverage": coverage,
            "primary_specificity_call_eligible": primary,
            "qa_quality_screen_call_eligible": quality,
        })
    return rows


def episode_counts(
    coverage: Sequence[dict[str, object]], field: str
) -> tuple[dict[str, int], int]:
    counts: dict[str, int] = Counter(
        str(row["episode_rank"]) for row in coverage if int(row[field]) == 1
    )
    return counts, sum(count >= 5 for count in counts.values())


def turnover_count(
    turnover_gate: Sequence[dict[str, str]], episode_eligible_counts: dict[str, int]
) -> int:
    return sum(
        int(row["turnover_analysis_gate_pass"]) == 1
        and episode_eligible_counts.get(row["old_episode_rank"], 0) >= 5
        and episode_eligible_counts.get(row["new_episode_rank"], 0) >= 5
        for row in turnover_gate
    )


def build_package(args, stage):
    summary_path = args.output_dir / "specificity_manifest_summary.json"

    presentations = read_csv(args.presentations)
    blocks = read_csv(args.high_qa_blocks)
    turnover_gate = read_csv(args.turnover_gate)
    if not presentations:
        raise SystemExit("no presentation rows")
    if len({row["event_id"] for row in presentations}) != len(presentations):
        raise SystemExit("duplicate event_id in presentations")
    event_by_id = {row["event_id"]: row for row in presentations}

    pre_units = build_pre_units(
        presentations, args.max_pre_words, args.min_pre_fragment_words
    )
    qa_units, qa_exclusions = build_qa_units(blocks, event_by_id)
    coverage = build_call_coverage(presentations, pre_units, qa_units)

    all_custom_ids = [str(row["custom_id"]) for row in [*pre_units, *qa_units]]
    if len(set(all_custom_ids)) != len(all_custom_ids):
        raise SystemExit("duplicate scoring-unit custom_id")
    if any(int(row["unit_word_count"]) <= 0 for row in [*pre_units, *qa_units]):
        raise SystemExit("zero-word scoring unit")

    pre_path = args.output_dir / "pre_specificity_scoring_units.csv"
    qa_path = args.output_dir / "qa_specificity_scoring_units.csv"
    coverage_path = args.output_dir / "specificity_call_coverage.csv"
    qa_exclusions_path = args.output_dir / "qa_specificity_excluded_non_substantive.csv"
    write_csv(stage / pre_path.name, pre_units, PRE_FIELDS)
    write_csv(stage / qa_path.name, qa_units, QA_FIELDS)
    write_csv(stage / coverage_path.name, coverage, COVERAGE_FIELDS)
    write_csv(
        stage / qa_exclusions_path.name,
        qa_exclusions,
        [
            "block_id", "event_id", "question_word_count", "ceo_answer_word_count",
            "exclusion_reasons", "analyst_question", "ceo_answer",
        ],
    )

    primary_counts, primary_episode_count = episode_counts(
        coverage, "primary_specificity_call_eligible"
    )
    quality_counts, quality_episode_count = episode_counts(
        coverage, "qa_quality_screen_call_eligible"
    )
    contract_files = sorted(path for path in args.contract_dir.iterdir() if path.is_file())
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "local_only_unscored_manifest_no_api_submission",
        "contract_version": "specificity_v1_20260720",
        "unit_definition": {
            "pre": "One confirmed CEO PRE turn, deterministically chunked above max_pre_words.",
            "qa": "One canonical high-quality analyst-question/CEO-answer block.",
            "max_pre_words": args.max_pre_words,
            "min_pre_fragment_words": args.min_pre_fragment_words,
        },
        "counts": {
            "source_calls": len(presentations),
            "pre_scoring_units": len(pre_units),
            "qa_scoring_units": len(qa_units),
            "qa_input_blocks_excluded_non_substantive": len(qa_exclusions),
            "total_scoring_units": len(pre_units) + len(qa_units),
            "primary_eligible_calls": sum(
                int(row["primary_specificity_call_eligible"]) for row in coverage
            ),
            "primary_episodes_with_at_least_5_calls": primary_episode_count,
            "primary_turnovers_with_both_episodes": turnover_count(
                turnover_gate, primary_counts
            ),
            "quality_screen_eligible_calls": sum(
                int(row["qa_quality_screen_call_eligible"]) for row in coverage
            ),
            "quality_screen_episodes_with_at_least_5_calls": quality_episode_count,
            "quality_screen_turnovers_with_both_episodes": turnover_count(
                turnover_gate, quality_counts
            ),
        },
        "artifacts": {
            "pre_scoring_units_csv": str(pre_path),
            "qa_scoring_units_csv": str(qa_path),
            "call_coverage_csv": str(coverage_path),
            "qa_exclusions_csv": str(qa_exclusions_path),
        },
        "provenance": {
            "builder_script": str(Path(__file__).resolve()),
            "presentations_csv": str(args.presentations.resolve()),
            "presentations_sha256": sha256(args.presentations),
            "high_qa_blocks_csv": str(args.high_qa_blocks.resolve()),
            "high_qa_blocks_sha256": sha256(args.high_qa_blocks),
            "turnover_gate_csv": str(args.turnover_gate.resolve()),
            "turnover_gate_sha256": sha256(args.turnover_gate),
            "contract_dir": str(args.contract_dir.resolve()),
            "contract_file_sha256": {
                path.name: sha256(path) for path in contract_files
            },
            "qa_non_substantive_gate": "whole_passage_v1_20261003",
        },
        "privacy": {
            "contains_licensed_transcript_text": True,
            "cloud_submitted": False,
            "handling": "Keep local on the thesis SSD until the license/API boundary is explicitly resolved.",
        },
    }
    (stage / summary_path.name).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = [
        "# Local Specificity Scoring Manifest v1",
        "",
        "## Status",
        "",
        "- Local-only, unscored manifest. No API or cloud submission was performed.",
        "- The PRE and Q&A unit CSVs contain licensed transcript text and must remain local.",
        "",
        "## Units",
        "",
        f"- Source calls: {len(presentations):,}",
        f"- PRE scoring units: {len(pre_units):,}",
        f"- High-quality Q&A scoring units: {len(qa_units):,}",
        f"- Input high blocks excluded as non-substantive: {len(qa_exclusions):,}",
        f"- Total model requests if every unit is scored once: {len(pre_units) + len(qa_units):,}",
        "",
        "## Eligibility Profile Before LLM Scoring",
        "",
        f"- Primary calls with PRE and at least one high Q&A unit: {summary['counts']['primary_eligible_calls']:,}",
        f"- Primary CEO episodes retaining at least five calls: {primary_episode_count:,}",
        f"- Primary turnovers retaining both episodes: {summary['counts']['primary_turnovers_with_both_episodes']:,}",
        f"- Q&A quality-screen calls: {summary['counts']['quality_screen_eligible_calls']:,}",
        f"- Quality-screen CEO episodes retaining at least five calls: {quality_episode_count:,}",
        f"- Quality-screen turnovers retaining both episodes: {summary['counts']['quality_screen_turnovers_with_both_episodes']:,}",
        "",
        "## Contract",
        "",
        f"- Contract directory: `{args.contract_dir.resolve()}`",
        "- Primary call aggregation after scoring: CEO-word-weighted mean within PRE and Q&A separately.",
        "- Equal-unit mean, median, and common 50-200 word support are pre-specified sensitivity measures.",
        "- The model/provider is not frozen and licensed text must not be submitted externally yet.",
    ]
    (stage / "specificity_manifest_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentations", required=True, type=Path)
    parser.add_argument("--high-qa-blocks", required=True, type=Path)
    parser.add_argument("--turnover-gate", required=True, type=Path)
    parser.add_argument("--contract-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pre-words", type=int, default=150)
    parser.add_argument("--min-pre-fragment-words", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Deprecated; existing packages cannot be replaced")
    args = parser.parse_args()
    if args.max_pre_words < 1 or args.min_pre_fragment_words < 1:
        parser.error("PRE chunk limits must be positive")
    args.output_dir = args.output_dir.absolute()
    with fresh_artifact_directory(args.output_dir) as stage:
        summary = build_package(args, stage)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
