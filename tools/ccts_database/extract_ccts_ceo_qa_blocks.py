#!/usr/bin/env python3
"""Extract auditable analyst-question -> CEO-answer blocks from CCTS turn rows.

This tool operates on a small CCTS extraction sample. It retains the full
answer context for audit, while keeping the CEO-only answer as the future
measurement unit. It does not score text or call an LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ccts_qa_episodes import collect_episode, procedure_kind, analyst_question_candidate, source_quality_reasons


CEO_RE = re.compile(r"\b(?:ceo|chief\s+executive\s+officer)\b", re.IGNORECASE)
ANALYST_RE = re.compile(r"\banalyst\b", re.IGNORECASE)
FORMER_CEO_RE = re.compile(r"\b(?:former|retired|ex[-\s])\b", re.IGNORECASE)
SPECIAL_CEO_RE = re.compile(r"\b(?:interim|co[-\s]?ceo|co[-\s]?chief\s+executive)\b", re.IGNORECASE)
OPERATOR_RE = re.compile(r"\boperator\b", re.IGNORECASE)
CEO_ASSISTANT_RE = re.compile(r"\b(?:ceo\s+assistant|assistant\s+to\s+(?:the\s+)?ceo)\b", re.IGNORECASE)
NEXT_QUESTION_RE = re.compile(
    r"\b(?:our|the|next|first|follow[-\s]?up)\s+question\b|\bwe(?:'|’)ll\s+take\s+(?:our\s+)?next\s+question\b|"
    r"^(?:and\s+)?next[,\s]+we(?:['’]ll|\s+will)?\s+go\s+to\s+the\s+line\s+of\s+\S+",
    re.IGNORECASE,
)
OPERATOR_HANDOFF_IN_TEXT_RE = re.compile(
    r"(?:operator\s*:\s*)?(?:thank you\.?\s*)?(?:our|the|next|first)\s+question\s+comes\s+from",
    re.IGNORECASE,
)
CLARIFICATION_ANSWER_RE = re.compile(
    r"\b(?:i (?:did not|didn't) hear|could you repeat|can you repeat|(?:please|you(?:'ll| will) have to) repeat yourself|"
    r"repeat (?:yourself|the question)|say that again|did you say|"
    r"i(?:'m| am) not sure i understand(?: the question)?|can you be more specific|"
    r"is that what i heard you say|is that what you meant|what was that your question|"
    r"could you (?:clarify|be more specific)|what do you mean by)\b",
    re.IGNORECASE,
)
CLOSING_ANSWER_RE = re.compile(
    r"^(?:(?:well|okay)[,.]?\s*)?(?:not at all\.\s*)?(?:in closing[,.]?\s*)?"
    r"(?:i just wanted to say\s+)?(?:(?:we )?appreciate (?:you|everyone)|thank you all|"
    r"thanks everyone).*\b(?:joining|call)\b",
    re.IGNORECASE,
)
PLEASANTRY_ANSWER_RE = re.compile(
    r"\(laughter\).*(?:thank you|called it|better start|good research)|"
    r"(?:thank you|called it|better start|good research).*\(laughter\)",
    re.IGNORECASE,
)

BLOCK_COLUMNS = [
    "block_id", "event_id", "sample_rank", "year", "start_date", "company_name", "company_ticker",
    "event_title", "question_start_sequence_id", "question_end_sequence_id", "answer_end_sequence_id",
    "analyst_speaker", "ceo_speaker", "shared_ceo_candidate_count", "question_turn_count",
    "ceo_answer_turn_count", "answer_context_turn_count", "question_word_count", "ceo_answer_word_count",
    "answer_context_word_count", "non_ceo_management_turn_count", "non_ceo_management_speakers",
    "quality_tier", "block_flags", "analyst_question", "ceo_answer", "answer_context",
    "extraction_version", "question_sequence_ids", "ceo_answer_sequence_ids",
    "context_sequence_ids", "procedural_sequence_ids", "applicability_status",
    "preceding_question_context",
]

EVENT_COLUMNS = [
    "event_id", "sample_rank", "year", "start_date", "company_name", "company_ticker", "event_title",
    "pre_ceo_speakers", "qa_ceo_speakers", "shared_ceo_speakers", "qa_analyst_turn_count",
    "analyst_question_episodes", "eligible_ceo_qa_blocks", "high_quality_blocks", "event_audit_status",
    "event_flags",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).lower()).strip()


def word_count(value: str) -> int:
    return len(clean(value).split())


def non_substantive_answer_flags(value: str) -> list[str]:
    text = clean(value)
    words = word_count(text)
    flags: list[str] = []
    # A clarification phrase inside a longer answer can precede substantive
    # disclosure, so only short clarification-only turns are excluded.
    if words <= 40 and CLARIFICATION_ANSWER_RE.search(text):
        flags.append("CEO_ANSWER_IS_CLARIFICATION_REQUEST")
    if CLOSING_ANSWER_RE.search(text):
        flags.append("CEO_ANSWER_IS_CLOSING_REMARKS")
    if words <= 25 and PLEASANTRY_ANSWER_RE.search(text):
        flags.append("CEO_ANSWER_IS_NONSUBSTANTIVE_PLEASANTRY")
    if words < 25 and re.search(r"(?:--|—)\s*$", text):
        flags.append("CEO_ANSWER_IS_INCOMPLETE_FRAGMENT")
    return flags


def label(row: Mapping[str, str]) -> str:
    return re.sub(r"\s*\[\d+\]\s*$", "", clean(row.get("text_name", ""))).strip()


def speaker_key(row: Mapping[str, str]) -> str:
    value = label(row)
    # CCTS labels normally follow "Name, Firm - Title". The left side is stable
    # across PRE/Q&A and preserves disambiguating firm information when needed.
    value = re.split(r"\s+-\s+", value, maxsplit=1)[0]
    return normalized(value)


def is_ceo(row: Mapping[str, str]) -> bool:
    value = label(row)
    if not CEO_RE.search(value) or FORMER_CEO_RE.search(value) or CEO_ASSISTANT_RE.search(value):
        return False
    return True


def is_analyst(row: Mapping[str, str]) -> bool:
    value = label(row)
    if ANALYST_RE.search(value):
        return True
    parts = re.split(r'\s+-\s+', re.sub(r'\s*\[\d+\]\s*$', '', value), maxsplit=1)
    if len(parts) != 2 or is_ceo(row):
        return False
    firm, title = parts
    return bool(re.fullmatch(r'(?:MD|Managing Director|Head|Director) of Equity Research', title, re.I) or
                (re.search(r'\bResearch Division\b', firm, re.I) and
                 re.fullmatch(r'MD|Managing Director', title, re.I)))


def is_operator(row: Mapping[str, str]) -> bool:
    return bool(OPERATOR_RE.search(label(row)))


def is_next_question_operator(row: Mapping[str, str]) -> bool:
    return is_operator(row) and bool(NEXT_QUESTION_RE.search(clean(row.get("text_contents", ""))))


def join_turns(rows: Iterable[Mapping[str, str]]) -> str:
    return "\n\n".join(clean(row.get("text_contents", "")) for row in rows if clean(row.get("text_contents", "")))


def read_turns(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["event_id"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("sequence_id") or 0))
    return dict(grouped)


def event_metadata(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    first = rows[0]
    return {column: first.get(column, "") for column in EVENT_COLUMNS if column in first}


def candidate_map(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    return {speaker_key(row): label(row) for row in rows if is_ceo(row) and speaker_key(row)}


def extract_event(rows: Sequence[Mapping[str, str]], *, participant_roles=None, validated_anchor_key=None, allow_anonymous_company_context=False) -> tuple[dict[str, str], list[dict[str, object]]]:
    if allow_anonymous_company_context and participant_roles is None:
        raise ValueError('Anonymous company context requires guarded extraction')
    sequence = [int(row["sequence_id"]) for row in rows]
    if len(set(sequence)) != len(sequence) or sequence != sorted(sequence):
        raise ValueError("Event requires unique ordered sequence_id values")
    if len({row["event_id"] for row in rows}) != 1:
        raise ValueError("extract_event requires one nonempty event")
    meta = event_metadata(rows)
    pre_rows = [row for row in rows if row.get("analysis_text_type") == "PRE"]
    qa_rows = [row for row in rows if row.get("analysis_text_type") == "Q&A"]
    pre_ceos = candidate_map(pre_rows)
    qa_ceos = candidate_map(qa_rows)
    shared_keys = sorted(set(pre_ceos) & set(qa_ceos))
    shared_ceos = {key: qa_ceos[key] for key in shared_keys}
    questioner = is_analyst
    eligible_questioner = is_analyst
    unresolved = None
    if participant_roles is not None:
        if not validated_anchor_key or validated_anchor_key not in shared_ceos:
            raise ValueError('Guarded extraction requires a validated shared issuer anchor')
        if any(str(r['event_id']) != str(rows[0]['event_id']) for r in participant_roles.values()):
            raise ValueError('Role map contains another event')
        shared_ceos = {validated_anchor_key: shared_ceos[validated_anchor_key]}
        def role(r):
            return participant_roles.get(speaker_key(r), {})
        questioner = lambda r: role(r).get('role') == 'external_questioner'
        eligible_questioner = lambda r: questioner(r) and role(r).get('external_subtype') == 'research_label' and not role(r).get('roster_quarantined', True)
        unresolved = lambda r: (role(r).get('role', 'unresolved') == 'unresolved'
                               and not (allow_anonymous_company_context
                                        and speaker_key(r) == 'unidentified company representative'))
    elif validated_anchor_key is not None:
        if not validated_anchor_key or validated_anchor_key not in shared_ceos:
            raise ValueError('Validated extraction requires an anchor present in both PRE and Q&A')
        shared_ceos = {validated_anchor_key: shared_ceos[validated_anchor_key]}

    flags: list[str] = []
    if not shared_ceos:
        flags.append("NO_SAME_CEO_IDENTIFIED_IN_PRE_AND_QA")
    if len(shared_ceos) > 1:
        flags.append("MULTIPLE_SHARED_CEO_CANDIDATES")
    for name in shared_ceos.values():
        if SPECIAL_CEO_RE.search(name):
            flags.append("SPECIAL_CEO_TITLE")

    blocks: list[dict[str, object]] = []
    episodes = 0
    pos = 0
    while pos < len(qa_rows):
        if not eligible_questioner(qa_rows[pos]):
            pos += 1
            continue
        question_rows, context_rows, procedural_rows, episode_flags, pos = collect_episode(
            qa_rows, pos, is_analyst=questioner, is_operator=is_operator,
            is_question_handoff=is_next_question_operator, speaker_key=speaker_key,
            is_unresolved=unresolved
        )
        episodes += 1
        if not context_rows:
            continue
        all_ceo_rows = [row for row in context_rows if speaker_key(row) in shared_ceos]
        if not all_ceo_rows:
            continue
        procedural_ceo = [row for row in all_ceo_rows if procedure_kind(row.get("text_contents", ""))]
        ceo_rows = [row for row in all_ceo_rows if row not in procedural_ceo]
        procedural_rows += procedural_ceo
        procedural_rows.sort(key=lambda r: int(r["sequence_id"]))

        answer_ceo_keys = {speaker_key(row) for row in ceo_rows}
        block_flags = list(flags) + sorted(episode_flags)
        if allow_anonymous_company_context and any(speaker_key(r) == 'unidentified company representative' for r in context_rows):
            block_flags.append('ANONYMOUS_COMPANY_CONTEXT_REVIEW')
        if len(answer_ceo_keys) > 1:
            block_flags.append("MULTIPLE_CEOS_IN_ANSWER_CONTEXT")
        question_text = join_turns(question_rows)
        ceo_text = join_turns(ceo_rows)
        context_text = join_turns(context_rows)
        # Keep uncertain source content intact, but never call it clean.
        for source_row in question_rows:
            block_flags.extend('QUESTION_' + reason.upper() + '_REVIEW'
                               for reason in source_quality_reasons(source_row.get('text_contents', '')))
        for source_row in ceo_rows:
            block_flags.extend('CEO_' + reason.upper() + '_REVIEW'
                               for reason in source_quality_reasons(source_row.get('text_contents', '')))
        if any(re.search(r'(?:--|\u2014)\s*$', row.get('text_contents', '')) for row in ceo_rows):
            block_flags.append('TRAILING_INTERRUPTED_CEO_TURN_REVIEW')
        if any(re.search(r'\((?:inaudible|technical difficult(?:y|ies))\)', row.get('text_contents', ''), re.I) for row in ceo_rows):
            block_flags.append('CEO_SOURCE_TEXT_GAP_REVIEW')
        if re.search(r'\bthank (?:everybody|everyone|you all) for joining us\b', ceo_text, re.I):
            block_flags.append('CEO_MIXED_CLOSING_REVIEW')
        preceding_question = ""
        if blocks and "UNRESOLVED_CEO_COUNTERQUESTION_REVIEW" in str(blocks[-1]["block_flags"]):
            previous_end = int(blocks[-1]["answer_end_sequence_id"])
            current_start = int(question_rows[0]["sequence_id"])
            intervening_operator = any(is_operator(r) and previous_end < int(r["sequence_id"]) < current_start for r in qa_rows)
            if blocks[-1]["analyst_speaker"] == label(question_rows[0]) and not intervening_operator:
                preceding_question = str(blocks[-1]["analyst_question"])
                block_flags.append("PRECEDING_QUESTION_CONTEXT_REVIEW")
        if not analyst_question_candidate(question_text):
            block_flags.append("ANALYST_QUESTION_APPLICABILITY_REVIEW")
        if not ceo_rows:
            block_flags.append("NO_SUBSTANTIVE_CEO_TURN")
        if word_count(question_text) < 8:
            block_flags.append("SHORT_ANALYST_QUESTION")
        if word_count(ceo_text) < 10:
            block_flags.append("SHORT_CEO_ANSWER")
        if OPERATOR_HANDOFF_IN_TEXT_RE.search(question_text):
            block_flags.append("QUESTION_CONTAINS_OPERATOR_HANDOFF")
        block_flags.extend(non_substantive_answer_flags(ceo_text))

        non_ceo_management = [
            row for row in context_rows
            if not is_operator(row) and not questioner(row) and speaker_key(row) not in shared_ceos
        ]
        if non_ceo_management:
            block_flags.append("NON_CEO_MANAGEMENT_CONTEXT")
        high_exclusion_flags = {
            "QUESTION_CONTAINS_OPERATOR_HANDOFF", "NO_SUBSTANTIVE_CEO_TURN",
            "CEO_ANSWER_IS_CLARIFICATION_REQUEST", "SPECIAL_CEO_TITLE", "MULTIPLE_SHARED_CEO_CANDIDATES",
            "MULTIPLE_CEOS_IN_ANSWER_CONTEXT", "CEO_ANSWER_IS_CLOSING_REMARKS",
            "CEO_ANSWER_IS_NONSUBSTANTIVE_PLEASANTRY", "CEO_ANSWER_IS_INCOMPLETE_FRAGMENT",
        }
        review_required = any(flag.endswith("_REVIEW") for flag in block_flags)
        if review_required:
            quality_tier = "review"
        elif len(shared_ceos) == 1 and not non_ceo_management and not high_exclusion_flags & set(block_flags):
            quality_tier = "high"
        elif len(shared_ceos) == 1:
            quality_tier = "usable_with_flag"
        else:
            quality_tier = "review"

        event_id = meta.get("event_id", "")
        blocks.append(
            {
                "block_id": f"ccts_{event_id}_episode_{question_rows[0]['sequence_id']}",
                "event_id": event_id,
                "sample_rank": meta.get("sample_rank", ""),
                "year": meta.get("year", ""),
                "start_date": meta.get("start_date", ""),
                "company_name": meta.get("company_name", ""),
                "company_ticker": meta.get("company_ticker", ""),
                "event_title": meta.get("event_title", ""),
                "question_start_sequence_id": question_rows[0].get("sequence_id", ""),
                "question_end_sequence_id": question_rows[-1].get("sequence_id", ""),
                "answer_end_sequence_id": context_rows[-1].get("sequence_id", ""),
                "analyst_speaker": label(question_rows[0]),
                "ceo_speaker": "; ".join(sorted({label(row) for row in all_ceo_rows})),
                "shared_ceo_candidate_count": len(shared_ceos),
                "question_turn_count": len(question_rows),
                "ceo_answer_turn_count": len(ceo_rows),
                "answer_context_turn_count": len(context_rows),
                "question_word_count": word_count(question_text),
                "ceo_answer_word_count": word_count(ceo_text),
                "answer_context_word_count": word_count(context_text),
                "non_ceo_management_turn_count": len(non_ceo_management),
                "non_ceo_management_speakers": "; ".join(sorted({label(row) for row in non_ceo_management})),
                "quality_tier": quality_tier,
                "block_flags": ";".join(sorted(set(block_flags))),
                "analyst_question": question_text,
                "ceo_answer": ceo_text,
                "answer_context": context_text,
                "extraction_version": (
                    "participant_guard_optin_20260906" if participant_roles is not None
                    else "external_speaker_gate_anchor_20260928_v1" if validated_anchor_key is not None
                    else "episode_coverage_guard_20260906"
                ),
                "question_sequence_ids": json.dumps([r["sequence_id"] for r in question_rows]),
                "ceo_answer_sequence_ids": json.dumps([r["sequence_id"] for r in ceo_rows]),
                "context_sequence_ids": json.dumps([r["sequence_id"] for r in context_rows]),
                "procedural_sequence_ids": json.dumps([r["sequence_id"] for r in procedural_rows]),
                "applicability_status": "review" if review_required else ("procedural_only" if not ceo_rows else "candidate_substantive"),
                "preceding_question_context": preceding_question,
            }
        )

    high_blocks = sum(block["quality_tier"] == "high" for block in blocks)
    if not shared_ceos:
        event_status = "exclude_no_same_ceo"
    elif not blocks:
        event_status = "review_no_ceo_answer_to_analyst_question"
    elif len(shared_ceos) > 1:
        event_status = "review_multiple_ceo_candidates"
    else:
        event_status = "usable"
    event_row = {
        "event_id": meta.get("event_id", ""), "sample_rank": meta.get("sample_rank", ""),
        "year": meta.get("year", ""), "start_date": meta.get("start_date", ""),
        "company_name": meta.get("company_name", ""), "company_ticker": meta.get("company_ticker", ""),
        "event_title": meta.get("event_title", ""),
        "pre_ceo_speakers": "; ".join(pre_ceos.values()), "qa_ceo_speakers": "; ".join(qa_ceos.values()),
        "shared_ceo_speakers": "; ".join(shared_ceos.values()),
        "qa_analyst_turn_count": sum(is_analyst(row) for row in qa_rows),
        "analyst_question_episodes": episodes, "eligible_ceo_qa_blocks": len(blocks),
        "high_quality_blocks": high_blocks, "event_audit_status": event_status,
        "event_flags": ";".join(sorted(set(flags))),
    }
    return event_row, blocks


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            text_fields = {"analyst_question", "ceo_answer", "answer_context", "preceding_question_context"}
            writer.writerow({column: row.get(column, "") if column in text_fields else clean(row.get(column, "")) for column in columns})


def spread_across_events(blocks: Sequence[Mapping[str, object]], limit: int) -> list[Mapping[str, object]]:
    """Select blocks across the time-ordered event sample before taking repeats."""
    by_event: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for block in sorted(blocks, key=lambda block: (str(block["event_id"]), str(block["block_id"]))):
        by_event[str(block["event_id"])].append(block)
    event_groups = list(by_event.values())
    if limit <= 0 or not event_groups:
        return []
    if limit <= len(event_groups):
        indexes = [round(i * (len(event_groups) - 1) / max(1, limit - 1)) for i in range(limit)]
        return [event_groups[index][0] for index in indexes]
    selected = [group[0] for group in event_groups]
    remaining = [block for group in event_groups for block in group[1:]]
    remaining.sort(key=lambda block: (str(block["event_id"]), str(block["block_id"])))
    extra = limit - len(selected)
    if remaining and extra:
        indexes = [round(i * (len(remaining) - 1) / max(1, extra - 1)) for i in range(min(extra, len(remaining)))]
        selected.extend(remaining[index] for index in indexes)
    return selected[:limit]


def write_manual_review(path: Path, blocks: Sequence[Mapping[str, object]], limit: int) -> None:
    review = [block for block in blocks if block["quality_tier"] == "review"]
    flagged = [block for block in blocks if block["quality_tier"] == "usable_with_flag"]
    high = [block for block in blocks if block["quality_tier"] == "high"]
    selected: list[Mapping[str, object]] = []
    selected.extend(spread_across_events(review, min(len(review), limit)))
    remaining = limit - len(selected)
    flagged_target = min(len(flagged), max(0, round(limit * 0.30)))
    selected.extend(spread_across_events(flagged, min(flagged_target, remaining)))
    remaining = limit - len(selected)
    selected.extend(spread_across_events(high, remaining))
    # If a bucket is smaller than its allocation, fill the remaining slots from
    # unused blocks while preserving broad event coverage.
    if len(selected) < limit:
        seen = {str(block["block_id"]) for block in selected}
        fallback = [block for block in blocks if str(block["block_id"]) not in seen]
        selected.extend(spread_across_events(fallback, limit - len(selected)))
    lines = ["# CCTS CEO Q&A Block Manual Audit", "", "One section below is one analyst question, the CEO-only response, and its full management-answer context.", ""]
    for index, block in enumerate(selected, start=1):
        lines.extend([
            f"## Block {index}: `{block['block_id']}`", "",
            f"- Event: `{block['event_id']}` | {block['company_name']} | {block['year']}",
            f"- Analyst: {block['analyst_speaker']}", f"- CEO: {block['ceo_speaker']}",
            f"- Quality: `{block['quality_tier']}` | Flags: `{block['block_flags'] or 'none'}`", "",
            "### Analyst Question", "", str(block["analyst_question"]), "",
            "### CEO Answer", "", str(block["ceo_answer"]), "",
            "### Full Answer Context", "", str(block["answer_context"]), "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manual-review-blocks", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    turns_by_event = read_turns(args.turns)
    event_rows: list[dict[str, str]] = []
    blocks: list[dict[str, object]] = []
    for event_id in sorted(turns_by_event, key=int):
        event_row, event_blocks = extract_event(turns_by_event[event_id])
        event_rows.append(event_row)
        blocks.extend(event_blocks)

    write_csv(args.output_dir / "ceo_qa_event_audit.csv", EVENT_COLUMNS, event_rows)
    write_csv(args.output_dir / "ceo_qa_blocks.csv", BLOCK_COLUMNS, blocks)
    write_manual_review(args.output_dir / "manual_block_audit.md", blocks, args.manual_review_blocks)
    status_counts = Counter(row["event_audit_status"] for row in event_rows)
    quality_counts = Counter(str(block["quality_tier"]) for block in blocks)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_turns": str(args.turns), "events": len(event_rows), "blocks": len(blocks),
        "event_status_counts": dict(status_counts), "block_quality_counts": dict(quality_counts),
        "artifacts": {"event_audit_csv": str(args.output_dir / "ceo_qa_event_audit.csv"), "blocks_csv": str(args.output_dir / "ceo_qa_blocks.csv"), "manual_audit_md": str(args.output_dir / "manual_block_audit.md")},
        "caveat": "High means one shared title-labelled CEO in PRE and Q&A and no other management turn in the answer context. It is an audit tier, not a final CEO-identity proof.",
    }
    (args.output_dir / "ceo_qa_block_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# CCTS CEO Q&A Block Audit", "", f"- Events audited: {len(event_rows):,}", f"- Extracted CEO Q&A blocks: {len(blocks):,}", "", "## Event Status"]
    lines.extend(f"- `{key}`: {value:,}" for key, value in status_counts.most_common())
    lines.extend(["", "## Block Quality"])
    lines.extend(f"- `{key}`: {value:,}" for key, value in quality_counts.most_common())
    lines.extend(["", "## Caveat", "", summary["caveat"]])
    (args.output_dir / "ceo_qa_block_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
