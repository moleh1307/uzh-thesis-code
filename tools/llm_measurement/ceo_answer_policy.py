"""CEO-owned answer policy and separately accounted Q&A coverage stages.

These functions consume verified source decisions/turn ledgers, not raw labels
or unaudited model responses. They do not classify speakers or infer content.
"""
from collections import Counter

POLICY_ID = "ceo_owned_answers_identified_management_v1_20261005"
CATEGORIES = {"assigned_answer", "procedure_only", "post_session_closing", "unresolved"}


def binary(value):
    if type(value) is int and value in (0, 1):
        return value
    if type(value) is str and value in ("0", "1"):
        return int(value)
    raise ValueError("Expected an explicit 0/1 source decision")


def answer_policy(decision):
    strict = binary(decision["primary_block_candidate"])
    broader = binary(decision["management_context_block_candidate"])
    status = decision["inclusion_status"]
    if strict and (not broader or status != "proposed_primary"):
        raise ValueError("Inconsistent strict-primary source decision")
    if broader and not strict and (status != "sensitivity_only" or
            decision["inclusion_reason"] != "identified_same_issuer_management_context"):
        raise ValueError("Broader inclusion requires identified management evidence")
    if not broader and status not in {"source_or_boundary_review", "exclude_procedural_or_empty"}:
        raise ValueError("Inconsistent excluded source decision")
    return {"policy_primary": broader, "strict_robustness": strict,
            "added_identified_management": int(broader and not strict)}


def qa_word_coverage(turn_ledger, eligible_blocks, score_status=None):
    """Account all raw CEO Q&A turns, including unresolved and excluded speech.

    A supplied score_status maps eligible block IDs to `valid`, `unscorable`,
    `invalid`, or `missing`, after technical/provenance validation upstream.
    None means inference has not run, not zero successful scores.
    """
    eligible_list = list(eligible_blocks)
    if len(eligible_list) != len(set(eligible_list)) or any(not x for x in eligible_list):
        raise ValueError("Duplicate/empty eligible block ID")
    eligible = set(eligible_list)
    totals, block_words = Counter(), Counter()
    seen = set()
    events = set()
    for row in turn_ledger:
        key = (row["event_id"], row["sequence_id"])
        if not all(key) or key in seen:
            raise ValueError("Duplicate/empty source turn identity")
        seen.add(key)
        events.add(row["event_id"])
        category, words = row["category"], row["word_count"]
        if category not in CATEGORIES or type(words) is not int or words < 0:
            raise ValueError("Invalid category or source word count")
        block = row.get("block_id", "")
        if (category == "assigned_answer") != bool(block):
            raise ValueError("Answer block identity and turn category disagree")
        totals[category] += words
        if block:
            block_words[block] += words
    if len(events) > 1:
        raise ValueError("Coverage must be computed separately for each call")
    if not eligible.issubset(block_words):
        raise ValueError("Eligible block absent from source ledger")
    words = sum(block_words[k] for k in eligible)
    raw = sum(totals.values())
    assigned = totals["assigned_answer"]
    unresolved = totals["unresolved"]
    ratio = lambda n, d: n / d if d else None
    result = {"raw_ceo_qa_words": raw, "assigned_answer_words": assigned,
              "procedure_only_words": totals["procedure_only"],
              "post_session_closing_words": totals["post_session_closing"],
              "unresolved_words": unresolved, "eligible_answer_words": words,
              "raw_qa_word_share": ratio(words, raw),
              "answer_inclusion_coverage": ratio(words, assigned),
              "answer_inclusion_if_all_unresolved_are_answers": ratio(words, assigned + unresolved),
              "answer_denominator_status": "unresolved_speech_present" if unresolved else "fully_accounted_in_supplied_ledger",
              "scoring_status": "not_run", "validly_scored_words": None,
              "scoring_coverage": None, "score_status_words": None}
    if score_status is not None:
        if set(score_status) != eligible or any(v not in {"valid", "unscorable", "invalid", "missing"}
                                               for v in score_status.values()):
            raise ValueError("Score status must account for every eligible block exactly")
        statuses = Counter()
        for block, status in score_status.items():
            statuses[status] += block_words[block]
        result.update(scoring_status="accounted", validly_scored_words=statuses["valid"],
                      scoring_coverage=ratio(statuses["valid"], words),
                      score_status_words={s: statuses[s] for s in ("valid", "unscorable", "invalid", "missing")})
    return result
