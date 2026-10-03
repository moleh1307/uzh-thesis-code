"""Current annotation convention; source missingness is never procedural zero."""
CLASSES = {"substantive", "mixed", "procedural_only", "uncertain"}


def human_reference(row):
    label = row.get("human_content_class", "").strip()
    ok, score = str(row.get("human_ok", "")).strip(), str(row.get("human_specificity", "")).strip()
    note = row.get("human_notes", "").strip()
    if label not in CLASSES:
        raise ValueError("missing/invalid current human content class")
    if label == "uncertain" and not ok and not score:
        if not note:
            raise ValueError("source-uninterpretable reference requires a note")
        return {"reference_state": "source_uninterpretable", "human_ok": None, "human_specificity": None}
    if label == "procedural_only" and ok == "0" and score == "0":
        return {"reference_state": "procedural_only", "human_ok": 0, "human_specificity": None}
    if label != "procedural_only" and ok == "1" and score in {"1", "2", "3", "4", "5"}:
        return {"reference_state": "scored", "human_ok": 1, "human_specificity": int(score)}
    raise ValueError("inconsistent or unfinished human reference")


CODING_RULES = """1. Classify only the CEO target, not courtesies in the analyst question.
   - substantive: interpretable CEO business information, assessments,
     expectations or explanations. Disclosure limits and ordinary answer
     organization are part of the business answer.
   - mixed: substantive content plus a distinct ceremonial/call-management
     passage, such as speaker self-identification, a formal welcome, handoff,
     or opening/closing a call segment. Brief thanks or greetings alone are incidental.
   - procedural_only: interpretable call administration/ceremony without business content.
   - uncertain: classification is uncertain, or the source cannot be interpreted.
2. For substantive/mixed and interpretable uncertain targets, set human_ok=1
   and human_specificity=1-5. Score the entire supplied CEO target; do not delete passages.
3. For procedural_only, set human_ok=0 and human_specificity=0. Zero is an
   eligibility marker, not a numerical specificity observation.
4. If the source is corrupted/incomplete and no assertion is recoverable
   without guessing, use uncertain, leave human_ok and human_specificity blank,
   and explain the source problem in human_notes. Do not impute zero.
5. PRE: score the CEO presentation segment. Q&A: use the analyst question as
   context but score the CEO answer only. Do not infer missing CEO content from the question.
6. Do not substitute correctness, favorability, confidence, outside knowledge,
   or answer responsiveness for specificity.
7. Specificity anchors (the unchanged model rubric):
   1 = generic/abstract with no materially bounding detail.
   2 = mostly vague with one weak detail.
   3 = mixed, with at least one meaningful concrete detail but important claims still broad.
   4 = concrete, with several relevant metrics, dates, milestones, mechanisms,
       conditions, or bounded commitments.
   5 = highly concrete throughout, with multiple precise and internally
       checkable details tightly bounding the claims.
   Here "mixed" in anchor 3 refers to specificity, not the content-class label.
"""
