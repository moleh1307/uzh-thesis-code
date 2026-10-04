"""Preserve complete per-person title evidence independently of turn ordering."""
import json
import re

GATE_VERSION = "v1.4_structured_title_evidence_20261004"
SPEAKER_LABEL_FIELDS = (
    "pre_ceo_speakers", "qa_ceo_speakers", "shared_ceo_speakers",
    "matched_pre_speakers", "matched_qa_speakers", "matched_shared_speakers",
)


SPECIAL_CEO_RE = re.compile(
    r"\b(?:co[-\s]?(?:ceo|chief\s+executive(?:\s+officer)?)|"
    r"(?:interim|acting)\s+(?:as\s+)?(?:the\s+)?"
    r"(?:(?:president|chairman|chair|cfo|chief\s+financial\s+officer)\s*(?:and|&|/)\s*)*"
    r"(?:(?:group|global)\s+)?(?:co[-\s]?)?(?:ceo|chief\s+executive(?:\s+officer)?))\b", re.I)


def candidate_evidence(rows, is_ceo, speaker_key, label):
    rows = list(rows)
    keys = {speaker_key(row) for row in rows if is_ceo(row) and speaker_key(row)}
    return {key: tuple(sorted({label(row) for row in rows if speaker_key(row) == key}))
            for key in sorted(keys)}


def evidence_labels(mapping, keys=None):
    return sorted({text for key in (sorted(mapping) if keys is None else keys)
                   for text in ((mapping[key],) if isinstance(mapping[key], str) else mapping[key])})


def shared_evidence(pre, qa, keys):
    return sorted(set(evidence_labels(pre, keys)) | set(evidence_labels(qa, keys)))


def validate_labels(values):
    if (not isinstance(values, (list, tuple))
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(values) != len(set(values))):
        raise ValueError("speaker evidence must be a list of distinct nonblank labels")
    return list(values)


def encode_labels(values):
    return json.dumps(validate_labels(values), ensure_ascii=True, separators=(",", ":"))


def label_evidence(row, field):
    try:
        values = validate_labels(json.loads(row[field + "_json"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid structured speaker evidence: {field}") from exc
    if row.get(field) != "; ".join(values):
        raise ValueError(f"speaker evidence display differs from structured labels: {field}")
    return values


def single_anchor(labels, speaker_key):
    values = validate_labels(labels)
    keys = {speaker_key({"text_name": value}) for value in values}
    if not values or len(keys) != 1 or not next(iter(keys)):
        raise ValueError("CEO title evidence must describe exactly one nonempty speaker key")
    return next(iter(keys))
