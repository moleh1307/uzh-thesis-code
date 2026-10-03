"""Preserve complete per-person title evidence independently of turn ordering."""
import re

GATE_VERSION = "v1.3_complete_title_evidence_20261003"


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


def single_anchor(labels, speaker_key):
    values = [value.strip() for value in labels.split(";") if value.strip()]
    keys = {speaker_key({"text_name": value}) for value in values}
    if not values or len(keys) != 1 or not next(iter(keys)):
        raise ValueError("CEO title evidence must describe exactly one nonempty speaker key")
    return next(iter(keys))
