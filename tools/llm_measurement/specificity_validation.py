"""Shared fail-closed contract and artifact checks; no historical output salvage."""
import hashlib
import json
import math
from pathlib import Path


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"ok": {"type": "integer", "enum": [0, 1]},
                   "specificity": {"type": "integer", "minimum": 0, "maximum": 5}},
    "required": ["ok", "specificity"],
    "oneOf": [{"properties": {"ok": {"const": 0}, "specificity": {"const": 0}}},
              {"properties": {"ok": {"const": 1}, "specificity": {"minimum": 1, "maximum": 5}}}],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError(f"duplicate JSON key: {k}")
            result[k] = v
        return result

    def nonfinite(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"non-finite JSON number: {value}")
        return result

    if not isinstance(text, str):
        raise ValueError("missing raw JSON text")
    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite, parse_float=finite_float)


def require_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("missing response schema")
    structural = {k: v for k, v in schema.items() if k not in {"title", "$schema", "description"}}
    if digest(structural) != digest(SCHEMA):
        raise ValueError("unsupported response schema; historical contracts cannot be upgraded implicitly")


def validate_score(value):
    if not isinstance(value, dict) or set(value) != {"ok", "specificity"}:
        return False, "output keys are not exactly ok,specificity"
    if type(value["ok"]) is not int or value["ok"] not in {0, 1}:
        return False, "ok is not integer 0 or 1"
    if type(value["specificity"]) is not int or not 0 <= value["specificity"] <= 5:
        return False, "specificity is not integer 0 through 5"
    if (value["ok"] == 0) != (value["specificity"] == 0):
        return False, "ok=0 requires specificity=0; ok=1 requires specificity 1 through 5"
    return True, ""


def decode_evidence(tokenizer, token_ids, eos_ids):
    """Remove exactly one verified terminal EOS, never arbitrary special tokens."""
    kwargs = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    full = tokenizer.decode(token_ids, **kwargs)
    terminal = token_ids[-1] if token_ids and token_ids[-1] in eos_ids else None
    specials = sorted(set(tokenizer.all_special_ids))
    if terminal is not None and terminal not in specials:
        raise ValueError("configured EOS is not a tokenizer special token")
    body = token_ids[:-1] if terminal is not None else token_ids
    payload = tokenizer.decode(body, **kwargs)
    suffix = tokenizer.decode([terminal], **kwargs) if terminal is not None else ""
    unexpected = [token for token in body if token in specials]
    evidence = {"raw_output": payload, "raw_output_with_special_tokens": full,
                "terminal_eos_token_id": terminal, "terminal_eos_text": suffix,
                "tokenizer_special_token_ids": specials, "unexpected_special_token_ids": unexpected}
    if full != payload + suffix:
        evidence["token_evidence_error"] = "terminal EOS decode is not an exact suffix"
    elif unexpected:
        evidence["token_evidence_error"] = "unexpected special/control token in generated payload"
    return evidence


def validate_result(row):
    if row is None:
        return False, "missing output"
    if row.get("status") != "completed" or row.get("validation_error") is not None:
        return False, str(row.get("validation_error") or "output status is not completed")
    if row.get("output_truncated") is not False or row.get("finish_reason") != "eos_token":
        return False, "missing or invalid EOS termination evidence"
    if row.get("input_truncated") is not False or row.get("input_context_rejected") is not False:
        return False, "input truncation/rejection evidence missing or invalid"
    try:
        ids = row["generated_token_ids"]
        specials = row["tokenizer_special_token_ids"]
        eos = row["terminal_eos_token_id"]
        configured = row["configured_eos_token_ids"]
        if (not ids or type(eos) is not int or
                not all(type(i) is int and i >= 0 for i in ids + specials + configured)
                or eos != ids[-1] or eos not in row["configured_eos_token_ids"]
                or eos not in specials or any(i in specials for i in ids[:-1])):
            raise ValueError("invalid generated-token/EOS evidence")
        if row.get("token_evidence_error") or row.get("unexpected_special_token_ids"):
            raise ValueError("unexpected special/control token or invalid decode evidence")
        if (not isinstance(row["terminal_eos_text"], str) or not row["terminal_eos_text"]
                or row["raw_output_with_special_tokens"] != row["raw_output"] + row["terminal_eos_text"]):
            raise ValueError("full decoded response differs from payload plus terminal EOS")
        parsed = strict_json(row.get("raw_output"))
        valid, error = validate_score(parsed)
        if not valid:
            return valid, error
        if digest(parsed) != digest(row.get("parsed")):
            raise ValueError("stored parsed output differs from strict whole-raw parse")
        return True, ""
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc)


def technical_retry_reason(row):
    if row.get("input_truncated") or row.get("input_context_rejected") or validate_result(row)[0]:
        return None
    if row.get("status") == "error":
        return "inference_error"
    if row.get("status") == "completed":
        return "output_truncation" if row.get("output_truncated") else "output_validation_failure"
    return None


def reconcile_attempts(rows):
    """Select final attempts by ID, never by score; retain original rows in the source."""
    selected = {}
    for row in rows:
        key = row.get("custom_id")
        attempt = row.get("attempt_number", 1)
        if not isinstance(key, str) or not key or type(attempt) is not int or attempt not in {1, 2}:
            raise ValueError("invalid attempt ID/number")
        previous = selected.get(key)
        if previous is None:
            if attempt != 1:
                raise ValueError("second attempt without original evidence")
            if row.get("retry_reason") is not None:
                raise ValueError("first attempt cannot claim a retry reason")
        else:
            reason = technical_retry_reason(previous)
            if (previous.get("attempt_number", 1) != 1 or attempt != 2 or not reason
                    or row.get("retry_reason") != reason
                    or previous.get("run_binding_sha256") != row.get("run_binding_sha256")):
                raise ValueError("duplicate, third, unbound or ineligible technical retry")
        selected[key] = row
    return selected


def attempt_ledger(rows):
    selected = reconcile_attempts(rows)
    return {"total_attempt_records": len(rows), "unique_request_ids": len(selected),
            "retried_ids": sorted(key for key, row in selected.items() if row.get("attempt_number") == 2),
            "first_attempt_invalid_ids": sorted(row["custom_id"] for row in rows
                if row.get("attempt_number", 1) == 1 and not validate_result(row)[0])}


def validate_retry_reservations(rows, reservations, request_ids):
    if not isinstance(reservations, dict):
        raise ValueError("retry reservations must be an object")
    originals = {row["custom_id"]: row for row in rows if row.get("attempt_number", 1) == 1}
    selected = reconcile_attempts(rows)
    for key, reservation in reservations.items():
        original = originals.get(key)
        if (key not in request_ids or not isinstance(reservation, dict)
                or type(reservation.get("attempt_number")) is not int
                or reservation["attempt_number"] != 2 or original is None
                or not technical_retry_reason(original)
                or reservation.get("retry_reason") != technical_retry_reason(original)):
            raise ValueError("invalid or ineligible retry reservation")
    for key, row in selected.items():
        if row.get("attempt_number") == 2 and key not in reservations:
            raise ValueError("second attempt lacks its persisted retry reservation")
    return sorted(key for key in reservations if selected[key].get("attempt_number", 1) != 2)


def verify_provenance(input_path, output_path, manifest_path, unit_manifest_path, requests, outputs):
    """File hashes bind exact bytes; row bindings bind each output to that run."""
    if manifest_path is None:
        raise ValueError("run manifest required for every supplied output")
    manifest = strict_json(Path(manifest_path).read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("run_binding"), dict):
        raise ValueError("missing object run binding")
    binding = manifest.get("run_binding", {})
    binding_hash = digest(binding)
    if manifest.get("run_binding_sha256") != binding_hash:
        raise ValueError("run manifest binding checksum mismatch")
    if (binding.get("input_sha256") != file_hash(input_path)
            or manifest.get("input_sha256") != file_hash(input_path)
            or manifest.get("output_sha256") != file_hash(output_path)
            or binding.get("unit_manifest_sha256") != file_hash(unit_manifest_path)):
        raise ValueError("input/output/unit-manifest provenance checksum mismatch")
    if (not isinstance(binding.get("execution_identity"), dict) or not binding["execution_identity"]
            or not isinstance(binding.get("contract"), str) or not binding["contract"]
            or manifest.get("contract") != binding.get("contract")):
        raise ValueError("missing execution identity or incompatible contract provenance")
    schemas = {}
    for row in requests:
        if not isinstance(row.get("custom_id"), str) or not row["custom_id"] or row["custom_id"] in schemas:
            raise ValueError("empty or duplicate request ID in provenance input")
        require_schema(row.get("response_schema"))
        schemas[row["custom_id"]] = digest(row["response_schema"])
    limit = binding.get("limit")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("invalid limited-scope binding")
    selected = requests if limit is None else requests[:limit]
    if len(selected) != len(requests):
        raise ValueError("limited run requires a separately frozen matching input; full-scope consumption refused")
    if binding.get("response_schema_hashes") != schemas:
        raise ValueError("response schema binding mismatch")
    for row in outputs:
        if row.get("run_binding_sha256") != binding_hash:
            raise ValueError("output row run identity mismatch")
    ledger = attempt_ledger(outputs)
    ledger["reserved_retries_without_result"] = validate_retry_reservations(
        outputs, manifest.get("retry_reservations", {}), schemas)
    return {"verified": True, "run_binding_sha256": binding_hash,
            "contract": binding["contract"], "attempt_ledger": ledger}
