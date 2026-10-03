#!/usr/bin/env python3
"""Run frozen specificity requests with a local Transformers causal LM.

This runner is deliberately provider-neutral and local-only. It preserves one
unit per request, validates the compact production contract without silently
repairing model output, and writes resumable row-level results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_execution_identity import prepare_backend
from specificity_validation import (decode_evidence, digest, require_schema,
                                    validate_result, validate_score, strict_json)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = strict_json(line)
            except ValueError as exc:
                raise SystemExit(f"invalid JSONL at line {line_number}: {exc}") from exc
            records.append(record)
    return records


def validate_request(record: dict[str, Any]) -> None:
    if set(record) != {"custom_id", "messages", "response_schema"}:
        raise SystemExit(f"unexpected request keys for {record.get('custom_id')}")
    if not isinstance(record["custom_id"], str) or not record["custom_id"]:
        raise SystemExit("request has empty custom_id")
    try:
        require_schema(record["response_schema"])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    messages = record["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise SystemExit(f"request must contain exactly two messages: {record['custom_id']}")
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise SystemExit(f"request roles must be system,user: {record['custom_id']}")
    if not all(isinstance(message.get("content"), str) for message in messages):
        raise SystemExit(f"request message content must be strings: {record['custom_id']}")
    user_input = strict_json(messages[1]["content"])
    if not isinstance(user_input, dict):
        raise SystemExit(f"user input must be an object: {record['custom_id']}")
    if user_input.get("unit_type") == "pre":
        if set(user_input) != {"unit_type", "ceo_presentation_segment"}:
            raise SystemExit(f"invalid PRE input shape: {record['custom_id']}")
    elif user_input.get("unit_type") == "qa":
        if set(user_input) != {"unit_type", "analyst_question", "ceo_answer"}:
            raise SystemExit(f"invalid Q&A input shape: {record['custom_id']}")
    else:
        raise SystemExit(f"invalid unit_type: {record['custom_id']}")
    if not all(isinstance(value, str) for value in user_input.values()):
        raise SystemExit(f"user input fields must be strings: {record['custom_id']}")


def validate_output(value: object) -> tuple[bool, str]:
    return validate_score(value)


def extract_json(raw_text: str) -> tuple[object | None, str | None]:
    try:
        return strict_json(raw_text), None
    except ValueError as exc:
        return None, f"invalid strict JSON: {exc}"


def termination_metadata(token_ids: list[int], eos_token_id, max_new_tokens: int) -> dict[str, Any]:
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = set(eos_token_id or [])
    ended = bool(token_ids and token_ids[-1] in eos_ids)
    truncated = not ended and len(token_ids) >= max_new_tokens
    return {
        "finish_reason": "eos_token" if ended else "length" if truncated else "unknown",
        "output_truncated": truncated,
        "generated_token_ids": token_ids,
        "configured_eos_token_ids": sorted(eos_ids),
    }


def context_limit(model_limit, tokenizer_limit) -> int:
    limits = [value for value in (model_limit, tokenizer_limit)
              if isinstance(value, int) and 0 < value < 1_000_000_000]
    if not limits:
        raise ValueError("no finite model/tokenizer context limit is available")
    return min(limits)


def atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def result_is_valid(result: dict[str, Any]) -> bool:
    return validate_result(result)[0]


def result_counts(expected_ids: set[str], completed: dict[str, dict[str, Any]],
                  processed: int) -> dict[str, int]:
    results = [completed[key] for key in expected_ids if key in completed]
    valid_count = sum(result_is_valid(row) for row in results)
    return {
        "requested": len(expected_ids),
        "processed_this_run": processed,
        "completed_status": sum(row.get("status") == "completed" for row in results),
        "row_errors": sum(row.get("status") == "error" for row in results),
        "strict_valid_outputs": valid_count,
        "invalid_or_error_outputs": len(results) - valid_count,
        "missing_outputs": len(expected_ids - completed.keys()),
        "unexpected_outputs": len(completed.keys() - expected_ids),
    }


def validate_only(input_path: Path) -> int:
    records = read_jsonl(input_path)
    custom_ids: set[str] = set()
    for record in records:
        validate_request(record)
        custom_id = record["custom_id"]
        if custom_id in custom_ids:
            raise SystemExit(f"duplicate custom_id: {custom_id}")
        custom_ids.add(custom_id)
    print(json.dumps({"status": "input_valid", "requests": len(records)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, help="existing local safetensors checkpoint directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--contract-version", default="specificity_v1_20260720")
    parser.add_argument("--settings", type=Path, help="reviewed decoding/runtime settings JSON")
    parser.add_argument("--execution-profile", type=Path, help="reviewed approved offline execution identity")
    parser.add_argument("--unit-manifest-csv", type=Path, help="exact unit metadata used by consumers")
    parser.add_argument("--capture-execution-profile", type=Path,
                        help="capture an unapproved candidate identity; load locally but do not score")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    input_hash = sha256_path(args.input_jsonl)
    records = read_jsonl(args.input_jsonl)
    if sha256_path(args.input_jsonl) != input_hash:
        raise SystemExit("input changed while being read")
    custom_ids: set[str] = set()
    for record in records:
        validate_request(record)
        custom_id = record["custom_id"]
        if custom_id in custom_ids:
            raise SystemExit(f"duplicate custom_id: {custom_id}")
        custom_ids.add(custom_id)
    if args.validate_only:
        print(json.dumps({"status": "input_valid", "requests": len(records)}, indent=2))
        return 0

    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise SystemExit("no scoring requests; an empty run cannot count as successful")
    if len({args.input_jsonl.resolve(), args.output_jsonl.resolve(), args.run_manifest.resolve()}) != 3:
        raise SystemExit("input, output and run manifest must be distinct files")
    custom_ids = {record["custom_id"] for record in records}

    if args.settings is None or (args.execution_profile is None and not args.capture_execution_profile):
        raise SystemExit("--settings and --execution-profile required before inference")
    if not args.capture_execution_profile and args.unit_manifest_csv is None:
        raise SystemExit("--unit-manifest-csv required to bind downstream metadata")
    readonly = [args.input_jsonl, args.settings, args.execution_profile, args.unit_manifest_csv]
    destinations = [args.output_jsonl, args.run_manifest, args.capture_execution_profile]
    paths = [p.resolve() for p in readonly + destinations if p is not None]
    if len(set(paths)) != len(paths):
        raise SystemExit("request/settings/profile/unit metadata and output destinations must be distinct")
    if args.run_manifest.exists() and not args.resume and not args.capture_execution_profile:
        raise SystemExit("run manifest already exists; choose a new path or matching --resume")
    if args.output_jsonl.exists() and not args.resume:
        raise SystemExit("output already exists; use a new path or an explicitly matching --resume")
    if args.resume and not args.capture_execution_profile and (
            not args.output_jsonl.exists() or not args.run_manifest.exists()):
        raise SystemExit("resume requires both original output and run manifest")
    unit_hash = None
    if args.unit_manifest_csv:
        unit_hash = sha256_path(args.unit_manifest_csv)
        with args.unit_manifest_csv.open(newline="", encoding="utf-8-sig") as handle:
            metadata = list(csv.DictReader(handle))
        ids = [row.get("custom_id") for row in metadata]
        all_request_ids = {r["custom_id"] for r in read_jsonl(args.input_jsonl)}
        if len(set(ids)) != len(ids) or set(ids) != all_request_ids:
            raise SystemExit("unit metadata IDs must match the frozen request input exactly")
    # Preflight must not rewrite an existing checkpoint or its first-run provenance.
    try:
        backend = prepare_backend(args)
    except (Exception, KeyboardInterrupt) as exc:
        if args.resume or args.run_manifest.exists() or args.capture_execution_profile:
            raise SystemExit(f"execution preflight rejected without modifying prior artifacts: {exc}") from exc
        interrupted_preflight = isinstance(exc, KeyboardInterrupt)
        failed = {"status": "interrupted_local_run" if interrupted_preflight else "failed_local_run", "stage": "preflight",
                  "fatal_error": f"{type(exc).__name__}: {exc}",
                  "counts": result_counts(custom_ids, {}, 0)}
        atomic_json_write(args.run_manifest, failed)
        return 130 if interrupted_preflight else 1
    identity = backend[2]
    if sha256_path(args.input_jsonl) != input_hash or (
            unit_hash is not None and sha256_path(args.unit_manifest_csv) != unit_hash):
        raise SystemExit("request or unit metadata changed during execution preflight")
    if args.capture_execution_profile:
        if args.capture_execution_profile.exists():
            raise SystemExit("candidate profile already exists; choose a new path")
        atomic_json_write(args.capture_execution_profile,
                          {"status": "candidate_requires_review", "identity": identity})
        return 0

    completed: dict[str, dict[str, Any]] = {}
    binding = {
        "input_sha256": input_hash,
        "runner_sha256": sha256_path(Path(__file__)),
        "model": args.model, "revision": args.revision or "unspecified",
        "contract": args.contract_version, "max_new_tokens": args.max_new_tokens,
        "seed": 0, "limit": args.limit,
        "execution_identity": identity,
        "execution_profile_sha256": sha256_path(args.execution_profile),
        "unit_manifest_sha256": unit_hash,
        "response_schema_hashes": {r["custom_id"]: digest(r["response_schema"]) for r in records},
        "validation_module_sha256": sha256_path(Path(__file__).with_name("specificity_validation.py")),
        "execution_module_sha256": sha256_path(Path(__file__).with_name("local_execution_identity.py")),
    }
    binding_hash = digest(binding)
    prior = None
    if args.resume and args.output_jsonl.exists():
        if not args.run_manifest.exists():
            raise SystemExit("resume requires original run manifest")
        prior = strict_json(args.run_manifest.read_text())
        if prior.get("run_binding") != binding or prior.get("run_binding_sha256") != binding_hash:
            raise SystemExit("resume execution identity differs; prior provenance left unchanged")
        if prior.get("output_sha256") is not None and prior["output_sha256"] != sha256_path(args.output_jsonl):
            raise SystemExit("resume output checksum differs; checkpoint left unchanged")
        for record in read_jsonl(args.output_jsonl):
            custom_id = record.get("custom_id")
            if custom_id not in custom_ids or custom_id in completed:
                raise SystemExit("resume file has an unknown or duplicate custom_id")
            if record.get("run_binding_sha256") != binding_hash:
                raise SystemExit("resume input/model/settings/runner binding differs or is absent")
            completed[custom_id] = record
    pending = [record for record in records if record["custom_id"] not in completed]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "created_at_utc": prior["created_at_utc"] if prior else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "running_local_run",
        "stage": "scoring",
        "input_jsonl": str(args.input_jsonl.resolve()),
        "input_sha256": input_hash,
        "output_jsonl": str(args.output_jsonl.resolve()),
        "output_sha256": None,
        "model": args.model,
        "revision": args.revision or "unspecified",
        "run_binding": binding,
        "run_binding_sha256": binding_hash,
        "output_validation": "whole_raw_JSON_no_salvage_duplicate_and_nonfinite_rejection",
        "completion_policy": "all_requested_ids_completed_schema_valid_and_eos_terminated",
        "contract": args.contract_version,
        "privacy": {"licensed_transcript_text_processed": True, "cloud_submitted": False},
        "counts": result_counts(custom_ids, completed, 0),
        "runtime": identity["runtime"],
        "effective_generation_config": identity["generation_config"],
        "actual_device_map": identity["device_map"],
        "first_run_provenance": (prior["first_run_provenance"] if prior else {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution_identity_sha256": digest(identity),
            "input_sha256": binding["input_sha256"], "run_binding_sha256": binding_hash}),
    }
    atomic_json_write(args.run_manifest, run_manifest)
    processed = 0
    fatal_error = None
    interrupted = False
    try:
        with args.output_jsonl.open("a", encoding="utf-8") as output_handle:
            for result in generate_results(args, pending, binding_hash, backend):
                output_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                completed[result["custom_id"]] = result
                processed += 1
                counts = run_manifest["counts"]
                valid = result_is_valid(result)
                counts["processed_this_run"] = processed
                counts["completed_status"] += int(result["status"] == "completed")
                counts["row_errors"] += int(result["status"] == "error")
                counts["strict_valid_outputs"] += int(valid)
                counts["invalid_or_error_outputs"] += int(not valid)
                counts["missing_outputs"] -= 1
                run_manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                atomic_json_write(args.run_manifest, run_manifest)
                print(f"{processed}/{len(pending)} {result['custom_id']} {result['status']} "
                      f"{result['validation_error'] or 'valid'}", flush=True)
    except KeyboardInterrupt:
        fatal_error = "KeyboardInterrupt: scoring interrupted"
        interrupted = True
    except Exception as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"

    counts = result_counts(custom_ids, completed, processed)
    success = (fatal_error is None and counts["strict_valid_outputs"] == counts["requested"]
               and counts["missing_outputs"] == 0 and counts["unexpected_outputs"] == 0)
    run_manifest.update({
        "status": ("interrupted_local_run" if interrupted else
                   "completed_local_run" if success else "failed_local_run"),
        "stage": "finished",
        "counts": counts,
        "fatal_error": fatal_error,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_sha256": sha256_path(args.output_jsonl) if args.output_jsonl.exists() else None,
    })
    atomic_json_write(args.run_manifest, run_manifest)
    print(json.dumps(run_manifest, indent=2))
    return 130 if interrupted else 0 if success else 1


def generate_results(args: argparse.Namespace, pending: list[dict[str, Any]],
                     binding_hash: str, backend):
    """Yield original row evidence; the caller persists it before reporting progress."""

    import torch
    tokenizer, model, identity = backend
    max_context = context_limit(getattr(model.config, "max_position_embeddings", None),
                                tokenizer.model_max_length)
    input_device = next(model.parameters()).device
    for record in pending:
        custom_id = record["custom_id"]
        started = time.perf_counter()
        result: dict[str, Any] = {
            "custom_id": custom_id,
            "status": "error",
            "raw_output": None,
            "parsed": None,
            "validation_error": None,
            "input_tokens": None,
            "output_tokens": None,
            "elapsed_seconds": None,
            "run_binding_sha256": binding_hash,
            "finish_reason": None,
            "output_truncated": None,
            "input_truncated": False,
            "input_context_rejected": False,
        }
        try:
            prompt = tokenizer.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=True
            )
            encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            input_length = int(encoded["input_ids"].shape[1])
            result["input_tokens"] = input_length
            if input_length + args.max_new_tokens > max_context:
                result["input_context_rejected"] = True
                raise ValueError("input plus output allowance exceeds context limit; not truncated")
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            output_tokens = generated[0, input_length:]
            eos = model.generation_config.eos_token_id
            if eos is None:
                eos = tokenizer.eos_token_id
            result.update(termination_metadata(output_tokens.tolist(), eos, args.max_new_tokens))
            result.update(decode_evidence(tokenizer, output_tokens.tolist(), result["configured_eos_token_ids"]))
            raw_output = result["raw_output"]
            parsed, parse_error = extract_json(raw_output)
            result["status"] = "completed"
            result["raw_output"] = raw_output
            result["input_tokens"] = input_length
            result["output_tokens"] = int(output_tokens.shape[0])
            if parse_error:
                result["validation_error"] = parse_error
            else:
                valid, validation_error = validate_output(parsed)
                result["parsed"] = parsed
                result["validation_error"] = None if valid else validation_error
            if result["output_truncated"]:
                result["validation_error"] = result["validation_error"] or "output reached token limit without EOS"
            elif result["finish_reason"] == "unknown":
                result["validation_error"] = result["validation_error"] or "generation termination is unknown"
            result["validation_error"] = result.get("token_evidence_error") or result["validation_error"]
        except Exception as exc:  # preserve row-level failure and continue
            result["validation_error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        yield result


if __name__ == "__main__":
    raise SystemExit(main())
