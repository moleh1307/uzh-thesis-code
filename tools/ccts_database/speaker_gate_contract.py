"""Byte-level binding for speaker-gate bundles and their external evidence."""
import hashlib
import re
from pathlib import Path


BUNDLE_VERSION = "speaker_gate_bound_bundle_v1"
INPUT_NAMES = (
    "turns", "event_assignments", "episode_manifest", "turnover_episode_map",
    "turnover_pairs", "execucomp_normalized", "call_sequences", "fetch_audit",
    "dedup_summary", "resolution_audit",
)
ARTIFACT_NAMES = (
    "event_speaker_gate_csv", "episode_speaker_gate_csv", "turnover_speaker_gate_csv",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_bindings(args):
    return {name: {"path": str(getattr(args, name).resolve()),
                   "sha256": sha256_file(getattr(args, name))}
            for name in INPUT_NAMES}


def verify_input_bindings(bindings, *, base_dir, turns_path=None, turns_sha256=None):
    if not isinstance(bindings, dict) or set(bindings) != set(INPUT_NAMES):
        raise ValueError("speaker gate input bindings are missing or incomplete; rebuild the gate")
    paths = {}
    for name in INPUT_NAMES:
        record = bindings[name]
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or not record["path"] or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError(f"invalid speaker gate input binding: {name}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = base_dir / path
        if name == "turns" and turns_path is not None:
            path = turns_path
        actual = turns_sha256 if name == "turns" and turns_sha256 is not None else sha256_file(path)
        if actual != record["sha256"]:
            raise ValueError(f"speaker gate input hash differs: {name}")
        paths[name] = path
    return paths


def verify_artifact_bindings(summary, paths):
    if summary.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("speaker gate bundle is unbound or unsupported; rebuild the gate")
    expected = summary.get("artifact_sha256")
    if not isinstance(expected, dict) or set(expected) != set(ARTIFACT_NAMES):
        raise ValueError("speaker gate artifact hashes are missing or incomplete")
    for name in ARTIFACT_NAMES:
        if sha256_file(paths[name]) != expected[name]:
            raise ValueError(f"speaker gate artifact hash differs: {name}")
