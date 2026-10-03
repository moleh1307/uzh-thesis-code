"""Launch one explicitly selected stage in screen; no automatic scoring/retry."""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from run_local_specificity import atomic_json_write, sha256_path


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(config, stage, job, session):
    if not re.fullmatch(r"uzh-specificity-[A-Za-z0-9_.-]+", session):
        raise ValueError("session must be a unique uzh-specificity-* name")
    required = {"python", "runner", "input_jsonl", "unit_manifest_csv", "model", "revision", "settings", "contract_version"}
    if set(config) - (required | {"execution_profile"}) or not required <= set(config):
        raise ValueError("invalid job configuration fields")
    for key in ("python", "runner", "input_jsonl", "unit_manifest_csv", "model", "settings"):
        path = Path(config[key])
        if not path.is_absolute() or not path.exists():
            raise ValueError(f"missing/nonabsolute job path: {key}")
    if not Path(config["model"]).is_dir() or not os.access(config["python"], os.X_OK):
        raise ValueError("model directory and executable Python required")
    settings = json.loads(Path(config["settings"]).read_text())
    if config["revision"] != settings["model"]["revision"]:
        raise ValueError("revision/settings mismatch")
    command = [config["python"], config["runner"], "--input-jsonl", config["input_jsonl"],
        "--unit-manifest-csv", config["unit_manifest_csv"], "--model", config["model"],
        "--revision", config["revision"], "--settings", config["settings"],
        "--contract-version", config["contract_version"], "--max-new-tokens", str(settings["decoding"]["max_new_tokens"]),
        "--output-jsonl", str(job / "results.jsonl"), "--run-manifest", str(job / "run.json")]
    paths = [Path(config[key]) for key in ("runner", "input_jsonl", "unit_manifest_csv", "settings")]
    # Bind code dependencies too; copying just the runner is not a valid deployment.
    root = Path(config["runner"]).parent
    paths += [root / name for name in ("specificity_validation.py", "local_execution_identity.py")]
    paths.append(root.parent / "csv_contract.py")
    if stage == "validate":
        command.append("--validate-only")
    elif stage == "capture":
        command += ["--capture-execution-profile", str(job / "candidate-execution-profile.json")]
    elif stage == "score":
        profile = Path(config.get("execution_profile", ""))
        if not profile.is_absolute() or not profile.is_file() or json.loads(profile.read_text()).get("status") != "approved":
            raise ValueError("score requires a separately reviewed approved execution profile")
        paths.append(profile)
        command += ["--execution-profile", str(profile)]
    else:
        raise ValueError("invalid stage")
    return {"schema": 1, "created_at_utc": now(), "stage": stage, "session": session, "command": command,
            "input_sha256": {str(path): sha256_path(path) for path in paths}}


def worker(job):
    job = job.resolve()
    status_path = job / "status.json"
    status = {"stage": "preflight", "status": "starting", "worker_pid": os.getpid(), "started_at_utc": now()}
    process = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    previous = {value: signal.signal(value, interrupted) for value in (signal.SIGTERM, signal.SIGHUP)}
    atomic_json_write(status_path, status)
    try:
        execution = json.loads((job / "execution.json").read_text())
        for path, expected in execution["input_sha256"].items():
            if sha256_path(Path(path)) != expected:
                raise ValueError("job input/code identity changed after launch preparation")
        with (job / "execution.log").open("x") as log:
            process = subprocess.Popen(execution["command"], stdout=log, stderr=subprocess.STDOUT)
            status.update(status="running", stage=execution["stage"], runner_pid=process.pid,
                          session=execution["session"])
            atomic_json_write(status_path, status)
            code = process.wait()
        status.update(exit_code=code, status="completed" if code == 0 else "failed", finished_at_utc=now())
        if code == 0 and execution["stage"] == "capture":
            profile = json.loads((job / "candidate-execution-profile.json").read_text())
            if profile.get("status") != "candidate_requires_review":
                raise ValueError("capture did not produce a reviewable candidate")
            status["status"] = "candidate_requires_review"
        if code == 0 and execution["stage"] == "score":
            result = json.loads((job / "run.json").read_text())
            if result.get("status") != "completed_local_run":
                raise ValueError("scorer did not produce a completed run manifest")
        atomic_json_write(status_path, status)
        return code
    except (Exception, KeyboardInterrupt) as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at_utc=now())
        atomic_json_write(status_path, status)
        return 1
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--stage", choices=("validate", "capture", "score"))
    parser.add_argument("--confirm-scoring", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if any((args.config, args.job_dir, args.session, args.stage, args.dry_run, args.confirm_scoring)):
            parser.error("worker cannot be combined with launch options")
        return worker(args.worker)
    if not all((args.config, args.job_dir, args.session, args.stage)):
        parser.error("config, job-dir, session and stage required")
    if args.stage == "score" and not args.confirm_scoring:
        parser.error("score requires explicit --confirm-scoring")
    job = args.job_dir.resolve()
    if job.exists():
        raise FileExistsError("job directory already exists; no automatic resume, restart or overwrite")
    execution = prepare(json.loads(args.config.read_text()), args.stage, job, args.session)
    if args.dry_run:
        print(json.dumps(execution, indent=2))
        return 0
    sessions = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    if re.search(r"\d+\." + re.escape(args.session) + r"\s", sessions.stdout):
        raise ValueError("screen session already exists")
    job.mkdir(parents=True, exist_ok=False)
    atomic_json_write(job / "execution.json", execution)
    atomic_json_write(job / "status.json", {"status": "queued", "stage": args.stage, "session": args.session})
    # Lowercase -d forks a daemon; uppercase -D keeps this caller waiting.
    result = subprocess.run(["screen", "-dmS", args.session, sys.executable, str(Path(__file__).resolve()),
                             "--worker", str(job)], capture_output=True, text=True)
    if result.returncode:
        atomic_json_write(job / "status.json", {"status": "launch_failed", "error": result.stderr})
        return 1
    print(json.dumps({"status": "launch_requested", "stage": args.stage, "session": args.session,
                      "job_dir": str(job), "status_file": str(job / "status.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
