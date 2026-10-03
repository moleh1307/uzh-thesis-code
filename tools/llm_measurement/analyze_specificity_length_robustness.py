#!/usr/bin/env python3
"""Run local length-robustness diagnostics for blinded specificity scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csv_contract import configure_csv

configure_csv()
from statistics import mean, median

import numpy as np
from annotation_contract import human_reference
from aggregate_local_specificity import index_unique


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def word_count(row: dict[str, str]) -> int:
    text = row["ceo_answer"] if row["unit_type"] == "qa" else row["ceo_presentation_segment"]
    return len(text.split())


def pearson(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def bootstrap_mean_ci(values: list[float], rng: random.Random, draws: int) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    estimates = []
    for _ in range(draws):
        estimates.append(mean(rng.choice(values) for _ in values))
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]


def independent_bootstrap_diff_ci(
    pre_scores: list[float], qa_scores: list[float], rng: random.Random, draws: int
) -> tuple[float, float]:
    if not pre_scores or not qa_scores:
        return float("nan"), float("nan")
    estimates = []
    for _ in range(draws):
        pre_mean = mean(rng.choice(pre_scores) for _ in pre_scores)
        qa_mean = mean(rng.choice(qa_scores) for _ in qa_scores)
        estimates.append(qa_mean - pre_mean)
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]


def optimal_ordered_matches(
    pre_rows: list[dict], qa_rows: list[dict], caliper: int
) -> list[tuple[dict, dict]]:
    """Maximize pair count, then minimize total word-distance for ordered 1D data."""
    pre = sorted(pre_rows, key=lambda row: (row["word_count"], row["audit_id"]))
    qa = sorted(qa_rows, key=lambda row: (row["word_count"], row["audit_id"]))
    n, m = len(pre), len(qa)
    best = [[(0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    action = [["" for _ in range(m + 1)] for _ in range(n + 1)]

    def preferred(candidate: tuple[int, int], incumbent: tuple[int, int]) -> bool:
        return candidate[0] > incumbent[0] or (
            candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
        )

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            chosen = best[i - 1][j]
            chosen_action = "skip_pre"
            if preferred(best[i][j - 1], chosen):
                chosen = best[i][j - 1]
                chosen_action = "skip_qa"
            distance = abs(pre[i - 1]["word_count"] - qa[j - 1]["word_count"])
            if distance <= caliper:
                prior = best[i - 1][j - 1]
                candidate = (prior[0] + 1, prior[1] + distance)
                if preferred(candidate, chosen):
                    chosen = candidate
                    chosen_action = "match"
            best[i][j] = chosen
            action[i][j] = chosen_action

    matches = []
    i, j = n, m
    while i and j:
        if action[i][j] == "match":
            matches.append((pre[i - 1], qa[j - 1]))
            i -= 1
            j -= 1
        elif action[i][j] == "skip_pre":
            i -= 1
        else:
            j -= 1
    return list(reversed(matches))


def match_by_period(rows: list[dict], caliper: int) -> list[dict]:
    periods = sorted({row["period_bin"] for row in rows})
    output = []
    pair_number = 0
    for period in periods:
        pre = [row for row in rows if row["period_bin"] == period and row["unit_type"] == "pre"]
        qa = [row for row in rows if row["period_bin"] == period and row["unit_type"] == "qa"]
        for pre_row, qa_row in optimal_ordered_matches(pre, qa, caliper):
            pair_number += 1
            output.append(
                {
                    "pair_id": f"LMATCH_{pair_number:03d}",
                    "period_bin": period,
                    "pre_audit_id": pre_row["audit_id"],
                    "qa_audit_id": qa_row["audit_id"],
                    "pre_words": pre_row["word_count"],
                    "qa_words": qa_row["word_count"],
                    "absolute_word_difference": abs(pre_row["word_count"] - qa_row["word_count"]),
                    "pre_score": pre_row["score"],
                    "qa_score": qa_row["score"],
                    "qa_minus_pre_score": qa_row["score"] - pre_row["score"],
                }
            )
    return output


def hc3_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    adjusted = residuals / np.clip(1.0 - leverage, 1e-8, None)
    meat = x.T @ ((adjusted**2)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    r2 = 1.0 - float(residuals.T @ residuals) / float(((y - y.mean()) ** 2).sum())
    return beta, covariance, r2


def normal_p_value(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def regression(rows: list[dict], include_interactions: bool = False) -> dict:
    if not rows or {row["unit_type"] for row in rows} != {"pre", "qa"}:
        return {"status": "insufficient_support", "reason": "both sections required"}
    periods = sorted({row["period_bin"] for row in rows})
    reference_period = periods[0]
    log_words = np.log(np.asarray([row["word_count"] for row in rows], dtype=float))
    log_reference = math.log(140.0)
    centered = log_words - log_reference
    qa = np.asarray([1.0 if row["unit_type"] == "qa" else 0.0 for row in rows])
    columns = [np.ones(len(rows)), qa, centered, centered**2]
    names = ["intercept_pre_at_140_words", "qa_indicator", "log_words_centered", "log_words_centered_squared"]
    if include_interactions:
        columns.append(qa * centered)
        names.append("qa_x_log_words_centered")
    for period in periods[1:]:
        columns.append(np.asarray([1.0 if row["period_bin"] == period else 0.0 for row in rows]))
        names.append(f"period_{period}")
    x = np.column_stack(columns)
    y = np.asarray([row["score"] for row in rows], dtype=float)
    if len(rows) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1] or np.std(y) == 0:
        return {"status": "insufficient_support", "reason": "rank, residual degrees of freedom or score variation"}
    leverage = np.sum((x @ np.linalg.pinv(x.T @ x)) * x, axis=1)
    if np.any(leverage >= 1 - 1e-8):
        return {"status": "insufficient_support", "reason": "unit leverage makes HC3 undefined"}
    beta, covariance, r2 = hc3_ols(x, y)
    estimates = []
    for index, name in enumerate(names):
        standard_error = math.sqrt(max(0.0, float(covariance[index, index])))
        z_value = float(beta[index]) / standard_error if standard_error else float("nan")
        estimates.append(
            {
                "term": name,
                "estimate": float(beta[index]),
                "hc3_standard_error": standard_error,
                "ci95_low": float(beta[index]) - 1.96 * standard_error,
                "ci95_high": float(beta[index]) + 1.96 * standard_error,
                "normal_approx_p": normal_p_value(z_value),
            }
        )
    return {
        "status": "estimated_diagnostic",
        "n": len(rows),
        "reference_period": reference_period,
        "word_reference": 140,
        "includes_qa_length_interactions": include_interactions,
        "r_squared": r2,
        "estimates": estimates,
        "coefficient_names": names,
        "coefficients": beta.tolist(),
        "hc3_covariance": covariance.tolist(),
    }


def qa_minus_pre_contrast(model: dict, words: int) -> dict:
    if model.get("status") == "insufficient_support":
        return {"status": "insufficient_support", "word_count": words, "estimate": None}
    names = model["coefficient_names"]
    vector = np.zeros(len(names))
    vector[names.index("qa_indicator")] = 1.0
    centered = math.log(float(words)) - math.log(float(model["word_reference"]))
    if model["includes_qa_length_interactions"]:
        vector[names.index("qa_x_log_words_centered")] = centered
    beta = np.asarray(model["coefficients"])
    covariance = np.asarray(model["hc3_covariance"])
    estimate = float(vector @ beta)
    standard_error = math.sqrt(max(0.0, float(vector @ covariance @ vector)))
    z_value = estimate / standard_error if standard_error else float("nan")
    return {
        "word_count": words,
        "estimate": estimate,
        "hc3_standard_error": standard_error,
        "ci95_low": estimate - 1.96 * standard_error,
        "ci95_high": estimate + 1.96 * standard_error,
        "normal_approx_p": normal_p_value(z_value),
    }


def load_diagnostic_rows(args):
    keys_raw = read_csv(args.key_csv)
    key_field = "custom_id" if keys_raw and "custom_id" in keys_raw[0] else "audit_id"
    keys = index_unique(keys_raw, key_field, "length metadata")
    if args.lane == "human_reference":
        if not args.coded_csv or args.unit_results_csv:
            raise ValueError("human_reference requires --coded-csv, not --unit-results-csv")
        raw = read_csv(args.coded_csv)
        id_field = "audit_id"
    else:
        if not args.unit_results_csv or not args.aggregation_summary or args.coded_csv:
            raise ValueError("model requires --unit-results-csv and --aggregation-summary")
        summary = json.loads(args.aggregation_summary.read_text())
        if summary.get("status") != "completed_diagnostic_aggregation":
            raise ValueError("model length diagnostics require technically valid aggregation")
        if summary.get("output_sha256", {}).get("specificity_unit_results.csv") != sha256(args.unit_results_csv):
            raise ValueError("unit results are not bound to the aggregation receipt")
        raw = read_csv(args.unit_results_csv)
        id_field = "custom_id"
    indexed = index_unique(raw, id_field, "length observations")
    if not indexed or set(indexed) != set(keys):
        raise ValueError("observation/metadata identity coverage mismatch")
    rows, exclusions = [], []
    for unit_id, row in indexed.items():
        key = keys[unit_id]
        kind = row.get("unit_type")
        if kind not in {"pre", "qa"} or key.get("unit_type") != kind or not key.get("period_bin"):
            raise ValueError(f"invalid section/period metadata: {unit_id}")
        if args.lane == "human_reference":
            reference = human_reference(row)
            state = reference["reference_state"]
            score = reference["human_specificity"]
            words = word_count(row)
        else:
            for field in ("event_id", "start_date"):
                if key.get(field) and row.get(field) != key[field]:
                    raise ValueError(f"model/metadata {field} mismatch: {unit_id}")
            if row.get("status") != "completed" or row.get("model_validation_error"):
                raise ValueError(f"technical failure in model unit ledger: {unit_id}")
            ok, value = row.get("model_ok"), row.get("model_specificity")
            if (ok, value) not in ({("0", "0")} | {("1", str(i)) for i in range(1, 6)}):
                raise ValueError(f"invalid current model score schema: {unit_id}")
            if row.get("scored") != ("True" if ok == "1" else "False"):
                raise ValueError(f"inconsistent scored flag: {unit_id}")
            state = "scored" if ok == "1" else "model_unscorable"
            score = int(value) if ok == "1" else None
            words = int(row["unit_word_count"])
        expected_words = key.get("unit_word_count", key.get("target_word_count", ""))
        if expected_words == "" or int(expected_words) != words:
            raise ValueError(f"word-count metadata mismatch: {unit_id}")
        if state != "scored":
            exclusions.append({"audit_id": unit_id, "unit_type": kind, "reference_state": state})
            continue
        if words <= 0:
            raise ValueError(f"scored target has no words: {unit_id}")
        rows.append({"audit_id": unit_id, "unit_type": kind, "period_bin": key["period_bin"],
                     "word_count": words, "score": score, "common_50_200": int(50 <= words <= 200)})
    return rows, exclusions


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded-csv", type=Path)
    parser.add_argument("--unit-results-csv", type=Path)
    parser.add_argument("--aggregation-summary", type=Path)
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-caliper", type=int, default=15)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--lane", choices=("human_reference", "model"), default="human_reference")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.primary_caliper < 0 or args.bootstrap_draws < 1:
        parser.error("caliper must be nonnegative and bootstrap draws positive")
    return args


def main():
    args = parse_args()
    rows, excluded = load_diagnostic_rows(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise ValueError("output directory is not empty; use a fresh directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    unit_fields = ["audit_id", "unit_type", "period_bin", "word_count", "score", "common_50_200"]
    write_csv(args.output_dir / "specificity_length_unit_diagnostics.csv", rows, unit_fields)
    write_csv(args.output_dir / "specificity_length_exclusions.csv", excluded,
              ["audit_id", "unit_type", "reference_state"])
    rng = random.Random(args.seed)
    pre = [row for row in rows if row["unit_type"] == "pre"]
    qa = [row for row in rows if row["unit_type"] == "qa"]
    pre_common = [row for row in pre if row["common_50_200"]]
    qa_common = [row for row in qa if row["common_50_200"]]
    avg = lambda subset, field: mean(row[field] for row in subset) if subset else None
    common_diff = (avg(qa_common, "score") - avg(pre_common, "score")
                   if pre_common and qa_common else None)
    common_ci = independent_bootstrap_diff_ci([row["score"] for row in pre_common],
                                             [row["score"] for row in qa_common], rng, args.bootstrap_draws)
    matching, primary_pairs = {}, []
    for caliper in dict.fromkeys((10, args.primary_caliper, 20, 25)):
        pairs = match_by_period(rows, caliper)
        differences = [row["qa_minus_pre_score"] for row in pairs]
        matching[str(caliper)] = {
            "pairs": len(pairs), "mean_absolute_word_difference": avg(pairs, "absolute_word_difference"),
            "maximum_absolute_word_difference": max((row["absolute_word_difference"] for row in pairs), default=None),
            "pre_mean": avg(pairs, "pre_score"), "qa_mean": avg(pairs, "qa_score"),
            "mean_paired_qa_minus_pre": mean(differences) if differences else None,
            "bootstrap_ci95": bootstrap_mean_ci(differences, rng, args.bootstrap_draws)}
        if caliper == args.primary_caliper:
            primary_pairs = pairs
    write_csv(args.output_dir / "specificity_length_matched_pairs.csv", primary_pairs,
        ["pair_id", "period_bin", "pre_audit_id", "qa_audit_id", "pre_words", "qa_words",
         "absolute_word_difference", "pre_score", "qa_score", "qa_minus_pre_score"])
    additive, interaction = regression(rows), regression(rows, include_interactions=True)
    correlations = {}
    for label, subset in (("all", rows), ("pre", pre), ("qa", qa)):
        scores, words = [row["score"] for row in subset], [row["word_count"] for row in subset]
        correlations[label] = {"pearson": pearson(scores, words),
                               "spearman": pearson(average_ranks(scores), average_ranks(words))}
    inputs = {"key_csv": args.key_csv}
    for key in ("coded_csv", "unit_results_csv", "aggregation_summary"):
        if getattr(args, key):
            inputs[key] = getattr(args, key)
    summary = json_safe({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_descriptive_length_diagnostic" if pre and qa else "insufficient_section_support",
        "lane": args.lane, "contrast": "Q&A minus PRE",
        "counts": {"units": len(rows), "excluded": len(excluded), "pre": len(pre), "qa": len(qa),
                   "pre_common_50_200": len(pre_common), "qa_common_50_200": len(qa_common)},
        "score_distribution": dict(sorted(Counter(row["score"] for row in rows).items())),
        "correlations": correlations,
        "common_support": {"pre_mean": avg(pre_common, "score"), "qa_mean": avg(qa_common, "score"),
                           "qa_minus_pre": common_diff, "independent_unit_bootstrap_ci95": common_ci},
        "period_and_length_matching": matching,
        "hc3_ols_additive": additive,
        "adjusted_qa_minus_pre_additive_at_140_words": qa_minus_pre_contrast(additive, 140),
        "hc3_ols_with_qa_length_interactions": interaction,
        "adjusted_qa_minus_pre_interaction_model": [qa_minus_pre_contrast(interaction, w) for w in (100, 140, 180)],
        "provenance": {"inputs": {key: str(path.resolve()) for key, path in inputs.items()},
                       "input_sha256": {key: sha256(path) for key, path in inputs.items()},
                       "script_sha256": sha256(Path(__file__)), "seed": args.seed,
                       "helper_sha256": {name: sha256(Path(__file__).parent / name)
                                          for name in ("annotation_contract.py", "aggregate_local_specificity.py")},
                       "bootstrap_draws": args.bootstrap_draws, "primary_caliper_words": args.primary_caliper,
                       "numpy_version": np.__version__},
        "caveats": ["Descriptive unit-level diagnostic, not construct validation or a causal thesis estimator.",
                    "Period/length matching can pair units from different calls; it is not the primary within-call contrast.",
                    "Bootstrap intervals resample units, not firms/calls; HC3 is not firm-clustered inference.",
                    "Undefined/empty-support metrics are null, never zero or evidence of passing.",
                    "Human references are one-rater measurements, not infallible ground truth.",
                    "No transcript text or scores are sent to an API or cloud service."]})
    path = args.output_dir / "specificity_length_robustness_summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = ["# Specificity Length-Robustness Diagnostic", "",
              f"- Status: {summary['status']}", f"- Lane: {args.lane}",
              f"- Scored units: {len(rows)}; excluded: {len(excluded)}",
              "- Contrast: Q&A minus PRE. Empty or undefined support is reported as null.", "",
              "## Common 50-200 Word Support", "",
              json.dumps(summary["common_support"], indent=2, allow_nan=False), "",
              "## Limits", "", *["- " + item for item in summary["caveats"]], "",
              "Full matching, regression, correlation and input-identity evidence is in the JSON summary."]
    (args.output_dir / "specificity_length_robustness_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
