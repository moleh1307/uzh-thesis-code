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
from statistics import mean, median

import numpy as np


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
    estimates = []
    for _ in range(draws):
        pre_mean = mean(rng.choice(pre_scores) for _ in pre_scores)
        qa_mean = mean(rng.choice(qa_scores) for _ in qa_scores)
        estimates.append(pre_mean - qa_mean)
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
                    "pre_minus_qa_score": pre_row["score"] - qa_row["score"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coded-csv", type=Path, required=True)
    parser.add_argument("--key-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-caliper", type=int, default=15)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--lane", choices=("codex_proxy", "human_initial"), default="codex_proxy")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output directory is not empty: {args.output_dir}; use --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coded = read_csv(args.coded_csv)
    keys = {row["audit_id"]: row for row in read_csv(args.key_csv)}
    if len(coded) != 300 or set(keys) != {row["audit_id"] for row in coded}:
        raise ValueError("expected a reconciled 300-unit coded/key sample")
    if any(row["human_ok"] != "1" or row["human_specificity"] not in {"1", "2", "3", "4", "5"} for row in coded):
        raise ValueError("all 300 units must be scorable before this diagnostic")

    if args.lane == "human_initial":
        status = "initial_human_length_diagnostic_recode_pending"
        lane_label = "frozen initial human coding; delayed 60-unit recode pending"
        lane_caveat = "These are the thesis author's initial blinded human scores; intra-rater reliability remains pending."
        closeout = (
            "Do not interpret these initial-human estimates as thesis findings. Reassess after the delayed "
            "human recode and production-model scoring."
        )
    else:
        status = "local_codex_reviewer_length_diagnostic_not_human_validation"
        lane_label = "local Codex-assisted reviewer diagnostic, not human validation"
        lane_caveat = "These are Codex-assisted reviewer scores, not independent human gold labels."
        closeout = (
            "Do not interpret these reviewer-only estimates as thesis findings. Repeat the same diagnostics "
            "after independent human coding and model scoring."
        )

    rows = []
    for row in coded:
        key = keys[row["audit_id"]]
        rows.append(
            {
                "audit_id": row["audit_id"],
                "unit_type": row["unit_type"],
                "period_bin": key["period_bin"],
                "word_count": word_count(row),
                "score": int(row["human_specificity"]),
                "common_50_200": int(50 <= word_count(row) <= 200),
            }
        )

    diagnostics_path = args.output_dir / "specificity_length_unit_diagnostics.csv"
    write_csv(diagnostics_path, rows, list(rows[0]))
    rng = random.Random(args.seed)
    pre = [row for row in rows if row["unit_type"] == "pre"]
    qa = [row for row in rows if row["unit_type"] == "qa"]
    pre_common = [row for row in pre if row["common_50_200"]]
    qa_common = [row for row in qa if row["common_50_200"]]
    common_diff = mean(row["score"] for row in pre_common) - mean(row["score"] for row in qa_common)
    common_ci = independent_bootstrap_diff_ci(
        [row["score"] for row in pre_common],
        [row["score"] for row in qa_common],
        rng,
        args.bootstrap_draws,
    )

    matching = {}
    primary_pairs = []
    for caliper in (10, args.primary_caliper, 20, 25):
        pairs = match_by_period(rows, caliper)
        differences = [row["pre_minus_qa_score"] for row in pairs]
        ci = bootstrap_mean_ci(differences, rng, args.bootstrap_draws)
        matching[str(caliper)] = {
            "pairs": len(pairs),
            "mean_absolute_word_difference": mean(row["absolute_word_difference"] for row in pairs),
            "maximum_absolute_word_difference": max(row["absolute_word_difference"] for row in pairs),
            "pre_mean": mean(row["pre_score"] for row in pairs),
            "qa_mean": mean(row["qa_score"] for row in pairs),
            "mean_paired_pre_minus_qa": mean(differences),
            "bootstrap_ci95": list(ci),
        }
        if caliper == args.primary_caliper:
            primary_pairs = pairs

    pairs_path = args.output_dir / "specificity_length_matched_pairs.csv"
    write_csv(pairs_path, primary_pairs, list(primary_pairs[0]))
    additive_model = regression(rows)
    interaction_model = regression(rows, include_interactions=True)
    additive_contrast = qa_minus_pre_contrast(additive_model, 140)
    interaction_contrasts = [qa_minus_pre_contrast(interaction_model, words) for words in (100, 140, 180)]
    correlations = {}
    for label, subset in (("all", rows), ("pre", pre), ("qa", qa)):
        scores = [row["score"] for row in subset]
        words = [row["word_count"] for row in subset]
        correlations[label] = {
            "pearson": pearson(scores, words),
            "spearman": pearson(average_ranks(scores), average_ranks(words)),
        }

    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": status,
        "lane": args.lane,
        "counts": {
            "units": len(rows),
            "pre": len(pre),
            "qa": len(qa),
            "pre_common_50_200": len(pre_common),
            "qa_common_50_200": len(qa_common),
        },
        "score_distribution": dict(sorted(Counter(row["score"] for row in rows).items())),
        "correlations": correlations,
        "common_support": {
            "pre_mean": mean(row["score"] for row in pre_common),
            "qa_mean": mean(row["score"] for row in qa_common),
            "pre_minus_qa": common_diff,
            "independent_bootstrap_ci95": list(common_ci),
        },
        "period_and_length_matching": matching,
        "hc3_ols_additive": additive_model,
        "adjusted_qa_minus_pre_additive_at_140_words": additive_contrast,
        "hc3_ols_with_qa_length_interactions": interaction_model,
        "adjusted_qa_minus_pre_interaction_model": interaction_contrasts,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "coded_csv": str(args.coded_csv.resolve()),
            "coded_csv_sha256": sha256(args.coded_csv),
            "key_csv": str(args.key_csv.resolve()),
            "key_csv_sha256": sha256(args.key_csv),
            "seed": args.seed,
            "bootstrap_draws": args.bootstrap_draws,
            "primary_caliper_words": args.primary_caliper,
            "numpy_version": np.__version__,
        },
        "caveats": [
            lane_caveat,
            "Length matching and regression reduce observed length imbalance but cannot distinguish reviewer bias from genuine differences in available concrete detail.",
            "The 1-5 score is ordinal; OLS is a transparent quasi-cardinal diagnostic, not the primary thesis estimator.",
            "No transcript text or score was sent to an API or cloud service.",
        ],
    }
    summary_path = args.output_dir / "specificity_length_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    primary = matching[str(args.primary_caliper)]
    report = f"""# Specificity Length-Robustness Diagnostic

## Status

- Lane: {lane_label}
- Units: {len(rows)} ({len(pre)} PRE, {len(qa)} Q&A)
- Cloud/API submission: none
- Primary length-matching caliper: {args.primary_caliper} words, exact calendar-period bin

## Raw Length Signal

| Segment | Pearson(score, words) | Spearman(score, words) |
| --- | ---: | ---: |
| PRE | {correlations['pre']['pearson']:.4f} | {correlations['pre']['spearman']:.4f} |
| Q&A | {correlations['qa']['pearson']:.4f} | {correlations['qa']['spearman']:.4f} |

The Q&A score-length relationship remains material. It may combine a real opportunity-for-detail mechanism with reviewer sensitivity to length.

## Common 50-200 Word Support

- PRE: N={len(pre_common)}, mean={mean(row['score'] for row in pre_common):.4f}
- Q&A: N={len(qa_common)}, mean={mean(row['score'] for row in qa_common):.4f}
- PRE minus Q&A: {common_diff:.4f}, independent bootstrap 95% CI [{common_ci[0]:.4f}, {common_ci[1]:.4f}]

## Period And Length Matching

Primary matching yields {primary['pairs']} disjoint PRE/Q&A pairs. Mean absolute word difference is {primary['mean_absolute_word_difference']:.2f}; maximum is {primary['maximum_absolute_word_difference']}.

- Matched PRE mean: {primary['pre_mean']:.4f}
- Matched Q&A mean: {primary['qa_mean']:.4f}
- Paired PRE minus Q&A: {primary['mean_paired_pre_minus_qa']:.4f}
- Paired bootstrap 95% CI: [{primary['bootstrap_ci95'][0]:.4f}, {primary['bootstrap_ci95'][1]:.4f}]

Caliper sensitivity:

| Caliper | Pairs | Mean absolute word gap | PRE minus Q&A | Bootstrap 95% CI |
| ---: | ---: | ---: | ---: | ---: |
"""
    for caliper, item in matching.items():
        report += f"| {caliper} | {item['pairs']} | {item['mean_absolute_word_difference']:.2f} | {item['mean_paired_pre_minus_qa']:.4f} | [{item['bootstrap_ci95'][0]:.4f}, {item['bootstrap_ci95'][1]:.4f}] |\n"
    report += f"""

## Flexible Length Control

The additive HC3 OLS controls for centered log word count, squared centered log word count, and calendar-period fixed effects.

- Adjusted Q&A minus PRE at 140 words: {additive_contrast['estimate']:.4f}
- HC3 95% CI: [{additive_contrast['ci95_low']:.4f}, {additive_contrast['ci95_high']:.4f}]
- Model R-squared: {additive_model['r_squared']:.4f}

Because PRE and Q&A have different raw score-length slopes, a parsimonious interaction model allows a separate linear Q&A log-length slope while retaining common curvature:

| Words | Adjusted Q&A minus PRE | HC3 95% CI |
| ---: | ---: | ---: |
"""
    for contrast in interaction_contrasts:
        report += f"| {contrast['word_count']} | {contrast['estimate']:.4f} | [{contrast['ci95_low']:.4f}, {contrast['ci95_high']:.4f}] |\n"
    report += f"""

- Interaction-model R-squared: {interaction_model['r_squared']:.4f}

## Decision

This diagnostic does not validate the construct. It checks whether the descriptive PRE/Q&A contrast disappears under observed length balancing. The Q&A score-length association remains a mandatory robustness issue for later human/model agreement and full-sample analysis.

{closeout}
"""
    report_path = args.output_dir / "specificity_length_robustness_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
