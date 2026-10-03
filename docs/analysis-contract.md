# Current Analysis Contract

As of 2026-10-03. This repairs consumers, not the frozen scoring prompt, schema,
model settings, source passages or human labels. Historical outputs remain unchanged.

## Sign And Support

Within each event and section, the primary score is
`sum(CEO_target_words * specificity) / sum(CEO_target_words)` over scored units.
Analyst-question words are not weights. The contrast is **Q&A minus PRE**.
Equal-unit means, medians and common 50-200-word support are sensitivity summaries.
A missing section produces a missing contrast, never zero or imputed data.

Across events, PRE, Q&A and contrast summaries use the same eligible event IDs
for each measure and equal event weights. Available-only section summaries are
separately named and must not be subtracted as the paired-event estimate.

For human/model comparisons, both call-level measures use exactly the same unit
IDs scored by both. Full-model and full-human support descriptions are separate.
Eligibility disagreements and source/technical missingness are not numerical
errors. Procedural human zero is excluded from specificity averages. Reviewed
uninterpretable sources retain blank status/score and a required note.
One-rater references are a comparator, not infallible ground truth or independent
validation. Sampled passages do not represent complete-call coverage.

## Inputs And Outputs

`prepare_scoring_unit_metadata.py` reconciles frozen request text with a local
source key by ID, type, word counts and content hash. It writes a separate,
label-free scoring manifest and receipt; it never edits frozen files.

```bash
python3 tools/llm_measurement/prepare_scoring_unit_metadata.py \
  --input-jsonl /path/to/frozen-requests.jsonl \
  --source-key /path/to/source-key.csv --output-dir /path/to/new-metadata
```

Aggregation requires exact request, output, unit metadata and run-manifest bindings.
For current references whose audit IDs equal request IDs, use `--human-coded`
without a key; a supplied key supports explicit legacy ID crosswalks.

```bash
python3 tools/llm_measurement/aggregate_local_specificity.py \
  --input-manifest /path/to/scoring_unit_metadata.csv \
  --input-jsonl /path/to/frozen-requests.jsonl \
  --output-jsonl /path/to/results.jsonl --run-manifest /path/to/run.json \
  --human-coded /path/to/human_reference_frozen.csv --output-dir /path/to/new-analysis
```

`specificity_call_level.csv` describes all model-scored units.
`specificity_model_common_support_call_level.csv` and
`specificity_human_call_level.csv` are the identical-support comparison.
`specificity_human_full_support_call_level.csv` is a separate human description.
The summary binds output CSV checksums and records support IDs and missingness.
Old `pre_minus_qa_*` fields are intentionally replaced by `qa_minus_pre_*`.

## Length Diagnostics

Use the pinned NumPy analysis environment. Human mode takes `--coded-csv`,
`--key-csv` and `--output-dir`; model mode takes `--lane model`,
`--unit-results-csv`, `--aggregation-summary`, `--key-csv` and `--output-dir`.
Model rows must be bound to a technically valid aggregation receipt.
Both modes enforce exact IDs, section/period metadata and CEO word counts.

These are descriptive unit-level diagnostics. Period/length matching can pair
different calls; it is not the primary within-call thesis design. Unit bootstrap
intervals and HC3 standard errors are not firm-clustered causal inference.
Empty support, rank deficiency and undefined statistics are reported as null or
`insufficient_support`, not as passing evidence. Exclusion ledgers distinguish
procedural units, source corruption and model unscorability. Seeds, settings,
input/script hashes and NumPy version are recorded. No further annotation round
is requested by this repair.
