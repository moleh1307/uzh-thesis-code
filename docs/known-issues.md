# Known Issues And Readiness

As of 2026-10-03. Targeted repairs or explicit deprecation address all sixteen tracked findings; this is not
production scoring approval. Prompt, schema and model settings are unchanged.

## Targeted Repairs

- #5: dashboard startup detects CSV/progress conflicts without rewriting either
  file; explicit CSV import validates labels/source and backs up conflicting
  progress. External edits block save/export. See [workflow](workflow.md).
- #1: runner success requires all requested IDs to have completed, schema-valid,
  nontruncated EOS-terminated results. Row failures, model-load failures and
  interruption produce non-success statuses/exits. Progress/final/error manifests
  are written atomically; row evidence is retained and flushed before progress.
- #6: audit fails incomplete/invalid runs, counts score distributions only from
  valid rows, and compares repeats only on valid comparable support. Missing and
  unexpected IDs, full requested counts and repeat exclusion counts are retained.
  Valid `ok=0` is an eligibility edge case, not a numerical score or execution error.
- #2: explicitly offline loading requires a reviewed execution profile binding
  settings, checkpoint bytes, runtime, tokenizer/template, effective generation
  configuration and actual device placement. Resume rejects identity drift before
  modifying prior artifacts and preserves first-run provenance.
- #3: full decoded text and generated IDs are preserved; only one verified terminal
  EOS is removed. Unexpected special tokens and non-JSON payloads are invalid.
- #7: both consumers check whole raw JSON, stored parsed equality, token/EOS evidence,
  exact schema and input/output/unit-metadata run bindings. Old unbound results are
  refused, not retrospectively certified. See [details](execution-validation.md).
- #14: reviewed source-uninterpretable annotations use `uncertain`, blank status
  and score, and a required source-quality note. They persist/import/export as
  reviewed missingness, distinct from unfinished rows and procedural zero.
- #4: ordinary resume still skips recorded attempts. Explicit `--resume
  --technical-retry` permits one additional same-settings technical attempt,
  with append-only evidence and pre-inference reservations. Consumers select
  the final attempt by ID, never by score, and retain first-failure ledgers.
- #8: CEO candidates retain all observed per-person labels across PRE/Q&A.
  Interim/acting/co-CEO evidence cannot be overwritten by a later ordinary
  title; Acting CFO alone is not Acting CEO. The gate version changed; see
  [reproduction](reproduction.md) before using a new gate.
- #9: exact CEO panels require one nonempty GVKEY for the CUSIP8 bridge and
  one covering exact tenure. Multiple/missing issuer identities stay in review,
  even if only one tenure covers the date; turnover windows recheck the guard.
- #10: tenure boundaries are parsed and checked within each contiguous annual
  episode. Equivalent date formats agree; conflicting, invalid or incompatible
  values remain in the raw-value audit and cannot be selected as exact min/max.
- #11: census runs bind mode, schemas, ordered event scope, filter bytes,
  database identity and code. Validated atomic count batches commit with hash
  markers; incomplete cells, duplicates and incompatible/unbound checkpoints
  are rejected. Uncomputed section-only signals are blank/absent/null, not zero.
- #12: fetched turns and audits materialize in manifest event order even when
  replacement batches span gaps between old shards. Within-event row order is
  retained; temporary disk-backed sorting avoids loading full transcript text.
- #13: each standalone CSV entry point configures the shared decoded-character
  bound of 64 Mi characters. Larger fields fail explicitly, never truncate.
- #15 (historical CLI): both sides require distinct confirmed events, consistent
  intended CEO identities, different executives and old-before-new call dates.
  Missing, duplicate, mislabelled or inconsistent support remains in review.
- #16 (historical PDF): deprecation option selected. The public compatibility
  entry point refuses every invocation without changing files. Archived private
  scripts are reference-only, not repaired or certified for future resume.

Synthetic regressions cover these repairs. They do not validate real scoring
quality or authorize inference. See [data integrity](data-integrity.md) for
resume compatibility and historical-workflow limits.

## Execution Limits

The old retry characterization test has been replaced by correct-behavior
regressions with fake inference. Passing these tests is not production approval.

The checkpoint inherits a repetition penalty of 1.05; it is now explicitly
guarded, alongside greedy single-beam decoding, and disclosed in the effective config.
Atomic runner status and an explicit detached named-screen launcher are
implemented. The launcher separates validation, profile capture and scoring;
capture is not approval and scoring requires explicit confirmation. Changed runner hashes intentionally reject old-checkpoint
resume; do not overwrite frozen snapshots or rerun them to bypass that guard.
Do not launch inference from this code repair.

## Historical Or Protocol-Specific Tools

- `aggregate_local_specificity.py` now uses Q&A-minus-PRE, CEO-word-weighted
  section means and exact common human/model scorable unit support for comparison.
  Available-section descriptions are separate from paired-event contrasts.
  Existing historical outputs are not rewritten. See [analysis contract](analysis-contract.md).
- The implemented production schema is exactly `ok` and `specificity`, not a
  three-field schema. Other historical schemas are deliberately incompatible.
- Length diagnostics accept current frozen references or receipt-bound model
  unit results, retain exclusion ledgers and report undefined statistics as null.
  NumPy is pinned in `requirements-analysis.txt`; use a separate Python >=3.10
  analysis environment, not the existing Python 3.9 GPU runtime. Unit bootstrap
  and HC3 results are descriptive, not firm-clustered thesis inference.
- Fresh evaluation/freezing tools encode the specific 100-unit protocol and
  require separately retained source manifests and proposal materials. They
  are not general-purpose tools for arbitrary sample sizes.
- The annotation-package generator now uses the current shared convention:
  incidental courtesies do not alone imply `mixed`, and uninterpretable sources
  have missing scores, not procedural zero. Existing annotations are unchanged;
  this repair does not request another annotation round.
- The settings JSON is consumed explicitly with `--settings`; its migration status
  still grants no execution approval. The reviewed profile is a separate required gate.
Closing tracked software findings does not resolve scientific/protocol
limitations or certify historical outputs. No historical output is recertified.

Only a reviewed subset of historical tests is migrated. No claim is made that
all original tests or the entire research pipeline are reproduced here.
