# Known Issues And Readiness

As of 2026-10-03. Targeted repairs address issues #5, #1, #6, #2, #3, #7, #14, #4 and #8; this is not
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

Synthetic regressions cover these repairs. They do not validate real scoring
quality, authorize inference or repair the remaining issues below.

## Execution Limits

The old retry characterization test has been replaced by correct-behavior
regressions with fake inference. Passing these tests is not production approval.

The checkpoint inherits a repetition penalty of 1.05; it is now explicitly
guarded, alongside greedy single-beam decoding, and disclosed in the effective config.
Atomic runner status is now implemented; the detached named-screen launcher
remains unimplemented. Changed runner hashes intentionally reject old-checkpoint
resume; do not overwrite frozen snapshots or rerun them to bypass that guard.
Do not launch inference from this code repair.

## Historical Or Protocol-Specific Tools

- `aggregate_local_specificity.py` retains historical PRE-minus-Q&A naming
  and old crosswalk assumptions. The planned current contrast is Q&A-minus-PRE
  on matched scorable support. Adapt and verify before using its outputs.
- The implemented production schema is exactly `ok` and `specificity`, not a
  three-field schema. Other historical schemas are deliberately incompatible.
- Length-robustness utilities still need current schema/support checks.
  The length tool requires NumPy; its environment is not locked here yet.
- Fresh evaluation/freezing tools encode the specific 100-unit protocol and
  require separately retained source manifests and proposal materials. They
  are not general-purpose tools for arbitrary sample sizes.
- The annotation-package generator includes historical instructions. For the
  current convention, incidental courtesies do not alone imply `mixed`, and
  uninterpretable sources have missing scores, not procedural zero. Do not
  start a new annotation round from generated historical instructions.
- The settings JSON is consumed explicitly with `--settings`; its migration status
  still grants no execution approval. The reviewed profile is a separate required gate.
Other open findings (#9-#13 and historical #15-#16) are tracked on GitHub;
these targeted repairs do not imply those workflows have been corrected.

Only a reviewed subset of historical tests is migrated. No claim is made that
all original tests or the entire research pipeline are reproduced here.
