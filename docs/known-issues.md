# Known Issues And Readiness

As of 2026-10-03. Targeted repairs address issues #5, #1 and #6; this is not
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

Synthetic regressions cover these repairs. They do not validate real scoring
quality, authorize inference or repair the remaining issues below.

## Runner Repairs Required Before Inference

1. #2: Resume binding does not enforce runtime, actual model/tokenizer identity or
   the effective generation configuration. Recorded versions are not guards.
2. #3: Decoding with `skip_special_tokens=True` can hide unexpected control tokens
   before strict JSON validation. Preserve the full response and allow only
   verified terminal EOS removal.
3. #4: Resume skips previously failed IDs instead of performing the allowed
   same-settings technical retry. Preserve both attempt records when repaired.

`tests/known_behavior` reproduces these behaviors with fake inference. Convert
these tests into correct-behavior regression tests when making each repair.
Passing ordinary helper tests is not evidence that the full runner is sound.

The checkpoint inherits a repetition penalty of 1.05; the baseline does not
explicitly pin or guard it. Preserve and disclose it when repairing the runner,
rather than silently substituting 1.0. Verify greedy single-beam decoding too.
Atomic runner status is now implemented; the detached named-screen launcher
remains unimplemented. Changed runner hashes intentionally reject old-checkpoint
resume; do not overwrite frozen snapshots or rerun them to bypass that guard.
Do not launch inference from this code repair.

## Historical Or Protocol-Specific Tools

- `aggregate_local_specificity.py` retains historical PRE-minus-Q&A naming
  and old crosswalk assumptions. The planned current contrast is Q&A-minus-PRE
  on matched scorable support. Adapt and verify before using its outputs.
- #7: audit/aggregation still trust parsed scores rather than independently
  validating whole raw output and provenance. The repaired audit's technical pass
  is limited to its historical two-field contract, not current-contract readiness.
- Audit and length-robustness utilities need current schema/support checks.
  The length tool requires NumPy; its environment is not locked here yet.
- Fresh evaluation/freezing tools encode the specific 100-unit protocol and
  require separately retained source manifests and proposal materials. They
  are not general-purpose tools for arbitrary sample sizes.
- The annotation-package generator includes historical instructions. For the
  current convention, incidental courtesies do not alone imply `mixed`, and
  uninterpretable sources have missing scores, not procedural zero. Do not
  start a new annotation round from generated historical instructions.
- The settings JSON is a reference artifact, not configuration automatically
  consumed or enforced by the current runner.
- #14: the dashboard still cannot import/save reviewed source-uninterpretable
  rows with blank scores; it rejects them rather than imputing a numerical zero.

Other open findings (#8-#13 and historical #15-#16) are tracked on GitHub;
these three repairs do not imply those workflows have been corrected.

Only a reviewed subset of historical tests is migrated. No claim is made that
all original tests or the entire research pipeline are reproduced here.
