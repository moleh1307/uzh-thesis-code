# Known Issues And Readiness

As of 2026-10-03. Migration preserves the reviewed runner; no scoring repair
has been folded silently into this baseline.

## Runner Repairs Required Before Inference

1. Invalid schema outputs and even all-row inference failures can still produce
   a success exit code and `completed_local_run` status.
2. Resume binding does not enforce runtime, actual model/tokenizer identity or
   the effective generation configuration. Recorded versions are not guards.
3. Decoding with `skip_special_tokens=True` can hide unexpected control tokens
   before strict JSON validation. Preserve the full response and allow only
   verified terminal EOS removal.
4. Resume skips previously failed IDs instead of performing the allowed
   same-settings technical retry. Preserve both attempt records when repaired.

`tests/known_behavior` reproduces these behaviors with fake inference. Convert
these tests into correct-behavior regression tests when making each repair.
Passing ordinary helper tests is not evidence that the full runner is sound.

The checkpoint inherits a repetition penalty of 1.05; the baseline does not
explicitly pin or guard it. Preserve and disclose it when repairing the runner,
rather than silently substituting 1.0. Verify greedy single-beam decoding too.
An atomic progress/final-status writer and detached named-screen launcher
remain to be implemented. Do not launch inference from this migration.

## Historical Or Protocol-Specific Tools

- `aggregate_local_specificity.py` retains historical PRE-minus-Q&A naming
  and old crosswalk assumptions. The planned current contrast is Q&A-minus-PRE
  on matched scorable support. Adapt and verify before using its outputs.
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

Only a reviewed subset of historical tests is migrated. No claim is made that
all original tests or the entire research pipeline are reproduced here.
