# UZH Thesis Code

Public code-only workspace for studying CEO communication in prepared
presentations (PRE) versus analyst Q&A, with externally anchored CEO changes.

## Status

Prepared on 2026-10-03 from the existing research scripts. This is a migration
baseline, not a completed analysis or a production scoring release. Software
tests generate no real model scores. All sixteen tracked software
findings have targeted repairs or explicit deprecation, backed by synthetic
regressions. This includes issuer/tenure review gates, atomic census resume,
manifest-ordered fetch materialization and standalone CSV limits. The retired
PDF batch workflow is disabled, not restored or migrated;
see [known issues](docs/known-issues.md) before running it. The subsequent readiness
repair adds current Q&A-minus-PRE/common-support analysis, pinned length diagnostics,
current annotation instructions and an explicit detached stage launcher. These
software checks do not establish scoring quality or approve population inference.

The priority integrity repair covers issues #23, #33 and #29: whole-passage
procedural selection, source-bound speaker-gate bundles, and per-call tenure
roster reconciliation. Historical unbound gates require an explicitly authorized
fresh build. The eleven findings from the subsequent review remain tracked as
open GitHub issues; this repair does not resolve the other eight findings.
See [data integrity](docs/data-integrity.md) for the new contracts.

## Included

- Pipeline/helper modules: CCTS retrieval, ExecuComp matching, CEO identity,
  PRE/Q&A extraction, sample construction, evaluation and scoring utilities.
- The local annotation dashboard's source, not its inputs or saved labels.
- The unchanged reviewed scoring prompt/schema and machine-neutral settings.
- Synthetic offline tests, including a small extraction-to-sample smoke test.

No real transcripts, financial datasets, human ratings, generated results,
credentials, model weights, emails or thesis papers belong in this repository.
Database access and data licensing must be arranged separately. Keep all
research outputs outside Git. Public visibility applies to the software only,
not to the underlying research data or private materials.

## Offline Checks

From the repository root, using Python 3.13:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tools/ccts_database/tests -v
.venv/bin/python -m unittest discover -s tests -v
```

Tests do not connect to a database, load weights, use a GPU or call an API.

## Reproduction And Workflow

See the [ordered script inventory](docs/reproduction.md),
[settings reference](configs/specificity-settings.reference.json) and
[coding workflow](docs/workflow.md).
The runner now requires a reviewed local execution profile and exact unit metadata;
see [execution and output checks](docs/execution-validation.md). Historical outputs
without the required evidence cannot be silently consumed by the repaired tools.

Use this checkout for new code changes. Original research snapshots remain historical evidence;
do not edit them or rerun old commands simply because this copy exists.

No open-source license has been selected. Public visibility does not itself
grant a redistribution license or permission to share private research materials.
