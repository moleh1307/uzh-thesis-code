# Reproduction Inventory

This is an ordered inventory, not a one-command rerun of the full thesis.
Entry points document their arguments with `--help`. Inputs, source snapshots
and run manifests must be supplied from the separately licensed local archive.
Historical defaults do not define an approved new run.

## Data And Sample Stages

All entries below are in `tools/ccts_database/`.

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | `build_ccts_event_level_census.py` | Event-level PRE/Q&A availability |
| 2 | `build_ccts_event_master.py` | Metadata and issuer identifiers |
| 3 | `normalize_execucomp_ceo_tenure.py` | External CEO tenure episodes |
| 4 | `build_ccts_execucomp_coverage.py` | CUSIP8 matching and tenure coverage |
| 5 | `build_execucomp_turnover_windows.py` | CEO transition support |
| 6 | `build_turnover_candidate_manifest.py` | Candidate event fetch list |
| 7 | `fetch_ccts_turn_sample.py` | Resumable source-turn retrieval |
| 8 | `derive_ccts_turns_unique_sequence.py` | Exact-alias deduplication with provenance |
| 9 | `build_ccts_execucomp_speaker_gate.py` | External CEO identity across PRE/Q&A |
| 10 | `extract_execucomp_confirmed_ceo_qa_blocks.py` | Anchored Q&A block extraction |
| 11 | `build_ccts_proposed_analysis_sample.py` | PRE units and matched call/episode support |

Required local helpers are retained: `extract_ccts_ceo_qa_blocks.py`,
`ccts_qa_episodes.py`, `build_ccts_ceo_presentation_representation.py`,
`validate_ccts_execucomp_speaker_identity.py`, and the measurement manifest
builder below. Their import layout is preserved.

The shared `ceo_title_evidence.py` helper retains every observed title per
speaker key in both sections, with deterministic ordering. Special CEO roles
are review flags even if another turn uses an ordinary CEO title. Concurrent
Acting/Interim CFO labels do not, by themselves, imply a special CEO role.
The speaker gate now declares `v1.3_complete_title_evidence_20261003`; strict
anchored Q&A extraction requires this version and checks every label resolves
to the same person. Saved v1.2 last-label gates cannot be retrospectively
certified. A newly authorized source-backed gate rebuild is required before
using this repair on research data; no real gate/sample was regenerated here.

## Measurement Stages

All entries below are in `tools/llm_measurement/`.

| Script | Purpose / limit |
| --- | --- |
| `build_specificity_scoring_manifest.py` | PRE/Q&A scoring units and imported splitting helper |
| `build_untouched_preqa_evaluation_manifest.py` | Deterministic source-clean evaluation selection |
| `prepare_fresh_evaluation_human_package.py` | Blinded package creation; historical instructions need review |
| `freeze_fresh_specificity_setup.py` | Protocol-specific prompt/settings/input archive |
| `freeze_fresh_specificity_reference.py` | Protocol-specific label and missingness freeze |
| `run_local_specificity.py` | Reviewed offline identity, full-output checks and explicit bounded technical retry |
| `audit_local_specificity_run.py` | Whole-raw/token/schema/provenance and repeat diagnostics |
| `aggregate_local_specificity.py` | Historical word-weighted aggregation; sign/support adaptation pending |
| `analyze_specificity_length_robustness.py` | Length sensitivity; current support and NumPy environment check pending |

`human_audit_dashboard/` contains the existing local interface source. It is
not started during repository preparation, and there is no request to score
the existing sample again.

## Local Configuration

Personal absolute paths were mechanically replaced in the copied code only:
the research SSD root becomes `data/`, and the credential-file default becomes
`secrets/ccts_database_credentials.md`. Both directories are ignored by Git.
Use explicit CLI paths where supported; otherwise `data/` may be an ignored
local symlink to the separately stored research archive. Some default paths
retain historical run-directory names for provenance, not portability claims.
Relative defaults assume execution from the repository root.

The two database clients accept `--credentials`. They retain the legacy
private Markdown credential reader (`dbhost`, `dbname`, `dbuser`, `dbpass`).
Never commit that file, including an example containing real values. An
environment-based credential interface is future work, not already supported.

The prompt and output schema in `configs/` are unchanged copies of the reviewed
deployment artifacts. Settings use generic local paths and disclose inherited
generation behavior. No actual request JSONL or reference ratings are copied.
See [execution validation](execution-validation.md) for the new required profile,
settings and unit-metadata flags. Aggregation now also requires `--input-jsonl`.
Do not reuse old unbound outputs or change scientific sign/support definitions
merely to make the repaired consumer accept a file.

## Verified Scope

Offline tests use constructed fixtures and temporary directories. The sample
test exercises extraction, source hash reconciliation, PRE construction and
repeat-byte equality, then rejects a tampered source artifact. This is a useful
software smoke test, not evidence of model or population measurement validity.

The initial migration excludes old PDF parsers, API cost experiments, model
comparison screens, participant-assisted diagnostics, real-text historical
tests and the uncertainty-dictionary benchmark. Add any needed component in a
separate reviewed change, including its dependencies and licensing checks.
