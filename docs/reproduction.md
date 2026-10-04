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
| 8 | `assemble_ccts_candidate_turns.py` | Combine saved source pairs in exact candidate-assignment order |
| 9 | `derive_ccts_turns_unique_sequence.py` | Exact-alias deduplication with provenance |
| 10 | `build_ccts_execucomp_speaker_gate.py` | External CEO identity across PRE/Q&A |
| 11 | `extract_execucomp_confirmed_ceo_qa_blocks.py` | Anchored Q&A block extraction |
| 12 | `build_ccts_proposed_analysis_sample.py` | PRE units and matched call/episode support |

### Combining Retained And Newly Retrieved Calls

Use the rebuilt candidate assignments and saved metadata, not a concatenation
of old and new transcript CSVs. The gate scans turns in assignment order.
All paths below are placeholders for separately licensed local inputs:

```bash
python3 tools/ccts_database/assemble_ccts_candidate_turns.py \
  --event-assignments /path/to/candidate_event_assignments.csv \
  --event-metadata /path/to/ccts_event_master.csv \
  --source /path/to/retained_raw_turns.csv /path/to/retained_fetch_audit.csv \
  --source /path/to/new_raw_turns.csv /path/to/new_fetch_audit.csv \
  --output-dir /path/to/fresh_candidate_assembly
```

The assembler selects exactly the assigned events, rejects source overlap,
checks raw row/section/sequence counts against the supplied audits, and links
only blank date/year/company header fields from metadata. Existing conflicting
dates/years are errors. It copies whole event CSV byte spans in assignment
order, preserves rows within each event, and verifies unchanged nonmetadata
payloads. It handles quoted, multiline and UTF-8 text without ad hoc line sorting.
A complete fresh package contains `candidate_raw_turns.csv`,
`candidate_raw_audit.csv`, `event_source_provenance.csv` and
`assembly_summary.json`; failures do not publish partial packages or overwrite
an existing package.

Run the existing exact-alias derivation on these assembled files, then build a
new speaker gate from the rebuilt assignments, episodes, turnover map/pairs,
global normalized roster and exact call sequences. Assembly does not certify
database or runner provenance: review original fetch checkpoint bindings
separately. Historical unbound fetch evidence remains unbound. It does not
confirm speakers, remove duplicates, extract Q&A blocks or score text.

The structured-label repair removes the semicolon delimiter ambiguity without
dropping legitimate titles or weakening identity checks. Such a failure in an
older producer/consumer run is not an approved gate; do not bypass the consumer
or silently upgrade old output files.
Tracking: [candidate-order assembly](https://github.com/moleh1307/uzh-thesis-code/issues/37)
and [semicolon-title anchors](https://github.com/moleh1307/uzh-thesis-code/issues/38).
The standalone derivation repairs are tracked separately as
[CSV header validation](https://github.com/moleh1307/uzh-thesis-code/issues/40)
and [raw/audit byte binding](https://github.com/moleh1307/uzh-thesis-code/issues/41).
Tracking issues remain open independently of implementation status.

Required local helpers are retained: `extract_ccts_ceo_qa_blocks.py`,
`ccts_qa_episodes.py`, `build_ccts_ceo_presentation_representation.py`,
`validate_ccts_execucomp_speaker_identity.py`, and the measurement manifest
builder below. Their import layout is preserved.

The shared `ceo_title_evidence.py` helper retains every observed title per
speaker key in both sections, with deterministic ordering. Special CEO roles
are review flags even if another turn uses an ordinary CEO title. Concurrent
Acting/Interim CFO labels do not, by themselves, imply a special CEO role.
The speaker gate now declares `v1.4_structured_title_evidence_20261004`. All six
speaker-label fields have `_json` companions containing arrays of full labels;
the semicolon-joined fields are readable displays only. Consumers reject
malformed, blank, duplicate or non-list evidence, check display consistency,
and require every matched/shared anchor label to resolve to one person/issuer.
Matched labels must be drawn from the shared evidence. Q&A outputs carry
`validated_ceo_speaker_json`; extraction and proposed-sample construction both
use the structured anchor rather than parsing display strings.
Saved v1.2/v1.3 gates cannot be retrospectively certified or consumed by the new
contract. A newly authorized source-backed gate rebuild is required before
using this repair on research data; no real gate/sample was regenerated here.

Exact-alias derivation now declares `exact_alias_bound_inputs_v2_20261004` in
its summary. Both transcript and audit CSVs must have unique, nonblank headers.
Their SHA-256 bindings are captured before parsing and compared again before
publication; changes prevent publication of the temporary package. The receipt
records the verified original bindings, not the hashes of replacement files.
The exact two-row alias-collapse rule and retained transcript text are unchanged.

Issuer matching, tenure normalization, census and fetch resume guards are also
repaired. Read [data integrity](data-integrity.md) before continuing any older
data-stage output directory. Existing files are not automatically upgraded.

## Measurement Stages

All entries below are in `tools/llm_measurement/`.

| Script | Purpose / limit |
| --- | --- |
| `build_specificity_scoring_manifest.py` | PRE/Q&A scoring units and imported splitting helper |
| `build_untouched_preqa_evaluation_manifest.py` | Deterministic source-clean evaluation selection |
| `prepare_fresh_evaluation_human_package.py` | Blinded package with current content-class/missingness instructions |
| `freeze_fresh_specificity_setup.py` | Protocol-specific prompt/settings/input archive |
| `freeze_fresh_specificity_reference.py` | Protocol-specific label and missingness freeze |
| `prepare_scoring_unit_metadata.py` | Label-free metadata derived from exact frozen request/source hashes |
| `run_local_specificity.py` | Reviewed offline identity, full-output checks and explicit bounded technical retry |
| `launch_local_specificity.py` | Explicit detached validation, capture or approved scoring stage |
| `audit_local_specificity_run.py` | Whole-raw/token/schema/provenance and repeat diagnostics |
| `aggregate_local_specificity.py` | Q&A-minus-PRE with CEO-word weighting and common reference/model support |
| `analyze_specificity_length_robustness.py` | Current-schema descriptive length sensitivity and exclusion ledgers |

`human_audit_dashboard/` contains the existing local interface source. It is
not started during repository preparation, and there is no request to score
the existing sample again.

New setup archives use `specificity_fresh100_complete_target_code_bundle_v2_20261003`.
Their `code/tools/` tree preserves local imports and includes `csv_contract.py`.
The freeze manifest lists three `code_entrypoints` and hashes every archived
file recursively. Run `code/tools/llm_measurement/run_local_specificity.py` for
archived validate-only checks; the freezer and human-package builder retain
the same relative tree. Required source/proposal artifacts and any model/runtime
dependencies remain separate. Archiving local code does not supply model weights,
an approved execution profile or authorization to score. Historical flat archives
are not rewritten or retrospectively declared standalone.

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
Do not reuse old unbound outputs. The current consumer explicitly changes historical
PRE-minus-Q&A output fields to `qa_minus_pre_*`; it does not rewrite old results.
See [analysis contract](analysis-contract.md) for support definitions and commands.
Length analysis requires the pinned `requirements-analysis.txt` in Python >=3.10;
keep that environment separate from the reviewed Python 3.9 GPU runtime.

## Verified Scope

Offline tests use constructed fixtures and temporary directories. The sample
test exercises extraction, source hash reconciliation, PRE construction and
repeat-byte equality, then rejects a tampered source artifact. This is a useful
software smoke test, not evidence of model or population measurement validity.

The initial migration excludes old PDF parsers, API cost experiments, model
comparison screens, participant-assisted diagnostics, real-text historical
tests and the uncertainty-dictionary benchmark. Add any needed component in a
separate reviewed change, including its dependencies and licensing checks.
The disabled compatibility entry point `tools/historical/ecc_run_parser_batches.py`
is a tombstone, not a migrated PDF parser. Historical PDF commands are obsolete.
