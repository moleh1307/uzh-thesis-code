# Data Integrity And Resume Contracts

Software repairs verified with synthetic fixtures only. Existing licensed
datasets, source files, labels, frozen runs and deployed code were not changed.
None of these checks establish measurement validity or approve inference.

## Issuer And Tenure Evidence

Exact coverage requires a single nonempty GVKEY behind CUSIP8 and one exact
CEO episode covering the call date. A second issuer remains ambiguity even if
only one has a covering tenure. Ambiguous/missing issuer identities appear in
the review ledger, not the exact panel. Turnover windows reject a mislabeled
exact input panel rather than accepting its filename as evidence.

Tenure dates are taken from the relevant contiguous CEOANN episode, not pooled
across an executive's career. Different raw spellings of the same parsed date
are consistent. Conflicting or unparseable values are retained in the audit;
annual placeholders for those cases have explicit review status, not exact
boundary status. Clearly episode-incompatible dates also require review.
Calendar-year compatibility allows one year for fiscal-year alignment; it is
a conservative diagnostic, not a source-authority hierarchy. No conflicting
date is resolved by choosing the minimum, maximum or a preferred record.

Historical exact panels require a separately authorized rebuild/review before
being used as evidence that these new guards were applied.

## Census Checkpoints

`census_run_config.json` binds mode, count schemas, year/limit/filter scope,
filter-file checksum, database host/name/user, SSL mode and implementation
hashes. It never stores a password. Metadata bytes are bound to that run.
Committed count batches bind the ordered event universe and metadata identity.
Batch size and timeouts can change without altering scientific scope.

Count batches are staged, flushed and atomically published; a checksum-backed
`.done.json` marker is the commit point. Uncommitted orphan batches are ignored
and may be fetched again. Damaged committed batches, duplicated/out-of-scope
IDs, different schemas/modes and missing count cells fail closed. The SQL
query must finish before absent GROUP BY events are recorded as measured zero.
A missing signal cell is never equivalent to a completed zero count.
Metadata publication has a bound staged-file recovery path for interruption.

Flat stage CSVs are materialized only when that stage is complete, in metadata
event order. Monitor stdout and the committed `.checkpoints/` markers during
a run; an older dashboard watching only the flat CSV will not show partial
stage progress. A final table requires complete text and participant coverage.
Section-only reports leave uncomputed diagnostics blank/absent/null.

Old directories without this binding are rejected unchanged. Do not pass
`--force` to bypass this guard on archival research folders: it deletes the
named run artifacts. Use a fresh, explicitly authorized output directory.
Source metadata changes during a resume are rejected, not silently merged.
These are run identities, not a database-wide transactional snapshot guarantee.

## Fetch Ordering And CSV Bound

Fetch shard storage remains resumable. Materialization uses manifest event
rank, not concatenated shard order. It preserves original within-event turn
order and validates shard ownership and unique audit rows. Temporary SQLite
sorting keeps transcript memory bounded and files closed; allow temporary
disk space for the staged text and final CSV. Old fetch CSVs are not rewritten
by this repair; a newly authorized materialization is required.

Every transcript CSV entry point explicitly sets `csv.field_size_limit` to
`64 * 1024 * 1024` **decoded characters per field**, via `tools/csv_contract.py`.
This is not a byte/file-size limit. Oversized fields raise `csv.Error`; nothing
is truncated. Tests run readers in separate processes with multiline quoted
fields, the exact upper bound and a field above it. The real corpus maximum
has not been measured, so this bound is not a population completeness claim.

## Historical Workflows

The older speaker-validation CLI retains its name helpers but now confirms a
turnover pair only with distinct confirmed old/new events, consistent expected
CEO identity on each side, different executive IDs and ordered call dates.
It is still historical; these checks do not replace the current speaker gate.

The count-only PDF batch workflow is **deprecated**, rather than reactivated.
`tools/historical/ecc_run_parser_batches.py` is an always-failing compatibility
entry point, including for old resume/force/dry-run options. It cannot reuse
stale batches, rewrite request lists, launch a parser or create outputs.
Private archived originals are reference-only and are not patched or certified.
No compatible-resume success path exists by design. Any future reactivation
requires a new reviewed implementation with ordered PDF identities, parser
version/options, validated artifact hashes and CSV-record-aware checks.
