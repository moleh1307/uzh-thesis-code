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
Every exact-panel event must map to one unique roster episode with matching
issuer/executive/episode keys and tenure boundaries. Its call date must lie
inside that exact tenure. Missing, duplicate, nonexact or conflicting evidence
fails before output publication and before call/quarter support is counted;
endpoint audit notes do not override these checks. Nonexact roster episodes
without exact-panel calls remain visible as adjacency-review evidence.

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

Metadata-rich CSV selection applies the same standard earnings event type and
minimum start-date year as database selection. Date/year disagreement fails
explicitly; eligible IDs must be unique and positive. Sorting precedes the
event limit. Summaries include local input/filter/type/year/limit exclusion
counts, distinct from database-wide metadata counts. Old checkpoints from
the prior selector are not compatible because the implementation hash changes.

## Fresh Package Publication

Event masters and specificity scoring manifests stage all artifacts under a
sibling temporary directory, then publish the completed directory. Existing
directories (including empty ones) and symlinks are rejected unchanged; choose
a fresh run name. The manifest builder's legacy `--force` flag no longer grants
replacement permission. Empty/all-filtered Q&A produces a complete diagnostic
package with stable headers and zero support; a header-only presentation input
is rejected before publication. Neither outcome implies scoring readiness.

Event-master `latest` uses a resolved absolute target, including when the output
root was relative. Its pointer is replaced only after a complete new run is
published; a real file/directory at `latest` is refused. A latest-update failure
can leave a complete unpointed run, not a partial published package. Cooperating
writers use a sibling publication lock. Abrupt process termination may leave a
stale lock; it is not automatically removed or interpreted as completed work.

## Fetch Ordering And CSV Bound

Fetch shard storage remains resumable. Materialization uses manifest event
rank, not concatenated shard order. It preserves original within-event turn
order and validates shard ownership and unique audit rows. Temporary SQLite
sorting keeps transcript memory bounded and files closed; allow temporary
disk space for the staged text and final CSV. Old fetch CSVs are not rewritten
by this repair; a newly authorized materialization is required.

New fetch runs use checkpoint schema 2. The run and each committed shard bind
manifest bytes/order, database host/name/user/port, SSL mode, runner/helper code
hashes and output schemas. Source/code drift, foreign shard bindings and old
unbound directories are refused before database contact or output rewriting.
Password bytes and credential-file hashes are never recorded; rotating only
the password permits same-source resume. Batch size, timeouts and a smoke limit
can change without altering the full manifest binding. Matching credentials are
read even when the requested scope is already checkpointed. These identities
do not guarantee the database itself has remained unchanged between batches.

## Model Length Metadata

For the model lane, `--key-csv` must have the exact SHA-256 of the aggregation's
`input_manifest`, and unit results must match the aggregation output hash.
Matching IDs, words and dates alone cannot certify period bins. A byte-identical
copy may be relocated; an edited or independently assembled key is rejected,
as is an older receipt without the input binding. Rebuild a separately authorized
aggregation from verified inputs rather than adding hashes to old evidence.
The human-reference lane remains distinct and retains its existing validation.

Every transcript CSV entry point explicitly sets `csv.field_size_limit` to
`64 * 1024 * 1024` **decoded characters per field**, via `tools/csv_contract.py`.
This is not a byte/file-size limit. Oversized fields raise `csv.Error`; nothing
is truncated. Tests run readers in separate processes with multiline quoted
fields, the exact upper bound and a field above it. The real corpus maximum
has not been measured, so this bound is not a population completeness claim.

## Historical Workflows

### Bound Speaker-Gate Bundles

New speaker gates declare `speaker_gate_bound_bundle_v1`, alongside the title
evidence gate version. Their summary binds all ten input files (including the
resolution audit) and each emitted gate CSV by SHA-256. Producers verify that
inputs remain unchanged across the build. Consumers verify the exact artifact
bytes, input bytes and complete external metadata reconciliation, then recompute
episode support and old/new turnover gates from the verified event rows.
Counts alone cannot certify executive IDs or pass flags. Duplicate identities,
substituted gate tables, mismatched source identities and inconsistent support
or flags fail before extraction writes outputs.

Historical unbound bundles are rejected, even if their title gate is current.
Do not add hashes to recertify them: a separately authorized fresh build is
required. Keep the bound source files available; moving the gate CSVs is allowed
when their bytes and source bindings remain unchanged. For a relocated source
bundle, update only the recorded paths to the same hash-verified input copies;
relative input paths resolve against the gate summary directory. These hashes
detect stale/mixed artifacts, not deliberate falsification of every source.

### Whole-Passage Procedural Eligibility

Extraction and scoring-manifest building use the shared `procedure_kind`
whole-passage check. A courtesy, laughter marker or clarification phrase inside
a business answer does not by itself exclude it or send it to source review.
Recognized fully procedural passages remain excluded; unknown nonprocedural
content remains visible, not automatically semantically validated. Interrupted
or corrupted source text and uncertain question/answer boundaries retain their
separate review flags. Existing labels, frozen samples and prompts are unchanged.
New block extraction records the `whole_passage_v2_20261004` suffix; the anchored
extractor records `external_execucomp_bound_speaker_gate_anchor_v3_20261004`.
The bounded saved-data rebuild exposed complete clarification passages that the
first whole-passage repair did not recognize. The new rule excludes a passage
only when every sentence is a recognized clarification or incidental apology.
Clarification followed by business information or a disclosure limit remains
visible. Unknown phrases are not inferred to be procedural. Preserve older
diagnostic output and rebuild affected extraction and selection outputs in a
fresh directory; no human ratings or scoring prompts need changing for this fix.

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
