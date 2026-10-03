# Execution And Output Checks

Repairs #2, #3, #7 and #4 are software checks, not approval to run the model or scale
the research. Prompt, two-field schema and decoding values are unchanged.
All sixteen tracked findings are closed; scientific validation and execution
approval remain separate from software readiness.

## Reviewed Local Identity

Scoring now requires `--settings`, `--execution-profile` and
`--unit-manifest-csv`. Model loading accepts an existing local safetensors
directory only, with `local_files_only=True` and `trust_remote_code=False`.
There is no automatic model download or remote revision resolution.

A profile contains `status: approved` and an `identity` object. The identity
binds the settings checksum, declared pinned revision/model ID, hashes of local
checkpoint and tokenizer resources, package/CUDA versions and GPU names,
effective tokenizer vocabulary/template, complete effective generation config,
actual device map, precision and quantization status. File hashes are checked
again after loading. Shard indexes cannot reference unbound external weights.

The configured runtime versions, token allowance, inherited repetition penalty
of 1.05, single beam and single return sequence are checked, not silently changed.
Greedy decoding is retained; inactive sampling controls remain unset. A profile
must also match effective placement/configuration after local backend loading.
Greedy decoding and an identical profile do not guarantee scientific reliability
or bit-identical outputs; they make execution differences explicit.

To capture a candidate for review, append the following to the normal runner
arguments (with a local model, reviewed settings and frozen request JSONL):

```bash
--capture-execution-profile /path/to/candidate-execution-profile.json
```

Capture loads the local backend and hashes weights but generates no response and
does not create a scoring output/run manifest. It can be slow and needs an
explicitly authorized detached compute session. Its status is
`candidate_requires_review`, which scoring rejects. Review checkpoint provenance
against the intended pinned revision, runtime, template/config and actual device
placement before creating a separate approved copy; do not merely relabel it to
bypass discrepancies. Capture alone never approves a scoring run.

For scoring, use explicit paths:

```bash
python3 tools/llm_measurement/run_local_specificity.py \
  --input-jsonl /path/to/frozen-requests.jsonl \
  --unit-manifest-csv /path/to/frozen-unit-metadata.csv \
  --output-jsonl /path/to/new-results.jsonl \
  --run-manifest /path/to/new-run.json \
  --model /path/to/local-checkpoint \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8 \
  --settings configs/specificity-settings.reference.json \
  --execution-profile /path/to/reviewed-execution-profile.json
```

Input-only `--validate-only` needs neither a profile nor model dependencies and
does not certify execution identity. Resume requires matching prior binding,
manifest and checkpoint checksum. Mismatch fails before any append or rewrite;
first-run creation/provenance is retained. Ordinary resume does not retry recorded rows.
For consumer-ready bounded runs, freeze a separate exact input/metadata subset
rather than consuming a `--limit` result as though it covered a larger file.

## Explicit Technical Retry

After all first attempts have been recorded, add `--resume --technical-retry`
to the otherwise identical scoring command to allow one additional attempt for
an inference error, invalid output/schema or output truncation. Input-context
rejection is not retryable. There is no retry for a valid low score, abstention
or disagreement with a human label, and no selection of the more favorable score.

The runner preserves original rows and appends attempt 2 with its technical
reason. Input, metadata, runner, validation code, backend and settings bindings
must still match. A retry reservation is written before inference: an interrupted
retry without a result remains explicitly unresolved and is not invoked again.
Consumers use the final recorded attempt per ID, validate every attempt's
binding, and report the original failures and unresolved reservations separately.
A second failure remains a failure; rerunning the flag cannot create attempt 3.
An interrupted first pass must finish missing first attempts with ordinary resume
before this separate retry pass. Changed code cannot resume an old bound run.

## Complete Response Evidence

Each completed result preserves generated token IDs, all tokenizer special IDs,
the full decoded response (no cleanup or special-token skipping), the JSON
payload, and its verified terminal EOS ID/text. Only one legitimate final EOS
is removed. A leading/embedded special token, unknown termination, truncation,
wrapper, reasoning prefix, trailing text, duplicate key, non-finite number,
extra field or invalid score combination fails validation. The stored parsed
object must equal the strict whole-payload parse, including value types.

## Consumers And Historical Artifacts

Audit and aggregation share `specificity_validation.py`. They verify exact
request/output/unit-metadata file hashes, response-schema hashes, the contract,
manifest binding checksum and every row's run binding. Repeat auditing requires
`--repeat-run-manifest` as well as repeat output, with matching execution binding.
Aggregation additionally requires `--input-jsonl`. The subsequent analysis-readiness
repair applies Q&A-minus-PRE and common-support comparisons; see
[analysis contract](analysis-contract.md). Historical PRE-minus-Q&A files remain historical.

Row failures/missingness remain in diagnostic ledgers and cannot contribute
scores. Valid `ok=0,specificity=0` remains an abstention. Provenance mismatch
stops consumption before generating reports, rather than retroactively inventing
a binding. Files with a different schema, missing full-response evidence or old
unbound provenance need a separately authorized migration decision; they are
not implicitly upgraded. Issue #7's reference to three fields was inaccurate:
the actual schema in this checkout is exactly `ok` and `specificity`.

Regression tests are synthetic and offline. Operational preflight receipts belong
in the private research archive, not this public repository. A remote deployment
does not imply model execution or measurement-quality validation.

## Detached Stage Launcher

`launch_local_specificity.py` takes an explicit JSON config with absolute paths
for `python`, `runner`, `input_jsonl`, `unit_manifest_csv`, `model`, `settings`,
plus `revision` and `contract_version`. Scoring additionally needs
`execution_profile`. Use `--dry-run` to inspect the exact command before launch.

```bash
python3 tools/llm_measurement/launch_local_specificity.py \
  --config /path/to/job-config.json --stage validate \
  --job-dir /path/to/new-job-directory --session uzh-specificity-validation
```

Every stage uses a fresh job directory and unique named `screen`. It binds input
and runner/helper hashes, writes `execution.json`, `execution.log` and atomic
`status.json`, and returns immediately after requesting launch. Inspect status
and logs separately; do not wait in an SSH session. `--stage capture` loads/hashes
the model without generating responses and ends at `candidate_requires_review`.
`--stage score --confirm-scoring` is the only scoring stage and requires a
separately reviewed approved profile. There is no automatic next stage, retry,
resume, model download or overwrite. Interruption cleanup terminates only the
worker's own child process; external SIGKILL cannot guarantee cleanup.
