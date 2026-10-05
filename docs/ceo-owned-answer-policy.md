# CEO-Owned Answers And Coverage

The prospectively approved policy `ceo_owned_answers_identified_management_v1_20261005`
uses source-valid, uniquely anchored CEO answers even when identified same-issuer
managers participate. Only CEO words are scored and weighted; analyst text is
context-only and other managers' text is not added to the scoring request.
The previous strict-only lane remains a robustness subset.

Saved source tables keep their original `primary` and `management_context` names.
Explicit policy overlays map `management_context` to the new primary and
`primary` to strict robustness. Existing identity/quality/boundary gates and
PRE representation remain unchanged. Review and procedure/empty units remain
excluded; whole eligible mixed answers are not trimmed.

## Offline Preparation

```bash
python3 tools/llm_measurement/build_ceo_owned_pilot.py \
  --strict-packet /path/to/reviewed-strict-packet \
  --source-dir /path/to/verified-source-publication \
  --trace-dir /path/to/audited-bounded-source-trace \
  --output-dir /path/to/new-policy-packet \
  --evaluation-id new-policy-evaluation

python3 tools/llm_measurement/verify_ceo_owned_pilot.py \
  --packet-dir /path/to/new-policy-packet
```

The builder requires the existing reviewed packet/source/trace contracts. It
preserves the strict requests as a byte-for-byte prefix, appends only the newly
eligible CEO-owned management answers, derives label-free metadata and publishes
to a fresh directory. The verifier reads back artifacts, original inputs,
coverage arithmetic, unchanged model settings and pre-score support overlays.
Old receipts that bound a mutable public checkout are verified against the
original producer Git commit; source data and prior packet files are checked
live. This preserves historical code evidence without requiring today's
documentation and tests to remain byte-identical forever. The new packet also
records exact current producer code hashes.
Neither command deploys, loads a model, downloads data or calls an API.

## Three Coverage Stages

- Raw Q&A word share: eligible CEO answer words / all raw anchored CEO Q&A words.
  Descriptive, not answer recall.
- Answer-inclusion coverage: eligible CEO answer words / source-linked
  nonprocedural candidate CEO answer words, including review-excluded candidates.
  Review words are reported separately because uncertain question boundaries
  prevent calling this verified valid-answer recall.
- Scoring coverage: validated scorable CEO words / eligible target words. Before
  inference it is null, not zero. Failed, missing and unscorable outputs remain
  separately accounted; only upstream technically/provenance-validated status
  maps may be supplied to `qa_word_coverage` after inference.

Procedure/empty targets and previously audited post-session closings remain in
the raw ledger but outside the answer denominator. Closing speech is not silently
reassigned to PRE. Speech lacking classification evidence remains unresolved;
a second denominator includes all unresolved words as an explicit sensitivity.
Bounded closing evidence must not be generalized to population transcripts.
No arbitrary new coverage threshold or claim of semantic completeness follows
from complete raw-word accounting.

The packet is diagnostic, not independent human validation or population-scoring
approval. Prompt/schema/model/decoding are unchanged. Capture and review a fresh
execution identity before a separately authorized named, detached remote run.
