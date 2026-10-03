# Coding Workflow

1. Open one issue for a bounded bug or implementation task. Describe the
   expected behavior and acceptance test without attaching real data.
2. Make a branch, implement the change and add synthetic regression coverage.
3. Run offline checks, inspect the diff for data/credentials and open a PR.
4. Review and merge. Tag only a verified executable milestone, not merely a
   migration baseline with unresolved execution failures.
5. Record the commit, environment, model revision, prompt/schema identity and
   input checksums in local/server run manifests. Results remain outside Git.

Code review by the same assistant is a second-eye check, not independent
scientific validation. Merging code does not authorize scaling the study.

For long remote inference, the intended workflow is a named `screen` session,
a log and atomic `status.json`: launch, detach, then inspect outputs later.
The launcher is not implemented in this baseline. No CI job should access
licensed databases, personal labels or shared compute.
