# GPT-5.6 Sol independent review

## Verdict

**REVIEW IN PROGRESS — NOT APPROVED, NOT MERGED.**

This is the designated review record. It remains deliberately non-approving
until the repaired candidate has passed the complete local gate set, a fresh
untuned A/B pair, the required GitHub checks, and post-merge clean-clone
verification. Historical green summaries are not substituted for those gates.

## Scope and method

The review began from `main` at
`03923083ed1619cde01816922e6c9968031d59ba`. Every tracked regular file was read
as text or bytes, every JSON file was parsed, and every retained PNG was opened
at original resolution. The exact per-path ledger, reviewed worktree SHA-256,
purpose, evidence category, verification route, finding links, and disposition
are generated in [`final_audit.md`](final_audit.md).

The active review authority was changed from GPT-6 Astra to GPT-5.6 Sol.
Remaining Astra references are historical authorship or the historical handoff
tag; none is an active assignment or approval.

## Reproduced findings and repairs

- `AAA-121`: the freeze did not enforce source identity. Repaired with a
  manifest-v2 scientific fingerprint and drift regressions.
- `AAA-122`: non-finite recomputation values could compare equal. Repaired with
  finite tolerances, full result/gate comparison, and mutation tests.
- `AAA-123`: every replica's raw records claimed replica 0 training seeds.
  Repaired by attaching each plan's actual replica lineage.
- `AAA-124`: all four historical v2.1 checksum manifests name omitted files and
  fail clean-clone integrity verification. Preserved as a negative limitation;
  new evidence is required.
- `AAA-125`: separate Actions checkouts could not enforce one durable formal
  batch claim. Formal roles are now confined to the maintained checkout; local
  reservation is persisted under a lock before observation.

Additional repairs reject unknown/coercive/non-finite step-record fields,
duplicate trial identities, unsafe checksum paths, negative tolerances, and
nondeterministic gzip headers. Dependencies and actions were updated to the
exact reviewed versions proposed by PRs 6–9, with the build backend locked.

## Evidence boundaries

- Historical PASS/FAIL summaries remain historical and are not fresh review
  evidence.
- A checksum verifies bytes only when every named file exists. Semantic replay
  is a separate claim and does not promise timestamp-identical directories.
- Development smoke evidence can validate execution mechanics but cannot
  approve the selected candidate.
- Package construction, payload inspection, installation, import/CLI smoke, and
  scientific confirmation are distinct gates.

## Final evidence

Pending. This section will record exact commands, exit statuses, test counts,
candidate/freeze identities, A/B batch IDs, gate outcomes, CI run IDs, review
PR and merge SHAs, and the post-merge clean-clone result. It will not be upgraded
from pending unless each corresponding artifact was freshly verified.

## Residual limitations

- Raw per-step logs are not a permanent public archive; Actions retention is 90
  days and semantic regeneration is weaker than durable byte preservation
  (`AAA-077`).
- Confirmation A and B intentionally share selected checkpoint lineages and use
  independent evaluation streams; they measure evaluation-stream generalization,
  not retraining sensitivity.
- CPU-only evidence does not establish GPU behavior, which the project does not
  claim or require.
