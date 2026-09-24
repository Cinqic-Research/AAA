# Independent review of PR #28 (2026-09-24)

This is an independent review of the Python-first transition, not the
implementer's `docs/aaa_python_self_review.md`. It began at 2026-09-24
05:47:35 UTC from canonical `main`
`f56caaf0c0c5e38882c1464f24e7a62441df1f32` and PR head
`da1903285d665cb3d9c185ebe711dd6344357166`. The initial PR tree had
740 tracked regular files; the staged review candidate has 749. The final
reviewed candidate is the PR head after
the remediation and review-record commits; its exact SHA and hosted CI result
are recorded in the PR discussion because a committed document cannot name
its own commit SHA. The associated `docs/final_audit.md` inventories that
final staged tree, not evidence that each row was freshly read by hand.

## Scope and independent methods

The reviewer re-fetched GitHub branches and tags, branch protection, PR
reviews/threads/checks, open PRs and issues. The four scientific/provenance
branches under `aaa-confirmation-claims/` were retained. `git fsck` passed.
The full PR diff and resulting tree were separately inspected. Every tracked
path was visited by a type-specific inventory pass: 221 Python files parsed
as AST, 429 JSON files strictly parsed, one gzip JSON-lines file decompressed
and strictly parsed, 77 Markdown/text files read for claim searches, four PNG
files decoded and verified, and four YAML workflows plus `CITATION.cff`
parsed. These parser checks establish syntax/integrity, not scientific
correctness. Current high-risk source and tests were reasoned through and
probed separately; frozen scientific code and evidence were checked against
canonical historical identities and targeted recomputation rather than
rewritten or represented as a fresh manual reread of every line.

Three independent read-only agents worked without using the implementer's
self-review as an answer key: Python generation/sandbox/causal boundary;
promotion and protected history; learner/evidence/checkpoints/packaging. A
fourth checked inventory and cross-document claims. Each reported exact
paths, probes and limits. The primary reviewer reproduced material failures
and verified the final repairs. No Python confirmation or attack identities
were generated or inspected.

## Findings, reproductions and repairs

| Ledger | Severity | Unmodified-candidate reproduction | Repair and evidence impact |
|---|---|---|---|
| `AAA-187` | high | Alter `online`/`syntax` `brier_sum`, regenerate the summary, retain all original records: recomputation said `PASS` while Brier changed to `0.25231798264709543`. | Recompute Brier, bins and per-class primitives from records; partial summary-only mode is labelled `SUMMARY_ONLY_NOT_VERIFIED`. Original evidence unchanged. |
| `AAA-188` | high | A checkpoint from an empty training pool resumed against a one-task pool and kept zero updates, versus two when fresh; changed `init_scale` changed reset-control weights after resume. | Bind checkpoint to specification and full training-task digest. Old unbound checkpoints retrain; retained evidence unchanged. |
| `AAA-189` | medium | Positive finite values `1e300` versus `1e-300` raised `math domain error`; `10**1000` raised overflow during validation. | Arithmetic failures return `INVALID_EVIDENCE`; nonfinite evaluator output returns `DISAGREEMENT`. Neither promotes. |
| `AAA-190` | medium | A caller-modified repair view with label `99` was accepted and reveal raised `IndexError`. | Require the exact view issued at presentation in commit and reveal. Ordinary run behavior is unchanged. |
| `AAA-191` | medium claim defect | Direct private child execution recovered an excluded builtin through Python reflection, while public `oracle.execute` correctly refused that source at AST validation. | Narrow claims: the child limits resources; it is not an independent hostile-code jail. No generated-program containment claim is upgraded. |

`tests/test_review_regressions.py` holds the new deterministic checks. The
record reader also now rejects nonstandard JSON constants. Documentation
reconciles the charter's historical dot-era wording, current runtime-state
accounting, and the scope of specification and output-root security claims.
The packaged specification remains project-controlled; its validator does
not range-check every numeric field. This is a documented limitation, not a
claim that arbitrary user specifications are safe.

## Storage and backup

Preflight on FLOWBOX matched the `Cinqic Storage` ext4 label and UUID
`3b337640-5967-4ef8-b7bb-44331c5bbbf4` to the live mount at
`/media/cinqic/Cinqic Storage`, confirmed writing, and observed about 412 GB
free (decimal bytes). The starting task directory was on the NVMe; all
substantive checkout, venv, validation and backup work moved to the HDD.
`tools/storage_preflight.py` and synthetic mount tests enforce the rule for
FLOWBOX without requiring the HDD on hosted CI. Root `AGENTS.md` and
`CLAUDE.md` point to one [work policy](agent_work_policy.md).

`tools/backup_repository.py` fetched GitHub refs and tags, checked live
canonical `main` and Git integrity, created a commit-addressed local Git
bundle, verified its SHA-256, cloned it in a fresh HDD directory and compared
restored commit and tree. Initial canonical-main backup:
`/media/cinqic/Cinqic Storage/AAA/archives/git/f56caaf0c0c5e38882c1464f24e7a62441df1f32/`,
bundle SHA-256
`d7efc0a5e8ed9aabb277c6517847ec5995dc11d86aa5cca341e2c6ddfe5f155f`.
This is a local backup on one HDD, not off-site preservation of omitted raw
artifacts. Two local-only observation-noise engineering runs (about 46 MB
each) remain outside Git; they were inventoried, not uploaded.

The connected Hugging Face identity is `Cinqic`, but OAuth grants only
`read-repos`, `read-mcp`, profile and jobs scopes. An HDD-local `hf` CLI
(version 1.32.0) has no login, and creating private Bucket
`Cinqic/aaa-backup` returned HTTP 401. No remote object exists or has been
restore-tested. **Hugging Face remote backup: BLOCKED BY AUTHORIZATION.** The
[backup policy](backup_policy.md) gives the exact upload and retrieval gate.

## Validation and limits

The original PR head had six successful required CPU jobs. The live
protection required a PR, strict up-to-date checks and resolved threads, with
zero required approvals. There were no prior reviews, review threads or open
issues. These were starting facts and must be rechecked against the final PR
head before merging.

Initial focused checks passed: 68/68 safety checks, 52/52 leakage probes,
golden keys, promotion self-test and successor comparison, four protected
identities and 471 protected historical files. Full-record recomputation
re-executed 1,100 distinct scored programs and passed. A full development
run produced the same 27,840-record digest
`fe489b739fb2d482daae4e7fa4409c84731f7473618369e721f2e6bb8c3b6369`,
cells and summary as retained negative development evidence. The first full
881-test run found one mismatch between the new nonfinite-result status and
an established regression; that was corrected and the focused tests passed.
The final local suite then passed **882 tests, one skipped**. Ruff lint and
format, mypy, 24-pin lock verification, exit-code contracts, promotion
self-test, Python identity/safety/leakage/golden/report/inventory, AAA-1K
parameter audit and full gradient check, Champion/loop integrity, v2
audit/registry/freshness/recomputation/report/parity, benchmark and noise
specification identities, and protected identities all passed. The command
transcript is retained locally at
`/media/cinqic/Cinqic Storage/AAA/validation/pr28-exact-validation.log`.
An installed wheel built with the locked backend passed package-data and
sandbox-child inspection, golden, safety, leakage, promotion, a 2,328-record
quick development run and independent recomputation from an unrelated HDD
directory containing spaces. Installed-package fingerprint and confirmation
both correctly refused with exit status 2. Clean-clone and exact-head hosted
CI results must be added to the PR discussion and task completion record
after they run.

Scientific scope remains unchanged: Python is the current specialization,
not a demonstrated general coding capability; the first v0 development
result is negative; no Python confirmation is admitted. The prospective
promotion successor is not a retroactive correction of frozen v2 evidence.
The retained v2 K5 discrepancy remains reproducible, and `aaa.1k.v2` remains
`NO_CHALLENGER`. About 105M parameters remains revisable long-term planning
guidance. No scaling or held-out decision was made in this review.

**Technical verdict:** the reproduced defects have concrete repairs and
regressions; protected history and negative development evidence are retained.
Final merge eligibility requires the exact-head local, clean-clone and hosted
checks recorded after this document is committed. Hugging Face write
authorization is a separate operational backup blocker, not a scientific
pass or fail.

The local review record was frozen at 2026-09-24 06:24 UTC, about 36 minutes
after start. Preflight and live-state establishment took roughly 3 minutes;
independent probes and repository coverage roughly 11 minutes; remediation,
policy and backup roughly 12 minutes; local validation and record preparation
roughly 10 minutes. Hosted CI, merge and post-merge backup follow as separate
phases in the PR/task completion record.
