# GPT-5.6 Sol self-audit — observation-noise engineering phase closure

## Current verdict (2026-09-19)

The repository engineering candidate is **APPROVED PENDING EXACT FINAL-HEAD
HOSTED CI AND POST-MERGE VERIFICATION**. The observation-noise scientific
outcome is **NOT EXECUTED**.

Sol's initial audit of inherited PR #11 was independent. After Sol changed the
implementation, this became a self-audit rather than a second independent
review. The confidence basis is adversarial regression coverage, an independent
recomputation path, a clean-checkout run, and exact hosted CI—not a claim of
human independence.

## Findings, repairs, and retained development evidence

The earlier review reproduced `AAA-135` through `AAA-149`. This closure also
reproduced hosted artifact-upload failure `AAA-150`: colon-bearing schedule
filenames survived the 442-test suite but were rejected by GitHub's artifact
service. Schedule storage now uses a full SHA-256 identifier while complete
trial identity remains in structured metadata and the index. `AAA-151` adds a
100 GB target-filesystem preflight before any formal remote reservation.

The corrected full 2 x 2 x 1 development search remains valid development
evidence: all four 235,776-record candidate archives independently verified,
no clipping refinement met every preregistered condition, and
`incumbent-no-refinement-v1` remains the selected no-refinement control. It does
not establish any formal observation-noise endpoint.

## Storage and experiment decision

The dedicated ext4 HDD was directly inspected and exercised. It had
425,141,489,664 available bytes before profiling, supports ordinary POSIX
permissions, symlinks and same-filesystem atomic rename, and retained a fresh
58,944-record/446-file run occupying 47,307,645 bytes. The run completed in
113.16 seconds at 150,844 KiB peak RSS and independently recomputed in 24.66
seconds. An interrupted attempt resumed to completion. SMART health is
`NOT VERIFIED` because `smartctl` is unavailable.

Linear scaling projects about 31.4 wall hours, 47.4 GB, and 447,000 files per
formal batch; A+B projects about 64 CPU hours plus 13.7 hours of independent
recomputation. Quick-scale object operations do not justify adding a packed
container format. They also do not justify the fixed replication count.

Formal A/B was therefore not executed. The v1.1 `0002` batch identities were
retired unobserved and cannot be reused. A future formal experiment must have a
new version, quantitative precision justification, final freeze, and fresh
batch identities before observation. External immutable archival is
`NOT VERIFIED`; it remains necessary for publication-grade durability claims
but is not an engineering merge gate for this internal phase.

## Identity and evidence boundary

The active protocol remains `aaa.observation_noise.v1.1`, hash
`546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c`,
and pins v2.1 commit
`25b6c32c9040d0f934314a2139993d12763afc99`. Final source fingerprint, lock,
commit, hosted CI, merge, and post-merge verification are recorded in
`handoff_sol.md` and `final_audit.md` after the exact final candidate is
generated.

---

# Historical GPT-5.6 Sol independent review — benchmark v2.1 (2026-09-12)

## Verdict

**APPROVED FOR NORMAL MERGE through PR #10, subject to required checks on the
exact final PR head. NOT YET MERGED at the time of this record.**

The scientific and engineering review gates are satisfied. This is not a
release approval and does not authorize a tag or published artifact. Normal
branch protection remains the integration gate; a post-merge clean-clone smoke
must still be recorded before the task is called complete.

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
- `AAA-126`: wheel and runtime versions disagreed. They now coherently report
  `0.2.0`, verified from an installed wheel outside the checkout.
- `AAA-127`: fresh A-0003 exposed self-induced Git dirtiness after durable
  reservation. The failed attempt is retained, B-0003 was cancelled unobserved,
  and the pre-claim source snapshot repair passed the new A/B pair.

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

- Local complete suite: 401 tests, exit 0, 78.855 seconds after the final
  scientific repair. An earlier instrumented 400-test run covered 89%; the new
  lifecycle regression then passed in focused and complete suites.
- Static gates: Ruff lint, Ruff format check, configured mypy scope, dependency
  lock validation (24/24 exact pins), diff whitespace, and confirmation exit-code
  contract all exited 0.
- Package: `aaa-0.2.0-py3-none-any.whl` built without isolation using the locked
  setuptools backend; payload included the canonical specification, license and
  console entry point. In a fresh venv outside the checkout, the exact lock was
  installed, the wheel was installed with `--no-deps`, import reported `0.2.0`,
  `spec-hash` resolved the packaged file, and the 24-pin check passed.
- Runtime smoke: a 2-replica/3-episode development run correctly reported thin
  coverage as `INSUFFICIENT_EVIDENCE`; recomputation reproduced its full status
  tree. Historical v1 smoke and headless animation/plot tests passed. These are
  mechanics/regression evidence, not confirmation.
- Frozen candidate: `11ba061d2c33a5f04e71fd10a186de3af887b180`, spec
  `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37`,
  lock `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7`.
- A-0004: source `11ba061d2c33`, runtime 666.191 seconds, confirmation family
  size 4, process exit 0, 14/14 required gates PASS, 4,058/4,058 full checksums,
  exact semantic recomputation, and 55/55 compact archive checksums.
- A evidence/bookkeeping commit: `dc4a6e1988c00dde3a3b27622c0af9982c395f1d`.
- B-0004: source `dc4a6e1988c0`, runtime 666.648 seconds, confirmation family
  size 5, process exit 0, 14/14 required gates PASS, 4,058/4,058 full checksums,
  exact semantic recomputation, and 55/55 compact archive checksums.
- GitHub PR #10: observed CPU CI runs `34718326277` and `34718328566` passed
  locked environment, fresh install, and Python 3.10/3.11/3.12/3.13. The final
  evidence/documentation head must receive the same required checks before merge.
- [`final_audit.md`](final_audit.md) is generated from the exact staged path set
  and includes every tracked regular file. Historical and generated evidence
  are explicitly categorized; a row is not itself scientific re-verification.

## Residual limitations

- Raw per-step logs are not a permanent public archive; Actions retention is 90
  days and semantic regeneration is weaker than durable byte preservation
  (`AAA-077`).
- Confirmation A and B intentionally share selected checkpoint lineages and use
  independent evaluation streams; they measure evaluation-stream generalization,
  not retraining sensitivity.
- CPU-only evidence does not establish GPU behavior, which the project does not
  claim or require.
