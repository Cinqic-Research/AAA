# AAA engineering phase-closure handoff

This is the active GPT-5.6 Sol self-audit handoff for the integrated
observation-noise engineering closure. Sol's initial review of inherited
work was independent; post-remediation verification is a self-audit backed
by adversarial tests, independent recomputation, clean-checkout execution,
and exact hosted CI. It is not human review or product-release approval.

## Final identities

| Item | Value |
|---|---|
| generated from local head | `d7511da06465a40d3b97e614e03ae34d5b312263` |
| protocol | `aaa.observation_noise.v1.1` |
| protocol hash | `546e2434cc15850779107c1a329af1e2f8581cd6a1c3b31807eecf5f6e7b427c` |
| scientific fingerprint | `f48f9d8f67d767b84c99d2fc53fd658e1e5e1362f1fac9f209520d0293135d7a` |
| dependency lock | `2b5a23094b0cd99df66fa534b2e979a7a3e8c8b393df71cd53f1345281ab45b6` |
| v2.1 reference commit | `25b6c32c9040d0f934314a2139993d12763afc99` |
| selected candidate | `incumbent-no-refinement-v1` |
| candidate configuration | `dcc82cb0396985976419733a817d02027c45683db4f76d3ea37ac48290f6c649` |
| formal observation-noise outcome | `NOT EXECUTED` |
| external durable archival | `NOT VERIFIED`; not required for this internal phase |

## Decisions

The complete option analysis is retained in `phase_closure_decisions.md`.
FontTools 4.65.0 was integrated before the final freeze. PR #11's
infrastructure is retained with portable schedule storage IDs and a 100 GB
formal-output preflight. PR #12 and PR #13 remain exploratory history and
are not part of the maintained scientific tree. Broad fingerprint coverage
and deterministic sharding remain in place.

Formal A/B was deliberately deferred. The v1.1 batches are:

| Batch | Role | Status |
|---|---|---|
| `observation-noise-a-0002` | `confirmation_a` | `retired_unobserved` |
| `observation-noise-b-0002` | `confirmation_b` | `retired_unobserved` |

They were never observed and may never be reactivated. A successor must
receive a new protocol version, precision justification, freeze, and fresh
batch identities before observation.

## Storage evidence

The dedicated ext4 HDD had 425,141,489,664
available bytes before profiling. POSIX permissions, symlinks, ordinary
writes, and same-filesystem atomic rename passed. SMART health is
`NOT_VERIFIED` because `smartctl` is unavailable.

The retained quick run wrote 446 files and
47,307,645 bytes in 113.16
seconds, then independently recomputed `PASS`. The linear formal projection
is 31.4 wall
hours and 47.4
GB per batch, about
446,755 files per batch.
That measurement supports local execution capacity and retaining sharding;
it does not justify the replication budget or establish external durability.

## Validation

- Complete suite: 445 tests, exit 0, 165.150 seconds.
- Coverage: 72% against the configured 70% floor.
- Lock, Ruff lint/format, mypy, exit-code contract, strict JSON parse,
  package build/payload, clean wheel install, installed CLI smoke,
  v2.1 smoke/recomputation, observation-noise smoke/recomputation, and
  zero-noise pinned-reference replay passed.
- Recorded exact-source hosted CI runs: source-identical head d7511da06465a40d3b97e614e03ae34d5b312263: push run 35450887407 and pull-request run 35450890204; all 12 jobs PASS; locked artifacts 10585989223 and 10586079089 uploaded successfully.
- Exact final remote main: pending merge and post-merge verification.

## Approval matrix

| Gate | Verdict |
|---|---|
| Repository engineering | APPROVED, subject to exact final-head CI when marked pending above |
| Test and CI integrity | APPROVED only when final-head hosted jobs are recorded green |
| Packaging and clean-install integrity | APPROVED when final validation record is present |
| Evidence integrity | APPROVED for the retained engineering/development evidence |
| Benchmark v2.1 historical state | LIMITED: valid confirmed results, historical raw-archive limitations retained |
| Observation-noise implementation | APPROVED when final-head artifact upload is green |
| Observation-noise scientific outcome | NOT EXECUTED |
| Local evidence retention | VERIFIED on the dedicated HDD |
| External durable archival | NOT VERIFIED; not required for current internal phase |
| Repository hygiene | APPROVED after final PR/branch closure |
| Ready for next AAA phase | APPROVED after exact final main and hosted CI verification |

The scientific `NOT EXECUTED` status is intentional and is not represented as
a negative, inconclusive, or successful experiment.
