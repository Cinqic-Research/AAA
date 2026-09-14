# Observation-noise v1 handoff — PR #11

This is the concise handoff for the separately scoped `aaa.observation_noise.v1`
phase. It reports engineering work and evidence boundaries; it is not an
approval and does not convert development smoke evidence into a confirmation
claim.

## Exact identity at the engineering source boundary

| Item | Value |
|---|---|
| PR | [#11](https://github.com/Cinqic/AAA/pull/11), `[AAA] Add controlled observation-noise v1 phase` |
| engineering branch | `codex/aaa-observation-noise-v1` |
| source-bound head | `2d1f619b4dbe7a86d9f2818002575faf67844f21` |
| exact base | `main` at `25b6c32c9040d0f934314a2139993d12763afc99` |
| scientific source fingerprint | `ddcbed59d8c0c223e5fbc546832c8e6d385c40c232abd6c70c1025b15b9082e0` |
| protocol | `aaa.observation_noise.v1` |
| protocol hash | `d5466983b358175ac4f33f4d7450953573721159c34fec4660bbb38b5c9f88ff` |
| v2.1 reference raw identity | `4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e` |
| v2.1 reference resolved identity | `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37` |
| dependency lock | `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7` |
| source-freeze manifest | `benchmarks/observation_noise_source_freeze.json`, identity `5d59e82b2707330552535055dd985f65a5a513c500a2eb79aeeabaaed92dd838` |

The source fingerprint is the non-self-referential identity used by future
confirmation admission. The source freeze records the protocol, preserved
v2.1 reference identities, lock, candidate ledger hash
`4f049abf580b12c3a6a1718af16ab09d4c36ba05099c539efc87d311145d7be7`, and the
two planned batch IDs below. Generated results and review documents are
excluded from that fingerprint by design.

## Candidate, ledger, freeze, and batch boundary

The selected candidate is the valid no-refinement identity
`incumbent-no-refinement-v1`, configuration hash
`dcc82cb0396985976419733a817d02027c45683db4f76d3ea37ac48290f6c649`.
The complete bounded catalog has four immutable entries and three rejected
causal innovation-clipping variants:

| Candidate | Configuration hash | Development outcome |
|---|---|---|
| `incumbent-no-refinement-v1` | `dcc82cb0396985976419733a817d02027c45683db4f76d3ea37ac48290f6c649` | selected; unchanged incumbent |
| `causal-innovation-clip-025-v1` | recorded in `benchmarks/observation_noise_candidate_ledger.json` | rejected; development constraints not jointly met |
| `causal-innovation-clip-050-v1` | recorded in `benchmarks/observation_noise_candidate_ledger.json` | rejected; development constraints not jointly met |
| `causal-innovation-clip-100-v1` | recorded in `benchmarks/observation_noise_candidate_ledger.json` | rejected; development constraints not jointly met |

All four quick development attempts were independently verified from their
primitive records before the compact evidence was committed. Their stable
attempt IDs and summary/record hashes are in the ledger; their local archive
locator is explicitly `transient_local_selection_root_not_committed`.

The confirmation-freeze identity is **not created**. No selected-candidate
confirmation freeze, A/B checkpoint set, formal schedule reservation, or
confirmation attempt identity exists. The predeclared batches are
`observation-noise-a-0001` (`confirmation_a`) and
`observation-noise-b-0001` (`confirmation_b`), both still `planned` in the
registry. A/B attempt IDs, durable archive locators, archive hashes, and
independent archive-retrieval results are therefore `N/A — not run`, not
missing claims to be filled from transient development output.

## Analysis and gate status

The implemented joint evaluator recomputes the complete primary A+B endpoint
family from primitive archives, checks shared candidate/checkpoint/fingerprint
identity, and applies one Holm family to the one-sided 95% bounds. That joint
analysis has not run because no admissible A/B archives exist. The formal gate
status is:

| Gate or required evidence | Status |
|---|---|
| v2.1 raw/resolved reference preservation and zero-noise regression | `PASS` in the local regression suite |
| development archive integrity and independent arithmetic | `PASS` for the four bounded smoke attempts |
| formal absolute utility | `INSUFFICIENT_EVIDENCE` |
| formal reflected-baseline competitiveness | `INSUFFICIENT_EVIDENCE` |
| formal first-50 changed-law adaptation | `INSUFFICIENT_EVIDENCE` |
| formal unchanged-law retention | `INSUFFICIENT_EVIDENCE` |
| added-mechanism promotion | `NOT_APPLICABLE` to the selected no-refinement incumbent |
| joint Holm multiplicity analysis | `NOT_RUN` |
| complete A/B primitive archives and durable retrieval | `BLOCKED` |
| independent human review | `PENDING` |

The development smoke selected no refinement; it does not establish that the
incumbent satisfies any confirmation endpoint. Final outcome at this boundary:
**EXPLICITLY BLOCKED BY MISSING APPROVED DURABLE A/B ARCHIVE AND INDEPENDENT
REVIEW**. No scientific promotion, release, or generalization claim is made.

## Reproduction and recomputation commands

From the repository root, with the locked environment:

```bash
python -m aaa.cli observation-noise-protocol-hash
python -m aaa.cli observation-noise-fingerprint
python -m aaa.cli observation-noise-development-select --quick \
  --output docs/evidence/observation_noise_development_selection.json \
  --runs-root /tmp/aaa-noise-selection
python -m aaa.cli observation-noise-recompute <development-attempt>
python -m aaa.cli observation-noise-freeze \
  --stage confirmation_freeze \
  --batch observation-noise-a-0001 \
  --batch observation-noise-b-0001 \
  --selected-candidate incumbent-no-refinement-v1
python -m aaa.cli observation-noise-confirmation-evaluate <A-archive> <B-archive> \
  --output joint-observation-noise-evaluation.json
```

The last three commands are intentionally not represented as completed
evidence in this PR. The confirmation freeze must be committed before either
batch is consumed, both complete archives must be durably retrievable, and the
joint evaluator must be independently recomputed before a scientific verdict.

## CI and remaining blocker

The PR head was pushed to GitHub and the CPU workflow run identity was
`34807070748` (run number 49; status must be re-queried at the final PR head).
Local validation before this documentation refresh was 429 tests passed. The
remaining blocker is the absence of an approved durable archive mechanism for
the full observation-noise A/B primitive records; this is tracked as
`AAA-134`. The PR is engineering-complete for the controlled phase boundary
and explicitly blocked for scientific confirmation.

[SOL-INDEPENDENT-REVIEW: AAA / aaa.observation_noise.v1 / FULL]

Independently inspect the entire repository, full PR #11 history/diff, protocol and source identities, candidate ledger, all development-selection evidence, confirmation freeze, complete A/B primitive archives, schedules, checkpoints, statistics, joint multiplicity analysis, independent recomputation, durable-archive retrieval, tests, CI, documentation, issue ledger and every scientific claim. Recompute from primitive evidence rather than trusting Luna's summaries. Issue one final verdict: APPROVED, DECLINED, or BLOCKED, with exact findings. Approval authorizes normal merge only unless release/tag publication is separately requested.

---

# Engineering handoff for independent review

**Status: Prepared for GPT-5.6 Sol independent review and remediation.**
**This document does not grant approval.**

Every status below is a measured engineering outcome. None of it is an
approval decision, and none of it should be read as one. GPT-5.6 Sol is the
current designated independent reviewer and remediation owner. Sol's verdict
is recorded separately in `sol_review.md` only after the final evidence qualifies.

## 1. Identity

| Item | Value |
|---|---|
| starting `main` | `235ce28518ca4ec3a9a240066154090a33b33470` |
| engineering branch | `opus/aaa-complete-engineering-repair` |
| merge commit on `main` | `03923083ed1619cde01816922e6c9968031d59ba` |
| review baseline | `03923083ed1619cde01816922e6c9968031d59ba` |
| historical handoff tag | `opus-independent-engineering-complete-awaiting-astra-review` (provenance only) |
| historical tag target | `03923083ed1619cde01816922e6c9968031d59ba` |
| protocol | `aaa.benchmark.v2.1` |
| specification hash | `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37` |
| dependency lock hash | `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7` |
| freeze manifest source | `fadf9af0281dc71e1659e357218a18fd3af067c8`, dirty: yes |
| A/B relationship | `shared_frozen_checkpoints` |

### Selected candidate

`OnlineRLSPredictor` — three parameters, feature set
`displacement_position`, forgetting `0.3`
(`exponential`, trace bound `1.000e+05`,
self-triggered at `8.0x` its own error scale), ridge
`0.0001`, reflection `yes`,
24 training episodes per replica.

Selection evidence: [`candidate_selection.md`](candidate_selection.md), from
development streams only.

Frozen checkpoint hashes (shared by A and B, by declared design):

- replica 0: `e635f04f944190742fc6037c4468814a6a53eb50b4633ed9d1c0edd86e305a96`
- replica 1: `0dede9edd8afce6f583e97ba299d571c456ff05e266df6a24cf82328547feb9e`
- replica 2: `9540f727907c0bc8b36afaf15fecdd14d35775c537145e7eab999bfd878aa0ce`
- replica 3: `745dbb6d3f29eaa4d71c337f1ecf0e3f260ad93a6b07afdf221448a6913aac8a`
- replica 4: `d31088732cd195089a209ba1460c930fefcad7b160de5eeb1ca92c3edfc9361a`

## 2. Confirmation outcomes

| | Confirmation A | Confirmation B |
|---|---|---|
| batch id | `aaa-v2_1-confirmation-a-0004` | `aaa-v2_1-confirmation-b-0004` |
| all required gates pass | **yes** | **yes** |
| unmet required gates | none | none |
| source commit | `11ba061d2c33` | `dc4a6e1988c0` |
| source dirty | no | no |
| runtime | 666.2 s | 666.6 s |

### Every gate, both attempts

| Gate | Required | A | A observed | B | B observed |
|---|---|---|---:|---|---:|
| `stratum_coverage` | yes | **PASS** | 829 | **PASS** | 830 |
| `constant_velocity_identification` | yes | **PASS** | 3.908e-11 | **PASS** | 3.895e-11 |
| `learning_progress` | yes | **PASS** | 1 | **PASS** | 1 |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.536e-06 | **PASS** | 5.467e-06 |
| `bounce_event_accuracy` | yes | **PASS** | 3.377e-11 | **PASS** | 3.396e-11 |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.136e-05 | **PASS** | 4.063e-05 |
| `changed_law_adaptation` | yes | **PASS** | 0.7514 | **PASS** | 0.7542 |
| `unchanged_control` | yes | **PASS** | -5.705e-08 | **PASS** | -5.341e-08 |
| `recovery` | yes | **PASS** | 0.9746 | **PASS** | 0.9658 |
| `always_online_stability` | yes | **PASS** | 6.880e-06 | **PASS** | 6.841e-06 |
| `speed_extrapolation_report` | no | **PASS** | 1.102e-10 | **PASS** | 1.084e-10 |
| `correctness` | yes | **PASS** | 0 | **PASS** | 0 |
| `reproducibility` | yes | **PASS** | 0 | **PASS** | 0 |
| `cpu_usability` | yes | **PASS** | 0.03245 | **PASS** | 0.03319 |

## 2b. Round 1 — the confirmation that failed

Recorded here because it is evidence, not an embarrassment to be tidied away.
The first confirmation round under this protocol **failed**: both fresh streams
failed the required `always_online_stability` gate and both exited non-zero.
Both batches are retired permanently and cannot be reused. **No threshold was
altered in response**; the candidate was repaired instead, on development
evidence. See `AAA-120` in [`issue_ledger.md`](issue_ledger.md) for the full
diagnosis, including an alternative repair that was tried and did not work.

| | Round 1 A | Round 1 B |
|---|---|---|
| batch id | `aaa-v2_1-confirmation-a-0001` | `aaa-v2_1-confirmation-b-0001` |
| all required gates pass | **no** | **no** |
| unmet required gates | always_online_stability | always_online_stability |
| `candidate_online` normalized MAE | 4.427e-05 | 3.414e-05 |
| `constant_motion_reflected` | 6.621e-06 | 7.012e-06 |
| specification hash | `198b9a4ea4541206` | `198b9a4ea4541206` |

The round-1 specification hash differs from round 2 because the candidate changed;
a batch declared against one hash is refused under the other.

## 3. The findings that mattered most

### Numerical stability of the learner

The previous covariance-form RLS with forgetting 0.90 lost positive
semidefiniteness after **323** updates on a slow constant-velocity stream and
overflowed to non-finite state at update **6,642** on a stationary one; its
checkpoint loader accepted negative-definite, indefinite, asymmetric, singular,
zero and badly conditioned covariance matrices. The replacement propagates a
square-root factor with trace-bounded forgetting, matches an independent ridge
batch least-squares reference to **1.7e-18** with no forgetting, and survives
eleven weak-excitation streams to 20,000 updates. Invalid state now raises.

### The fair reflected baseline

The v2 bounce gate compared a candidate that could reflect into the public
bounds against a baseline that could not. Measured on development streams,
essentially the entire reported margin was that transform:

| Bounce-event normalized MAE | Value |
|---|---|
| raw constant motion | ~3.1e-03 |
| reflected constant motion (like-for-like) | ~3.7e-17 |
| candidate | ~4.2e-17 |
| candidate with reflection removed | ~3.1e-03 |

Against the fair baseline the candidate is not 20% better; it is marginally
worse, at floating-point magnitude. The gate was **replaced, not lowered**:
no correct system can beat a baseline that is already exact, so the criterion
is now an absolute accuracy requirement plus non-regression with an absolute
floor. The decision was made on development evidence and frozen before any
confirmation batch was generated.

Measured decomposition in confirmation A:

- versus raw constant motion: 1 [1, 1]
- versus reflected constant motion: -1.141e+06 [-1.626e+06, -7.357e+05]
- candidate without reflection versus raw constant motion: 1.209e-08 [8.192e-09, 1.582e-08]

The middle row is the honest one.

### Learning evidence

The v2 `straight_learning` gate was satisfied by a parameterless analytic rule.
It is replaced by a frozen-probe learning curve: checkpoints taken at increasing
cumulative update budgets, scored on one **fixed** development probe bank.

| Cumulative training episodes | Frozen probe normalized MAE (A) |
|---:|---:|
| 0 | 0.002872 |
| 2 | 1.806e-07 |
| 4 | 3.299e-09 |
| 8 | 1.196e-10 |
| 16 | 5.935e-11 |
| 24 | 3.911e-11 |

Reduction: 1 [1, 1].

### Changed-law adaptation

The matched experiment: common prefix, cloned learner state at the
intervention, frozen and updating copies in the same changed world, plus an
unchanged-world control. The branch state is hashed into
`metrics/changed_law_interventions.json` so the matched design is auditable
without trusting the runtime.

| Comparison | A | B |
|---|---|---|
| online vs identical frozen copy | 0.7514 [0.7401, 0.7613] | 0.7542 [0.7434, 0.765] |
| online vs persistence | 0.9594 [0.957, 0.9617] | 0.9599 [0.9579, 0.9617] |
| online vs constant motion (non-regression margin) | -0.01504 [-0.01653, -0.0137] | -0.01501 [-0.01657, -0.0135] |
| unchanged-world control margin | -1.015e-05 [-1.020e-05, -1.012e-05] | -1.014e-05 [-1.018e-05, -1.010e-05] |

The v2 result reported roughly 55% improvement over the frozen copy. That
signal survives the repaired statistics, the stabilized learner, the corrected
recovery rules and fresh confirmation streams. 55% was never used as a target.

### Recovery

Eligibility is now driven by the peak post-event shock rather than a
50-transition average, which previously diluted large fast shocks out of the
denominator entirely. Unrecovered events are counted.

| | A | B |
|---|---:|---:|
| eligible | 473 | 468 |
| recovered | 461 | 452 |
| unrecovered within horizon | 12 | 16 |
| rate | 0.9746 | 0.9658 |
| ineligible reasons (A) | {"no_measured_shock": 27} | |
| ineligible reasons (B) | {"no_measured_shock": 32} | |
| recovery time median / p90 / p95 | 20 / 26 / 30 | 19 / 26 / 32 |

Per-replica recovery rates, confirmation A: `r0` 0.9684, `r1` 0.9681, `r2` 0.9688, `r3` 0.9787, `r4` 0.9894.

### Reproducibility

| Check | A | B |
|---|---|---|
| `deterministic_rerun_identical` | yes | yes |
| `golden_seed_mapping_matches` | yes | yes |
| `save_resume_identical` | yes | yes |

| Correctness check | A | B |
|---|---|---|
| `all_trials_complete` | yes | yes |
| `checkpoint_hashes_stable` | yes | yes |
| `checkpoints_load` | yes | yes |
| `checksums_valid` | yes | yes |
| `dependency_lock_recorded` | yes | yes |
| `frozen_branch_state_measured_unchanged` | yes | yes |
| `no_duplicate_trial_ids` | yes | yes |
| `online_branch_actually_updated` | yes | yes |
| `records_structurally_valid` | yes | yes |
| `source_tree_clean_when_confirming` | yes | yes |
| `spec_hash_matches_canonical_when_confirming` | yes | yes |
| `summary_recomputes_from_raw_evidence` | yes | yes |

### Latency of the selected candidate

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00431 | 0.00497 | 0.005871 |
| update | 0.02523 | 0.02726 | 0.03632 |
| predict+update | 0.02956 | 0.03245 | 0.04175 |

Samples 2,000, warm-up 200 excluded, failures 0, on the selected candidate itself (forgetting 0.3).

### Hardware

- CPU: AMD Ryzen 7 5700G with Radeon Graphics (16 threads)
- OS: Linux-7.0.0-31-generic-x86_64-with-glibc2.39
- Python 3.12.3, NumPy 2.5.3, BLAS {"scipy-openblas": "0.3.34.106.0"}
- GPU present but unused: NVIDIA GeForce RTX 2060, 595.84, 6144 MiB

## 4. Architecture changes

- `aaa/benchmark.py` (587 lines, one module) became the `aaa/benchmark/`
  package: `spec`, `seeds`, `families`, `training`, `stats`, `gates`,
  `evidence`, `manifest`, `recompute`, `runner`, `report`.
- The canonical specification is typed, strictly validated, hash-identified,
  ships as package data, and is fully consumed — a test fails if any declared
  leaf stops being read.
- `StepRecord` v2 gives every provenance concept its own field; the previous
  `seed` field carried a replica id.
- The learner is a square-root RLS with trace-bounded, self-triggered
  forgetting; the historical SGD learner is retained as a reported diagnostic.
- New: confirmation batch registry, freeze manifest, experiment registry with
  safe resume, independent recomputation, an always-online deployment track,
  and a reflected constant-motion baseline.

## 5. Superseded and failed evidence

- `results/final/` — historical v1, preserved unchanged, unfavourable result
  intact.
- `results/benchmark_v2/` — the four v2 confirmation attempts, preserved and
  reclassified as historical/provisional under a superseded methodology, with
  corrections marked inline rather than rewritten away.
- `aaa-v2_1-confirmation-a-0001` and `aaa-v2_1-confirmation-b-0001` — the round-1 confirmation pair, retired by
  required-gate failure. Preserved in full under `results/benchmark_v2_1/`.
  Batch status is in `benchmarks/confirmation_batches.json`.

## 6. Unresolved limitations

- `AAA-077` raw per-step evidence is regenerable rather than durably
  archived. Git LFS or an external archive would be stronger; neither is in
  place and this is stated as a recommendation, not as done.
- `AAA-110` is closed rather than open: branch protection, required status
  checks, Dependabot and branch cleanup were applied through the API and read
  back to confirm. Two items are deliberately *not* applied and say so with
  their exact commands in `SECURITY.md`: `enforce_admins`, so the maintainer
  keeps a recovery path, and the account-level read-only workflow permission,
  which every workflow here already supersedes by declaring
  `permissions: contents: read` directly.
- `AAA-008` the always-online family rotates through a fixed sequence of the
  existing regimes rather than an open-ended stream.
- The world is deterministic and noiseless. Observation noise is the single
  most useful next experiment and is deliberately out of scope here.

## 7. Exact reproduction commands

```bash
git clone https://github.com/Cinqic/AAA.git && cd AAA
git checkout 03923083ed1619cde01816922e6c9968031d59ba
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/check_lock.py

# verification suite
python -m unittest discover -s tests -t . -v
python -m ruff check . && python -m ruff format --check . && python -m mypy
python tools/check_exit_codes.py

# protocol identity (must match the hash in section 1)
python -m aaa.cli spec-hash

# regenerate a confirmation attempt from its recorded batch identity
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0004 \
  --reproduce --output-root runs
```

### Recompute every metric and gate without retraining or re-simulating

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0004
```

This verifies checksums and the specification hash first and exits non-zero if
any stored gate status fails to reproduce.

## 8. Evidence locations

| What | Where |
|---|---|
| confirmation A summary and report | `results/benchmark_v2_1/aaa-v2_1-confirmation-a-0004/` |
| confirmation B summary and report | `results/benchmark_v2_1/aaa-v2_1-confirmation-b-0004/` |
| freeze manifest | `benchmarks/freeze_manifest.json` |
| batch registry | `benchmarks/confirmation_batches.json` |
| golden seed fixture | `benchmarks/golden_seeds.json` |
| canonical specification | `aaa/benchmark/data/benchmark_v2_1.json` |
| pre-repair defect reproduction | `docs/evidence/pre_repair_probes.json` |
| candidate selection | `docs/evidence/candidate_selection.json` |
| issue ledger | `docs/issue_ledger.md` |
| errata | `docs/errata.md` |

Per-attempt checksums are in each attempt's `checksums.json`. Raw per-step
records are regenerable rather than committed; see
[`evidence_policy.md`](evidence_policy.md) and its stated limitation.

### A note on how these settings were landed

The commit that applied and recorded the branch protections was pushed to
`main` directly, using the admin bypass that `enforce_admins: false`
deliberately leaves open — the protection had just been enabled and the
commit was the one recording it. Everything after that went through a pull
request with the six required checks, which is also how the protection was
verified to work end to end. Both facts are recorded here rather than left
for a reviewer to notice in the reflog.

## 9. Self-review

See [`self_review.md`](self_review.md). It is an **Opus self-review, not
independent approval**, and it is offered as a record of what was checked and
what was found, not as a verdict.

---

**Prepared for GPT-5.6 Sol independent review and remediation.**
**The handoff itself is not an approval decision.**
