# Engineering handoff for independent review

**Status: Opus engineering complete. Awaiting independent Astra review.**
**This document does not grant approval.**

Every status below is a measured engineering outcome. None of it is an
approval decision, and none of it should be read as one. GPT-6 Astra is the
designated independent reviewer and is the only reviewer authorized to issue
an APPROVED or DECLINED decision.

## 1. Identity

| Item | Value |
|---|---|
| starting `main` | `235ce28518ca4ec3a9a240066154090a33b33470` |
| engineering branch | `opus/aaa-complete-engineering-repair` |
| merge commit on `main` | `7c11639656267787045c2a849ff51426bba9a4d7` |
| tag | `opus-independent-engineering-complete-awaiting-astra-review` |
| tag target | `61bd516ee475b8f21f56235875faf47ebda363f9` |
| protocol | `aaa.benchmark.v2.1` |
| specification hash | `f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37` |
| dependency lock hash | `6370808d7f23a04fce4f86b136210e1669063d6ef8113509381eec6655730ec3` |
| freeze manifest source | `2fe2f2f3704c0194d233d5a6f9f135d401f0ddd9`, dirty: no |
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
| batch id | `aaa-v2_1-confirmation-a-0002` | `aaa-v2_1-confirmation-b-0002` |
| all required gates pass | **yes** | **yes** |
| unmet required gates | none | none |
| source commit | `f1b94cc4084e` | `936b6bec6ef1` |
| source dirty | no | no |
| runtime | 288.1 s | 291.7 s |

### Every gate, both attempts

| Gate | Required | A | A observed | B | B observed |
|---|---|---|---:|---|---:|
| `stratum_coverage` | yes | **PASS** | 820 | **PASS** | 826 |
| `constant_velocity_identification` | yes | **PASS** | 3.904e-11 | **PASS** | 3.888e-11 |
| `learning_progress` | yes | **PASS** | 1 | **PASS** | 1 |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.347e-06 | **PASS** | 5.480e-06 |
| `bounce_event_accuracy` | yes | **PASS** | 3.396e-11 | **PASS** | 3.381e-11 |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.450e-05 | **PASS** | 4.273e-05 |
| `changed_law_adaptation` | yes | **PASS** | 0.7635 | **PASS** | 0.7517 |
| `unchanged_control` | yes | **PASS** | -4.906e-08 | **PASS** | -6.766e-08 |
| `recovery` | yes | **PASS** | 0.9679 | **PASS** | 0.9688 |
| `always_online_stability` | yes | **PASS** | 6.712e-06 | **PASS** | 7.308e-06 |
| `speed_extrapolation_report` | no | **PASS** | 1.113e-10 | **PASS** | 1.111e-10 |
| `correctness` | yes | **PASS** | 0 | **PASS** | 0 |
| `reproducibility` | yes | **PASS** | 0 | **PASS** | 0 |
| `cpu_usability` | yes | **PASS** | 0.03093 | **PASS** | 0.03424 |

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
- versus reflected constant motion: -1.233e+06 [-1.896e+06, -8.283e+05]
- candidate without reflection versus raw constant motion: 1.198e-08 [8.181e-09, 1.565e-08]

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
| online vs identical frozen copy | 0.7635 [0.7515, 0.7735] | 0.7517 [0.7376, 0.7626] |
| online vs persistence | 0.9617 [0.9593, 0.9643] | 0.9601 [0.9579, 0.9621] |
| online vs constant motion (non-regression margin) | -0.01581 [-0.01771, -0.01405] | -0.0146 [-0.01612, -0.01313] |
| unchanged-world control margin | -1.012e-05 [-1.017e-05, -1.009e-05] | -1.016e-05 [-1.021e-05, -1.012e-05] |

The v2 result reported roughly 55% improvement over the frozen copy. That
signal survives the repaired statistics, the stabilized learner, the corrected
recovery rules and fresh confirmation streams. 55% was never used as a target.

### Recovery

Eligibility is now driven by the peak post-event shock rather than a
50-transition average, which previously diluted large fast shocks out of the
denominator entirely. Unrecovered events are counted.

| | A | B |
|---|---:|---:|
| eligible | 467 | 480 |
| recovered | 452 | 465 |
| unrecovered within horizon | 15 | 15 |
| rate | 0.9679 | 0.9688 |
| ineligible reasons (A) | {"no_measured_shock": 33} | |
| ineligible reasons (B) | {"no_measured_shock": 20} | |
| recovery time median / p90 / p95 | 19 / 25 / 29.45 | 19 / 25 / 30 |

Per-replica recovery rates, confirmation A: `r0` 0.9783, `r1` 0.9785, `r2` 0.9588, `r3` 0.9785, `r4` 0.9457.

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
| predict | 0.00416 | 0.00452 | 0.00532 |
| update | 0.02485 | 0.02621 | 0.03262 |
| predict+update | 0.02902 | 0.03093 | 0.0373 |

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
git checkout opus-independent-engineering-complete-awaiting-astra-review
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
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0002 \
  --reproduce --output-root runs
```

### Recompute every metric and gate without retraining or re-simulating

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0002
```

This verifies checksums and the specification hash first and exits non-zero if
any stored gate status fails to reproduce.

## 8. Evidence locations

| What | Where |
|---|---|
| confirmation A summary and report | `results/benchmark_v2_1/aaa-v2_1-confirmation-a-0002/` |
| confirmation B summary and report | `results/benchmark_v2_1/aaa-v2_1-confirmation-b-0002/` |
| freeze manifest | `benchmarks/freeze_manifest.json` |
| batch registry | `benchmarks/confirmation_batches.json` |
| golden seed fixture | `benchmarks/golden_seeds.json` |
| canonical specification | `aaa/benchmark/data/benchmark_v2_1.json` |
| pre-repair defect reproduction | `docs/evidence/pre_repair_probes.json` |
| learner diagnosis | `docs/evidence/diagnosis/` |
| candidate selection | `docs/evidence/candidate_selection.json` |
| issue ledger | `docs/issue_ledger.md` |
| errata | `docs/errata.md` |

Per-attempt checksums are in each attempt's `checksums.json`. Raw per-step
records are regenerable rather than committed; see
[`evidence_policy.md`](evidence_policy.md) and its stated limitation.

## 9. Self-review

See [`self_review.md`](self_review.md). It is an **Opus self-review, not
independent approval**, and it is offered as a record of what was checked and
what was found, not as a verdict.

---

**Opus independent engineering complete. Awaiting Astra independent review.**
**No independent approval decision has been made.**
