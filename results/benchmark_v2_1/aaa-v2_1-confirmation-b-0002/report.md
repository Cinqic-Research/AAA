# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-b-0002`

Role: `confirmation_b`. Confirmation batch: `aaa-v2_1-confirmation-b-0002`.
Specification hash: `f8e1090bf5b1aeb0…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: yes**

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 826 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.888e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.480e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 3.381e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.273e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.7517 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -6.766e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9688 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **PASS** | 7.308e-06 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 1.111e-10 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03424 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | -9.550e-06 [-9.886e-06, -9.102e-06] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 3.381e-11 [2.349e-11, 4.437e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.206e-08 [8.293e-09, 1.587e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.171e+06 [-1.744e+06, -7.476e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.0146 [-0.01612, -0.01313] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.7517 [0.7376, 0.7626] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9601 [0.9579, 0.9621] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.888e-11 [3.468e-11, 4.325e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.480e-06 [5.116e-06, 5.863e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.427e-05 [-1.463e-05, -1.394e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 1.111e-10 [1.006e-10, 1.217e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.016e-05 [-1.021e-05, -1.012e-05] | n/a | 5 | 500 |

### Multiple-comparison treatment

Method `holm_bonferroni`, family size 4 (primary comparisons plus previously spent confirmation attempts), alpha 0.05.

| Comparison | p | adjusted p | rejected |
|---|---:|---:|---|
| `changed_law.online_vs_frozen` | 0.0002499 | 0.0009998 | yes |
| `changed_law.online_vs_persistence` | 0.0002499 | 0.0009998 | yes |
| `learning_progress.reduction` | 0.0002499 | 0.0009998 | yes |

## Learning progress

| Cumulative training episodes | Frozen probe MAE |
|---:|---:|
| 0 | 0.002872 |
| 2 | 1.806e-07 |
| 4 | 3.299e-09 |
| 8 | 1.196e-10 |
| 16 | 5.935e-11 |
| 24 | 3.911e-11 |

Every row is the same fixed development probe bank scored by a frozen checkpoint, so a changing
episode difficulty cannot masquerade as learning.

## Family measurements

### `constant_velocity`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.00279 | 0.002803 | 0.003842 | 0.003949 | n/a | n/a | n/a | 0.002799 | -0.0001173 |
| `constant_motion` | 2.557e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.212e-19 | 4.979e-20 |
| `constant_motion_reflected` | 2.557e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.212e-19 | 4.979e-20 |
| `zero_control` | 0.00279 | 0.002803 | 0.003842 | 0.003949 | n/a | n/a | n/a | 0.002799 | -0.0001173 |
| `legacy_linear_sgd` | 0.004404 | 0.004268 | 0.009703 | 0.01134 | n/a | n/a | n/a | 0.006013 | 0.001587 |
| `candidate_frozen` | 3.888e-11 | 3.850e-11 | 6.372e-11 | 7.170e-11 | n/a | n/a | n/a | 4.612e-11 | -4.592e-12 |
| `candidate_no_reflect` | 3.888e-11 | 3.850e-11 | 6.372e-11 | 7.170e-11 | n/a | n/a | n/a | 4.612e-11 | -4.592e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 826; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002796 | 0.002764 | 0.00388 | 0.003949 | 0.001393 | 0.003152 | n/a | 0.002821 | -2.512e-06 |
| `constant_motion` | 1.640e-05 | 0 | 0 | 1.388e-17 | 0.002801 | 0.006396 | n/a | 1.694e-05 | 2.604e-07 |
| `constant_motion_reflected` | 5.480e-06 | 0 | 0 | 3.469e-18 | 2.887e-17 | 2.220e-16 | n/a | 5.827e-06 | 1.253e-07 |
| `zero_control` | 0.002796 | 0.002764 | 0.00388 | 0.003949 | 0.001393 | 0.003152 | n/a | 0.002821 | -2.512e-06 |
| `legacy_linear_sgd` | 0.004453 | 0.004537 | 0.009799 | 0.01165 | 0.004737 | 0.01036 | n/a | 0.005959 | 0.001655 |
| `candidate_frozen` | 5.480e-06 | 3.918e-11 | 6.561e-11 | 7.436e-11 | 3.381e-11 | 6.622e-11 | n/a | 5.827e-06 | 1.253e-07 |
| `candidate_no_reflect` | 1.640e-05 | 3.925e-11 | 6.596e-11 | 7.564e-11 | 0.002801 | 0.006396 | n/a | 1.694e-05 | 2.604e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 924; event episodes: 490; no-event episodes excluded from event statistics: 10.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003033 | 0.002915 | 0.006371 | 0.006976 | 0.001695 | 0.004907 | 0.003202 | 0.003212 | 3.670e-05 |
| `constant_motion` | 2.707e-05 | 0 | 0 | 5.551e-17 | 0.003546 | 0.009986 | 4.993e-05 | 2.972e-05 | 1.219e-07 |
| `constant_motion_reflected` | 1.113e-05 | 0 | 0 | 1.388e-17 | 2.198e-17 | 2.220e-16 | 4.273e-05 | 1.147e-05 | -2.863e-08 |
| `zero_control` | 0.003033 | 0.002915 | 0.006371 | 0.006976 | 0.001695 | 0.004907 | 0.003202 | 0.003212 | 3.670e-05 |
| `legacy_linear_sgd` | 0.004738 | 0.004547 | 0.01059 | 0.01286 | 0.004806 | 0.01085 | 0.00512 | 0.006062 | 0.001714 |
| `candidate_frozen` | 1.113e-05 | 3.955e-11 | 9.285e-11 | 1.152e-10 | 4.274e-11 | 1.006e-10 | 4.273e-05 | 1.147e-05 | -2.863e-08 |
| `candidate_no_reflect` | 2.707e-05 | 3.962e-11 | 9.391e-11 | 1.214e-10 | 0.003546 | 0.009986 | 4.993e-05 | 2.972e-05 | 1.219e-07 |
| `candidate_online` | 1.202e-05 | 2.427e-08 | 4.064e-06 | 6.809e-06 | 8.923e-07 | 4.896e-06 | 4.599e-05 | 1.234e-05 | -2.852e-08 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007938 | 0.007951 | 0.009762 | 0.009957 | n/a | n/a | n/a | 0.008034 | 5.496e-05 |
| `constant_motion` | 7.643e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.171e-19 | -4.365e-20 |
| `constant_motion_reflected` | 7.643e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.171e-19 | -4.365e-20 |
| `zero_control` | 0.007938 | 0.007951 | 0.009762 | 0.009957 | n/a | n/a | n/a | 0.008034 | 5.496e-05 |
| `legacy_linear_sgd` | 0.008421 | 0.009296 | 0.01493 | 0.01656 | n/a | n/a | n/a | 0.008873 | 0.00165 |
| `candidate_frozen` | 1.111e-10 | 1.106e-10 | 1.472e-10 | 1.702e-10 | n/a | n/a | n/a | 1.304e-10 | -3.311e-12 |
| `candidate_no_reflect` | 1.111e-10 | 1.106e-10 | 1.472e-10 | 1.702e-10 | n/a | n/a | n/a | 1.304e-10 | -3.311e-12 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 152; event episodes: 152; no-event episodes excluded from event statistics: 348.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002845 | 0.002773 | 0.00514 | 0.006055 | 0.001858 | 0.004031 | 0.003231 | 0.002918 | -0.0001194 |
| `constant_motion` | 1.380e-05 | 0 | 0 | 1.388e-17 | 0.003789 | 0.008192 | 4.222e-05 | 1.695e-05 | 2.951e-06 |
| `constant_motion_reflected` | 6.235e-06 | 0 | 0 | 6.939e-18 | 2.337e-17 | 2.220e-16 | 3.416e-05 | 8.035e-06 | 9.586e-07 |
| `zero_control` | 0.002845 | 0.002773 | 0.00514 | 0.006055 | 0.001858 | 0.004031 | 0.003231 | 0.002918 | -0.0001194 |
| `legacy_linear_sgd` | 0.004582 | 0.004472 | 0.0102 | 0.0123 | 0.004714 | 0.01069 | 0.004987 | 0.006435 | 0.001503 |
| `candidate_frozen` | 6.235e-06 | 3.949e-11 | 7.502e-11 | 9.836e-11 | 4.674e-11 | 9.482e-11 | 3.416e-05 | 8.035e-06 | 9.586e-07 |
| `candidate_no_reflect` | 1.380e-05 | 3.951e-11 | 7.577e-11 | 1.046e-10 | 0.003789 | 0.008192 | 4.222e-05 | 1.695e-05 | 2.951e-06 |
| `candidate_online` | 7.308e-06 | 1.097e-07 | 5.849e-06 | 1.994e-05 | 8.394e-07 | 5.054e-06 | 3.826e-05 | 9.052e-06 | 1.155e-06 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001337 | 0.0006879 | 0.004859 | 0.007876 | n/a | n/a | 0.002031 | 0.001457 | -2.171e-06 |
| `constant_motion` | 0.0002209 | 0.0001046 | 0.0008325 | 0.001532 | n/a | n/a | 0.000339 | 0.0002407 | -1.258e-07 |
| `constant_motion_reflected` | 0.0002209 | 0.0001046 | 0.0008325 | 0.001532 | n/a | n/a | 0.000339 | 0.0002407 | -1.258e-07 |
| `frozen` | 0.0002125 | 9.968e-05 | 0.0008021 | 0.00149 | n/a | n/a | 0.0003265 | 0.0002313 | -1.668e-07 |
| `online` | 4.134e-05 | 1.374e-07 | 0.0001918 | 0.0009455 | n/a | n/a | 8.107e-05 | 4.252e-05 | -2.301e-07 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007623 | 0.0006838 | 0.001708 | 0.001948 | n/a | n/a | 0.0006363 | 0.0008001 | -3.322e-06 |
| `constant_motion` | 2.324e-05 | 1.961e-05 | 5.713e-05 | 6.787e-05 | n/a | n/a | 2.995e-05 | 2.453e-05 | -8.286e-08 |
| `constant_motion_reflected` | 2.324e-05 | 1.961e-05 | 5.713e-05 | 6.787e-05 | n/a | n/a | 2.995e-05 | 2.453e-05 | -8.286e-08 |
| `frozen` | 9.618e-07 | 1.015e-07 | 4.479e-06 | 1.887e-05 | n/a | n/a | 6.602e-07 | 1.192e-06 | -7.058e-08 |
| `online` | 8.942e-07 | 9.123e-08 | 4.357e-06 | 1.736e-05 | n/a | n/a | 6.283e-07 | 1.114e-06 | -6.827e-08 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 20 |
| `recovered` | 465 |
| `unrecovered` | 15 |

Eligible: 480; recovered: 465; unrecovered within horizon: 15.
Recovery time (transitions): median 19, p90 25, p95 30, max 49.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9789 |
| 1 | 0.9899 |
| 2 | 0.9789 |
| 3 | 0.9474 |
| 4 | 0.9479 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.171e+06 [-1.744e+06, -7.476e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.206e-08 [8.293e-09, 1.587e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00437 | 0.005131 | 0.007752 |
| update | 0.02568 | 0.02844 | 0.03555 |
| predict+update | 0.03007 | 0.03424 | 0.04103 |

Samples: 2,000; failures: 0; candidate forgetting factor: 0.3.

## Verification

| Check | Result |
|---|---|
| correctness: `all_trials_complete` | yes |
| correctness: `checkpoint_hashes_stable` | yes |
| correctness: `checkpoints_load` | yes |
| correctness: `checksums_valid` | yes |
| correctness: `dependency_lock_recorded` | yes |
| correctness: `frozen_branch_state_measured_unchanged` | yes |
| correctness: `no_duplicate_trial_ids` | yes |
| correctness: `online_branch_actually_updated` | yes |
| correctness: `records_structurally_valid` | yes |
| correctness: `source_tree_clean_when_confirming` | yes |
| correctness: `spec_hash_matches_canonical_when_confirming` | yes |
| correctness: `summary_recomputes_from_raw_evidence` | yes |
| reproducibility: `deterministic_rerun_identical` | yes |
| reproducibility: `golden_seed_mapping_matches` | yes |
| reproducibility: `save_resume_identical` | yes |

## Provenance

- Source commit: `936b6bec6ef1e86a5224c0b070ee3540d3030a12`; dirty: no
- Tree hash: `d36ef665d2cc6b5892dee6f8309650a0f1c5a6267f77232ddef04474e6f0a97b`
- Dependency lock hash: `6370808d7f23a04fce4f86b136210e1669063d6ef8113509381eec6655730ec3`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `e635f04f9441, 0dede9edd8af, 9540f727907c, 745dbb6d3f29, d31088732cd1`
- Runtime: 291.7 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_b --batch-id aaa-v2_1-confirmation-b-0002 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-b-0002
```
