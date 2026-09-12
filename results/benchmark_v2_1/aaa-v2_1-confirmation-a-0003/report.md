# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-a-0003`

Role: `confirmation_a`. Confirmation batch: `aaa-v2_1-confirmation-a-0003`.
Specification hash: `f8e1090bf5b1aeb0…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: no**

Unmet required gates: `correctness`.

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 823 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.900e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.575e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 3.379e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.384e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.7551 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -4.894e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9662 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **PASS** | 7.372e-06 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 1.107e-10 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **FAIL** | 1 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03387 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | -1.000e-05 [-1.013e-05, -9.868e-06] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 3.379e-11 [2.341e-11, 4.454e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.181e-08 [8.161e-09, 1.513e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.240e+06 [-1.851e+06, -7.968e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01493 [-0.01635, -0.01364] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.7551 [0.7429, 0.7654] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9603 [0.9579, 0.9625] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.900e-11 [3.464e-11, 4.351e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.575e-06 [5.220e-06, 5.939e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.438e-05 [-1.474e-05, -1.406e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 1.107e-10 [9.821e-11, 1.221e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.014e-05 [-1.018e-05, -1.009e-05] | n/a | 5 | 500 |

### Multiple-comparison treatment

Method `holm_bonferroni`, family size 5 (primary comparisons plus previously spent confirmation attempts), alpha 0.05.

| Comparison | p | adjusted p | rejected |
|---|---:|---:|---|
| `changed_law.online_vs_frozen` | 0.0002499 | 0.00125 | yes |
| `changed_law.online_vs_persistence` | 0.0002499 | 0.00125 | yes |
| `learning_progress.reduction` | 0.0002499 | 0.00125 | yes |

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
| `persistence` | 0.002798 | 0.002798 | 0.00388 | 0.003955 | n/a | n/a | n/a | 0.002809 | -0.0001032 |
| `constant_motion` | 1.858e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.663e-19 | 1.444e-20 |
| `constant_motion_reflected` | 1.858e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.663e-19 | 1.444e-20 |
| `zero_control` | 0.002798 | 0.002798 | 0.00388 | 0.003955 | n/a | n/a | n/a | 0.002809 | -0.0001032 |
| `legacy_linear_sgd` | 0.004416 | 0.004259 | 0.009756 | 0.01153 | n/a | n/a | n/a | 0.005948 | 0.00163 |
| `candidate_frozen` | 3.900e-11 | 3.879e-11 | 6.537e-11 | 7.253e-11 | n/a | n/a | n/a | 4.635e-11 | -4.466e-12 |
| `candidate_no_reflect` | 3.900e-11 | 3.879e-11 | 6.537e-11 | 7.253e-11 | n/a | n/a | n/a | 4.635e-11 | -4.466e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 823; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002786 | 0.002768 | 0.003844 | 0.003933 | 0.001366 | 0.003158 | n/a | 0.002802 | -4.385e-06 |
| `constant_motion` | 1.640e-05 | 0 | 0 | 1.388e-17 | 0.002855 | 0.006199 | n/a | 1.682e-05 | 1.089e-07 |
| `constant_motion_reflected` | 5.575e-06 | 0 | 0 | 3.469e-18 | 2.724e-17 | 2.220e-16 | n/a | 5.795e-06 | 4.481e-08 |
| `zero_control` | 0.002786 | 0.002768 | 0.003844 | 0.003933 | 0.001366 | 0.003158 | n/a | 0.002802 | -4.385e-06 |
| `legacy_linear_sgd` | 0.004473 | 0.004568 | 0.009808 | 0.01168 | 0.004828 | 0.01014 | n/a | 0.006007 | 0.001656 |
| `candidate_frozen` | 5.575e-06 | 3.932e-11 | 6.572e-11 | 7.503e-11 | 3.379e-11 | 6.650e-11 | n/a | 5.795e-06 | 4.481e-08 |
| `candidate_no_reflect` | 1.640e-05 | 3.938e-11 | 6.611e-11 | 7.622e-11 | 0.002855 | 0.006199 | n/a | 1.682e-05 | 1.089e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 894; event episodes: 484; no-event episodes excluded from event statistics: 16.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003023 | 0.002888 | 0.00628 | 0.007004 | 0.001671 | 0.005 | 0.003166 | 0.003148 | 2.585e-05 |
| `constant_motion` | 2.631e-05 | 0 | 0 | 5.551e-17 | 0.003488 | 0.00956 | 5.243e-05 | 2.832e-05 | -1.742e-08 |
| `constant_motion_reflected` | 1.118e-05 | 0 | 0 | 1.388e-17 | 2.638e-17 | 2.220e-16 | 4.384e-05 | 1.188e-05 | 9.601e-08 |
| `zero_control` | 0.003023 | 0.002888 | 0.00628 | 0.007004 | 0.001671 | 0.005 | 0.003166 | 0.003148 | 2.585e-05 |
| `legacy_linear_sgd` | 0.004764 | 0.004623 | 0.01055 | 0.01263 | 0.004962 | 0.01098 | 0.005072 | 0.006096 | 0.001716 |
| `candidate_frozen` | 1.118e-05 | 3.960e-11 | 9.338e-11 | 1.191e-10 | 4.375e-11 | 1.040e-10 | 4.384e-05 | 1.188e-05 | 9.601e-08 |
| `candidate_no_reflect` | 2.631e-05 | 3.968e-11 | 9.434e-11 | 1.241e-10 | 0.003488 | 0.00956 | 5.243e-05 | 2.832e-05 | -1.742e-08 |
| `candidate_online` | 1.208e-05 | 1.919e-08 | 4.096e-06 | 6.682e-06 | 9.248e-07 | 4.672e-06 | 4.711e-05 | 1.275e-05 | 5.468e-08 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007951 | 0.007862 | 0.009814 | 0.009966 | n/a | n/a | n/a | 0.008034 | -1.434e-05 |
| `constant_motion` | 7.484e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.232e-19 | -1.003e-20 |
| `constant_motion_reflected` | 7.484e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.232e-19 | -1.003e-20 |
| `zero_control` | 0.007951 | 0.007862 | 0.009814 | 0.009966 | n/a | n/a | n/a | 0.008034 | -1.434e-05 |
| `legacy_linear_sgd` | 0.008049 | 0.007894 | 0.01477 | 0.01637 | n/a | n/a | n/a | 0.008495 | 0.001578 |
| `candidate_frozen` | 1.107e-10 | 1.087e-10 | 1.507e-10 | 1.679e-10 | n/a | n/a | n/a | 1.306e-10 | -2.683e-12 |
| `candidate_no_reflect` | 1.107e-10 | 1.087e-10 | 1.507e-10 | 1.679e-10 | n/a | n/a | n/a | 1.306e-10 | -2.683e-12 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 161; event episodes: 161; no-event episodes excluded from event statistics: 339.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.00288 | 0.002823 | 0.005045 | 0.006154 | 0.001817 | 0.003744 | 0.003198 | 0.002939 | 4.400e-05 |
| `constant_motion` | 1.476e-05 | 0 | 0 | 2.776e-17 | 0.003835 | 0.008828 | 5.328e-05 | 1.682e-05 | -3.288e-06 |
| `constant_motion_reflected` | 6.703e-06 | 0 | 0 | 7.008e-18 | 2.069e-17 | 2.220e-16 | 3.890e-05 | 7.792e-06 | -9.866e-07 |
| `zero_control` | 0.00288 | 0.002823 | 0.005045 | 0.006154 | 0.001817 | 0.003744 | 0.003198 | 0.002939 | 4.400e-05 |
| `legacy_linear_sgd` | 0.004822 | 0.004987 | 0.01036 | 0.01222 | 0.005256 | 0.01064 | 0.005437 | 0.006201 | 0.001881 |
| `candidate_frozen` | 6.703e-06 | 4.009e-11 | 7.495e-11 | 9.941e-11 | 4.663e-11 | 8.492e-11 | 3.890e-05 | 7.792e-06 | -9.866e-07 |
| `candidate_no_reflect` | 1.476e-05 | 4.013e-11 | 7.543e-11 | 1.080e-10 | 0.003835 | 0.008828 | 5.328e-05 | 1.682e-05 | -3.288e-06 |
| `candidate_online` | 7.372e-06 | 8.448e-08 | 2.609e-06 | 1.873e-05 | 3.697e-07 | 8.259e-07 | 4.233e-05 | 8.568e-06 | -1.041e-06 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001365 | 0.0006837 | 0.005032 | 0.007757 | n/a | n/a | 0.002051 | 0.001539 | -2.831e-05 |
| `constant_motion` | 0.0002277 | 0.0001056 | 0.0008641 | 0.001469 | n/a | n/a | 0.0003453 | 0.0002517 | -7.182e-07 |
| `constant_motion_reflected` | 0.0002277 | 0.0001056 | 0.0008641 | 0.001469 | n/a | n/a | 0.0003453 | 0.0002517 | -7.182e-07 |
| `frozen` | 0.0002192 | 0.0001009 | 0.000837 | 0.001427 | n/a | n/a | 0.0003328 | 0.0002418 | -8.345e-07 |
| `online` | 4.170e-05 | 1.354e-07 | 0.0002091 | 0.001007 | n/a | n/a | 8.148e-05 | 4.305e-05 | -9.343e-07 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007596 | 0.0006834 | 0.001695 | 0.001934 | n/a | n/a | 0.0006272 | 0.0008284 | -4.320e-05 |
| `constant_motion` | 2.323e-05 | 1.951e-05 | 5.775e-05 | 6.795e-05 | n/a | n/a | 3.016e-05 | 2.549e-05 | -6.225e-07 |
| `constant_motion_reflected` | 2.323e-05 | 1.951e-05 | 5.775e-05 | 6.795e-05 | n/a | n/a | 3.016e-05 | 2.549e-05 | -6.225e-07 |
| `frozen` | 8.768e-07 | 1.021e-07 | 3.562e-06 | 1.726e-05 | n/a | n/a | 6.238e-07 | 1.155e-06 | -1.577e-07 |
| `online` | 8.278e-07 | 9.053e-08 | 3.305e-06 | 1.684e-05 | n/a | n/a | 5.932e-07 | 1.093e-06 | -1.498e-07 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 26 |
| `recovered` | 458 |
| `unrecovered` | 16 |

Eligible: 474; recovered: 458; unrecovered within horizon: 16.
Recovery time (transitions): median 19, p90 24, p95 28, max 44.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9574 |
| 1 | 0.9684 |
| 2 | 0.9677 |
| 3 | 0.9691 |
| 4 | 0.9684 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.240e+06 [-1.851e+06, -7.968e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.181e-08 [8.161e-09, 1.513e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.0048 | 0.00551 | 0.00662 |
| update | 0.0259 | 0.02828 | 0.03798 |
| predict+update | 0.0307 | 0.03387 | 0.04464 |

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
| correctness: `source_tree_clean_when_confirming` | no |
| correctness: `spec_hash_matches_canonical_when_confirming` | yes |
| correctness: `summary_recomputes_from_raw_evidence` | yes |
| reproducibility: `deterministic_rerun_identical` | yes |
| reproducibility: `golden_seed_mapping_matches` | yes |
| reproducibility: `save_resume_identical` | yes |

## Provenance

- Source commit: `8ae3f59c79518ece58356a141a6368ea479af693`; dirty: yes
- Tree hash: `b8171465a4f9388ae8ed29ca8b1f00805cc6e40655479048ebfb85373db820cc`
- Dependency lock hash: `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `e635f04f9441, 0dede9edd8af, 9540f727907c, 745dbb6d3f29, d31088732cd1`
- Runtime: 666 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0003 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0003
```
