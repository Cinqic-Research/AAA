# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-a-0004`

Role: `confirmation_a`. Confirmation batch: `aaa-v2_1-confirmation-a-0004`.
Specification hash: `f8e1090bf5b1aeb0…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: yes**

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 829 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.908e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.536e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 3.377e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.136e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.7514 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -5.705e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9746 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **PASS** | 6.880e-06 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 1.102e-10 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03245 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | -9.877e-06 [-1.003e-05, -9.731e-06] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 3.377e-11 [2.324e-11, 4.457e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.209e-08 [8.192e-09, 1.582e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.141e+06 [-1.626e+06, -7.357e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01504 [-0.01653, -0.0137] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.7514 [0.7401, 0.7613] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9594 [0.957, 0.9617] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.908e-11 [3.476e-11, 4.374e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.536e-06 [5.155e-06, 5.920e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.414e-05 [-1.439e-05, -1.391e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 1.102e-10 [9.989e-11, 1.210e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.015e-05 [-1.020e-05, -1.012e-05] | n/a | 5 | 500 |

### Multiple-comparison treatment

Method `holm_bonferroni`, family size 6 (primary comparisons plus previously spent confirmation attempts), alpha 0.05.

| Comparison | p | adjusted p | rejected |
|---|---:|---:|---|
| `changed_law.online_vs_frozen` | 0.0002499 | 0.0015 | yes |
| `changed_law.online_vs_persistence` | 0.0002499 | 0.0015 | yes |
| `learning_progress.reduction` | 0.0002499 | 0.0015 | yes |

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
| `persistence` | 0.002806 | 0.002801 | 0.003866 | 0.003945 | n/a | n/a | n/a | 0.002819 | -0.000113 |
| `constant_motion` | 2.402e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.879e-19 | -2.757e-20 |
| `constant_motion_reflected` | 2.402e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.879e-19 | -2.757e-20 |
| `zero_control` | 0.002806 | 0.002801 | 0.003866 | 0.003945 | n/a | n/a | n/a | 0.002819 | -0.000113 |
| `legacy_linear_sgd` | 0.00441 | 0.004253 | 0.009596 | 0.01153 | n/a | n/a | n/a | 0.006007 | 0.00161 |
| `candidate_frozen` | 3.908e-11 | 3.935e-11 | 6.512e-11 | 7.393e-11 | n/a | n/a | n/a | 4.689e-11 | -4.533e-12 |
| `candidate_no_reflect` | 3.908e-11 | 3.935e-11 | 6.512e-11 | 7.393e-11 | n/a | n/a | n/a | 4.689e-11 | -4.533e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 829; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002781 | 0.002768 | 0.003817 | 0.003948 | 0.001362 | 0.003242 | n/a | 0.002806 | 5.166e-07 |
| `constant_motion` | 1.642e-05 | 0 | 0 | 1.388e-17 | 0.002785 | 0.006322 | n/a | 1.670e-05 | 3.107e-07 |
| `constant_motion_reflected` | 5.536e-06 | 0 | 0 | 3.469e-18 | 2.961e-17 | 2.220e-16 | n/a | 5.802e-06 | 1.191e-07 |
| `zero_control` | 0.002781 | 0.002768 | 0.003817 | 0.003948 | 0.001362 | 0.003242 | n/a | 0.002806 | 5.166e-07 |
| `legacy_linear_sgd` | 0.004471 | 0.004569 | 0.009827 | 0.01163 | 0.004734 | 0.01033 | n/a | 0.005992 | 0.001655 |
| `candidate_frozen` | 5.536e-06 | 3.922e-11 | 6.547e-11 | 7.422e-11 | 3.377e-11 | 6.665e-11 | n/a | 5.803e-06 | 1.191e-07 |
| `candidate_no_reflect` | 1.642e-05 | 3.928e-11 | 6.584e-11 | 7.537e-11 | 0.002785 | 0.006322 | n/a | 1.670e-05 | 3.107e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 909; event episodes: 491; no-event episodes excluded from event statistics: 9.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003051 | 0.002904 | 0.006374 | 0.006976 | 0.001751 | 0.004864 | 0.003213 | 0.003183 | 3.204e-05 |
| `constant_motion` | 2.672e-05 | 0 | 0 | 5.551e-17 | 0.003574 | 0.009991 | 4.828e-05 | 2.851e-05 | -2.726e-07 |
| `constant_motion_reflected` | 1.098e-05 | 0 | 0 | 1.388e-17 | 3.030e-17 | 2.220e-16 | 4.136e-05 | 1.136e-05 | 1.852e-07 |
| `zero_control` | 0.003051 | 0.002904 | 0.006374 | 0.006976 | 0.001751 | 0.004864 | 0.003213 | 0.003183 | 3.204e-05 |
| `legacy_linear_sgd` | 0.004764 | 0.004476 | 0.01058 | 0.01284 | 0.004857 | 0.01078 | 0.005076 | 0.006009 | 0.001763 |
| `candidate_frozen` | 1.098e-05 | 4.014e-11 | 9.250e-11 | 1.171e-10 | 4.406e-11 | 9.898e-11 | 4.136e-05 | 1.136e-05 | 1.852e-07 |
| `candidate_no_reflect` | 2.672e-05 | 4.022e-11 | 9.348e-11 | 1.246e-10 | 0.003574 | 0.009991 | 4.828e-05 | 2.851e-05 | -2.726e-07 |
| `candidate_online` | 1.185e-05 | 2.778e-08 | 3.886e-06 | 6.676e-06 | 8.427e-07 | 4.392e-06 | 4.449e-05 | 1.218e-05 | 1.669e-07 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007878 | 0.007846 | 0.009748 | 0.009959 | n/a | n/a | n/a | 0.007975 | -0.0002213 |
| `constant_motion` | 7.629e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 8.364e-19 | 2.733e-20 |
| `constant_motion_reflected` | 7.629e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 8.364e-19 | 2.733e-20 |
| `zero_control` | 0.007878 | 0.007846 | 0.009748 | 0.009959 | n/a | n/a | n/a | 0.007975 | -0.0002213 |
| `legacy_linear_sgd` | 0.007951 | 0.007559 | 0.01488 | 0.01674 | n/a | n/a | n/a | 0.008429 | 0.001314 |
| `candidate_frozen` | 1.102e-10 | 1.097e-10 | 1.486e-10 | 1.643e-10 | n/a | n/a | n/a | 1.293e-10 | -5.725e-12 |
| `candidate_no_reflect` | 1.102e-10 | 1.097e-10 | 1.486e-10 | 1.643e-10 | n/a | n/a | n/a | 1.293e-10 | -5.725e-12 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 148; event episodes: 148; no-event episodes excluded from event statistics: 352.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002805 | 0.002728 | 0.004957 | 0.006112 | 0.001887 | 0.00445 | 0.003164 | 0.002963 | -0.0001013 |
| `constant_motion` | 1.337e-05 | 0 | 0 | 2.776e-17 | 0.003452 | 0.008247 | 3.962e-05 | 1.574e-05 | -2.270e-07 |
| `constant_motion_reflected` | 6.143e-06 | 0 | 0 | 6.939e-18 | 2.400e-17 | 2.220e-16 | 3.324e-05 | 7.269e-06 | -2.302e-08 |
| `zero_control` | 0.002805 | 0.002728 | 0.004957 | 0.006112 | 0.001887 | 0.00445 | 0.003164 | 0.002963 | -0.0001013 |
| `legacy_linear_sgd` | 0.004527 | 0.003885 | 0.009847 | 0.01174 | 0.004724 | 0.01053 | 0.004623 | 0.006069 | 0.001627 |
| `candidate_frozen` | 6.143e-06 | 3.826e-11 | 7.081e-11 | 1.030e-10 | 4.331e-11 | 8.563e-11 | 3.324e-05 | 7.269e-06 | -2.303e-08 |
| `candidate_no_reflect` | 1.337e-05 | 3.830e-11 | 7.125e-11 | 1.097e-10 | 0.003452 | 0.008247 | 3.962e-05 | 1.574e-05 | -2.270e-07 |
| `candidate_online` | 6.880e-06 | 9.271e-08 | 3.404e-06 | 1.718e-05 | 2.938e-07 | 1.282e-06 | 3.643e-05 | 8.188e-06 | -5.777e-09 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001366 | 0.0006679 | 0.00507 | 0.00798 | n/a | n/a | 0.002064 | 0.001502 | 2.923e-05 |
| `constant_motion` | 0.0002281 | 0.0001044 | 0.0008683 | 0.00155 | n/a | n/a | 0.0003495 | 0.0002517 | 9.490e-07 |
| `constant_motion_reflected` | 0.0002281 | 0.0001044 | 0.0008683 | 0.00155 | n/a | n/a | 0.0003495 | 0.0002517 | 9.490e-07 |
| `frozen` | 0.0002196 | 9.963e-05 | 0.0008355 | 0.001507 | n/a | n/a | 0.0003369 | 0.0002424 | 1.013e-06 |
| `online` | 4.267e-05 | 1.442e-07 | 0.0002066 | 0.000996 | n/a | n/a | 8.376e-05 | 4.571e-05 | 1.658e-06 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007687 | 0.0006885 | 0.001738 | 0.001958 | n/a | n/a | 0.0006357 | 0.0007965 | 4.914e-05 |
| `constant_motion` | 2.346e-05 | 1.950e-05 | 5.840e-05 | 6.828e-05 | n/a | n/a | 3.042e-05 | 2.451e-05 | 5.912e-07 |
| `constant_motion_reflected` | 2.346e-05 | 1.950e-05 | 5.840e-05 | 6.828e-05 | n/a | n/a | 3.042e-05 | 2.451e-05 | 5.912e-07 |
| `frozen` | 9.720e-07 | 9.925e-08 | 4.785e-06 | 1.650e-05 | n/a | n/a | 6.748e-07 | 1.225e-06 | 1.584e-09 |
| `online` | 9.150e-07 | 8.767e-08 | 4.693e-06 | 1.577e-05 | n/a | n/a | 6.443e-07 | 1.145e-06 | -1.746e-09 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 27 |
| `recovered` | 461 |
| `unrecovered` | 12 |

Eligible: 473; recovered: 461; unrecovered within horizon: 12.
Recovery time (transitions): median 20, p90 26, p95 30, max 50.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9684 |
| 1 | 0.9681 |
| 2 | 0.9688 |
| 3 | 0.9787 |
| 4 | 0.9894 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.141e+06 [-1.626e+06, -7.357e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.209e-08 [8.192e-09, 1.582e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00431 | 0.00497 | 0.005871 |
| update | 0.02523 | 0.02726 | 0.03632 |
| predict+update | 0.02956 | 0.03245 | 0.04175 |

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

- Source commit: `11ba061d2c33a5f04e71fd10a186de3af887b180`; dirty: no
- Tree hash: `aeaff526fde5d136c000debe1ce6fa8c9207ad43ece24d37a75ad5b572028d6a`
- Dependency lock hash: `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `e635f04f9441, 0dede9edd8af, 9540f727907c, 745dbb6d3f29, d31088732cd1`
- Runtime: 666.2 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0004 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0004
```
