# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-a-0001`

Role: `confirmation_a`. Confirmation batch: `aaa-v2_1-confirmation-a-0001`.
Specification hash: `198b9a4ea4541206…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: no**

Unmet required gates: `always_online_stability`.

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 824 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.295e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.493e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 2.899e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.423e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.6633 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -6.776e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9622 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **FAIL** | 4.427e-05 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 9.385e-11 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03029 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | 2.699e-05 [-4.287e-06, 0.0001096] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 2.899e-11 [1.969e-11, 3.927e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.059e-08 [6.918e-09, 1.458e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.125e+06 [-1.675e+06, -7.241e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01335 [-0.01469, -0.01191] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.6633 [0.6513, 0.6734] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9449 [0.9427, 0.9471] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.295e-11 [2.931e-11, 3.695e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.493e-06 [5.046e-06, 5.945e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.442e-05 [-1.476e-05, -1.412e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 9.385e-11 [8.450e-11, 1.044e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.017e-05 [-1.023e-05, -1.013e-05] | n/a | 5 | 500 |

### Multiple-comparison treatment

Method `holm_bonferroni`, family size 3 (primary comparisons plus previously spent confirmation attempts), alpha 0.05.

| Comparison | p | adjusted p | rejected |
|---|---:|---:|---|
| `changed_law.online_vs_frozen` | 0.0002499 | 0.0007498 | yes |
| `changed_law.online_vs_persistence` | 0.0002499 | 0.0007498 | yes |
| `learning_progress.reduction` | 0.0002499 | 0.0007498 | yes |

## Learning progress

| Cumulative training episodes | Frozen probe MAE |
|---:|---:|
| 0 | 0.002872 |
| 2 | 2.202e-07 |
| 4 | 3.258e-09 |
| 8 | 1.066e-10 |
| 16 | 5.131e-11 |
| 24 | 3.340e-11 |

Every row is the same fixed development probe bank scored by a frozen checkpoint, so a changing
episode difficulty cannot masquerade as learning.

## Family measurements

### `constant_velocity`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002791 | 0.002799 | 0.003834 | 0.003931 | n/a | n/a | n/a | 0.002814 | -0.000117 |
| `constant_motion` | 2.711e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 4.182e-19 | -2.204e-20 |
| `constant_motion_reflected` | 2.711e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 4.182e-19 | -2.204e-20 |
| `zero_control` | 0.002791 | 0.002799 | 0.003834 | 0.003931 | n/a | n/a | n/a | 0.002814 | -0.000117 |
| `legacy_linear_sgd` | 0.004424 | 0.004108 | 0.009736 | 0.01133 | n/a | n/a | n/a | 0.005972 | 0.001621 |
| `candidate_frozen` | 3.295e-11 | 3.267e-11 | 5.443e-11 | 6.439e-11 | n/a | n/a | n/a | 3.984e-11 | -3.294e-12 |
| `candidate_no_reflect` | 3.295e-11 | 3.267e-11 | 5.443e-11 | 6.439e-11 | n/a | n/a | n/a | 3.984e-11 | -3.294e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 824; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002788 | 0.002768 | 0.003843 | 0.003957 | 0.001396 | 0.003051 | n/a | 0.002808 | -6.170e-06 |
| `constant_motion` | 1.633e-05 | 0 | 0 | 1.388e-17 | 0.002734 | 0.006201 | n/a | 1.689e-05 | -7.445e-08 |
| `constant_motion_reflected` | 5.493e-06 | 0 | 0 | 3.469e-18 | 2.576e-17 | 2.220e-16 | n/a | 6.033e-06 | -1.453e-07 |
| `zero_control` | 0.002788 | 0.002768 | 0.003843 | 0.003957 | 0.001396 | 0.003051 | n/a | 0.002808 | -6.170e-06 |
| `legacy_linear_sgd` | 0.004483 | 0.004603 | 0.009802 | 0.01168 | 0.00477 | 0.01029 | n/a | 0.006014 | 0.001665 |
| `candidate_frozen` | 5.493e-06 | 3.313e-11 | 5.574e-11 | 6.595e-11 | 2.899e-11 | 6.026e-11 | n/a | 6.033e-06 | -1.453e-07 |
| `candidate_no_reflect` | 1.633e-05 | 3.319e-11 | 5.611e-11 | 6.763e-11 | 0.002734 | 0.006201 | n/a | 1.689e-05 | -7.445e-08 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 943; event episodes: 488; no-event episodes excluded from event statistics: 12.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003118 | 0.002982 | 0.006455 | 0.006965 | 0.001764 | 0.005107 | 0.003361 | 0.003361 | -2.124e-05 |
| `constant_motion` | 2.820e-05 | 0 | 0 | 5.551e-17 | 0.003489 | 0.01017 | 5.431e-05 | 3.208e-05 | 3.522e-08 |
| `constant_motion_reflected` | 1.153e-05 | 0 | 0 | 1.388e-17 | 2.889e-17 | 2.220e-16 | 4.423e-05 | 1.270e-05 | -3.761e-07 |
| `zero_control` | 0.003118 | 0.002982 | 0.006455 | 0.006965 | 0.001764 | 0.005107 | 0.003361 | 0.003361 | -2.124e-05 |
| `legacy_linear_sgd` | 0.004712 | 0.004347 | 0.01072 | 0.01307 | 0.004897 | 0.01059 | 0.005127 | 0.006095 | 0.001641 |
| `candidate_frozen` | 1.153e-05 | 3.514e-11 | 8.082e-11 | 1.025e-10 | 3.828e-11 | 8.668e-11 | 4.423e-05 | 1.270e-05 | -3.761e-07 |
| `candidate_no_reflect` | 2.820e-05 | 3.520e-11 | 8.159e-11 | 1.084e-10 | 0.003489 | 0.01017 | 5.431e-05 | 3.208e-05 | 3.522e-08 |
| `candidate_online` | 3.439e-05 | 7.235e-07 | 8.157e-06 | 7.193e-05 | 8.529e-07 | 4.463e-06 | 6.557e-05 | 5.228e-05 | 3.462e-06 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007952 | 0.007936 | 0.009814 | 0.009976 | n/a | n/a | n/a | 0.008012 | 0.0001066 |
| `constant_motion` | 6.386e-19 | 0 | 0 | 1.388e-17 | n/a | n/a | n/a | 7.183e-19 | 4.126e-21 |
| `constant_motion_reflected` | 6.386e-19 | 0 | 0 | 1.388e-17 | n/a | n/a | n/a | 7.183e-19 | 4.126e-21 |
| `zero_control` | 0.007952 | 0.007936 | 0.009814 | 0.009976 | n/a | n/a | n/a | 0.008012 | 0.0001066 |
| `legacy_linear_sgd` | 0.007983 | 0.007641 | 0.01471 | 0.01656 | n/a | n/a | n/a | 0.008187 | 0.001757 |
| `candidate_frozen` | 9.385e-11 | 9.246e-11 | 1.290e-10 | 1.459e-10 | n/a | n/a | n/a | 1.138e-10 | -2.914e-13 |
| `candidate_no_reflect` | 9.385e-11 | 9.246e-11 | 1.290e-10 | 1.459e-10 | n/a | n/a | n/a | 1.138e-10 | -2.914e-13 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 154; event episodes: 154; no-event episodes excluded from event statistics: 346.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002811 | 0.002739 | 0.004895 | 0.00622 | 0.001712 | 0.003857 | 0.003093 | 0.002918 | 1.363e-05 |
| `constant_motion` | 1.419e-05 | 0 | 0 | 2.776e-17 | 0.00372 | 0.007458 | 4.028e-05 | 1.601e-05 | 4.356e-07 |
| `constant_motion_reflected` | 6.621e-06 | 0 | 0 | 6.939e-18 | 2.307e-17 | 2.220e-16 | 3.419e-05 | 7.389e-06 | 2.391e-07 |
| `zero_control` | 0.002811 | 0.002739 | 0.004895 | 0.00622 | 0.001712 | 0.003857 | 0.003093 | 0.002918 | 1.363e-05 |
| `legacy_linear_sgd` | 0.004472 | 0.004477 | 0.01018 | 0.01239 | 0.004665 | 0.01055 | 0.005033 | 0.005723 | 0.001644 |
| `candidate_frozen` | 6.621e-06 | 3.275e-11 | 6.108e-11 | 8.294e-11 | 4.078e-11 | 7.483e-11 | 3.419e-05 | 7.389e-06 | 2.391e-07 |
| `candidate_no_reflect` | 1.419e-05 | 3.277e-11 | 6.149e-11 | 8.562e-11 | 0.00372 | 0.007458 | 4.028e-05 | 1.601e-05 | 4.356e-07 |
| `candidate_online` | 4.427e-05 | 6.656e-07 | 1.588e-05 | 0.0001369 | 1.619e-06 | 8.972e-06 | 4.673e-05 | 0.0001519 | -5.339e-06 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001316 | 0.0006598 | 0.004893 | 0.007645 | n/a | n/a | 0.002028 | 0.001424 | 2.813e-06 |
| `constant_motion` | 0.0002198 | 0.0001024 | 0.0008397 | 0.001489 | n/a | n/a | 0.0003442 | 0.0002374 | 4.761e-07 |
| `constant_motion_reflected` | 0.0002198 | 0.0001024 | 0.0008397 | 0.001489 | n/a | n/a | 0.0003442 | 0.0002374 | 4.761e-07 |
| `frozen` | 0.0002116 | 9.778e-05 | 0.0008127 | 0.001451 | n/a | n/a | 0.0003317 | 0.0002284 | 4.946e-07 |
| `online` | 5.675e-05 | 3.781e-07 | 0.0003186 | 0.001114 | n/a | n/a | 0.0001117 | 6.167e-05 | 2.592e-07 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007674 | 0.0006992 | 0.001714 | 0.001933 | n/a | n/a | 0.000634 | 0.0008089 | -2.398e-06 |
| `constant_motion` | 2.339e-05 | 2.017e-05 | 5.832e-05 | 6.760e-05 | n/a | n/a | 3.031e-05 | 2.465e-05 | 4.261e-07 |
| `constant_motion_reflected` | 2.339e-05 | 2.017e-05 | 5.832e-05 | 6.760e-05 | n/a | n/a | 3.031e-05 | 2.465e-05 | 4.261e-07 |
| `frozen` | 1.057e-06 | 1.768e-07 | 5.235e-06 | 1.366e-05 | n/a | n/a | 8.176e-07 | 1.610e-06 | 2.211e-08 |
| `online` | 9.893e-07 | 1.541e-07 | 5.084e-06 | 1.318e-05 | n/a | n/a | 7.668e-07 | 1.516e-06 | 2.048e-08 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 24 |
| `recovered` | 458 |
| `unrecovered` | 18 |

Eligible: 476; recovered: 458; unrecovered within horizon: 18.
Recovery time (transitions): median 27, p90 35, p95 40, max 50.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9175 |
| 1 | 0.9789 |
| 2 | 0.9681 |
| 3 | 0.9895 |
| 4 | 0.9579 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.125e+06 [-1.675e+06, -7.241e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.059e-08 [6.918e-09, 1.458e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00407 | 0.00454 | 0.00517 |
| update | 0.02398 | 0.02574 | 0.03199 |
| predict+update | 0.02807 | 0.03029 | 0.03635 |

Samples: 2,000; failures: 0; candidate forgetting factor: 0.5.

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

- Source commit: `fc439ccb86affdd53b979eaa50d9ff022237f6c7`; dirty: no
- Tree hash: `d73f80534d17614825f3e5ec3095ad686a236eb7c72908bf8085d6c15b6baae6`
- Dependency lock hash: `6370808d7f23a04fce4f86b136210e1669063d6ef8113509381eec6655730ec3`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `10e68600faf5, adadfab63d0d, 8b168a5096cd, 57f0718d11e8, 2f22182509b4`
- Runtime: 287.4 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0001 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0001
```
