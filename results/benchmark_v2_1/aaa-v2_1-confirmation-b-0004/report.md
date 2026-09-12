# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-b-0004`

Role: `confirmation_b`. Confirmation batch: `aaa-v2_1-confirmation-b-0004`.
Specification hash: `f8e1090bf5b1aeb0…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: yes**

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 830 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.895e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.467e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 3.396e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.063e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.7542 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -5.341e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9658 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **PASS** | 6.841e-06 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 1.084e-10 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03319 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | -1.002e-05 [-1.016e-05, -9.872e-06] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 3.396e-11 [2.331e-11, 4.499e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.200e-08 [8.092e-09, 1.609e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.237e+06 [-1.761e+06, -8.128e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01501 [-0.01657, -0.0135] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.7542 [0.7434, 0.765] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9599 [0.9579, 0.9617] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.895e-11 [3.464e-11, 4.339e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.467e-06 [5.020e-06, 5.915e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.406e-05 [-1.432e-05, -1.382e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 1.084e-10 [9.765e-11, 1.202e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.014e-05 [-1.018e-05, -1.010e-05] | n/a | 5 | 500 |

### Multiple-comparison treatment

Method `holm_bonferroni`, family size 7 (primary comparisons plus previously spent confirmation attempts), alpha 0.05.

| Comparison | p | adjusted p | rejected |
|---|---:|---:|---|
| `changed_law.online_vs_frozen` | 0.0002499 | 0.00175 | yes |
| `changed_law.online_vs_persistence` | 0.0002499 | 0.00175 | yes |
| `learning_progress.reduction` | 0.0002499 | 0.00175 | yes |

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
| `persistence` | 0.002797 | 0.002803 | 0.00388 | 0.003951 | n/a | n/a | n/a | 0.002816 | -0.0001058 |
| `constant_motion` | 2.525e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.207e-19 | 3.873e-20 |
| `constant_motion_reflected` | 2.525e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.207e-19 | 3.873e-20 |
| `zero_control` | 0.002797 | 0.002803 | 0.00388 | 0.003951 | n/a | n/a | n/a | 0.002816 | -0.0001058 |
| `legacy_linear_sgd` | 0.004434 | 0.004204 | 0.009736 | 0.01149 | n/a | n/a | n/a | 0.005968 | 0.001624 |
| `candidate_frozen` | 3.895e-11 | 3.924e-11 | 6.445e-11 | 7.376e-11 | n/a | n/a | n/a | 4.597e-11 | -4.553e-12 |
| `candidate_no_reflect` | 3.895e-11 | 3.924e-11 | 6.445e-11 | 7.376e-11 | n/a | n/a | n/a | 4.597e-11 | -4.553e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 830; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002795 | 0.002769 | 0.003863 | 0.003936 | 0.00142 | 0.003298 | n/a | 0.002822 | -3.219e-07 |
| `constant_motion` | 1.654e-05 | 0 | 0 | 1.388e-17 | 0.002824 | 0.006317 | n/a | 1.682e-05 | 3.500e-07 |
| `constant_motion_reflected` | 5.467e-06 | 0 | 0 | 3.469e-18 | 2.746e-17 | 2.220e-16 | n/a | 5.726e-06 | 1.045e-07 |
| `zero_control` | 0.002795 | 0.002769 | 0.003863 | 0.003936 | 0.00142 | 0.003298 | n/a | 0.002822 | -3.219e-07 |
| `legacy_linear_sgd` | 0.00449 | 0.004602 | 0.00981 | 0.01166 | 0.004861 | 0.01011 | n/a | 0.006024 | 0.001662 |
| `candidate_frozen` | 5.467e-06 | 3.928e-11 | 6.612e-11 | 7.455e-11 | 3.396e-11 | 6.758e-11 | n/a | 5.726e-06 | 1.045e-07 |
| `candidate_no_reflect` | 1.654e-05 | 3.935e-11 | 6.649e-11 | 7.570e-11 | 0.002824 | 0.006317 | n/a | 1.682e-05 | 3.500e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 887; event episodes: 483; no-event episodes excluded from event statistics: 17.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002939 | 0.002771 | 0.006302 | 0.006941 | 0.001704 | 0.004899 | 0.003058 | 0.003125 | 2.336e-05 |
| `constant_motion` | 2.559e-05 | 0 | 0 | 2.776e-17 | 0.003475 | 0.01033 | 4.519e-05 | 2.869e-05 | -1.466e-07 |
| `constant_motion_reflected` | 1.074e-05 | 0 | 0 | 1.388e-17 | 2.774e-17 | 2.220e-16 | 4.063e-05 | 1.179e-05 | -1.289e-07 |
| `zero_control` | 0.002939 | 0.002771 | 0.006302 | 0.006941 | 0.001704 | 0.004899 | 0.003058 | 0.003125 | 2.336e-05 |
| `legacy_linear_sgd` | 0.004646 | 0.004331 | 0.01058 | 0.01284 | 0.004838 | 0.01098 | 0.004746 | 0.005939 | 0.001754 |
| `candidate_frozen` | 1.074e-05 | 3.862e-11 | 9.227e-11 | 1.152e-10 | 4.205e-11 | 1.000e-10 | 4.063e-05 | 1.179e-05 | -1.289e-07 |
| `candidate_no_reflect` | 2.559e-05 | 3.869e-11 | 9.321e-11 | 1.193e-10 | 0.003475 | 0.01033 | 4.519e-05 | 2.869e-05 | -1.466e-07 |
| `candidate_online` | 1.162e-05 | 2.608e-08 | 3.914e-06 | 6.745e-06 | 8.662e-07 | 4.943e-06 | 4.377e-05 | 1.257e-05 | -1.284e-07 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007772 | 0.007572 | 0.009759 | 0.009924 | n/a | n/a | n/a | 0.007917 | 0.0005674 |
| `constant_motion` | 6.756e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 8.266e-19 | -1.364e-20 |
| `constant_motion_reflected` | 6.756e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 8.266e-19 | -1.364e-20 |
| `zero_control` | 0.007772 | 0.007572 | 0.009759 | 0.009924 | n/a | n/a | n/a | 0.007917 | 0.0005674 |
| `legacy_linear_sgd` | 0.008034 | 0.008563 | 0.01446 | 0.01585 | n/a | n/a | n/a | 0.008528 | 0.002306 |
| `candidate_frozen` | 1.084e-10 | 1.071e-10 | 1.487e-10 | 1.712e-10 | n/a | n/a | n/a | 1.300e-10 | 4.758e-12 |
| `candidate_no_reflect` | 1.084e-10 | 1.071e-10 | 1.487e-10 | 1.712e-10 | n/a | n/a | n/a | 1.300e-10 | 4.758e-12 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 164; event episodes: 164; no-event episodes excluded from event statistics: 336.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002847 | 0.002783 | 0.005186 | 0.006304 | 0.002026 | 0.004697 | 0.003202 | 0.003101 | 3.220e-05 |
| `constant_motion` | 1.457e-05 | 0 | 0 | 2.776e-17 | 0.0035 | 0.009179 | 4.771e-05 | 1.927e-05 | -5.940e-07 |
| `constant_motion_reflected` | 6.240e-06 | 0 | 0 | 6.939e-18 | 2.031e-17 | 2.220e-16 | 3.529e-05 | 7.555e-06 | -3.491e-07 |
| `zero_control` | 0.002847 | 0.002783 | 0.005186 | 0.006304 | 0.002026 | 0.004697 | 0.003202 | 0.003101 | 3.220e-05 |
| `legacy_linear_sgd` | 0.004609 | 0.004723 | 0.01024 | 0.01229 | 0.00493 | 0.01125 | 0.005321 | 0.006512 | 0.001694 |
| `candidate_frozen` | 6.240e-06 | 3.793e-11 | 7.531e-11 | 1.056e-10 | 4.687e-11 | 9.993e-11 | 3.529e-05 | 7.555e-06 | -3.491e-07 |
| `candidate_no_reflect` | 1.457e-05 | 3.797e-11 | 7.655e-11 | 1.125e-10 | 0.0035 | 0.009179 | 4.771e-05 | 1.927e-05 | -5.940e-07 |
| `candidate_online` | 6.841e-06 | 7.623e-08 | 2.575e-06 | 1.652e-05 | 2.604e-07 | 6.016e-07 | 3.838e-05 | 8.086e-06 | -4.388e-07 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001359 | 0.0006578 | 0.00514 | 0.008175 | n/a | n/a | 0.002052 | 0.001516 | -3.596e-06 |
| `constant_motion` | 0.000228 | 0.0001012 | 0.0008863 | 0.00164 | n/a | n/a | 0.0003475 | 0.0002587 | 2.120e-07 |
| `constant_motion_reflected` | 0.000228 | 0.0001012 | 0.0008863 | 0.00164 | n/a | n/a | 0.0003475 | 0.0002587 | 2.120e-07 |
| `frozen` | 0.0002194 | 9.651e-05 | 0.0008583 | 0.001593 | n/a | n/a | 0.000335 | 0.0002494 | 2.183e-07 |
| `online` | 4.214e-05 | 1.525e-07 | 0.0001902 | 0.0009897 | n/a | n/a | 8.233e-05 | 4.477e-05 | 2.511e-08 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007594 | 0.0006683 | 0.001733 | 0.00196 | n/a | n/a | 0.0006268 | 0.0007824 | -2.225e-06 |
| `constant_motion` | 2.318e-05 | 1.919e-05 | 5.871e-05 | 6.854e-05 | n/a | n/a | 3.008e-05 | 2.380e-05 | -6.488e-07 |
| `constant_motion_reflected` | 2.318e-05 | 1.919e-05 | 5.871e-05 | 6.854e-05 | n/a | n/a | 3.008e-05 | 2.380e-05 | -6.488e-07 |
| `frozen` | 8.292e-07 | 1.006e-07 | 3.900e-06 | 1.415e-05 | n/a | n/a | 6.264e-07 | 1.172e-06 | 3.858e-08 |
| `online` | 7.758e-07 | 8.856e-08 | 3.608e-06 | 1.372e-05 | n/a | n/a | 5.916e-07 | 1.106e-06 | 3.375e-08 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 32 |
| `recovered` | 452 |
| `unrecovered` | 16 |

Eligible: 468; recovered: 452; unrecovered within horizon: 16.
Recovery time (transitions): median 19, p90 26, p95 32, max 47.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9894 |
| 1 | 0.9574 |
| 2 | 0.9574 |
| 3 | 0.9574 |
| 4 | 0.9674 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.237e+06 [-1.761e+06, -8.128e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.200e-08 [8.092e-09, 1.609e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.0045 | 0.00528 | 0.00611 |
| update | 0.02637 | 0.02823 | 0.03519 |
| predict+update | 0.03094 | 0.03319 | 0.04147 |

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

- Source commit: `dc4a6e1988c00dde3a3b27622c0af9982c395f1d`; dirty: no
- Tree hash: `4b57e9c988e3d47f158840997bbb79a9b0fe34b10bd837fe1654dfb310779685`
- Dependency lock hash: `2aa52a7deefd05403b9fb6441bef42e13e47a4384baaf780f3b4b879bfcf3bc7`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `e635f04f9441, 0dede9edd8af, 9540f727907c, 745dbb6d3f29, d31088732cd1`
- Runtime: 666.6 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_b --batch-id aaa-v2_1-confirmation-b-0004 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-b-0004
```
