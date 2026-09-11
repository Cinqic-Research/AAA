# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-a-0002`

Role: `confirmation_a`. Confirmation batch: `aaa-v2_1-confirmation-a-0002`.
Specification hash: `f8e1090bf5b1aeb0…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: yes**

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 820 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.904e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.347e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 3.396e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.450e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.7635 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -4.906e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9679 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **PASS** | 6.712e-06 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 1.113e-10 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03093 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | -9.967e-06 [-1.014e-05, -9.800e-06] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 3.396e-11 [2.343e-11, 4.494e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.198e-08 [8.181e-09, 1.565e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -1.233e+06 [-1.896e+06, -8.283e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01581 [-0.01771, -0.01405] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.7635 [0.7515, 0.7735] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9617 [0.9593, 0.9643] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.904e-11 [3.484e-11, 4.346e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.347e-06 [4.958e-06, 5.741e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.445e-05 [-1.485e-05, -1.407e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 1.113e-10 [9.871e-11, 1.229e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.012e-05 [-1.017e-05, -1.009e-05] | n/a | 5 | 500 |

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
| `persistence` | 0.002797 | 0.002799 | 0.003855 | 0.003941 | n/a | n/a | n/a | 0.00281 | -0.0001104 |
| `constant_motion` | 2.244e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.151e-19 | 1.397e-20 |
| `constant_motion_reflected` | 2.244e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 3.151e-19 | 1.397e-20 |
| `zero_control` | 0.002797 | 0.002799 | 0.003855 | 0.003941 | n/a | n/a | n/a | 0.00281 | -0.0001104 |
| `legacy_linear_sgd` | 0.004406 | 0.004219 | 0.00966 | 0.01159 | n/a | n/a | n/a | 0.005973 | 0.001629 |
| `candidate_frozen` | 3.904e-11 | 3.854e-11 | 6.469e-11 | 7.407e-11 | n/a | n/a | n/a | 4.640e-11 | -4.607e-12 |
| `candidate_no_reflect` | 3.904e-11 | 3.854e-11 | 6.469e-11 | 7.407e-11 | n/a | n/a | n/a | 4.640e-11 | -4.607e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 820; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002794 | 0.002769 | 0.00385 | 0.003946 | 0.001436 | 0.00314 | n/a | 0.002822 | 1.293e-06 |
| `constant_motion` | 1.628e-05 | 0 | 0 | 1.388e-17 | 0.002828 | 0.006204 | n/a | 1.655e-05 | 1.372e-07 |
| `constant_motion_reflected` | 5.347e-06 | 0 | 0 | 3.469e-18 | 2.753e-17 | 2.220e-16 | n/a | 5.611e-06 | 7.101e-08 |
| `zero_control` | 0.002794 | 0.002769 | 0.00385 | 0.003946 | 0.001436 | 0.00314 | n/a | 0.002822 | 1.293e-06 |
| `legacy_linear_sgd` | 0.004481 | 0.004596 | 0.009818 | 0.01163 | 0.004806 | 0.01026 | n/a | 0.005987 | 0.001664 |
| `candidate_frozen` | 5.347e-06 | 3.922e-11 | 6.548e-11 | 7.426e-11 | 3.396e-11 | 6.685e-11 | n/a | 5.611e-06 | 7.100e-08 |
| `candidate_no_reflect` | 1.628e-05 | 3.928e-11 | 6.588e-11 | 7.545e-11 | 0.002828 | 0.006204 | n/a | 1.655e-05 | 1.372e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 934; event episodes: 491; no-event episodes excluded from event statistics: 9.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003085 | 0.002955 | 0.006437 | 0.006968 | 0.001808 | 0.00516 | 0.003277 | 0.003349 | 2.560e-05 |
| `constant_motion` | 2.797e-05 | 0 | 0 | 5.551e-17 | 0.003597 | 0.01034 | 5.548e-05 | 3.198e-05 | 2.437e-07 |
| `constant_motion_reflected` | 1.146e-05 | 0 | 0 | 1.388e-17 | 2.744e-17 | 2.220e-16 | 4.450e-05 | 1.275e-05 | -1.157e-07 |
| `zero_control` | 0.003085 | 0.002955 | 0.006437 | 0.006968 | 0.001808 | 0.00516 | 0.003277 | 0.003349 | 2.560e-05 |
| `legacy_linear_sgd` | 0.004825 | 0.004609 | 0.01061 | 0.01276 | 0.004805 | 0.01068 | 0.005089 | 0.006139 | 0.001743 |
| `candidate_frozen` | 1.146e-05 | 4.076e-11 | 9.551e-11 | 1.156e-10 | 4.462e-11 | 1.028e-10 | 4.450e-05 | 1.275e-05 | -1.157e-07 |
| `candidate_no_reflect` | 2.797e-05 | 4.084e-11 | 9.649e-11 | 1.230e-10 | 0.003597 | 0.01034 | 5.548e-05 | 3.198e-05 | 2.437e-07 |
| `candidate_online` | 1.232e-05 | 2.142e-08 | 3.981e-06 | 6.682e-06 | 8.557e-07 | 4.783e-06 | 4.768e-05 | 1.362e-05 | -8.779e-08 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007888 | 0.007823 | 0.009785 | 0.009934 | n/a | n/a | n/a | 0.00803 | -0.0008092 |
| `constant_motion` | 7.490e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.171e-19 | 7.877e-21 |
| `constant_motion_reflected` | 7.490e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.171e-19 | 7.877e-21 |
| `zero_control` | 0.007888 | 0.007823 | 0.009785 | 0.009934 | n/a | n/a | n/a | 0.00803 | -0.0008092 |
| `legacy_linear_sgd` | 0.008043 | 0.007774 | 0.01477 | 0.01643 | n/a | n/a | n/a | 0.009019 | 0.0008336 |
| `candidate_frozen` | 1.113e-10 | 1.112e-10 | 1.499e-10 | 1.644e-10 | n/a | n/a | n/a | 1.312e-10 | -1.427e-11 |
| `candidate_no_reflect` | 1.113e-10 | 1.112e-10 | 1.499e-10 | 1.644e-10 | n/a | n/a | n/a | 1.312e-10 | -1.427e-11 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 144; event episodes: 144; no-event episodes excluded from event statistics: 356.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002834 | 0.002766 | 0.004795 | 0.006427 | 0.001983 | 0.00418 | 0.003169 | 0.002952 | -7.343e-05 |
| `constant_motion` | 1.342e-05 | 0 | 0 | 1.388e-17 | 0.00359 | 0.007792 | 4.787e-05 | 1.542e-05 | 4.279e-07 |
| `constant_motion_reflected` | 6.072e-06 | 0 | 0 | 6.939e-18 | 2.467e-17 | 2.220e-16 | 3.464e-05 | 7.225e-06 | -3.602e-07 |
| `zero_control` | 0.002834 | 0.002766 | 0.004795 | 0.006427 | 0.001983 | 0.00418 | 0.003169 | 0.002952 | -7.343e-05 |
| `legacy_linear_sgd` | 0.004564 | 0.004545 | 0.01012 | 0.01225 | 0.00525 | 0.01086 | 0.005248 | 0.006221 | 0.001655 |
| `candidate_frozen` | 6.072e-06 | 3.949e-11 | 7.410e-11 | 9.990e-11 | 4.728e-11 | 9.536e-11 | 3.464e-05 | 7.225e-06 | -3.602e-07 |
| `candidate_no_reflect` | 1.342e-05 | 3.953e-11 | 7.465e-11 | 1.040e-10 | 0.00359 | 0.007792 | 4.787e-05 | 1.543e-05 | 4.278e-07 |
| `candidate_online` | 6.712e-06 | 7.527e-08 | 2.489e-06 | 1.807e-05 | 3.343e-07 | 1.022e-06 | 3.814e-05 | 7.696e-06 | -4.573e-07 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001443 | 0.0007387 | 0.005258 | 0.008313 | n/a | n/a | 0.002154 | 0.001608 | -4.476e-06 |
| `constant_motion` | 0.0002404 | 0.0001119 | 0.0009197 | 0.001672 | n/a | n/a | 0.0003623 | 0.0002833 | 6.879e-07 |
| `constant_motion_reflected` | 0.0002404 | 0.0001119 | 0.0009197 | 0.001672 | n/a | n/a | 0.0003623 | 0.0002833 | 6.879e-07 |
| `frozen` | 0.0002313 | 0.0001074 | 0.0008896 | 0.001626 | n/a | n/a | 0.000349 | 0.0002737 | 6.337e-07 |
| `online` | 4.203e-05 | 1.338e-07 | 0.0001977 | 0.0009867 | n/a | n/a | 8.256e-05 | 4.905e-05 | -1.832e-07 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007755 | 0.0007083 | 0.001713 | 0.001948 | n/a | n/a | 0.0006537 | 0.0008086 | -7.260e-06 |
| `constant_motion` | 2.357e-05 | 2.015e-05 | 5.789e-05 | 6.821e-05 | n/a | n/a | 3.016e-05 | 2.465e-05 | 2.006e-07 |
| `constant_motion_reflected` | 2.357e-05 | 2.015e-05 | 5.789e-05 | 6.821e-05 | n/a | n/a | 3.016e-05 | 2.465e-05 | 2.006e-07 |
| `frozen` | 7.407e-07 | 9.146e-08 | 3.339e-06 | 1.385e-05 | n/a | n/a | 5.450e-07 | 1.007e-06 | -1.358e-08 |
| `online` | 6.916e-07 | 8.125e-08 | 3.100e-06 | 1.268e-05 | n/a | n/a | 5.156e-07 | 9.493e-07 | -1.336e-08 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 33 |
| `recovered` | 452 |
| `unrecovered` | 15 |

Eligible: 467; recovered: 452; unrecovered within horizon: 15.
Recovery time (transitions): median 19, p90 25, p95 29.45, max 47.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9783 |
| 1 | 0.9785 |
| 2 | 0.9588 |
| 3 | 0.9785 |
| 4 | 0.9457 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -1.233e+06 [-1.896e+06, -8.283e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.198e-08 [8.181e-09, 1.565e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00416 | 0.00452 | 0.00532 |
| update | 0.02485 | 0.02621 | 0.03262 |
| predict+update | 0.02902 | 0.03093 | 0.0373 |

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

- Source commit: `f1b94cc4084e7e7fcdb73a38ef9c6ab8093ced85`; dirty: no
- Tree hash: `95778623903d46f74f7ea7df1e8f3cd4ab5a1cdf7ed51a6983c22f40ee362e54`
- Dependency lock hash: `6370808d7f23a04fce4f86b136210e1669063d6ef8113509381eec6655730ec3`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `e635f04f9441, 0dede9edd8af, 9540f727907c, 745dbb6d3f29, d31088732cd1`
- Runtime: 288.1 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0002 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-a-0002
```
