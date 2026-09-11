# AAA benchmark aaa.benchmark.v2.1 — `aaa-v2_1-confirmation-b-0001`

Role: `confirmation_b`. Confirmation batch: `aaa-v2_1-confirmation-b-0001`.
Specification hash: `198b9a4ea4541206…`.

This is an engineering measurement report. It is not an approval decision, and it does not
claim general intelligence, physical understanding, or independent goal formation.

## Required-gate status

**All required gates pass: no**

Unmet required gates: `always_online_stability`.

| Gate | Required | Status | Observed | Description |
|---|---|---|---:|---|
| `stratum_coverage` | yes | **PASS** | 827 | Every required direction x position x speed stratum is present with the declared minimum episode and replica counts; both walls are observed; bounce and eligible change events meet their minimums. |
| `constant_velocity_identification` | yes | **PASS** | 3.307e-11 | The candidate identifies the constant-velocity law: in-domain straight-motion normalized MAE and per-stratum p95 stay inside the declared absolute limits. |
| `learning_progress` | yes | **PASS** | 1 | Frozen checkpoints taken at increasing cumulative update budgets are scored on one fixed development probe bank. |
| `frozen_prediction_accuracy` | yes | **PASS** | 5.405e-06 | Frozen generalization to the unfamiliar bouncing regime stays inside a declared absolute accuracy limit. |
| `bounce_event_accuracy` | yes | **PASS** | 2.918e-11 | Bounce-transition accuracy against an absolute limit, plus non-regression against the like-for-like reflected constant-motion baseline that uses the identical public boundary policy. |
| `speed_change_frozen_robustness` | yes | **PASS** | 4.661e-05 | With updates disabled, the candidate's 50-transition post-event error does not regress against the reflected constant-motion baseline. |
| `changed_law_adaptation` | yes | **PASS** | 0.6624 | In the matched changed-law experiment the updating copy must beat its identical frozen copy and persistence on 50-transition cumulative error by the declared margin, and must not regress against constant motion. |
| `unchanged_control` | yes | **PASS** | -6.790e-08 | Continuing to update in an unchanged world must not degrade the model relative to its frozen twin. |
| `recovery` | yes | **PASS** | 0.9578 | Eligibility is defined by the peak post-event shock; every episode falls into exactly one status and unrecovered events are counted, never dropped. |
| `always_online_stability` | yes | **FAIL** | 3.414e-05 | A single continuously updating instance operating across regimes with no evaluator mode switching must not regress against the reflected constant-motion baseline. |
| `speed_extrapolation_report` | no | **PASS** | 9.363e-11 | Reported out-of-distribution accuracy above the training speed range. |
| `correctness` | yes | **PASS** | 0 | The retained evidence is complete, internally consistent, schema-valid, checksum-valid, and recomputes to the stored summary. |
| `reproducibility` | yes | **PASS** | 0 | An independently initialized duplicate mini experiment, a save/resume equivalence check and a golden seed-mapping fixture are actually executed and compared. |
| `cpu_usability` | yes | **PASS** | 0.03011 | The selected candidate itself is benchmarked for predict, update and combined latency on the recorded CPU. |

## Uncertainty intervals

| Comparison | Estimate [95% interval] | p | replicas | episodes |
|---|---|---:|---:|---:|
| `always_online_stability.margin` | 1.643e-05 [-5.048e-06, 6.268e-05] | n/a | 5 | 500 |
| `bounce.candidate_event_mae` | 2.918e-11 [1.984e-11, 3.949e-11] | n/a | 5 | 500 |
| `bounce.decomposition_no_reflect_vs_raw_constant_motion` | 1.001e-08 [6.802e-09, 1.351e-08] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_persistence` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_raw_constant_motion` | 1 [1, 1] | 0.0002499 | 5 | 500 |
| `bounce.decomposition_vs_reflected_constant_motion` | -9.321e+05 [-1.258e+06, -6.218e+05] | 1 | 5 | 500 |
| `bounce.parity_margin` | -1.000e-06 [-1.000e-06, -1.000e-06] | n/a | 5 | 500 |
| `changed_law.online_vs_constant_motion_margin` | -0.01317 [-0.01434, -0.01213] | n/a | 5 | 500 |
| `changed_law.online_vs_frozen` | 0.6624 [0.6485, 0.6749] | 0.0002499 | 5 | 500 |
| `changed_law.online_vs_persistence` | 0.9461 [0.9433, 0.9486] | 0.0002499 | 5 | 500 |
| `constant_velocity_identification.mean` | 3.307e-11 [2.947e-11, 3.708e-11] | n/a | 5 | 500 |
| `frozen_prediction_accuracy.mean` | 5.405e-06 [4.997e-06, 5.829e-06] | n/a | 5 | 500 |
| `learning_progress.reduction` | 1 [1, 1] | 0.0002499 | 5 | 40 |
| `speed_change_frozen_robustness.margin` | -1.466e-05 [-1.497e-05, -1.438e-05] | n/a | 5 | 500 |
| `speed_extrapolation_report.mean` | 9.363e-11 [8.516e-11, 1.029e-10] | n/a | 5 | 500 |
| `unchanged_control.margin` | -1.018e-05 [-1.022e-05, -1.014e-05] | n/a | 5 | 500 |

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
| `persistence` | 0.002804 | 0.002805 | 0.003895 | 0.003947 | n/a | n/a | n/a | 0.002822 | -0.0001109 |
| `constant_motion` | 2.575e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.813e-19 | -1.669e-20 |
| `constant_motion_reflected` | 2.575e-19 | 0 | 0 | 0 | n/a | n/a | n/a | 2.813e-19 | -1.669e-20 |
| `zero_control` | 0.002804 | 0.002805 | 0.003895 | 0.003947 | n/a | n/a | n/a | 0.002822 | -0.0001109 |
| `legacy_linear_sgd` | 0.004434 | 0.004188 | 0.009848 | 0.01167 | n/a | n/a | n/a | 0.005979 | 0.00162 |
| `candidate_frozen` | 3.307e-11 | 3.285e-11 | 5.478e-11 | 6.360e-11 | n/a | n/a | n/a | 4.015e-11 | -3.287e-12 |
| `candidate_no_reflect` | 3.307e-11 | 3.285e-11 | 5.478e-11 | 6.360e-11 | n/a | n/a | n/a | 4.015e-11 | -3.287e-12 |

### `bouncing`

Episodes: 500; replicas: 5; bounce events: 827; event episodes: 500; no-event episodes excluded from event statistics: 0.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002795 | 0.002767 | 0.003878 | 0.003933 | 0.001374 | 0.003233 | n/a | 0.002804 | -1.282e-05 |
| `constant_motion` | 1.649e-05 | 0 | 0 | 1.388e-17 | 0.00291 | 0.006524 | n/a | 1.683e-05 | 3.200e-07 |
| `constant_motion_reflected` | 5.405e-06 | 0 | 0 | 3.469e-18 | 3.131e-17 | 2.220e-16 | n/a | 5.923e-06 | 2.644e-07 |
| `zero_control` | 0.002795 | 0.002767 | 0.003878 | 0.003933 | 0.001374 | 0.003233 | n/a | 0.002804 | -1.282e-05 |
| `legacy_linear_sgd` | 0.004475 | 0.004601 | 0.009822 | 0.01167 | 0.004724 | 0.00977 | n/a | 0.006049 | 0.001629 |
| `candidate_frozen` | 5.405e-06 | 3.334e-11 | 5.588e-11 | 6.577e-11 | 2.918e-11 | 5.915e-11 | n/a | 5.923e-06 | 2.644e-07 |
| `candidate_no_reflect` | 1.649e-05 | 3.339e-11 | 5.626e-11 | 6.742e-11 | 0.00291 | 0.006524 | n/a | 1.683e-05 | 3.200e-07 |

### `speed_change`

Episodes: 500; replicas: 5; bounce events: 981; event episodes: 492; no-event episodes excluded from event statistics: 8.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.003242 | 0.003121 | 0.006516 | 0.006995 | 0.001855 | 0.00511 | 0.003537 | 0.003323 | -9.444e-06 |
| `constant_motion` | 3.050e-05 | 0 | 0 | 5.551e-17 | 0.003728 | 0.0105 | 5.912e-05 | 3.196e-05 | -4.381e-07 |
| `constant_motion_reflected` | 1.242e-05 | 0 | 0 | 1.388e-17 | 2.287e-17 | 2.220e-16 | 4.661e-05 | 1.320e-05 | -9.708e-08 |
| `zero_control` | 0.003242 | 0.003121 | 0.006516 | 0.006995 | 0.001855 | 0.00511 | 0.003537 | 0.003323 | -9.444e-06 |
| `legacy_linear_sgd` | 0.004768 | 0.004331 | 0.01075 | 0.01295 | 0.004929 | 0.01059 | 0.005229 | 0.005975 | 0.001708 |
| `candidate_frozen` | 1.242e-05 | 3.555e-11 | 8.234e-11 | 1.024e-10 | 4.014e-11 | 8.818e-11 | 4.661e-05 | 1.320e-05 | -9.708e-08 |
| `candidate_no_reflect` | 3.050e-05 | 3.562e-11 | 8.316e-11 | 1.083e-10 | 0.003728 | 0.0105 | 5.912e-05 | 3.196e-05 | -4.381e-07 |
| `candidate_online` | 4.916e-05 | 7.309e-07 | 8.890e-06 | 0.0001068 | 8.460e-07 | 4.871e-06 | 7.168e-05 | 7.848e-05 | -3.598e-06 |

### `speed_extrapolation`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.007884 | 0.007895 | 0.009707 | 0.009906 | n/a | n/a | n/a | 0.008093 | 0.0004414 |
| `constant_motion` | 6.553e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.264e-19 | 6.564e-21 |
| `constant_motion_reflected` | 6.553e-19 | 0 | 0 | 2.776e-17 | n/a | n/a | n/a | 9.264e-19 | 6.564e-21 |
| `zero_control` | 0.007884 | 0.007895 | 0.009707 | 0.009906 | n/a | n/a | n/a | 0.008093 | 0.0004414 |
| `legacy_linear_sgd` | 0.008184 | 0.008997 | 0.01468 | 0.01622 | n/a | n/a | n/a | 0.008426 | 0.002122 |
| `candidate_frozen` | 9.363e-11 | 9.189e-11 | 1.281e-10 | 1.465e-10 | n/a | n/a | n/a | 1.109e-10 | 3.152e-12 |
| `candidate_no_reflect` | 9.363e-11 | 9.189e-11 | 1.281e-10 | 1.465e-10 | n/a | n/a | n/a | 1.109e-10 | 3.152e-12 |

### `always_online`

Episodes: 500; replicas: 5; bounce events: 160; event episodes: 160; no-event episodes excluded from event statistics: 340.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.002859 | 0.002832 | 0.005199 | 0.00626 | 0.001803 | 0.004539 | 0.003259 | 0.002961 | -0.0001463 |
| `constant_motion` | 1.483e-05 | 0 | 0 | 2.776e-17 | 0.003541 | 0.007781 | 5.178e-05 | 1.896e-05 | 1.863e-06 |
| `constant_motion_reflected` | 7.012e-06 | 0 | 0 | 6.939e-18 | 2.776e-17 | 2.220e-16 | 3.890e-05 | 8.801e-06 | 1.321e-07 |
| `zero_control` | 0.002859 | 0.002832 | 0.005199 | 0.00626 | 0.001803 | 0.004539 | 0.003259 | 0.002961 | -0.0001463 |
| `legacy_linear_sgd` | 0.00452 | 0.004594 | 0.01 | 0.01198 | 0.004614 | 0.009813 | 0.005143 | 0.006401 | 0.001409 |
| `candidate_frozen` | 7.012e-06 | 3.382e-11 | 6.264e-11 | 8.913e-11 | 3.964e-11 | 7.884e-11 | 3.890e-05 | 8.801e-06 | 1.321e-07 |
| `candidate_no_reflect` | 1.483e-05 | 3.385e-11 | 6.336e-11 | 9.280e-11 | 0.003541 | 0.007781 | 5.178e-05 | 1.896e-05 | 1.863e-06 |
| `candidate_online` | 3.414e-05 | 7.134e-07 | 1.563e-05 | 0.0001373 | 1.508e-06 | 6.220e-06 | 5.919e-05 | 8.784e-05 | -1.208e-06 |

### `changed_law:changed`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.001348 | 0.0006651 | 0.005033 | 0.007858 | n/a | n/a | 0.002048 | 0.001444 | 1.976e-05 |
| `constant_motion` | 0.0002218 | 0.0001026 | 0.0008516 | 0.001479 | n/a | n/a | 0.0003397 | 0.0002426 | -1.016e-07 |
| `constant_motion_reflected` | 0.0002218 | 0.0001026 | 0.0008516 | 0.001479 | n/a | n/a | 0.0003397 | 0.0002426 | -1.016e-07 |
| `frozen` | 0.0002134 | 9.795e-05 | 0.000823 | 0.001436 | n/a | n/a | 0.0003271 | 0.0002335 | -1.430e-07 |
| `online` | 5.621e-05 | 3.871e-07 | 0.0003157 | 0.001074 | n/a | n/a | 0.0001104 | 5.874e-05 | 1.305e-06 |

### `changed_law:unchanged`

Episodes: 500; replicas: 5; bounce events: 0; event episodes: 0; no-event episodes excluded from event statistics: 500.

| Predictor | MAE | median | p95 | p99 | event MAE | event p95 | post-change MAE | worst replica | signed bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `persistence` | 0.0007835 | 0.0007156 | 0.00175 | 0.001948 | n/a | n/a | 0.000646 | 0.0008027 | 3.321e-05 |
| `constant_motion` | 2.392e-05 | 2.038e-05 | 5.972e-05 | 6.859e-05 | n/a | n/a | 3.106e-05 | 2.452e-05 | 1.797e-07 |
| `constant_motion_reflected` | 2.392e-05 | 2.038e-05 | 5.972e-05 | 6.859e-05 | n/a | n/a | 3.106e-05 | 2.452e-05 | 1.797e-07 |
| `frozen` | 1.088e-06 | 1.632e-07 | 5.297e-06 | 1.638e-05 | n/a | n/a | 8.074e-07 | 1.286e-06 | -2.414e-07 |
| `online` | 1.020e-06 | 1.440e-07 | 5.265e-06 | 1.603e-05 | n/a | n/a | 7.605e-07 | 1.205e-06 | -2.236e-07 |

## Recovery

| Status | Episodes |
|---|---:|
| `no_measured_shock` | 26 |
| `recovered` | 454 |
| `unrecovered` | 20 |

Eligible: 474; recovered: 454; unrecovered within horizon: 20.
Recovery time (transitions): median 28, p90 34, p95 39, max 50.

| Replica | Recovery rate |
|---|---:|
| 0 | 0.9588 |
| 1 | 0.9468 |
| 2 | 0.9565 |
| 3 | 0.9574 |
| 4 | 0.9691 |

## Where any bounce advantage comes from

| Comparison | Relative improvement [95% interval] |
|---|---|
| `vs_raw_constant_motion` | 1 [1, 1] |
| `vs_reflected_constant_motion` | -9.321e+05 [-1.258e+06, -6.218e+05] |
| `vs_persistence` | 1 [1, 1] |
| `no_reflect_vs_raw_constant_motion` | 1.001e-08 [6.802e-09, 1.351e-08] |

Reflection is programmed public knowledge of the observation format. The reflected
constant-motion baseline uses the identical policy, so the difference between the two
comparisons above is the part of any apparent advantage that is the boundary transform
rather than a learned parameter.

## Latency (selected candidate, CPU)

| Operation | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| predict | 0.00406 | 0.00453 | 0.00519 |
| update | 0.02405 | 0.02562 | 0.03288 |
| predict+update | 0.02814 | 0.03011 | 0.03783 |

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

- Source commit: `a1d8f0371e26e6d637117c412dca0ff8a3573895`; dirty: no
- Tree hash: `5baf65671dab4af89b821957563abef7f56b8e9e386100c0efd92995bba7be21`
- Dependency lock hash: `6370808d7f23a04fce4f86b136210e1669063d6ef8113509381eec6655730ec3`
- Checkpoint relationship: `shared_frozen_checkpoints`
- Checkpoint hashes: `10e68600faf5, adadfab63d0d, 8b168a5096cd, 57f0718d11e8, 2f22182509b4`
- Runtime: 287.3 s

## Reproduction

```bash
python -m aaa.cli benchmark --role confirmation_b --batch-id aaa-v2_1-confirmation-b-0001 --output-root runs
```

Recompute every metric and gate from the retained raw evidence, without retraining or
re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/aaa-v2_1-confirmation-b-0001
```
