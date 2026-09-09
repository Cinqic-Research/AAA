# AAA benchmark v2 — confirmation-b-20260909-clean

Role: `confirmation_b`. This is an engineering benchmark report, not a product or scientific-approval claim.

## Gate outcomes

| Gate | Status | Observed | Threshold |
|---|---|---:|---|
| straight_learning | PASS | 0.9999999999999819 | >= 0.99 relative reduction from zero control |
| straight_prediction | PASS | 1.4249285700683213e-17 | MAE <= 1e-5 and p95 <= 1e-4 in every direction/speed stratum |
| bouncing_prediction | PASS | 0.9999999999999781 | >= 0.20 event-balanced improvement; overall and non-bounce non-regression |
| speed_change_response | PASS | 4.286348608451759e-05 | <= 1.10 * constant motion + 1e-5; first surprise included |
| changed_law_adaptation | PASS | 0.5510346291758392 | >= 0.20 lower cumulative 50-transition error than frozen and persistence |
| unchanged_control | PASS | -4.527489494421393e-18 | increase <= max(10% frozen, 1e-5) |
| recovery | PASS | 1.0 | >= 0.90 by 50 transitions with >=100 eligible events |
| bounce_coverage | PASS | 1679.0 | >= 500 events across >=100 episodes and both walls |
| correctness | PASS | None | all recorded outputs finite and all requested episodes complete |
| reproducibility | PASS | None | stable role/family/replica/episode streams and versioned checkpoint state |
| cpu_usability | PASS | 0.022004499999999996 | predict+update p95 <= 5 ms on the recorded CPU |

Overall required-gate status: **True**.

## Family measurements

### constant_velocity

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0007850835168991554 | 0.0011479140176627128 | None |
| constant_motion | 6.463805087534484e-20 | 0.0 | None |
| zero_control | 0.0007850835168991554 | 0.0011479140176627128 | None |
| adaptive_rls | 1.4249285700683213e-17 | 1.1102230246251565e-16 | None |

### bouncing

Episodes: `500`; replicas: `5`; bounce events: `1679`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0027878531197428583 | 0.0038667617567568757 | None |
| constant_motion | 1.6571705160135597e-05 | 0.0 | None |
| zero_control | 0.0027878531197428583 | 0.0038667617567568757 | None |
| adaptive_rls | 5.6031826070283105e-06 | 1.1102230246251565e-16 | None |

### speed_change

Episodes: `500`; replicas: `5`; bounce events: `1768`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0029408492052287833 | 0.006387642185408795 | 0.003103097309177344 |
| constant_motion | 2.382036914343285e-05 | 0.0 | 5.417373457760883e-05 |
| zero_control | 0.0029408492052287833 | 0.006387642185408795 | 0.003103097309177344 |
| adaptive_rls | 9.049809271571318e-06 | 1.1102230246251565e-16 | 4.286348608451759e-05 |

### dynamics_change_changed

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0005188036494836596 | 0.0015165044889663443 | 0.0005188036494836634 |
| constant_motion | 8.498513595447225e-05 | 0.000252635753147018 | 8.498513595447175e-05 |
| frozen | 8.181115425798758e-05 | 0.00024292311605610681 | 8.181115425798778e-05 |
| online | 3.6730375208990006e-05 | 0.00012928925981151571 | 3.673037520898997e-05 |

### dynamics_change_unchanged

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.00022826199162987227 | 0.0005186949395515672 | 0.00022826199162987403 |
| constant_motion | 9.665288348155979e-06 | 1.9720652943300276e-05 | 9.665288348155979e-06 |
| frozen | 3.1141755840735644e-17 | 1.1102230246251565e-16 | 3.1141755840735644e-17 |
| online | 2.661426634631425e-17 | 1.1102230246251565e-16 | 2.661426634631425e-17 |

## Reproduction

` .venv/bin/python -m aaa.cli benchmark-v2 --role confirmation_a --attempt-id <new-id> --output-root runs `

Raw step-level predictions are compressed under `raw/`; checksums and source-tree identity are recorded in the attempt directory. Confirmation A and B must both be run from the committed selected source without intervening tuning.
