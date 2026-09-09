# AAA benchmark v2 — confirmation-a-20260909-clean

Role: `confirmation_a`. This is an engineering benchmark report, not a product or scientific-approval claim.

## Gate outcomes

| Gate | Status | Observed | Threshold |
|---|---|---:|---|
| straight_learning | PASS | 0.999999999999983 | >= 0.99 relative reduction from zero control |
| straight_prediction | PASS | 1.3738404736659242e-17 | MAE <= 1e-5 and p95 <= 1e-4 in every direction/speed stratum |
| bouncing_prediction | PASS | 0.9999999999999788 | >= 0.20 event-balanced improvement; overall and non-bounce non-regression |
| speed_change_response | PASS | 4.196641193482164e-05 | <= 1.10 * constant motion + 1e-5; first surprise included |
| changed_law_adaptation | PASS | 0.5544973337801926 | >= 0.20 lower cumulative 50-transition error than frozen and persistence |
| unchanged_control | PASS | -4.012346010995316e-18 | increase <= max(10% frozen, 1e-5) |
| recovery | PASS | 1.0 | >= 0.90 by 50 transitions with >=100 eligible events |
| bounce_coverage | PASS | 1681.0 | >= 500 events across >=100 episodes and both walls |
| correctness | PASS | None | all recorded outputs finite and all requested episodes complete |
| reproducibility | PASS | None | stable role/family/replica/episode streams and versioned checkpoint state |
| cpu_usability | PASS | 0.023441999999999998 | predict+update p95 <= 5 ms on the recorded CPU |

Overall required-gate status: **True**.

## Family measurements

### constant_velocity

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0008071350046165316 | 0.0011588042356848595 | None |
| constant_motion | 6.664719388424453e-20 | 0.0 | None |
| zero_control | 0.0008071350046165316 | 0.0011588042356848595 | None |
| adaptive_rls | 1.3738404736659242e-17 | 1.1102230246251565e-16 | None |

### bouncing

Episodes: `500`; replicas: `5`; bounce events: `1681`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0027894885115962538 | 0.003858786040020573 | None |
| constant_motion | 1.6651204947947177e-05 | 0.0 | None |
| zero_control | 0.0027894885115962538 | 0.003858786040020573 | None |
| adaptive_rls | 5.562156462765703e-06 | 1.1102230246251565e-16 | None |

### speed_change

Episodes: `500`; replicas: `5`; bounce events: `1763`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.0029129162950068065 | 0.006231784017926845 | 0.0031063032880990333 |
| constant_motion | 2.3435835426016958e-05 | 0.0 | 5.105090048488472e-05 |
| zero_control | 0.0029129162950068065 | 0.006231784017926845 | 0.0031063032880990333 |
| adaptive_rls | 9.098083165002456e-06 | 1.1102230246251565e-16 | 4.196641193482164e-05 |

### dynamics_change_changed

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.000540876512371954 | 0.001590020683877838 | 0.0005408765123719566 |
| constant_motion | 8.832630164465226e-05 | 0.00026580485154807654 | 8.832630164465219e-05 |
| frozen | 8.50383258054385e-05 | 0.000255264746374545 | 8.503832580543861e-05 |
| online | 3.78848008771915e-05 | 0.00013864220335567664 | 3.788480087719171e-05 |

### dynamics_change_unchanged

Episodes: `500`; replicas: `5`; bounce events: `0`.

| Predictor | MAE | p95 | post-change MAE |
|---|---:|---:|---:|
| persistence | 0.00021945246106471874 | 0.0004958373810840383 | 0.00021945246106472017 |
| constant_motion | 9.567266614426539e-06 | 1.9656589439437377e-05 | 9.567266614426539e-06 |
| frozen | 3.0591085220521564e-17 | 1.1102230246251565e-16 | 3.0591085220521564e-17 |
| online | 2.6578739209526248e-17 | 1.1102230246251565e-16 | 2.6578739209526248e-17 |

## Reproduction

` .venv/bin/python -m aaa.cli benchmark-v2 --role confirmation_a --attempt-id <new-id> --output-root runs `

Raw step-level predictions are compressed under `raw/`; checksums and source-tree identity are recorded in the attempt directory. Confirmation A and B must both be run from the committed selected source without intervening tuning.
