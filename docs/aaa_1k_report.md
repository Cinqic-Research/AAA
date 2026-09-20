# AAA-1K: a 994-parameter recurrent predictive core

AAA-1K is a research seed, not Juniper and not an agent. It is one primitive: a persistent gated recurrent predictor that learns online, keeps a hidden state, and estimates how wrong it expects to be. Everything below describes what was measured on a moving dot.

## Identity

| Item | Value |
|---|---|
| phase | `aaa.1k.v1` |
| scientific fingerprint | `677dfe4941b74a7793cd0532b4adcd6a1fc99512edf143adb20741f39d2deace` |
| files covered | 23 |
| trainable parameters | 994 |
| selected configuration | lr=0.03, T=4, lambda=0.25 |
| replicas per family | 32 |
| evaluation streams | 192 |

## Parameter and state accounting

| Scalar category | Count |
|---|---|
| hidden state scalars | 16 |
| optimizer state scalars | 0 |
| tbptt buffer scalars | 0 |
| total adaptive state scalars | 1010 |
| trainable parameters | 994 |

The optimizer is plain SGD and holds no state, so the adaptive footprint is the parameters, the 16-value hidden state, and the truncation buffer. There is no hidden second model.

## Development selection

The unclipped divergence probe put the boundary at a learning rate of 0.3. The declared stability margin therefore restricted the search to [0.001, 0.003, 0.01, 0.03], which **eliminated the best-performing configurations on development data**. That cost is real and is reported rather than quietly avoided: the rule was declared before the numbers existed, and a rule that never binds is not a rule.

Best eliminated configuration: `lr=0.1;T=32;lambda=0.25` at 1.138e-03 development error. Selected: `lr=0.03;T=4;lambda=0.25` at 1.434e-03, a 26% development penalty paid for the margin.

Auxiliary-weight decision: the predeclared default was retained: no alternative beat it by more than 2% on development data.

## Capability vector

Reported as separate dimensions. There is deliberately no combined score: one number is the most efficient way to hide a failure inside a success.

| Question | Verdict | Effect (normalized error) [95% CI] |
|---|---|---|
| Q1 online learning | POSITIVE | +5.19e-04 [+4.47e-04, +5.91e-04] (176/192 streams) |
| Q3 hidden state | POSITIVE | +4.25e-04 [+3.77e-04, +4.76e-04] (64/64 streams) |
| Q4 gating | NEGATIVE | -1.89e-04 [-2.37e-04, -1.42e-04] (44/192 streams) |
| Q2 online vs frozen | POSITIVE | +9.76e-04 [+7.80e-04, +1.20e-03] (96/96 streams) |

### Q1 -- can it learn online?

Overall: **POSITIVE**, +5.19e-04 [+4.47e-04, +5.91e-04] (176/192 streams).

| Family | Verdict | First quarter minus last quarter |
|---|---|---|
| aba_v1 | POSITIVE | +3.99e-04 [+3.31e-04, +4.62e-04] (32/32 streams) |
| coarse_speed_v1 | POSITIVE | +3.04e-04 [+2.34e-04, +3.70e-04] (29/32 streams) |
| motion_compat | POSITIVE | +7.39e-04 [+6.33e-04, +8.48e-04] (96/96 streams) |
| occlusion_v1 | POSITIVE | +1.90e-04 [+1.67e-05, +3.72e-04] (19/32 streams) |

### Q2 -- does continued learning help after a change?

Overall: **POSITIVE**, +9.76e-04 [+7.80e-04, +1.20e-03] (96/96 streams).

Every branch started from a clone whose complete model-state hash matched its twin: `clone_hashes_matched = True`. The frozen arm kept running its recurrence and its previous-error input; only its weights stopped moving.

| Family | Verdict | Frozen minus online |
|---|---|---|
| aba_v1 | POSITIVE | +1.53e-03 [+1.05e-03, +2.09e-03] (32/32 streams) |
| motion_compat | POSITIVE | +6.97e-04 [+5.83e-04, +8.23e-04] (64/64 streams) |

### Q3 -- is persistent recurrent state worth anything?

| Comparison | Verdict | Control minus AAA-1K |
|---|---|---|
| vs state-reset ablation, memory families | POSITIVE | +6.70e-04 [+6.14e-04, +7.23e-04] (64/64 streams) |
| vs stateless MLP, memory families | POSITIVE | +4.25e-04 [+3.77e-04, +4.76e-04] (64/64 streams) |
| vs state-reset, all families | POSITIVE | +4.80e-04 [+4.45e-04, +5.14e-04] (192/192 streams) |
| vs stateless MLP, all families | POSITIVE | +1.21e-04 [+8.63e-05, +1.58e-04] (111/192 streams) |

Restricted to steps whose target was never shown to the agent:

| Comparison | Verdict | Control minus AAA-1K |
|---|---|---|
| vs state-reset | POSITIVE | +9.66e-04 [+7.53e-04, +1.19e-03] (31/32 streams) |
| vs stateless MLP | POSITIVE | +1.34e-03 [+1.11e-03, +1.57e-03] (32/32 streams) |

### Q4 -- do the gates earn their parameters?

the ungated control has 954 parameters against 994, so this compares mechanisms at close to matched capacity, not a large model against a small one.

| Scope | Verdict | Ungated RNN minus AAA-1K |
|---|---|---|
| all families | NEGATIVE | -1.89e-04 [-2.37e-04, -1.42e-04] (44/192 streams) |
| memory families | NEGATIVE | -3.75e-04 [-5.02e-04, -2.49e-04] (31/64 streams) |

### Q5 -- what survives A, then B, then A again?

A2 tail minus A1 tail; positive means the model came back worse.

| Arm | A1 tail | B tail | A2 tail | A2 - A1 |
|---|---|---|---|---|
| aaa1k_frozen_recurrent | 9.357e-05 | 3.609e-04 | 3.734e-05 | -5.623e-05 |
| aaa1k_gru | 9.406e-05 | 2.870e-04 | 3.500e-05 | -5.907e-05 |
| aaa1k_no_error_input | 1.289e-04 | 6.823e-04 | 4.484e-05 | -8.408e-05 |
| aaa1k_state_reset | 3.297e-04 | 4.695e-04 | 7.118e-05 | -2.585e-04 |
| constant_motion_reflected | 4.597e-06 | 4.432e-04 | 9.331e-06 | +4.734e-06 |
| mlp_control | 7.560e-05 | 3.188e-04 | 4.886e-05 | -2.674e-05 |
| rls_online | 2.816e-05 | 1.143e-09 | 9.931e-06 | -1.823e-05 |
| rnn_control | 8.027e-05 | 1.055e-04 | 4.300e-05 | -3.727e-05 |

### Q6 -- can it estimate its own error?

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| rank correlation (Spearman) | 0.472 | 0.445 | 0.035 | 0.921 |
| linear correlation (Pearson) | 0.358 | 0.399 | -0.052 | 0.713 |
| calibration slope | 0.964 | 0.830 | -0.059 | 2.604 |
| bias (predicted minus realized) | 0.193 | 0.317 | -0.597 | 0.473 |

30 of 192 streams had a monotone quintile calibration table. this is a learned error-magnitude estimate, not a calibrated predictive distribution and not a Bayesian posterior.

### Q7 -- does it beat the simple alternatives?

a win requires the whole 95% interval above zero, declared in advance.

| Family | constant_motion | constant_motion_reflected | dead_reckoning | linear_fit | persistence | rls_online |
|---|---|---|---|---|---|---|
| aba_v1 | NEGATIVE | NEGATIVE | NEGATIVE | POSITIVE | POSITIVE | NEGATIVE |
| coarse_speed_v1 | POSITIVE | POSITIVE | NEGATIVE | NEGATIVE | POSITIVE | POSITIVE |
| motion_compat | NEGATIVE | NEGATIVE | NEGATIVE | POSITIVE | POSITIVE | NEGATIVE |
| occlusion_v1 | POSITIVE | POSITIVE | NEGATIVE | NEGATIVE | POSITIVE | NEGATIVE |

Baselines AAA-1K beat outright, by family:

- `aba_v1`: linear_fit, persistence
- `coarse_speed_v1`: constant_motion, constant_motion_reflected, persistence, rls_online
- `motion_compat`: linear_fit, persistence
- `occlusion_v1`: constant_motion, constant_motion_reflected, persistence

## Numerical stability and cost

| Measure | Value |
|---|---|
| gradient-clip activations | 12234 |
| non-finite events | 0 |
| trained steps | 53225 |
| targets skipped as unavailable | 1815 |
| scored transitions | 55040 |
| transitions per second | 756 |
| wall seconds | 72.8 |
| arms per stream | 12 |

Gradient clipping is a declared mechanism with a declared threshold, not a silent safety net, so its activation count is part of the result.

## What these numbers do and do not support

| Claim | What it would mean |
|---|---|
| implementation works | the code executes correctly and deterministically |
| neural learning occurred | weights changed in a useful direction on some measured stream |
| hidden state helps | persistent recurrence beat matched state-reset and stateless controls |
| online adaptation helps | continued learning beat an identical frozen-weight clone |
| retention exists | learning a new regime did not completely erase an old one |
| error estimation informative | predicted error magnitude tracked realized error |
| baseline competitiveness | the model beat the specified simple alternatives |
| generalization | held-out trajectories, regimes or families support a generalization claim |

None of these implies intelligence, general autonomy, causal understanding, physical understanding, AGI or consciousness. This is a 994-parameter network predicting where a dot goes next.

