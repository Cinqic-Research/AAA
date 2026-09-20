# AAA-1K: a 994-parameter recurrent predictive core

AAA-1K is a research seed, not Juniper and not an agent. It is one primitive: a persistent gated recurrent predictor that learns online, keeps a hidden state, and estimates how wrong it expects to be. Everything below describes what was measured on a moving dot.

## Identity

| Item | Value |
|---|---|
| phase | `aaa.1k.v1` |
| scientific fingerprint | `88738d9762f9721c4c32dd36dc2f93b13f11769aa5895c29879699c26952e2bd` |
| files covered | 28 |
| trainable parameters | 994 |
| selected configuration | lr=0.03, T=4, lambda=0.25 |
| replicas per family | 32 |
| evaluation streams | 192 |

## Parameter and state accounting

| Scalar category | Count |
|---|---|
| hidden state scalars | 16 |
| optimizer state scalars | 0 |
| tbptt buffer scalars capacity | 404 |
| tbptt buffer scalars current | 0 |
| total adaptive state scalars | 1414 |
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

**The declared precision objective was NOT met.** The development pilot measured a primary effect of 7.13e-05 with a per-stream spread of 2.22e-04; resolving a quarter of that effect at 95% would have needed 599 replicas per family, and the declared bound of 32 was applied. Every interval below is therefore wider than the design asked for, and effects near zero should be read as unresolved rather than absent.

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

**Important qualification, from an adversarial probe.** Branching at a point where *nothing changes* reproduces 95% of this effect. Q2 therefore measures continued learning in general far more than it measures adaptation specific to the change. The headline number is real; the natural reading of it is wrong. See `docs/evidence/aaa_1k_adversarial_probes.json` and `docs/aaa_1k_self_review.md`.

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

**Read the capacity match carefully.** Parameter counts are close (954 against 994), but the ungated control carries 28 hidden units to the gated model's 16. Matching on parameters buys the ungated arm more state, which is exactly the trade a gate costs you. The comparison is the honest one for a fixed parameter budget, and it is not a comparison at matched hidden width.

| Scope | Verdict | Ungated RNN minus AAA-1K |
|---|---|---|
| all families | NEGATIVE | -1.89e-04 [-2.37e-04, -1.42e-04] (44/192 streams) |
| memory families | NEGATIVE | -3.75e-04 [-5.02e-04, -2.49e-04] (31/64 streams) |

### Q5 -- what survives A, then B, then A again?

A2 tail minus A1 tail; positive means the model came back worse.

**Read this table carefully.** Every learning arm came back *better* than it left, so no catastrophic forgetting was detected. But A1 is the first segment a learner ever sees and A2 is the third, so by A2 the model has had three times as much experience in total. This design cannot separate "it retained A" from "it kept getting better at everything", and the next phase needs a fixed frozen probe bank measured at both boundaries to do so. The non-learning arms are the control: `constant_motion_reflected` is flat across A1 and A2, as it must be.

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
| transitions per second | 759 |
| wall seconds | 72.5 |
| arms per stream | 12 |

Gradient clipping is a declared mechanism with a declared threshold, not a silent safety net, so its activation count is part of the result.

## What these numbers do and do not support

| Claim | Status | What it would mean | What was actually measured |
|---|---|---|---|
| implementation works | **SUPPORTED** | the code executes correctly and deterministically | 0 non-finite events over 55040 scored transitions; the suite includes exhaustive finite-difference gradient checks and a resume-equals-uninterrupted test |
| neural learning occurred | **SUPPORTED** | weights changed in a useful direction on some measured stream | Q1, +5.19e-04 [+4.47e-04, +5.91e-04] (176/192 streams) |
| hidden state helps | **SUPPORTED** | persistent recurrence beat matched state-reset and stateless controls | Q3 against the stateless control on the memory families, +4.25e-04 [+3.77e-04, +4.76e-04] (64/64 streams) |
| online adaptation helps | **SUPPORTED** | continued learning beat an identical frozen-weight clone | Q2, +9.76e-04 [+7.80e-04, +1.20e-03] (96/96 streams) |
| retention exists | **SUPPORTED, WITH A CONFOUND** | learning a new regime did not completely erase an old one | the A2 tail was 5.91e-05 *lower* than the A1 tail, so no catastrophic forgetting was detected -- but the model has also had twice as much total experience by A2, so this run cannot separate retention from continued learning |
| error estimation informative | **SUPPORTED, WEAKLY** | predicted error magnitude tracked realized error | mean rank correlation 0.47 between the predicted and realized error magnitude, but only 30 of 192 streams had a monotone quintile table |
| baseline competitiveness | **MIXED** | the model beat the specified simple alternatives | beat constant_motion, constant_motion_reflected, linear_fit, persistence, rls_online on at least one family; lost to constant_motion, constant_motion_reflected, dead_reckoning, linear_fit, rls_online on at least one family |
| generalization | **NOT CLAIMED** | held-out trajectories, regimes or families support a generalization claim | evaluation streams are held out from development, which supports a claim about unseen trajectories of the *same* families only. No unseen family was tested, so nothing here supports generalization to a new kind of world |

Every effect above is conditional on a single model initialization: the evaluation gives every arm the same initialization seed so that an ablation differs from the primary in exactly one mechanism, and the intervals therefore resample streams but not initializations. A probe across five initializations found every comparison keeping its sign while magnitudes varied by up to a factor of two.

None of these implies intelligence, general autonomy, causal understanding, physical understanding, AGI or consciousness. This is a 994-parameter network predicting where a dot goes next.

