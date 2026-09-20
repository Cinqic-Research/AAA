# AAA-1K round 2: a 994-parameter recurrent predictive core

AAA-1K is a research seed, not Juniper and not an agent. It is one primitive: a persistent gated recurrent predictor that learns online, keeps a hidden state, and estimates how wrong it expects to be. Everything below describes what was measured on a moving dot.

**This round supersedes round 1.** Round 1 is retained unchanged at
`docs/evidence/aaa_1k_evaluation_round1_superseded.json`. Four defects in it
were found by probing its own conclusions, and all four are repaired here
rather than annotated.

| What | Round 1 | Round 2 | Tracked as |
|---|---|---|---|
| adaptation (Q2) | a single online/frozen branch, which a control showed was 95% reproduced where nothing changed | difference-of-differences against a bit-identical unchanged world | `AAA-153` |
| retention (Q5) | segment tails, confounded with three times the accumulated experience | a fixed frozen probe bank asked the same questions at every checkpoint | `AAA-154` |
| uncertainty | intervals resampled streams only, treating the starting weights as fixed by nature | a crossed bootstrap over initializations and streams | `AAA-155` |
| baseline fairness | one architecture's clip threshold imposed on all arms | each architecture on its own rule-selected learning rate and clip | `AAA-156` |

## Identity and design

| Item | Value |
|---|---|
| phase | `aaa.1k.v1` |
| scientific fingerprint | `60ab9aa04f2e7f99359e1ec2b41a0e86af58f0c78727307017ab3b17ad90978a` |
| trainable parameters | 994 |
| initializations | 5 |
| streams per family | 24 |
| evaluation cells | 720 |
| adaptation trials | 24 |
| retention trials | 12 |
| stream identities | fresh; no round-1 stream is reused |

Each architecture runs on the hyperparameters the same declared rules select for it:

| Architecture | learning rate | TBPTT | lambda | gradient clip |
|---|---|---|---|---|
| AAA1KGRU | 0.03 | 4 | 0.25 | 10 |
| StatelessMLPControl | 0.01 | 4 | 0.25 | none |
| VanillaRNNControl | 0.01 | 4 | 0.25 | none |

Ablations of the gated model share its row exactly, because an ablation is the same architecture with one mechanism removed.

## Development selection

The unclipped divergence probe put the gated model's boundary at a learning rate of 0.3, and the declared stability margin restricted the search to [0.001, 0.003, 0.01, 0.03]. That rule **eliminated the best-performing development configurations**, which is what a rule that can bind looks like.

Best eliminated configuration: `lr=0.1;T=32;lambda=0.25` at 1.138e-03 development error, discarded for sitting inside the margin.

Stage 3 then selected the clip threshold instead of asserting it. The originally declared 1.0 activated on 24% of updates and cost 30% of development error against the best stable threshold.

## Capability vector

| Question | Verdict | Effect (normalized error) [95% CI] |
|---|---|---|
| Q1 online learning | POSITIVE | +2.94e-04 [+2.17e-04, +3.70e-04] (119/144 streams) |
| Q2 adaptation (difference-of-differences) | POSITIVE | +9.50e-04 [+5.63e-04, +1.33e-03] (16/24 streams) |
| Q3 hidden state vs stateless control | POSITIVE | +2.49e-04 [+9.87e-05, +3.47e-04] (42/48 streams) |
| Q4 gating vs ungated control | INCONCLUSIVE | -4.51e-05 [-1.19e-04, +1.39e-05] (117/144 streams) |
| Q5 forgetting (probe bank) | NEGATIVE | -3.74e-04 [-7.99e-04, -9.52e-07] (4/12 streams) |

For Q5 a *negative* effect is the good direction: it means probe-bank error on regime A fell while the model was training on regime B.

### Achieved precision

Reported against the effect actually measured, rather than only against a target sized from a pilot estimate of an effect nobody had seen.

| Question | Effect | CI half-width | Resolution |
|---|---|---|---|
| Q1 online learning | +2.94e-04 | 7.61e-05 | 26% of the effect |
| Q2 adaptation | +9.50e-04 | 3.86e-04 | 41% of the effect |
| Q3 hidden state | +2.49e-04 | 1.24e-04 | 50% of the effect |
| Q4 gating | -4.51e-05 | 6.63e-05 | 147% of the effect |

### Q1 -- can it learn online?

| Comparison | Verdict | Effect |
|---|---|---|
| all families | POSITIVE | +2.94e-04 [+2.17e-04, +3.70e-04] (119/144 streams) |

### Q3 -- is persistent recurrent state worth anything?

| Comparison | Verdict | Effect |
|---|---|---|
| vs state-reset ablation, memory families | INCONCLUSIVE | +1.26e-04 [-3.74e-05, +2.11e-04] (39/48 streams) |
| vs stateless control, memory families | POSITIVE | +2.49e-04 [+9.87e-05, +3.47e-04] (42/48 streams) |
| vs no-previous-error ablation, memory families | INCONCLUSIVE | +4.80e-05 [-1.16e-04, +9.88e-05] (46/48 streams) |
| vs frozen-recurrent-weights ablation, memory families | INCONCLUSIVE | +5.20e-06 [-7.92e-06, +1.15e-05] (39/48 streams) |
| vs state-reset, all families | POSITIVE | +1.80e-04 [+1.26e-04, +2.15e-04] (135/144 streams) |
| vs stateless control, all families | POSITIVE | +2.19e-04 [+1.73e-04, +2.64e-04] (138/144 streams) |

Restricted to steps whose target was never shown to the agent:

| Comparison | Verdict | Effect |
|---|---|---|
| vs state-reset | POSITIVE | +4.94e-04 [+2.85e-04, +7.31e-04] (22/24 streams) |
| vs stateless control | POSITIVE | +1.22e-03 [+1.01e-03, +1.43e-03] (24/24 streams) |

### Q2 -- does continued learning help *because the world changed*?

two bit-identical worlds diverging at a declared step, one changed and one not; the online-minus-frozen advantage on the unchanged world is subtracted from the advantage on the changed one.

| Component | Verdict | Effect |
|---|---|---|
| adaptation (changed minus control) | POSITIVE | +9.50e-04 [+5.63e-04, +1.33e-03] (16/24 streams) |
| continued learning (control alone) | POSITIVE | +9.75e-04 [+6.48e-04, +1.32e-03] (24/24 streams) |

Adaptation accounts for 49% of the total online advantage; the rest is the ordinary benefit of continuing to learn, which round 1 reported as if it were all adaptation. Every paired trial's two trunks reached an identical model state at the branch: `trunks_matched = True`.

### Q4 -- do the gates earn their parameters?

954 parameters against 994, but 28 hidden units against 16. Matching on parameters necessarily buys the ungated arm more state, because that is what a gate costs. This is the honest comparison at a fixed parameter budget and is not a comparison at matched hidden width.

| Scope | Verdict | Effect (ungated minus gated) |
|---|---|---|
| all families | INCONCLUSIVE | -4.51e-05 [-1.19e-04, +1.39e-05] (117/144 streams) |
| memory families | NEGATIVE | -2.53e-04 [-4.45e-04, -1.04e-04] (24/48 streams) |

**The mean and the median disagree, and that is the finding.** The median stream favours the gated model (+5.59e-05) while the mean favours the ungated one (-4.51e-05): the gated model is slightly better on 117 of 144 streams and much worse on the rest. A single averaged number would have reported only half of that.

**The ungated control is the less stable architecture.** Run without a clip it diverges at a learning rate of 0.1, against 0.3 for the gated model, so the same declared stability margin allows it only 0.01 where the gated model gets 0.03. Round 1 ran both at the gated model's rate and reported that gating loses. Tuned separately, the comparison is inconclusive once each architecture is tuned separately, so Q4's negative result is weaker than the shared-rate comparison suggested.

### Q5 -- what survives learning a new regime?

a fixed bank of held-out regime-A episodes, never trained on, evaluated by a frozen clone with its hidden state reset, at the end of each of A1, B and A2.

| Probe-bank error | Mean |
|---|---|
| after A1 | 1.061e-03 |
| after B | 6.865e-04 |
| after A2 | 6.854e-04 |

Forgetting: **NEGATIVE**, -3.74e-04 [-7.99e-04, -9.52e-07] (4/12 streams). A positive value would mean regime-A ability degraded while learning regime B. The probe bank is identical at all three checkpoints, so accumulated experience cannot flatter the later measurements -- which is exactly what round 1's design could not rule out.

### Q6 -- can it estimate its own error?

| Measure | Mean | Median | Min | Max |
|---|---|---|---|---|
| rank correlation (Spearman) | 0.437 | 0.426 | -0.098 | 1.000 |
| linear correlation (Pearson) | 0.303 | 0.331 | -0.011 | 0.670 |
| calibration slope | 0.769 | 0.705 | -0.089 | 2.628 |
| bias (predicted minus realized) | 0.247 | 0.349 | -1.281 | 0.476 |

106 of 720 cells had a monotone quintile calibration table. a learned error-magnitude estimate, not a calibrated predictive distribution and not a Bayesian posterior.

### Q7 -- does it beat the simple alternatives?

a win requires the whole 95% interval above zero, declared in advance.

| Family | constant_motion | constant_motion_reflected | dead_reckoning | linear_fit | persistence | rls_online |
|---|---|---|---|---|---|---|
| aba_v1 | NEGATIVE | NEGATIVE | NEGATIVE | POSITIVE | POSITIVE | NEGATIVE |
| coarse_speed_v1 | POSITIVE | POSITIVE | INCONCLUSIVE | NEGATIVE | POSITIVE | POSITIVE |
| motion_compat | NEGATIVE | NEGATIVE | NEGATIVE | POSITIVE | POSITIVE | NEGATIVE |
| occlusion_v1 | POSITIVE | POSITIVE | NEGATIVE | NEGATIVE | POSITIVE | NEGATIVE |

Baselines AAA-1K beat outright:

- `aba_v1`: linear_fit, persistence
- `coarse_speed_v1`: constant_motion, constant_motion_reflected, persistence, rls_online
- `motion_compat`: linear_fit, persistence
- `occlusion_v1`: constant_motion, constant_motion_reflected, persistence

Baselines AAA-1K lost to outright:

- `aba_v1`: constant_motion, constant_motion_reflected, dead_reckoning, rls_online
- `coarse_speed_v1`: linear_fit
- `motion_compat`: constant_motion, constant_motion_reflected, dead_reckoning, rls_online
- `occlusion_v1`: dead_reckoning, linear_fit, rls_online

## What the development probes settled

**clipping_probe** -- is the declared gradient clip shaping the optimization or guarding it?

the declared threshold of 1.0 activates on 24% of updates and costs 29% of development error against the best stable threshold. At that rate it is a hyperparameter deciding what gets learned, not a guard, and asserting it rather than selecting it was a defect. The threshold is now chosen by stage 3 of the development selection..

**coarse_speed_decomposition** -- does coarse_speed_v1 measure hidden-regime inference or coarse-observation integration?

with the speed held fixed the recurrent model is 6.32e-04 *worse* than the stateless control, and it is better only once the speed starts switching. `coarse_speed_v1` is therefore measuring inference of a hidden regime, not tolerance to a coarse grid -- the quantizer alone hands the advantage to the stateless arm.

**per_architecture_selection** -- does Q4's negative result survive giving each architecture its own learning rate?

the comparison is inconclusive once each architecture is tuned separately, so Q4's negative result is weaker than the shared-rate comparison suggested.

## Numerical stability and cost

| Measure | Value |
|---|---|
| gradient-clip activations | 552 |
| mean clip rate | 0.20% |
| max clip rate on any cell | 16.07% |
| non-finite events | 0 |
| trained steps | 199575 |
| targets skipped as unavailable | 6825 |
| scored transitions | 206400 |
| transitions per second | 752 |
| wall seconds | 274.4 |

Round 1 clipped on 22% of updates at a threshold that was asserted rather than selected, and a probe found that threshold costing 29% of development error. The threshold is now chosen by the same rule as every other hyperparameter, and the activation rate above is what a guard rather than a decision-maker looks like.

## What these numbers do and do not support

| Claim | Status | What was actually measured |
|---|---|---|
| implementation works | **SUPPORTED** | 0 non-finite events over 206400 scored transitions; every parameter of every model is finite-difference verified and a resumed run is bitwise identical to an uninterrupted one |
| neural learning occurred | **SUPPORTED** | Q1, +2.94e-04 [+2.17e-04, +3.70e-04] (119/144 streams) |
| hidden state helps | **SUPPORTED** | Q3 against the stateless control on the memory families, +2.49e-04 [+9.87e-05, +3.47e-04] (42/48 streams); each control runs on its own rule-selected hyperparameters |
| gating helps | **INCONCLUSIVE** | Q4, -4.51e-05 [-1.19e-04, +1.39e-05] (117/144 streams). Round 1 reported this as CONTRADICTED; that result did not survive giving each architecture its own rule-selected learning rate and clip |
| online adaptation helps | **SUPPORTED** | Q2 difference-of-differences against a bit-identical unchanged world, +9.50e-04 [+5.63e-04, +1.33e-03] (16/24 streams) |
| retention exists | **SUPPORTED** | probe-bank error after regime B minus after regime A1, -3.74e-04 [-7.99e-04, -9.52e-07] (4/12 streams); a positive value would be forgetting |
| error estimation informative | **SUPPORTED, WEAKLY** | mean rank correlation 0.44 between predicted and realized error magnitude; only 106 of 720 cells had a monotone quintile table |
| baseline competitiveness | **MIXED** | beat constant_motion, constant_motion_reflected, linear_fit, persistence, rls_online on at least one family; lost to constant_motion, constant_motion_reflected, dead_reckoning, linear_fit, rls_online on at least one family |
| generalization | **NOT CLAIMED** | evaluation streams are held out from development and from round 1, which supports a claim about unseen trajectories of the same families only. No unseen family was tested |

None of these implies intelligence, general autonomy, causal understanding, physical understanding, AGI or consciousness. This is a 994-parameter network predicting where a dot goes next.
