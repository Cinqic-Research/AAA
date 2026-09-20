# AAA-1K adversarial self-review

An attempt to prove the phase's own conclusions wrong. It succeeded four times,
and all four findings are now repaired rather than annotated. Round 1's evidence
is retained unchanged at
[`evidence/aaa_1k_evaluation_round1_superseded.json`](evidence/aaa_1k_evaluation_round1_superseded.json);
round 2 is the corrected result.

Everything below is reproducible:

```bash
python -m research.aaa_1k adversarial-probes --selection <selection> --output <path>
python -m research.aaa_1k characterize       --selection <selection> --output <path>
```

This is an implementer's self-review. It is **not** independent review, and
nothing here should be read as approval.

---

## Defects found, and what they did to the result

### SR-1. Q2 measured continued learning, not adaptation — REPAIRED (`AAA-153`)

**Probe.** Branch an online/frozen pair at step 60, where nothing whatsoever
happens, and compare against the declared change point at step 100.

| Branch point | Frozen minus online | 95% interval |
|---|---|---|
| declared change | +8.07e-04 | [+6.17e-04, +1.03e-03] |
| quiet control | +7.65e-04 | [+5.77e-04, +9.76e-04] |

The control reproduced **95%** of the effect.

**Repair.** A `paired_change_v1` family emits two streams that are
bit-identical until a declared step, after which one changes speed and the
other does not. A single model is driven through the shared prefix, so both
variants branch from the same model state — asserted by complete state hash,
not assumed. The estimator is `advantage(changed) − advantage(control)`.

**What it changed.** Adaptation is real, and smaller than round 1 implied:
**+9.50e-04 [+5.63e-04, +1.33e-03]**, about 65% of the total online advantage.
Round 1's claim was directionally right and overstated by roughly a third.

### SR-2. Q5 did not measure retention — REPAIRED (`AAA-154`)

**Problem.** Every learning arm returned to regime A better than it left, which
reads as "no forgetting" but is confounded: by A2 the model has had three times
as much total experience.

**Repair.** A fixed bank of eight held-out regime-A episodes, generated once and
never trained on, evaluated by a frozen clone with its hidden state reset, at
the end of each of A1, B and A2. The same questions at every checkpoint, so
accumulated experience cannot flatter the later ones.

**What it changed.** Probe error after B minus after A1 is
**−3.74e-04 [−7.99e-04, −9.52e-07]**. Regime-A ability *improved* while the
model trained on regime B. There is no forgetting here to target, which is why
no replay mechanism was added — adding one would have decorated a
non-problem.

### SR-3. Intervals ignored initialization variance — REPAIRED (`AAA-155`)

**Problem.** Round 1 gave every arm one initialization seed, deliberately, then
resampled only streams. A five-seed probe found magnitudes varying by up to a
factor of two while the intervals claimed a precision that did not account for
it.

**Repair.** Five initializations, and a crossed bootstrap that resamples
initializations and streams independently. Per-initialization effects are
reported alongside the mean.

**What it changed.** Intervals widened, and every conclusion now states whether
all five initializations agreed on the sign. Q3's do.

### SR-4. The controls were handicapped — REPAIRED (`AAA-156`)

**This is the one that would have produced a false headline.**

The gradient-clip threshold was *asserted* at 1.0, never selected. A probe found
it activating on 24% of updates and costing 29% of development error. So I made
it a selected hyperparameter — and then applied the *gated model's* selected
threshold to every arm.

The stateless control destabilized on `coarse_speed_v1`: mean error **1.6e-01**
against the gated model's 2.2e-03. The crossed comparison duly reported a
hidden-state advantage of **+1.28e-02** — thirty times the true effect. Had I
not looked at the per-arm table, "hidden state helps, decisively" would have
gone into the report.

**Repair.** Stage 3a selects the learning rate *and* the clip separately for
each architecture, by the same declared rules, on the same development streams.
Ablations of the gated model still share its hyperparameters exactly, because
an ablation is the same architecture with one mechanism removed.

**What it changed.** The hidden-state effect fell to its honest **+2.49e-04**,
and **Q4 moved from `NEGATIVE` to `INCONCLUSIVE`**: the round-1 finding that
gating loses did not survive giving the ungated control a fair learning rate.

---

## What the development probes settled

**The clip was deciding, not guarding.** At the declared 1.0 it fired on 24% of
updates and cost 29% of development error. The selected threshold of 10.0 fires
on under 1%. That is the difference between a hyperparameter and a safety net.

**`coarse_speed_v1` really does measure hidden-regime inference.** With the
speed held fixed — the quantizer alone — the recurrent model is **6.32e-04
worse** than the stateless control. It only becomes better once the speed starts
switching. My worry that the family measured nothing but tolerance to a coarse
grid was wrong, and the opposite is true: the grid alone favours the stateless
arm.

**Gating buys stability, which round 1 never noticed.** Run without a clip, the
ungated control diverges at a learning rate of 0.1; the gated model survives to
0.3. The same declared stability margin therefore allows the ungated arm only
0.01 where the gated model gets 0.03. Round 1 ran both at the gated model's rate
and concluded that gating loses. Tuned separately, the comparison is
inconclusive — and the *reason* the ungated arm looked good is that it was being
run at a rate its own stability rule forbids.

---

## Defects caught before they reached any evaluation

**A family that would have measured nothing.** The third family was originally
specified as showing only every other step. Under the declared rule that a
learner updates only on transitions whose both ends it was shown, no two
consecutive steps are ever observed after the warm-up — the learner would have
trained **zero times**. The benchmark would have run and produced plausible
numbers. Replaced with `coarse_speed_v1` (`D-7`).

**A stability rule that could not fire.** The margin rule initially required
only that the next higher learning rate also be stable. With clipping on,
nothing in the grid ever diverges, so the rule was vacuous — exactly the disease
the v2.1 repair existed to cure. A stage-0 probe with clipping disabled found a
real boundary at `lr = 0.3`, and the repaired rule then eliminated the four best
development configurations (`D-8`).

**An input that could not be measured.** Normalized as the brief specified,
input 3 was numerically inert and its ablation was identical to the full model
to five decimal places (`D-3`).

**Baselines penalised by a gap they did not cause.** The first observation
tracker exposed a raw displacement across a gap, handing every rule a
displacement four times too large and then scoring it on the overshoot (`SR-7`
in round 1).

---

## Hypotheses tested that did not find a defect

| Checked | Method | Outcome |
|---|---|---|
| future leakage | `accept_observation` is the single channel and carries only `float \| None` | clean |
| evaluator metadata reaching a learner | regime, event, latent truth and hidden speed live only on `StreamStep` | clean |
| training on targets never shown | `trained_steps` asserted equal to transitions with both ends observed; an injected greedy agent is caught | clean |
| a retention probe that learns | injected a clone that still updates; the read-only assertion fires | caught |
| a broken DiD pairing | injected an unpaired control stream; the identical-trunk assertion fires | caught |
| state sharing between clones | array identities checked; an injected shallow clone is caught | clean |
| online/frozen asymmetry beyond weights | both arms branch from one `state_dict`; clone hashes asserted equal | clean |
| parameter miscounting | recomputed from arrays, from the formula, and against the declared constant | 994 / 982 / 954 |
| gradient correctness | every parameter of every model finite-differenced exhaustively | max absolute error 8.7e-10 |
| a broken temporal gradient | injected a backward pass keeping only the last transition | caught |
| stop-gradient not in effect | a reference that lets the auxiliary target move | disagrees, as it must |
| boundary-transform advantage | the GRU and the reflecting baselines share one public `reflect_prediction` | matched |
| an unfair incumbent | the RLS arm's ten declared parameters asserted equal to the frozen v2.1 block | unmodified |
| seed reuse across roles | five namespaces, 200 indices each, asserted pairwise disjoint | disjoint |
| round-1 stream reuse | round 2 draws from declared fresh offsets | no overlap |
| tuning on evaluation | selection and characterization read `development_env` only | separated |
| serialization / resume mismatch | a run interrupted at step 60 and resumed | bitwise identical |
| the renderer mutating model state | complete state hash across a render | unchanged |
| vacuous success | empty comparisons raise; under-sampled calibration returns `INSUFFICIENT_EVIDENCE` | cannot pass vacuously |
| broken legacy commands | the full suite plus every `aaa.cli` path | unaffected |

---

## What I would still challenge if I were reviewing this

1. **Q4 is inconclusive overall but negative on the memory families**
   (−2.53e-04 [−4.45e-04, −1.04e-04]). "Gating does not clearly help, and looks
   actively unhelpful where memory matters, but the ungated arm is the less
   stable architecture" is three claims, and only the first two are measured
   here.
2. **Q4's mean and median disagree in sign.** The gated model is slightly better
   on 117 of 144 streams and much worse on the rest. That structure deserves a
   distributional analysis this phase does not have.
3. **`occlusion_v1` and `coarse_speed_v1` were designed by the same implementer
   whose model they evaluate**, and reviewed by nobody. That remains the weakest
   part of the evidence, and the decomposition probe only addresses one of the
   two.
4. **Q6 is weak.** A mean rank correlation of 0.44 with 106 of 720 cells showing
   a monotone quintile table, and a head that over-predicts. "The model knows
   when it is about to be wrong" would be an overstatement.
5. **Beating `rls_online` on `coarse_speed_v1`** is beating it outside its
   declared operating envelope; the v2.1 candidate was frozen for smooth, fully
   observed motion.
6. **Five initializations is a small second level.** The crossed interval is
   honest about the design but is itself estimated from five points.

## What I would do next, in order

1. A width-matched gating comparison (16 ungated units) to separate "gating" from
   "fewer hidden units at the same parameter count".
2. A distributional analysis of Q4's minority of large ungated wins — that is
   where the mechanism is hiding.
3. An independently designed memory benchmark, from someone who is not also
   building the model.
4. A benchmark with genuine long-range dependence, so the truncation horizon
   matters and UORO becomes worth implementing (`D-12`).
5. Leave the architecture alone. It is inconclusive against a *smaller* control
   and loses to parameterless analytic rules; that is not a capacity problem.
