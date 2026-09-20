# AAA-1K adversarial self-review

This is an attempt to prove the phase's own conclusions wrong. Two probes found
something substantial, and one of them changes how the headline Q2 result
should be read. Both are reproducible:

```bash
python -m research.aaa_1k adversarial-probes \
    --selection docs/evidence/aaa_1k_development_selection.json \
    --output docs/evidence/aaa_1k_adversarial_probes.json
```

This is an implementer's self-review. It is **not** independent review, and
nothing here should be read as approval.

---

## Findings that changed a claim

### SR-1. Q2 does not measure what its name suggests — CONFIRMED DEFECT IN THE READING

**Hypothesis tested.** "An online arm beats its frozen twin after an
unannounced change" is also exactly what you would observe if continued
learning simply helped everywhere, change or no change. Q2's design cannot tell
those apart, because the model is still improving on every family throughout
every stream (that is Q1's result).

**Probe.** Branch the same streams at step 60, where nothing whatsoever
happens, and compare against the declared change point at step 100.

| Branch point | Frozen minus online | 95% interval |
|---|---|---|
| declared change (step 100) | +8.07e-04 | [+6.17e-04, +1.03e-03] |
| quiet control (step 60) | +7.65e-04 | [+5.77e-04, +9.76e-04] |

**The control reproduces 95% of the effect.**

**Disposition.** The measurement is correct; the natural interpretation of it is
not. Q2 supports "continued updating helps" and does **not** support "the model
adapts to change". The report now says so at the point of the claim. Isolating
adaptation needs a design this phase does not have: a matched pair of streams
identical up to the branch, one of which then changes and one of which does
not, with the effect defined as the *difference of differences*. That is a
concrete, cheap next experiment.

### SR-2. Every headline effect is conditional on one initialization — CONFIRMED LIMITATION

**Hypothesis tested.** The evaluation gives every arm the same `model_init`
seed, deliberately, so that an ablation differs from the primary in exactly one
mechanism. The bootstrap then resamples streams. Initialization variance is
therefore **entirely unsampled**, and the intervals are narrower than the
uncertainty that actually exists.

**Probe.** Repeat the key comparisons across five initializations on 16 memory-
family streams.

| seed index | Q3 vs stateless MLP | Q3 vs state reset | Q4 vs ungated RNN |
|---|---|---|---|
| 0 | +4.43e-04 | +7.06e-04 | -3.34e-04 |
| 1 | +2.98e-04 | +7.27e-04 | -5.02e-04 |
| 2 | +3.44e-04 | +6.94e-04 | -3.90e-04 |
| 3 | +3.60e-04 | +6.69e-04 | -4.95e-04 |
| 4 | +2.26e-04 | +5.41e-04 | -4.89e-04 |

**Every comparison kept its sign.** Magnitudes vary by up to a factor of two.

**Disposition.** The *directions* of Q3 and Q4 are robust to initialization;
the *magnitudes* in the report are conditional on one. The report states this.
A proper design would treat initialization as a second resampling level in a
hierarchical bootstrap, which is what the v2.1 protocol already does for
replicas and episodes, and which this phase should have copied.

### SR-3. Q4's capacity match gives the ungated arm more state — DISCLOSED, NOT A DEFECT

The ungated control has 954 parameters against the GRU's 994, but 28 hidden
units against 16. Matching on parameter count necessarily buys the ungated arm
more state, because that is precisely what a gate costs. The comparison is the
right one for a fixed parameter budget and is the one the phase brief
specified, but it is not a comparison at matched hidden width, and the report
now says so. A width-matched comparison (16 ungated units, ~370 parameters)
would answer a different and also interesting question.

---

## Defects found and fixed during the phase

### SR-4. A benchmark family that would have measured nothing — FIXED BEFORE ANY EVALUATION

The third family was originally specified as reflecting motion with a hidden
speed regime observed on every *other* step. Under the declared rule that a
learner updates only on transitions whose both ends it was shown, no two
consecutive steps are ever observed after the warm-up, so the learner would
have trained **zero times** on that family. The benchmark would have run,
produced plausible numbers, and measured nothing at all.

Caught by reasoning through the training rule before generating any evaluation
stream. Replaced with `coarse_speed_v1`, which hides observation *precision*
rather than observation *presence*. Recorded as `D-7`.

### SR-5. A stability rule that could not fire — FIXED BEFORE SELECTION

The stability-margin rule initially required only that the next higher learning
rate in the grid also be stable. With the declared gradient clip at 1.0,
nothing in the declared grid ever diverges, so the rule was vacuous — a gate
that cannot fail, which is exactly the disease the v2.1 repair existed to cure.

Fixed by adding a stage-0 probe with clipping disabled, which located a real
boundary at `lr = 0.3`, and by requiring two grid steps of margin. The repaired
rule then **eliminated the four best-performing configurations**, costing 26%
of development accuracy. Recorded as `D-8`.

### SR-6. An input that could not be measured — FIXED BEFORE SELECTION

Normalized as the brief specified, input 3 was numerically inert, and the
`zero_error_input` ablation was identical to the full model to five decimal
places. The mechanism the brief asked to be tested could not have been measured
either way. Rescaled to the public displacement constant; the ablation now
separates. Recorded as `D-3`.

### SR-7. Baselines penalised by a gap they did not cause — FIXED BEFORE EVALUATION

The first observation tracker exposed a raw displacement between the last two
*observed* positions. After a gap of four steps, that hands every rule a
displacement four times too large and then scores the rule on the overshoot.
Fixed by dividing by the elapsed step count, so the feature is always a
per-step velocity. Applies identically to the neural arms and the analytic
baselines.

### SR-8. A state-footprint figure measured on an empty buffer — FIXED

`state_footprint()` reported the TBPTT buffer's *current occupancy*, which is
zero for a freshly constructed model, so the first report claimed a total
adaptive footprint of 1010 scalars. The honest figure is the buffer at
capacity: 1414. Both are now reported.

---

## Hypotheses tested that did not find a defect

| Checked | Method | Outcome |
|---|---|---|
| future leakage | `accept_observation` is the single channel and carries only `float | None`; a recording agent asserts nothing else arrives | clean |
| evaluator metadata reaching a learner | regime, event, latent truth and hidden speed exist only on `StreamStep`, which the runner never passes on | clean |
| targets never shown being trained against | `trained_steps` is asserted equal to the count of transitions with both ends observed; an injected greedy agent trains more and the test catches it | clean |
| state sharing between clones | every array identity checked; mutating a clone does not touch the original; an injected shallow clone is caught | clean |
| online/frozen asymmetry beyond weights | both arms branch from one `state_dict`, and the clone hashes are asserted equal at every branch | clean |
| improper hidden-state resets | the hidden state is reset only on `begin_episode` or the declared ablation; a frozen arm's hidden state is asserted to keep moving | clean |
| parameter miscounting | recomputed from array sizes, from the formula, and against the declared constant | 994 / 982 / 954, all three agree |
| gradient correctness | every parameter of every model finite-differenced exhaustively against a sequence loss | max absolute error 8.7e-10 |
| a broken temporal gradient hiding behind a correct one-step gradient | injected a backward pass that keeps only the last transition | caught |
| stop-gradient not actually in effect | a finite-difference reference that lets the auxiliary target move | disagrees, as it must |
| boundary-transform advantage | the GRU, reflected constant motion and dead reckoning use the identical public `reflect_prediction` | matched; the unreflected variant is also reported |
| an unfair incumbent | the RLS arm's ten declared parameters are asserted equal to the frozen v2.1 specification's `candidate` block | unmodified |
| seed reuse across roles | five namespaces, 200 indices each, asserted pairwise disjoint | disjoint |
| tuning on evaluation | selection reads only `development_env` streams; evaluation reads only `evaluation_env` | separated |
| serialization / resume mismatch | a run interrupted at step 60 and resumed is asserted bitwise identical to the uninterrupted run | identical |
| the renderer mutating model state | complete state hash compared across a render | unchanged |
| `all([]) is True` style vacuity | empty comparisons raise; under-sampled calibration returns `INSUFFICIENT_EVIDENCE` | cannot pass vacuously |
| broken legacy commands | the full 538-test suite, `aaa spec-hash`, the v2.1 development benchmark, `recompute` and the v1 smoke all run | unaffected |

---

## Claims I would challenge if I were reviewing this

1. **Q2's headline.** SR-1. The number is right and the word "adaptation"
   should not be attached to it.
2. **Q7's wins against `rls_online` on `coarse_speed_v1`.** The RLS candidate
   was selected and frozen for smooth, fully observed motion. Beating it on a
   quantized staircase is beating it outside its declared operating envelope.
   That is worth knowing and is not evidence that the GRU is the better
   learner.
3. **Q5 entirely.** The A2 tail is lower than the A1 tail for every learning
   arm, which reads as "no forgetting" but is confounded with "three times as
   much total experience". This phase did not measure retention. It should be
   redesigned around a frozen probe bank evaluated at both segment boundaries,
   and until then `retention_exists` should be read as `NOT MEASURED` rather
   than as supported.
4. **Q6's usefulness.** A mean rank correlation of 0.47 with only 30 of 192
   streams showing a monotone quintile table is a weak signal, and the head
   over-predicts (bias +0.19 in normalized units, visible as the flat orange
   line above the red one on the dashboard). "The model knows when it is about
   to be wrong" would be an overstatement.
5. **The precision objective.** It was not met, by a factor of nearly twenty.
   Everything in the report is wider than the design asked for.

---

## What I would do next, in order

1. Redesign Q2 as a difference-of-differences against a matched no-change
   stream (SR-1). Cheap, and it converts a misleading result into a real one.
2. Redesign Q5 around a frozen probe bank measured at both segment boundaries,
   then — and only then — consider replay.
3. Make initialization a second level of the bootstrap (SR-2).
4. Build a benchmark with genuine long-range dependence, so that the truncation
   horizon matters and UORO becomes worth implementing (`D-12`).
5. Leave the architecture alone. It loses to a *smaller* ungated control and to
   parameterless analytic rules; that is not a capacity problem.
