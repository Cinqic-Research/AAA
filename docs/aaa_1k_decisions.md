# AAA-1K decision log

Decisions taken during the phase, with the option that was rejected and the
reason. Where a decision departs from the phase brief it says so explicitly.
Where a decision was forced by a measurement, the measurement is recorded.

The ordering rule applied throughout, declared in the brief: scientific
validity → causal correctness → reproducibility → simplicity → long-term AAA
usefulness → performance.

---

## D-1. Start from `main`, not from the PR #12 branch

**Decision.** Base the phase on `main` at the phase-closure merge commit.

**Context.** The working tree was checked out on `opus/aaa-playground-tiny-nn`,
the closed PR #12 branch, which diverged before the benchmark-v2.1 integration
and before the observation-noise phase. Its tree lacks `aaa/benchmark/`,
`aaa/noise/`, `docs/final_audit.md`, `docs/phase_closure_decisions.md` and
`docs/sol_review.md` entirely.

**Rejected.** Continuing from the PR #12 branch, which would have silently
reverted two merged phases.

**Note for the reviewer.** The phase brief's reading list named several files
that do not exist on the PR #12 branch. They all exist on `main`. The brief's
file list is accurate for `main` except for `docs/handoff_astra.md`, which is
`docs/handoff_sol.md` on current `main`.

---

## D-2. A separate `research/aaa_1k/` package, not an extension of `aaa/`

**Decision.** Put the phase in `research/aaa_1k/`, importing only the stable
shared parts of `aaa/` (`config`, `environment`, `predictors`).

**Reason.** `aaa/` carries a completed, frozen scientific phase with a
committed freeze manifest. Mixing a new exploratory phase into those modules
would make every future v2.1 reproduction question which code produced which
evidence. Nothing in `aaa/` is modified by this phase.

**Consequence accepted.** The repository-wide v2.1 and observation-noise
fingerprints cover *every* tracked file, so adding this package changes both.
See D-9.

---

## D-3. Input 3 is normalized by `displacement_scale`, not by `interval_width`

**Decision.** Normalize the previous signed prediction error by
`displacement_scale = dt * 0.2 = 0.004`, deviating from the brief's
`/ interval_width`.

**Measurement that forced it.** With `/ L` the input carries values around
0.003 against a centered position of order 0.5 and a velocity feature of order
1. A development probe across three seeds and two families found the
`zero_error_input` ablation **identical to the full model to five decimal
places** (occlusion: 0.00525 vs 0.00525; coarse: 0.00376 vs 0.00376). The
mechanism the brief asked to be tested could not have been measured either way.

After rescaling, the same probe separates them (occlusion at lr=0.01: 0.00356
full vs 0.00387 ablated).

**Reason.** The brief says "conceptually" and asks for "the same kind of public
normalization discipline already established in AAA". The prediction error is a
displacement-like quantity; input 2 is a displacement normalized by
`displacement_scale`. An input that is numerically inert is not a test of a
mechanism, and scientific validity outranks literal compliance.

---

## D-4. Output 1 is normalized by `displacement_scale`, not by `interval_width`

**Decision.** The primary output is a displacement in units of
`displacement_scale`; the evaluator converts to a position identically for
every arm.

**Reason.** The v2.1 RLS learner normalizes its target by `L`, which is fine
for an exact-fit recursive estimator but gives SGD targets of order 0.004 and
correspondingly tiny gradients. Normalizing by `displacement_scale` puts
targets at order 1 without changing what is predicted. All *reported* metrics
remain in AAA's canonical `/L` normalized units, so every comparison is on the
project's existing scale.

---

## D-5. Unobserved steps are held at the last observed position

**Decision.** During a gap the feature pipeline presents the last observed
position unchanged; input 2 and input 3 are then exactly zero.

**Rejected.** Rolling the agent's own prediction forward as its "known"
position. That would let a *stateless* model keep extrapolating through a gap
using its own output, which would make the occlusion family unable to
distinguish a model with memory from one without — destroying the only family
built to test recurrence.

**Consequence accepted and declared.** The all-zero input pair is the
missingness code. It is a weak code: a genuine zero displacement and a genuine
zero error look the same. Fixing it properly needs a fourth input, which would
change the frozen 994-parameter architecture, so it is recorded as a
limitation instead.

---

## D-6. Learning only from genuine one-step transitions

**Decision.** A learner updates only when both ends of a transition were shown
to it. Applied identically to the neural arms and the RLS arm.

**Reason.** A revealed position following a gap is a multi-step displacement
from the held input position, not a sample of the one-step law. Fitting it
teaches the model something false. This is `AAA-120`'s lesson — a window whose
displacement feature is not a sample of the law damages a continuously updating
learner — applied to a different cause.

---

## D-7. `coarse_speed_v1` replaced a subsampled-observation design

**Decision.** The third family quantizes observations to a coarse public grid
rather than showing only every other step.

**Defect in the original design, caught before any evaluation stream ran.**
With `observe_every = 2`, no two consecutive steps are ever observed after the
warm-up, so under D-6 the learner would have trained **zero** times on that
family. The benchmark would have run, produced numbers, and measured nothing.

**Reason for the replacement.** Quantization at 0.005 against per-step motion
of 0.0024–0.0060 makes a single transition a poor estimate of the current
speed while leaving every step a legitimate training target. What is hidden is
the *precision* of the observation rather than its presence.

**Boundary.** This is a memory task, not an observation-noise study. The
quantizer is deterministic and public. `aaa.observation_noise` is a separate
phase with its own protocol and is not touched here.

---

## D-8. The stability-margin rule uses an unclipped divergence probe

**Decision.** Add a stage-0 probe that runs the learning-rate grid with
gradient clipping *disabled*, and require the selected rate to sit at least two
grid positions below the lowest rate that diverged.

**Defect this fixed.** The rule as first written asked only that the next
higher rate in the *clipped* grid also be stable. With clipping at 1.0 nothing
in the declared grid ever diverges, so the rule was vacuous — a gate that
cannot fail, which is precisely the disease the v2.1 repair was about.

**Measurement.** Unclipped, the boundary is at `lr = 0.3`; `lr = 0.1` produced
a worst-case transient normalized error of 0.9, nearly the whole interval.

**Cost, paid and reported.** The rule eliminated the four best-performing
configurations. Development error rose from 1.138e-03 (`lr=0.1`) to 1.434e-03
(`lr=0.03`), a 26% penalty. The rule was declared before the numbers existed,
and a rule that never binds is not a rule.

---

## D-9. Phase-scoped identity, and no exclusion to preserve an old hash

**Decision.** Give AAA-1K its own fingerprint over its own source, the shared
`aaa/` modules it executes, its tests, its protocol documents and the
dependency lock. Leave both repository-wide fingerprints exactly as they are.

**Explicitly rejected.** Adding `research/` to the excluded prefixes in
`aaa/benchmark/source_identity.py` or `aaa/noise/scientific_identity.py` so
that the committed v2.1 and observation-noise freezes keep matching. That is a
convenience exclusion, and the brief forbids it for good reason.

**Consequence, stated rather than hidden.** After this phase the working tree
no longer reproduces the fingerprints recorded in
`benchmarks/freeze_manifest.json` and
`benchmarks/observation_noise_source_freeze.json`. That is correct behaviour:
those manifests describe the trees that produced their evidence, and those
trees are reachable at the annotated tags
`aaa-pre-next-phase-2026-09-19` and `aaa-pre-next-phase-closure-2026-09-19`.
Historical experiments stay reproducible from their historical checkout; new
experiments stay reproducible from the new phase identity. One hash is not
asked to describe two different worlds. Tracked as `AAA-152`.

---

## D-10. Plain SGD retained

**Decision.** Keep plain SGD. Do not evaluate Adam or RMSProp.

**Reason.** No development configuration inside the declared stability margin
was unstable or unusably slow, so the brief's condition for considering a more
sophisticated optimizer was never met. Optimizer state is separately counted
and is zero; a momentum or second-moment optimizer would double or triple the
model's adaptive footprint for a model whose premise is that its state is small
enough to inspect.

---

## D-11. `lambda_error = 0.25` retained

**Decision.** Keep the predeclared auxiliary weight.

**Evidence.** The declared ablation over `{0.0, 0.1, 0.25, 0.5}` at the
selected `(lr, T)` gave 1.430e-03, 1.431e-03, 1.434e-03 and 1.441e-03. Turning
the auxiliary head off entirely is 0.3% better on the primary objective —
inside the predeclared 2% practical margin, so the default stands.

**Worth stating plainly:** the self-error head costs essentially nothing on the
primary predictor. That is a useful result in its own right, and it is the
answer to "does the uncertainty objective harm the predictor" being *no*
rather than being unmeasured.

---

## D-12. The TBPTT horizon is reported as having almost no effect

**Decision.** Select `T = 4` and say why the choice barely matters.

**Measurement.** Across `T` in `{4, 8, 16, 32}` at the selected learning rate,
development error was identical to four decimal places. A direct probe of the
gradient's dependence on depth found the norm changing by 7% between `T = 1`
and `T = 4` and by under 2% from `T = 4` to `T = 32`: the update gate sits near
0.5, so hidden-state influence decays by roughly half per step and there is no
long-range credit to assign on these streams.

**Consequence.** `T = 4` was chosen by the declared tie-break, and the honest
reading is that this benchmark does not exercise temporal credit assignment.
A future AAA benchmark that genuinely requires long-range dependence is the
prerequisite for any UORO or RTRL work.

---

## D-13. Replication bounded below the declared precision target

**Decision.** Run 32 replicas per family and report that the precision
objective was **not met**.

**Measurement.** The development pilot measured a primary effect of 7.13e-05
with a per-stream spread of 2.22e-04. Resolving a quarter of that effect at 95%
needs 599 replicas per family; the declared bound of 32 was applied.

**Reason for bounding rather than running 599.** 599 replicas across six
families is roughly 3,600 streams and about 25 minutes of CPU — affordable, and
that is exactly why the bound exists. The pilot effect is one particular paired
comparison on development data; sizing an evaluation budget from it to three
significant figures would be letting pilot noise choose the design. The bound
was declared before the pilot ran.

**Consequence.** Every interval in the report is wider than the design asked
for. Effects near zero are unresolved rather than absent, and the report says
so at the top of the capability vector.

---

## D-14. No interactive key bindings on the visualization

**Decision.** The dashboard renders to a file. There is no interaction loop and
no key handler.

**Reason.** PR #12 discovered that Matplotlib binds its own keys to unrelated
actions, and the fix required releasing the default bindings. The cheapest way
not to reproduce that defect is not to add the feature. The dashboard's job is
to show real model state, not to be a product.

---

## D-15. Round 2 rather than a patched round 1

**Decision.** Retain round 1 unchanged and run a second evaluation round on
fresh stream identities, rather than re-analysing or re-running round 1.

**Reason.** The four repairs change what is measured (Q2, Q5), how uncertainty
is estimated (the crossed bootstrap) and what the controls run on (per-
architecture hyperparameters). Applying new designs to already-observed streams
would consume evaluation identities twice, which §22 of the phase brief forbids
for exactly this reason. Round 2 draws from declared fresh offsets in the
`evaluation_env` namespace: 10,000 for streams, 20,000 for adaptation trials and
30,000 for retention trials.

**Evidence impact.** Round 1 is `superseded`, not deleted, and is retained with
its report at `docs/evidence/aaa_1k_evaluation_round1_superseded.json` and
`docs/aaa_1k_report_round1_superseded.md`. Its rendered report carries a banner
naming its four defects.

---

## D-16. The gradient clip became a selected hyperparameter

**Decision.** Add stage 3 to the development selection and choose the clip
threshold by rule.

**Measurement that forced it.** The originally declared threshold of 1.0
activated on 24% of updates and cost 29% of development error against the best
stable alternative. A mechanism that active is deciding what gets learned.

**Rule.** The most conservative threshold whose development error is within the
2% practical margin of the best stable one — so given two thresholds that
perform the same, the tighter one wins, because its cost is bounded and its
benefit is insurance. Selected: 10.0 for the gated model, activating on under
1% of updates.

**Rejected.** Leaving the clip at 1.0 and reporting the cost as a limitation.
An asserted hyperparameter that a probe shows is load-bearing is a defect, not
a limitation.

---

## D-17. Hyperparameters are selected per architecture, ablations are not

**Decision.** The learning rate and clip are selected separately for the gated
model, the stateless control and the ungated control. The three ablations of the
gated model share its configuration exactly.

**Measurement that forced it.** Applying the gated model's selected clip to
every arm destabilized the stateless control on `coarse_speed_v1` — mean error
1.6e-01 against 2.2e-03 — and inflated the reported hidden-state advantage to
thirty times its true value. That number would have been published as "hidden
state helps, decisively".

**Reason for the asymmetry.** A *control* is a different architecture, and
imposing another architecture's hyperparameters on it turns a comparison into a
handicap. An *ablation* is the same architecture with one mechanism removed, and
must differ in exactly that mechanism or it stops being an ablation.

**Consequence.** Q4 moved from `NEGATIVE` to `INCONCLUSIVE`. The round-1 finding
that gating loses did not survive a fair comparison, and the reason the ungated
arm looked good is that it was being run at a learning rate its own stability
rule forbids.

---

## D-18. No replay, because there is no forgetting to target

**Decision.** Do not add experience replay, now that retention is measured
properly.

**Measurement.** Probe-bank error on regime A *fell* while the model trained on
regime B: −3.74e-04 [−7.99e-04, −9.52e-07].

**Reason.** Round 1 deferred replay because forgetting had not been measured.
Round 2 measured it and found none on this benchmark. Adding a replay buffer now
would be a mechanism with no failure mode to fix, and would have to be justified
by measured inadequacy rather than by the literature recommending it. Rolnick et
al.'s CLEAR remains the right reference *when* a benchmark produces forgetting;
this one does not.

