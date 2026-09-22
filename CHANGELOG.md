# Research history

## Independent repository review (2026-09-22)

Engineering, automation and documentation consistency only. No model, target
rule, threshold, estimand, gate, specification, seed mapping, identity,
retained evidence or scientific claim was changed. The `aaa.1k.v1`
fingerprint (`5ce6e019…`) and the v2.1 specification hash are unchanged.

- `champion1 verify` now recomputes the live phase fingerprint instead of
  comparing Champion 1's record with a copy of itself (`AAA-174`).
- Dependabot no longer opens `pip` pull requests. `requirements-lock.txt` is
  inside the phase fingerprint, and the old rule ignored only patch updates
  (`AAA-175`).
- Active loop documentation now says the independent review completed and
  `aaa.loop.v1` is current, and the README lists the PR #20 review record.
  Historical records are not relabelled (`AAA-176`).
- Stale ledger statuses are updated with dated notes. `AAA-149` and `AAA-150`
  are verified in hosted CI. Nine observation-noise entries now say
  confirmation was not executed, instead of "pending", because the v1.1 batches
  were retired unobserved (`AAA-177`).

## Development hardware and compute strategy documented (2026-09-22)

Documentation and project governance only. No implementation, protocol,
threshold, identity, evidence file or scientific claim was changed.

- Added [`docs/hardware.md`](docs/hardware.md): FLOWBOX, the current primary
  local development workstation, with its CPU, GPU, RAM, storage roles and OS.
- Recorded the boundary explicitly: FLOWBOX contains a discrete GPU, the
  current validated AAA implementation remains CPU-only, and no retained
  evidence is GPU-generated. GPU support would be separate future engineering
  with its own numerical reproduction standard.
- Recorded the owner's current planning ceiling of approximately 125 million
  trainable parameters for this FLOWBOX configuration. It is a project
  constraint, not a scientific result, a target, or a guarantee of
  trainability, and it is deliberately not encoded as a gate or test.
  *Complexity must earn its keep* remains the operative discipline.
- Recorded that FLOWBOX is expected to receive future upgrades, and that a
  dedicated server for running Juniper models is planned but does not exist
  and has no frozen specification.
- Linked the new document from the README, reproduction and dependency docs
  without replacing per-experiment hardware provenance, and regenerated the
  tracked-file audit inventory.
- `docs/aaa_charter.md` is deliberately unchanged: it is inside the frozen
  `aaa.1k.v1` phase fingerprint that the Champion 0 and Champion 1 records
  cite, so the charter cross-reference is made from `docs/hardware.md`
  instead. The reason is recorded there.

## aaa.loop.v1 — independent review and governance transition (2026-09-22)

- Independently verified iteration 0006's freeze, fresh identities,
  confirmation primitives, decision, Champion 1 record, round-3 replay and
  platform-scoped reproduction policy.
- Promoted only the current governance version. Every iteration-0001--0006
  artifact remains immutable and labelled `aaa.loop.v0-pilot`.
- Reconciled current protocol, reproduction, registry, evidence-policy and
  tracked-file-audit documentation after the first real outer cycle.

## Loop iterations 0004–0006 — audit response and Champion 1 (2026-09-22)

Answers the first four steps of the 2026-09-21 external research audit.

- **R-02 online TBPTT.** The live rule backpropagates stale activations
  through current matrices, so it is not the "exact realized-trajectory"
  gradient `model.py` claims (`AAA-169`). It is within 7×10⁻⁵ of that gradient,
  and M2 is identical under every rule, including T = 1.
- **R-01 M2's mechanism.** Four predeclared diagnosis rounds on fresh blocks
  falsified the pilot's loop-gain hypothesis and single-step overshoot. They
  established a self-confirming target-unfolding frame lock (54/54 diverged
  cells; `AAA-170`).
- **Prospective rules.** The pilot's seven gaps became hard rules, and
  stability became the primary gate. Every criterion is uncertainty-aware and
  three-valued (R-04). The candidate budget was precommitted: c7 and c8
  (0004) and c9 (0005) were REJECTED at the screen.
- **R-03 the first real outer path.** c10 (0006) passed screen and attack, was
  frozen, durably claimed on the remote, confirmed fresh (long-coarse
  divergence 21.3% → 0%; bitwise identical on every standard family),
  recomputed independently, and PROMOTED.
- **Champion 1** is Champion 0 plus c10: the same 994 parameters, 1414 state
  scalars and phase fingerprint; `research/aaa_1k` is untouched. Its round-3
  capability vector is measured by re-running round 3. Only round 3's two
  burst cells change (−56%, −62%; now confirmed as M2), and only one Q7
  verdict flips (an improvement).
- **Attempt 1 of the confirmation aborted before observation** on an admission
  defect (`AAA-171`). The identities were burned, not reused. The defect is
  fixed with an end-to-end outer-path test.
- **Open:** `AAA-172` (the v2.1 RLS learner uses the same own-prediction
  unfolding), M1, and audit items R-05 to R-11. R-12 is satisfied for this
  work; R-13 and R-14 are closed by the independent review and refreshed
  tracked-file audit.

## PR #19 independent final review remediation (2026-09-21)

- Repaired confirmation freeze provenance so every frozen scientific byte and
  executable bit must exist in the recorded commit; the complete loop package,
  including CLI admission, is now frozen (`AAA-165`).
- Replaced recoverable local-only confirmation consumption with atomic,
  immutable remote Git ownership before observation (`AAA-166`).
- Hardened iteration records against unsafe paths, symlinks, malformed hashes,
  duplicate/contradictory artifacts and duplicate candidate IDs (`AAA-167`).
- Weakened H13/M1 to partial mechanistic support without altering observed
  evidence, and fixed prospective pilot-gap rules without rewriting L-4
  (`AAA-168`). Champion 0 and all eight rejected candidates remain unchanged;
  no confirmation identity was reserved or spent.

## aaa.loop.v0-pilot — Birth of The Loop (2026-09-21)

A pilot of AAA's iterative-improvement loop — observe, classify, diagnose,
hypothesize, falsify, intervene minimally, attack, confirm fresh, decide,
preserve — run on AAA-1K from its reviewed round-3 state. It lives in
`research/aaa_1k_loop/`, outside the AAA-1K fingerprint, so Champion 0 is
byte-for-byte the reviewed model. Nothing in `research/aaa_1k/`, `aaa/`,
`results/`, or any AAA-1K evidence or report changed.

**It said no, eight times.** Three iterations tried six model candidates and
two corrected claims, every one parameter-neutral. All were rejected by rules
committed before their data existed: five at development screening, three at
attack. No freeze was written and no confirmation identity was spent, so the
outer promotion loop exists only as code tested by failure injection. That is
the pilot's main limitation, stated rather than worked around.

**What it found about AAA-1K.** Q4's "minority tail" is one whole family,
`coarse_speed_v1`, where zero gate biases leave the gated core without a
sign-alternating mode to track a quantized phase (supported on fresh
diagnostic streams; a keep-gate bias of −2 closes 73% of the gap without a
parameter, and trades away occlusion). A previously unmeasured long-horizon
runaway through the previous-error input affects about a third of long
fixed-speed coarse runs. The documented claim that `coarse_speed_v1` measures
regime inference only is unsupported (`AAA-162`). Capacity is not the
bottleneck: a 354-parameter ungated RNN beats the 994-parameter champion on
the family studied.

**What it found about itself.** Classification redirected the work before any
model change; attack found an initialization-dependent failure development had
missed; a reproduction check caught a labelling defect in the loop's own code
(`AAA-164`). Seven protocol gaps are recorded before the protocol may become
permanent. See [`docs/loop_pilot_report.md`](docs/loop_pilot_report.md).

## aaa.1k.v1 — a recurrent predictive seed (2026-09-19)

A new versioned research phase in `research/aaa_1k/`, isolated from the frozen
v2.1 core and the observation-noise phase. Nothing under `aaa/`, `benchmarks/`
or `results/` is modified, and no historical evidence is regenerated.

**A 994-parameter GRU, written out by hand.** Three inputs, sixteen hidden
units, two outputs, one bias vector per gate. No framework: at this size a
framework costs transparency and buys nothing. Every parameter of every model
is finite-difference verified against a *sequence* loss, exhaustively, to a
maximum absolute error of 8.7e-10 — because a recurrent model can have a
perfect one-step gradient and a broken temporal one, and an injected
single-step backward pass is shown to be caught.

**The optimizer holds nothing.** Plain SGD, zero optimizer state, counted and
reported separately from the 16 hidden scalars and the 404-scalar truncation
buffer. A "1K model" with a hidden 100K of optimizer state would deserve
ridicule.

**Mechanisms, not architectures.** Three ablations of the model itself — state
reset, previous-error input removed, recurrent weights frozen — plus a
982-parameter stateless MLP and a 954-parameter ungated RNN, all sharing one
initialization seed so each differs in exactly one mechanism. The design is
adapted from Foucault & Meyniel (2021), whose conclusion it then contradicts.

**Two new benchmark families that recurrence could actually matter on**, since
a GRU losing to a closed-form extrapolator on smooth motion teaches very
little: `occlusion_v1` (periodic observation gaps; during a gap the displacement
and error inputs are exactly zero for every arm) and `coarse_speed_v1` (a hidden
speed regime behind a coarse deterministic quantizer). Plus `aba_v1`, one
unlabelled A→B→A stream. A dead-reckoning baseline that carries velocity
perfectly and for free was added so that "the network remembered the velocity"
has to beat something that also remembered it.

**The results, including the ones that are not flattering.** The model learns
online on every family. Persistent hidden state helps, most clearly on steps
whose target it never saw. **Gating's predictive benefit is inconclusive**
under the corrected fair comparison: the earlier apparent ungated win did not
survive per-architecture hyperparameter selection, while gating separately
shows a stability advantage. The model loses to reflected constant motion,
dead reckoning and the existing three-parameter RLS learner on smooth fully
observed motion. Its self-error head is weakly informative and over-predicts.

**A gate that fired.** The stability-margin rule was rewritten mid-phase after
it was found to be vacuous — with the declared gradient clip in place, nothing
in the search grid ever diverges, so "require the next higher rate to be
stable" could not fail. A stage-0 probe with clipping disabled located a real
boundary at `lr = 0.3`, and the repaired rule then **eliminated the four best
development configurations**, costing 26% of development accuracy. The rule was
declared before the numbers existed.

**The evaluation ran three times, and round 3 is current.** Round 1 was
completed, probed, and found to contain four design defects. All four were
repaired and re-measured on fresh stream identities; round 1 is retained,
superseded, with its report and a banner naming what was wrong. Independent
review then superseded round 2 because Q1 did not identify weight learning and
Q2/Q5 flattened reused initialization identities.

**Q2 did not measure adaptation.** Branching an online/frozen pair where
*nothing happens* reproduced 95% of the effect, so the result was mostly about
continuing to learn. Repaired with a `paired_change_v1` family that emits two
streams bit-identical until a declared step, after which one changes and one
does not; the estimator is the difference of the two advantages, so the ordinary
benefit of learning cancels. Round 3 measures
`+5.12e-04 [+1.69e-04, +8.67e-04]`, 43% of the combined adaptation plus
ordinary continued-learning advantage (`AAA-153`, `AAA-160`).

**Q5 did not measure retention.** The A→B→A comparison confounded "came back
intact" with "had three times the experience". Repaired with a fixed bank of
held-out regime-A episodes, never trained on, evaluated by a frozen clone at
every segment boundary. Round 3 measures no statistically resolved forgetting:
`-2.02e-04 [-6.42e-04, +1.26e-04]`. This is not general retention immunity and
no replay was added (`AAA-154`, `AAA-160`).

**The intervals were too narrow.** They resampled streams while treating the
starting weights as fixed by nature. Repaired with five initializations and a
crossed bootstrap over both factors, plus achieved precision reported against
the effect that was measured rather than a pilot-sized target (`AAA-155`).

**The controls were handicapped, and it nearly produced a false headline.** The
gradient clip was asserted rather than selected; a probe found it firing on 24%
of updates and costing 29% of development error. Making it a selected
hyperparameter was right — applying the *gated model's* threshold to every arm
was not. The stateless control destabilized on one family, reaching a mean error
of `1.6e-01` against the gated model's `2.2e-03`, and the comparison reported a
hidden-state advantage thirty times too large. Repaired by selecting the
learning rate and clip separately per architecture, while ablations keep sharing
the primary's configuration exactly. With fair hyperparameters the hidden-state
effect is `+2.49e-04` and **Q4 moves from `NEGATIVE` to `INCONCLUSIVE`**: the
finding that gating loses did not survive giving the ungated control a fair
learning rate (`AAA-156`).

**Gating buys stability, which round 1 never noticed.** Unclipped, the ungated
control diverges at a learning rate of 0.1 where the gated model survives to
0.3, so the same stability margin allows it only a third of the rate.

**Three open questions closed by development probes.** `coarse_speed_v1` really
does measure hidden-regime inference — with the speed fixed the recurrent model
is *worse* than the stateless control. The ungated control's win was largely a
shared-learning-rate artefact. The clip rate was not masking a problem; it was
the problem.

Also: disjoint hashed seed namespaces; a paired stream-level bootstrap that
never resamples steps inside a trajectory; an error-head calibration analysis
kept separate from prediction accuracy; a capability vector with no combined
score; phase-scoped non-self-referential source identity with lineage metadata
(`parent_model_id: null` — this is the lineage root); a read-only Matplotlib
dashboard showing all sixteen real hidden activations and both gate vectors,
with no interactive key bindings, because PR #12 already found what Matplotlib
does with those; extensive deliberate failure-injection tests; and mypy,
coverage and packaging extended to cover `research/`.

Adding this phase necessarily changes both repository-wide source fingerprints.
No exclusion was added to prevent that; see `AAA-152`.

## Observation-noise engineering phase closure (2026-09-19)

The current engineering line fixes the hosted artifact filename defect with a
portable full-SHA-256 schedule storage ID; authoritative trial identity remains
in structured metadata and the schedule index. Formal runs now refuse to reserve
a batch unless their output filesystem has at least 100 GB available.

The dedicated Cinqic HDD was directly qualified as an ext4 working filesystem.
A retained 58,944-record sharded run occupied 47,307,645 bytes, completed in
113.16 seconds with 150,844 KiB peak RSS, and independently recomputed in 24.66
seconds. Linear scaling projects about 31.4 wall hours and 47.4 GB per formal
batch, roughly 64 CPU hours plus 13.7 recomputation hours for A+B, and about
447,000 files per batch. Quick-scale creation, traversal, resume, checksum, and
cleanup measurements did not justify introducing a new packed-container format.

Formal A/B was deliberately not executed. Local capacity is no longer the
blocker, but the fixed replication budget lacks a quantitative precision basis.
The v1.1 A/B identities were retired unobserved so they cannot be reused. The
scientific outcome is `NOT EXECUTED`; external immutable archival is
`NOT VERIFIED` and is not a gate for this internal engineering closure. Any
successor experiment must be versioned, precision-justified, frozen, and given
fresh batch identities before observation.

The FontTools 4.65.0 lock update was incorporated before the final source
freeze. The Playground/tiny-NN PR and archive/future-design draft were retained
as exploratory PR history rather than added to the maintained scientific tree.

## Observation-noise v1.1 independent repair boundary (2026-09-14)

Sol independently reproduced fifteen observation-noise review findings
(`AAA-135` through `AAA-149`) before repairing the implementation. The phase is
now explicitly versioned `aaa.observation_noise.v1.1`; v1 is retained as an
unobserved superseded protocol because its adaptation prose and machine
operator disagreed at the exact 10% boundary.

The corrected full 2 x 2 x 1 development selection retained and independently
verified four 235,776-record candidate archives. None of the three innovation
clipping refinements met every preregistered constraint, so the unchanged
`incumbent-no-refinement-v1` control remains selected. This is a development
decision, not confirmation evidence.

Record production and verification are bounded-memory and use deterministic
per-trial gzip shards. A measured quick pilot reduced peak RSS from about 2.67
GiB to 148 MiB and retained one primitive copy with exact timing-neutral
semantic equivalence. Formal admission now requires a committed freeze, fixed
4,000 draws, frozen batch identities, and an atomic append-only remote claim.
The verifier requires complete checksums and rejects non-finite values,
incomplete cells, unsafe paths, symlinks, ambiguous record layouts, and
tampered schedules or statistics.

Formal A/B remains deliberately unobserved. One batch is projected at 38.5 GB
of compressed records plus schedules and metadata, and no approved immutable
archive/retrieval destination exists. The phase is therefore `BLOCKED`, not
negative or inconclusive, and PR #11 is not approved for merge at this boundary.

## Observation-noise v1 completion boundary (2026-09-14, superseded)

The observation-noise phase now has a non-self-referential scientific source
fingerprint. It hashes the tracked scientific source map, rejects untracked
scientific files and symlinks, normalizes only mutable batch lifecycle fields,
and excludes generated result/freeze/review artifacts. Committing the exact
freeze therefore cannot invalidate its own recorded identity; Git commit and
tree values remain provenance only.

Development refinement is bounded by the declared 12-configuration/3-mechanism
budget. Four immutable candidate identities were evaluated on the committed
development plan. All three innovation-clipping refinements were rejected on
the preregistered constraints, so `incumbent-no-refinement-v1` is the selected
candidate. No A/B conclusion follows from this development result.

The joint confirmation evaluator is present as
`observation-noise-confirmation-evaluate <A> <B>`. It independently verifies
both archives, requires shared candidate/fingerprint/training compatibility,
reconstructs absolute utility, baseline competitiveness, first-50 adaptation,
and unchanged-law retention claims, and applies one Holm family across A+B.
No confirmation freeze or A/B archive was executed in this completion boundary.

Dates are the dates of the work, not of any release. Nothing here is a product
announcement.

## Observation-noise v1 design freeze (2026-09-13)

Added the separately versioned `aaa.observation_noise.v1` protocol and a
controlled sensor channel for next-position prediction. The implementation
keeps latent truth and noisy observations separate, caches one schedule sample
per timestamp, updates only from the revealed noisy observation, records full
primitive evidence, and independently recomputes its arithmetic. Development
smoke output is explicitly inconclusive; no refinement or capability promotion
is claimed.

## v2.1 — benchmark protocol repair (2026-09-10 / 2026-09-11)

A full engineering and scientific-methodology repair following independent
reviews by GPT-6 Astra, Claude Opus 5 and GPT-5.6 Sol. Every finding was
reproduced against the pre-repair tree before anything was changed; the probe
output is committed at `docs/evidence/pre_repair_probes.json` and every item is
tracked in `docs/issue_ledger.md`.

**The benchmark can now say no.** Formal confirmation returns a non-zero exit
status when any required gate is not `PASS`, including `NOT_VERIFIED` and
`INSUFFICIENT_EVIDENCE`. Absence of evidence is never converted into success.

**Gates measure what they are named after.** `straight_learning` — which a
parameterless analytic rule satisfied — is replaced by three separate claims.
The bounce criterion no longer compares a candidate that may reflect against a
baseline that may not; development measurement showed essentially the entire
reported margin was the reflection transform.

**The specification is executable.** 26 declared values that the runner never
read are now consumed, under a typed strict schema, with a test that fails if
any declared leaf stops being used.

**The learner is numerically sound.** Covariance-form RLS with forgetting lost
positive semidefiniteness at update 323 and overflowed at 6,642 on weakly
exciting input. It is replaced by a square-root formulation with trace-bounded
forgetting, verified against an independent batch least-squares reference.
Invalid state is refused rather than silently repaired.

**Statistics match their estimands.** A genuinely hierarchical paired bootstrap
that recomputes the gate's own statistic on every draw, with declared event
weighting, no-event episodes excluded from event resampling, intervals on
non-regression claims, Holm-Bonferroni multiplicity that counts repeated
confirmation attempts, and missing intervals reported rather than assumed.

**Confirmation discipline is mechanical.** Predeclared immutable batch
identities that *are* the seed input, a batch registry with a lifecycle, a
freeze manifest checked before a confirmation runs, a clean-source-tree
requirement, and invariants enforced by the runner rather than only the CLI.

**Evidence is auditable.** Versioned artifact schemas, an experiment registry
with safe resume, and a `recompute` command that rebuilds every metric and gate
from retained raw evidence without retraining or re-simulating.

Also: recovery redefined from peak shock with predictor-specific references and
a six-way status partition; one normalized-unit contract; multi-wall bounce
accounting; a headless, testable animation with checkpoint-type dispatch; plots
through the object-oriented Matplotlib API with bounds-derived axes; a real
learning curve on a fixed frozen probe bank; an always-online deployment track;
in-domain training with extrapolation measured separately; formatter, linter,
type checker and coverage with no blanket ignores; and CI that installs the
lock file and proves it.

**The repaired benchmark said no, and that is the headline.** The first v2.1
confirmation round failed: confirmation A and confirmation B, on independent
fresh streams, both failed the required `always_online_stability` gate and both
exited non-zero. Thirteen of fourteen gates passed in each. The failure was
traced on development data to a precise mechanism — a continuously updating
instance is damaged by the one window after each wall contact whose
displacement feature is a folded difference — and repaired in the candidate.
Both batches were retired permanently, no threshold was touched, and the failed
attempts are committed. A first attempt at the repair did not work and is
recorded too. See `AAA-120` in `docs/issue_ledger.md`.

Round 2, with the repaired candidate and newly declared batches against a new
specification hash, is recorded alongside it. Both rounds are in
`benchmarks/confirmation_batches.json` and `results/benchmark_v2_1/`.

Historical v1 and v2 evidence is preserved. The v2 confirmations are
reclassified as historical/provisional under a superseded methodology; see
`docs/errata.md`.

## v2 — benchmark introduction (2026-09-09)

Introduced a versioned benchmark separate from the v1 snapshot, with four
families, compressed step-level evidence, checkpoint lineage and gate
decisions. Superseded; see `docs/errata.md` for which of its claims were
affected.

## v1 — initial prototype (2026-09-09)

The original moving-dot evaluation with an online linear SGD learner over
absolute-position features. The result was unfavourable and is preserved
unchanged in `results/final/`: the learner did not convincingly improve with
experience, generalized poorly when frozen, and lost to constant motion.
Continued updates did improve it relative to its matched frozen copy after a
change, but it remained inferior to the analytic baseline. The learner is
retained as the `legacy_linear_sgd` diagnostic arm and reproduces that
behaviour under the repaired protocol.
