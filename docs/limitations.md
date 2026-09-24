# Research limitations

Stated plainly, because a benchmark that cannot say what it does not show is
not measuring much.

## Current phase: `aaa.python.v0`

- **No Python capability is demonstrated.** On retained development evidence
  the minimal learner beats chance, beats its memory-disabled control in four
  families and never beats the majority baseline. Surface heuristics beat it
  on syntax and localization. Online updating is not resolved better than the
  frozen twin in any family and is resolved worse on `output`
  ([report](aaa_python_development_report.md)).
- **Adaptation and retention are unresolved**, not absent. Three
  initializations x four streams give wide intervals.
- **The environment is small by design.** A frozen subset (no imports,
  attributes, `while`, strings beyond short literals, recursion, `**` or `/`),
  generated programs of at most 40 lines, and five task families. Nothing here
  speaks to real code, libraries, repositories or free-form generation.
- **Tasks were designed by the same implementer as the learner.** Two
  construction cues were found and removed during development (a fixed fault
  variable name, and a single risky line per program). Others may remain; the
  heuristic baselines exist to expose them, and they still do on syntax and
  localization.
- **Syntax validity is largely a surface property**, and a lint heuristic
  scores 0.97. That family measures whether a learner can match an easy
  deterministic rule, not deep knowledge.
- **The repair family is solvable by tool use** (running the two visible
  tests), which v0 deliberately withholds. A tool-using baseline is the next
  rung.
- **The boundary is against accidental leakage**, not a sandbox against an
  in-process learner written to subvert Python. Programs, not learners, are
  sandboxed.
- **Sandbox limits depend on the platform.** CPU, address-space, file-size and
  process-count limits are applied and recorded where `resource` provides
  them. Elsewhere they are recorded as absent, never claimed.
- **Cross-version equality** of answer keys is tested on the golden sample of
  100 tasks on each CI interpreter. The full pools are regenerated and
  compared only on the interpreter that runs them.
- **Formal confirmation is NOT EXECUTED**, and cannot be: v0 declares no
  candidate, criteria or confirmation identities.

## Dot-era limitations

The rest of this document describes the moving-dot benchmark family, which is
retained as evidence and as regression and mechanistic benchmarks
([archive](dot_benchmark_archive.md)).

## The environment

One dimension. One moving dot. Deterministic, noiseless, fully observed
positions. Two dynamical laws: constant velocity with reflection, and a damped
harmonic oscillator. Both are analytically simple, and one of them —
constant-velocity motion with reflection — is *exactly* solved by a
parameterless baseline given the public boundary map. That is why the reflected
constant-motion baseline exists and why the bounce criterion is an absolute
accuracy requirement rather than a relative one.

## The learner

Three parameters. A linear map from `[1, scaled displacement, centered
position]` to the next normalized displacement. It has no explicit bounce
detector, no change detector that the evaluator informs, and no memory beyond
four positions and its own covariance state. It cannot represent a law outside
that family.

## What the results can and cannot support

These claims are kept separate, and none implies another:

1. the implementation runs correctly;
2. the model improves with experience;
3. what it learned generalizes to unfamiliar episodes;
4. continued updates help after a change;
5. it beats a baseline.

In particular:

- **Boundary handling is programmed, not learned.** Reflection is public
  knowledge of the observation format, supplied by the programmer.
- **The oscillator's pre-change dynamics are not held out.** The candidate is
  allowed to learn them from its own observations during the prefix. The
  post-change coefficients and the intervention are held out.
- **"Autonomous" means unattended.** The observe / predict / score / update loop
  runs without intervention after launch. It does not mean general
  intelligence, physical understanding, or independent goal formation, and
  nothing here is evidence for any of those.
- **Five training replicas is a routine engineering minimum**, labelled as such.
  Five checkpoints evaluated on 500 episodes is 500 episodes from five
  learners.

## What the benchmark has already refused

The first confirmation round under this protocol failed. Both fresh streams
failed `always_online_stability`: a continuously updating instance regressed
against the reflected constant-motion baseline, because it was fitting the one
window after each wall contact whose displacement feature is a folded
difference. That was repaired in the candidate, on development evidence, with
the failed attempts kept and no threshold touched.

Worth stating plainly: the thirteen gates that passed in that round include
every *frozen* track. The gate that failed is the only one that asks what
happens when the system is left running. That asymmetry is the most interesting
thing this version has measured.

## Known open items

- Raw per-step evidence is regenerable rather than durably archived
  (`AAA-077`).
- The always-online family rotates through a fixed sequence of the existing
  regimes; it is not an open-ended stream (`AAA-008`).

## Historical next-experiment rationale

Before the separate noise phase existed, controlled observation noise was the
next experiment proposed for benchmark v2.1. That rationale remains historical:
perfect observations make an analytic extrapolator unusually strong. The
experiment is now implemented as `aaa.observation_noise.v1.1`, but its formal
scientific result is not established.

## Observation-noise phase status

`aaa.observation_noise.v1.1` has a corrected full 2 x 2 x 1 development search
and a smaller smoke path. Together they demonstrate schedule generation,
causal noisy updates, matched branch construction, direct realized stratum
identities, primitive recomputation, bounded-memory sharding, uncertainty and
raw-data plots. They do not establish a supported operating envelope.
Confirmation A/B, adjusted uncertainty over the fixed formal replication plan,
durable full-archive publication and retrieval are not represented.

The earlier quick refinement smoke was not a valid completion of the frozen
2 x 2 x 1 development selection plan. The repaired full selection has now run:
all four candidate archives independently verified, no refinement had the
preregistered practical gain or adjusted positive evidence, and the unchanged
incumbent was retained as the explicit no-refinement control. This does not
establish that the incumbent meets any confirmation endpoint. The source
fingerprint is non-self-referential and fail-closed. The dedicated HDD
establishes local execution capacity, not immutability, off-site retention, or
an independent failure domain. External durable archival remains
`NOT VERIFIED`, but is not required for the current internal engineering phase.
The measured HDD profile projects about 31.4 wall hours per formal batch and
the replication count has no quantitative precision justification. The
declared v1.1 batches were therefore retired before observation. Full A/B
execution, joint Holm analysis, and durable retrieval are `NOT EXECUTED` or
`NOT VERIFIED`, never silently treated as passing. Sol's AI review is recorded
separately and does not claim independent human review.

The noise model corrupts observations only. It does not study process noise,
dropout, bias, irregular sampling, hidden state, actions, goals, language,
vision, or general intelligence. A favorable detector response to a sensor
shift is not evidence that a physical-law change was detected.

## AAA-1K (`aaa.1k.v1`)

The 994-parameter recurrent phase has its own limitations, and they are
different in kind from the v2.1 ones. Its evaluation ran three times. Rounds 1
and 2 are retained as superseded evidence; round 3 repairs Q1 identification
and Q2/Q5 initialization dependence on fresh identities.

**What round 3 shows.** Online weight updating beats a matched frozen copy.
Persistent recurrent
state beating both a state-reset ablation and a matched-capacity stateless
control, most clearly on steps whose target the agent never saw. Genuine
adaptation, isolated from ordinary continued learning by a
difference-of-differences against a bit-identical unchanged world, accounting
for about 43% of the combined advantage. No statistically resolved forgetting
on the fixed probe bank; the interval crosses zero.

**What it does not show.** Generalization beyond unseen trajectories of the same
four families; no unseen family was tested. Whether gating pays for itself:
round 1's negative result did not survive giving the ungated control its own
rule-selected learning rate, and the honest answer is now inconclusive overall
and negative on the memory families.

**The most dangerous thing that nearly happened.** Selecting a gradient-clip
threshold on the gated model and applying it to every arm destabilized the
stateless control on one family — mean error `1.6e-01` against the gated model's
`2.2e-03` — and inflated the reported hidden-state advantage to thirty times its
true value. It was caught by reading the per-arm table rather than the summary.
Hyperparameters are now selected per architecture (`AAA-156`).

**Where the evidence is thin.** Five initializations is a small second bootstrap
level. Q4's mean and median disagree in sign, so a minority of streams carries
the aggregate. The self-error head is weakly informative and over-predicts.

**Mechanisms that are programmed, not learned.** Boundary reflection, target
unfolding, the holding of unobserved steps, and the rule that a learner updates
only on transitions whose both ends it was shown. All four are public knowledge
of the observation format applied identically to every arm, and the
dead-reckoning and reflected constant-motion baselines exist so that none of
them is mistaken for a capability.

**Benchmark provenance, still the weakest point.** `occlusion_v1` and
`coarse_speed_v1` are new, designed by the same implementer whose model they
evaluate, and were not independently designed or preregistered before
evaluation. Independent Sol review later examined the implementation and added
characterization, but that does not remove the benchmark-origin limitation or
provide external replication. A decomposition probe was taken to establish
that `coarse_speed_v1` genuinely tests hidden-regime inference, because with
the speed held fixed the gated model is *worse* than the stateless control.
**That conclusion is unsupported (`AAA-162`):** the probe tested only the gated
model. With the ungated control, memory keeps most of its advantage at fixed
speed at the declared construction, and how much of the family is regime
inference depends on the speed-to-quantum ratio. What `coarse_speed_v1`
isolates is unresolved. `occlusion_v1` has no equivalent check.

**The most useful next experiments** were, in order: a distributional analysis
of the minority of streams where the ungated control wins large; an
independently designed memory benchmark; and a benchmark with genuine
long-range dependence. The first was done by the loop pilot below: the
"minority" is one whole family.

## What the loop pilot measured about AAA-1K (2026-09-21)

The improvement-loop pilot ([`loop_pilot_report.md`](loop_pilot_report.md))
promoted nothing; Champion 0 is unchanged. Its measurements were made on
diagnostic, development and attack identities, never on confirmation
evidence, so they are characterizations rather than confirmed claims.

- **Q4's negative sign is one family.** Recomputed from round-3 primitives,
  all 24 `coarse_speed_v1` streams favour the ungated control in every
  initialization, and without that family the Q4 aggregate is positive.
- **Why: the gated core's zero-bias operating point.** With `z = r = 0.5` the
  one-step Jacobian has no sign-alternating mode, so the champion cannot track
  the slow regime's sub-quantum phase within an episode and behaves like its
  own stateless ablation there. A keep-gate bias of −2 closes about 73% of the
  gap without a single extra parameter, but trades away occlusion accuracy
  (+4–5%) and fails on some unseen initializations. Capacity is not the
  bottleneck: a 354-parameter ungated RNN beats the 994-parameter champion on
  this family.
- **A long-horizon runaway (M2), since repaired in Champion 1.** On 1120-step
  quantized streams Champion 0's online learning diverges (mean error above
  twice persistence) in about a fifth to a third of cells. Iteration 0004
  showed the cause is a self-confirming target-unfolding frame lock (`AAA-170`),
  not the previous-error feedback gain the pilot suspected. Champion 1
  (iteration 0006) removes it on fresh confirmation (21.3% → 0%) and is
  bitwise identical to Champion 0 everywhere else tested. Round 3 never
  evaluated coarse streams longer than 280 steps, and its clip statistics
  ("under 1% of updates") describe that horizon only.
- **The previous-error input matters more than round 3 showed.** Removing it
  regresses `aba_v1` by 41% and `dynamics_change` by 39% in development.

## After iterations 0004–0006 (2026-09-22)

- **Champion 1** is Champion 0 with one target-construction rule changed
  (reach-gated unfolding). It lives in `research/aaa_1k_loop`, and
  `research/aaa_1k` (`aaa.1k.v1`) is unchanged. Its round-3 capability vector
  is measured, not inherited: it equals Champion 0's except for one Q7 verdict
  that improves.
- **Its M2 repair is tested at 1120 steps** on the quantized coarse, smooth
  bouncing and occlusion families and nearby quantized constructions. Other
  wall geometries, observation noise, missing observations next to a wall,
  and accelerations at a wall are untested. The gate trusts the tracker's
  one-step velocity estimate.
- **The online TBPTT rule is an approximation** (cached activations, current
  weights; `AAA-169`). It is numerically negligible here, but it is not the
  exact truncated gradient the model docstring claims.
- **The v2.1 core RLS learner** uses the same own-prediction unfolding and is
  untested for the lock (`AAA-172`). *Tested in `aaa.1k.v2`: no lock, no
  stall (see below).*
- **M1** (the Q4 memory-family deficit) is unchanged in Champion 1. *Diagnosed
  in `aaa.1k.v2` as a coarse-observation operating-point effect (see below).*

## After `aaa.1k.v2` (2026-09-23)

- **No 1K challenger was found, but the search was bounded by its own
  stability rule.** Each candidate's unclipped reference diverged at lr 0.1 on
  the longer v2 streams, capping eligible learning rates at 0.01 (Elman 0.003).
  Clipped configurations at lr 0.03-0.1 looked better on development and were
  never confirmed (V2-D15). Testing them needs a new, predeclared stability rule
  and fresh identities.
- **Round 3's adaptation claim (Q2) did not replicate** on fresh confirmation
  identities: -1.8e-05 [-3.9e-04, +4.3e-04]. Round 3's evidence is unchanged,
  but "continued learning helps because the world changed" is no longer a
  supported property of Champion 1. A diagnostic replay of all 120 historical
  cells found exact primitive parity between the historical and batched v2
  measurement paths, including Champion 1. This narrows the discrepancy to
  fresh identities and their statistical variation on the tested design; it
  does not turn the replay into new confirmation evidence.
- **M1 remains.** Champion 1 is 28% worse than an ungated Elman cell on
  `v1_coarse_speed` and similarly on the other coarse-observation families. A
  keep-gate bias closes most of the gap but costs occlusion accuracy, and width
  does not help (capacity `NOT_CAPACITY_LIMITED`).
- **The error head is a weak uncertainty signal** on fresh identities (rank
  correlation about 0.30, slope about 0.40); its loss term has no material
  measured accuracy effect at the reported precision.
- **External benchmarks are descriptive.** On Monash, AAA-1K models learn
  online per series and forecast recursively, unlike the published offline
  methods, and they are not competitive with the best of them. The
  `aus_elec_demand` mapping to the archive's results row is by name only.
- **CPU and CUDA agree numerically only on stable configurations.** Elman and
  LRU cells near their stability edge diverge across backends while agreeing on
  failure classification. Each backend is bitwise deterministic. Confirmation
  evidence is CPU-only, with bitwise reproduction on Zen 3 and verdict-level
  reproduction elsewhere (`AAA-173`).
- **`AAA-180`**: the v2 confirmation's K5 omitted Monash, and its two K5
  implementations disagree on a single-series dataset. There was no effect
  here (no challenger). *Repaired prospectively (2026-09-23): the frozen
  aaa.1k.v2 defect is retained and forbidden for future promotion, and the
  versioned successor `aaa.promotion.crossed.v1` is tested
  ([promotion contract](promotion_contract.md)).*
