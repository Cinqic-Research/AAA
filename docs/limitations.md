# Research limitations

Stated plainly, because a benchmark that cannot say what it does not show is
not measuring much.

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
evaluate, and reviewed by nobody. A decomposition probe established that
`coarse_speed_v1` genuinely tests hidden-regime inference — with the speed held
fixed the recurrent model is *worse* than the stateless control — but
`occlusion_v1` has no equivalent check, and neither has been seen by anyone
else.

**The most useful next experiments**, in order: a width-matched gating
comparison, to separate "gating" from "fewer hidden units at the same parameter
count"; a distributional analysis of the minority of streams where the ungated
control wins large; an independently designed memory benchmark; and a benchmark
with genuine long-range dependence, without which the truncation horizon does
not matter and an unbiased online recurrent learner has nothing to fix.
