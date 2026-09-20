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
different in kind from the v2.1 ones.

**What it does not show.** It does not show adaptation. The online-versus-frozen
result that looked like adaptation reproduces at 95% strength when the branch
is placed where nothing happens, so it measures continued learning
(`AAA-153`). It does not show retention: the A/B/A design confounds "came back
to A intact" with "had three times as much total experience by then"
(`AAA-154`). It does not show generalization beyond unseen trajectories of the
same four families; no unseen family was tested.

**What it does show, within those bounds.** Online learning on every family.
Persistent recurrent state beating both a state-reset ablation and a
matched-capacity stateless control, most clearly on steps whose target the
agent never saw. Gating *failing* to pay for itself against a smaller ungated
control -- the opposite of what the small-gated-network literature reports for
stochastic, changing environments, and the most interesting negative result of
the phase.

**Where the evidence is thin.** The declared precision objective was missed by
a factor of nearly twenty: 599 replicas per family were indicated, 32 were run
under a declared bound. Every interval is wider than the design asked for, and
effects near zero are unresolved rather than absent. Every effect is also
conditional on a single model initialization; a five-seed probe found the
directions stable and the magnitudes varying by up to a factor of two.

**Mechanisms that are programmed, not learned.** Boundary reflection, target
unfolding, the holding of unobserved steps, and the rule that a learner updates
only on transitions whose both ends it was shown. All four are public knowledge
of the observation format applied identically to every arm, and the
dead-reckoning and reflected constant-motion baselines exist so that none of
them is mistaken for a capability.

**Benchmark provenance.** `occlusion_v1` and `coarse_speed_v1` are new,
designed by the same implementer whose model they evaluate, and reviewed by
nobody. That is the weakest kind of benchmark, and it is the first thing an
independent reviewer should attack.

**Gradient clipping activated on 22% of updates.** It is a declared mechanism
with a declared threshold, but at that rate it is shaping the optimization
rather than merely guarding it.

**The most useful next experiments**, in order: a difference-of-differences
design against a matched no-change stream, so adaptation can be separated from
learning; a frozen probe bank measured at both A/B/A boundaries, so retention
can be separated from accumulated experience; initialization as a second
bootstrap level; and a benchmark with genuine long-range dependence, without
which the truncation horizon does not matter and an unbiased online recurrent
learner has nothing to fix.

