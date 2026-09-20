# AAA-1K literature review: what was adopted, what was rejected, and why

> **Current-status note (2026-09-20).** Statements below about round 1 and
> round 2 are historical where labeled. Independent review superseded round 2
> because Q1 did not causally isolate weight learning and Q2/Q5 did not model
> reused initialization identities. Round 3 changes those estimands/statistics
> without changing the model or its selected hyperparameters. The fixed-budget
> gating result remains inconclusive; a separate width-matched development
> characterization slightly favors the ungated arm but its interval reaches
> zero. Retention is stated only as what the fixed probe bank measured, never as
> general immunity to catastrophic forgetting.

Conducted before the AAA-1K design was frozen. The rule applied throughout: do
not copy an architecture because a large lab uses it. Extract the *mechanism*,
ask whether it is testable at 994 parameters on a moving dot, and reject it
when it is not.

## Gated recurrence

**Chung, Gulcehre, Cho & Bengio (2014), "Empirical Evaluation of Gated
Recurrent Neural Networks on Sequence Modeling"** ([arXiv:1412.3555](https://arxiv.org/abs/1412.3555)).
Gated units beat plain tanh units on polyphonic music and speech modelling, and
the GRU is comparable to LSTM with fewer parameters and no separate cell state.

*Adopted:* the GRU rather than the LSTM. At a fixed parameter budget the GRU's
simpler state gives more hidden units and one fewer thing to serialize.

*Noted:* the evidence is from tasks with long-range structure and large models.
It does not predict anything about a 16-unit network on one-dimensional motion,
which is why the ungated control exists.

**Foucault & Meyniel (2021), "Gated recurrence enables simple and accurate
sequence prediction in stochastic, changing, and structured environments"**
([eLife 10:e71801](https://elifesciences.org/articles/71801)). Networks of
**eleven** recurrent units reach 99% of optimal prediction performance in
changing environments, beating delta-rule and leaky heuristics. The authors
isolate three mechanisms — gating, lateral connections, recurrent weight
training — and ablate each while keeping the other two. Removing gating cost
about 6x, removing lateral connections about 4x, removing recurrent weight
training about 12x. Removing any of them particularly damaged the ability to
adjust the effective learning rate around change points.

*Adopted, and this is the single largest influence on AAA-1K's design:* the
hidden size (16 is the same order as their 11), and above all the **ablation
methodology**. AAA-1K's `aaa1k_state_reset`, `aaa1k_no_error_input` and
`aaa1k_frozen_recurrent` arms, plus the stateless and ungated controls, are a
direct adaptation of "test the mechanisms, do not admire the diagram".

*Result worth recording:* their ablations found gating strongly beneficial.
AAA-1K's did not — the ungated 954-parameter control beat the gated
994-parameter model. Their environments were stochastic and structured; AAA's
are deterministic and smooth. That contrast is the finding, not a contradiction.

## Recurrent state-space world models

**Hafner, Pasukonis, Ba & Lillicrap (2023), DreamerV3, "Mastering Diverse
Control Tasks through World Models"** ([arXiv:2301.04104](https://arxiv.org/abs/2301.04104)).
A learned world model with a deterministic recurrent state plus a stochastic
state, trained across domains with one fixed hyperparameter set, made viable by
normalization, balancing and transformation techniques (symlog, two-hot,
percentile return normalization).

*Adopted:* the attitude toward normalization. AAA-1K's insistence that every
feature and target be normalized by a *declared public constant*, and the
observation that a badly scaled input silently disables the mechanism it
carries (see input 3), is the same lesson at a much smaller scale.

*Rejected:* the stochastic latent, the imagination rollout, the actor-critic,
and the whole apparatus of reinforcement learning. AAA-1K has no actions, no
rewards and no policy. A stochastic latent at 994 parameters would add sampling
noise to an experiment whose entire value is that it is deterministic and
inspectable.

## Joint-embedding predictive architectures

**V-JEPA 2 (2025)** ([arXiv:2506.09985](https://arxiv.org/abs/2506.09985)).
Predict in latent space rather than pixel space; the learned representation
then supports planning toward image goals.

*Adopted:* nothing directly, and that is deliberate. *Noted for the programme:*
latent-space prediction is the right idea once AAA has perception worth
compressing. AAA-1K's observation is a single scalar. There is nothing to
embed, and a joint-embedding objective on a one-dimensional observation would
be an elaborate way to learn the identity function.

## Predictive coding and prediction-error-driven learning

**Rao & Ballard (1999)** and the broader predictive-coding literature: cortical
systems are modelled as propagating prediction *errors* rather than raw signals.

*Adopted, in a deliberately weak form:* input 3 feeds the model's own previous
signed prediction error back as an ordinary observation. That is the one
mechanism from this literature that is cheap, testable and directly continuous
with AAA's existing predict / reveal / error / adapt loop.

*Explicitly not claimed:* AAA-1K is not a predictive-coding network. It has no
hierarchy, no precision weighting, no top-down generative pathway, and no
biological interpretation. It is a GRU that is told how wrong it was.

## Online recurrent learning: TBPTT versus RTRL and UORO

**Williams & Peng (1990)** for truncated BPTT. **Tallec & Ollivier (2017),
"Unbiased Online Recurrent Optimization"** ([arXiv:1702.05043](https://arxiv.org/abs/1702.05043)):
truncated BPTT gives *biased* gradients, and when a parameter has positive
short-term but negative long-term influence, truncated BPTT can diverge unless
the truncation span greatly exceeds the intrinsic temporal range. UORO provides
an unbiased estimate at roughly TBPTT's cost, at the price of variance, by
maintaining a rank-one stochastic approximation of the influence matrix. RTRL
is exact but costs `O(H^2 * P)` — for this model, 16 x 994 x 16 numbers carried
forward at every step.

*Adopted:* truncated BPTT, as the reference implementation. It is established,
inspectable, straightforward to verify against finite differences, and
appropriate for a first recurrent implementation. Its truncation bias is
documented rather than glossed over.

*Rejected for this phase, with a measurement rather than an opinion:* RTRL's
influence matrix would be roughly 254,000 carried scalars against 994
parameters — a 256x state footprint for a model whose entire premise is that
its state is small enough to inspect. UORO would add a stochastic estimator,
and therefore a new RNG stream and a new variance source, to an experiment
built on determinism.

More decisively: the development divergence probe measured the truncation
horizon's actual contribution. Beyond `T = 4` the gradient norm changes by
under 2%, and across `T` in `{4, 8, 16, 32}` the development error was
identical to four decimal places. On these streams the update gate sits near
0.5, so the hidden state's influence decays by roughly a factor of two per
step, and there is no long-range credit to assign. **Adding an unbiased online
estimator to remove a bias that was measured to be negligible would be
sophistication for its own sake.** If a future AAA benchmark shows genuine
long-range dependence — and one should be built that does — UORO becomes the
right next experiment, and this paragraph is the reason why.

## Catastrophic forgetting and replay

**Rolnick, Ahuja, Schwarz, Lillicrap & Wayne (2019), "Experience Replay for
Continual Learning" (CLEAR)** ([arXiv:1811.11682](https://arxiv.org/abs/1811.11682)).
Replay buffers plus behavioural cloning substantially reduce catastrophic
forgetting without task identity being signalled to the model; bounded buffers
with random eviction work nearly as well as unbounded ones.

*Adopted:* the experimental design. `aba_v1` is an unlabelled A→B→A stream, and
the model is never told a transition occurred — exactly CLEAR's "no explicit
task boundary" condition.

*Deliberately not implemented:* replay itself. AAA-1K's job in this phase is to
**measure whether forgetting exists**, not to prevent it. Adding replay now
would make the result prettier and would remove the failure mode a later phase
needs to target. The measurement came back inconclusive for a design reason —
see the confound recorded in [`aaa_1k_report.md`](aaa_1k_report.md) — and
fixing that design is the prerequisite for any replay work.

## Function-preserving network growth

**Chen, Goodfellow & Shlens (2015), "Net2Net"** ([arXiv:1511.05641](https://arxiv.org/abs/1511.05641)).
Net2WiderNet and Net2DeeperNet transfer knowledge into a larger network while
*preserving the function it computes*, so the larger network starts exactly
where the smaller one finished.

*Adopted:* only the metadata. Checkpoints carry a `parent_model_id` (`null`,
since AAA-1K is the lineage root), architecture id, parameter count and state
hash, so a later phase that widens this network can say precisely what it grew
from.

*Rejected for this phase:* growth itself. Nothing here is inadequate in a way
that more parameters would fix — the model loses to a *smaller* ungated control
and to parameterless analytic rules. Growing it would be treating a measurement
problem as a capacity problem.

## Summary of what crossed into the design

| Source | Mechanism taken | Mechanism refused |
|---|---|---|
| Chung et al. 2014 | GRU over LSTM at a fixed budget | — |
| Foucault & Meyniel 2021 | mechanism-by-mechanism ablation; small hidden size | their conclusion, which AAA-1K's own ablation contradicts |
| DreamerV3 | disciplined public normalization | stochastic latents, imagination, RL |
| V-JEPA 2 | — | latent-space prediction, premature with a scalar observation |
| predictive coding | prediction error as an input | hierarchy, precision weighting, any biological claim |
| Tallec & Ollivier 2017 | explicit statement of TBPTT's bias | UORO, after measuring the bias as negligible here |
| Rolnick et al. 2019 | unlabelled A→B→A design | replay, until forgetting is actually demonstrated |
| Chen et al. 2015 | lineage metadata | growth, until inadequacy is measured |
