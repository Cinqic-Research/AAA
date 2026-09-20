# The AAA research charter

This document exists because the repository's own history is misleading about
its scope. AAA's benchmark is one moving dot on a line, and every result so far
is about that dot. A future implementer reading only the code would reasonably
conclude that the dot *is* the project. It is not.

## The three names

**AAA** — Accurate Autonomous Adaptation — is the research programme.

**Juniper** is the persistent artificial agent the programme is ultimately
intended to develop. Nothing in this repository is Juniper. Nothing in this
repository is a component of Juniper yet, either; things become components by
surviving measurement.

**AUTONOMY** is the one-word long-term objective.

## What AAA is not trying to be

- a frontier language model;
- a chatbot;
- an AGI benchmark chaser;
- a language-only model;
- one large monolithic Transformer.

## The long-term question

How far can a persistent artificial agent develop autonomous capabilities,
using available research, experimental evidence, and the hardware Cinqic
actually has?

The intended Juniper loop is roughly:

> perceive → maintain internal state and beliefs → predict → reason and
> simulate → choose goals and subgoals → plan → decide → act → observe
> consequences → evaluate → adapt → consolidate memory → inspect itself →
> improve → repeat

Research along that path may eventually involve vision, audio, multimodal
representation learning, persistent latent state, world models, partial
observability, causal reasoning, memory, continual learning, exploration,
autonomous goal generation, planning, computer use, games, embodiment, a
persistent self-model, architecture inspection, controlled self-improvement,
neural-network growth, and externally enforced mandatory guardrails.

**None of that is implemented, and most of it is not designed.** Listing a
direction is not the same as having taken it.

## Where the dot fits

The moving-dot benchmark is **one benchmark family**, not the programme. It has
been an unusually good one for its purpose, because it is small enough that a
wrong answer is visible. The repository's most valuable output to date is not a
model; it is the discovery, under three independent reviews, that a set of
gates which all reported `PASS` could not have reported anything else. See
[`issue_ledger.md`](issue_ledger.md).

## Where AAA-1K fits

AAA-1K is a **research seed**: a persistent recurrent predictive core that
learns online, maintains hidden state, estimates its own predictive error,
adapts to changing dynamics, and is small enough that every parameter,
activation, gradient and state transition can be inspected.

It is 994 parameters. It predicts where a dot goes next. It is the smallest
thing that could plausibly survive into a later architecture as a component,
and it exists so that the programme has one such component whose behaviour is
measured rather than assumed.

The lineage so far, and what each step actually taught:

| Model | Parameters | What it established |
|---|---|---|
| v1 online linear SGD | 5 | a learner can lose to an analytic baseline, and that is a result |
| v2.1 square-root RLS | 3 | online adaptation, numerically soundly, with gates that can fail |
| PR #12 TinyMLP | 17 | online neural updates beat the same network frozen -- and still lost badly to simpler methods |
| AAA-1K GRU | 994 | see [`aaa_1k_report.md`](aaa_1k_report.md) |

Each entry earned its parameters or is recorded as not having earned them.

## Autonomy is not unlimited authority

"Autonomous" in AAA has always meant one thing: the observe / predict / score /
update loop runs unattended after launch. It has never meant general
intelligence, physical understanding, or independent goal formation, and this
charter does not expand it.

As the programme moves toward agents that choose goals and take actions, the
distinction becomes a safety property rather than a definitional one:

- **Mandatory guardrails must live outside self-modifiable model cognition.**
  A constraint a system can rewrite is not a constraint. Any future
  self-improvement work has to place its limits in code the improving component
  cannot reach, and has to demonstrate that it cannot reach it.
- **Controlled self-improvement means controlled.** Architecture inspection and
  neural growth are listed as research directions, not as permissions.
- **Capability claims stay empirical and benchmark-specific.** "AAA-1K beat a
  stateless control on an occlusion benchmark" is a sentence about an occlusion
  benchmark.

## Claim discipline

Every AAA phase inherits these, and AAA-1K is held to them:

1. predict before reveal — no future observation influences a scored prediction;
2. no evaluator leakage — no scenario label, change indicator, event flag,
   hidden coefficient or latent truth reaches a learner;
3. implementation success is not learning;
4. learning is not generalization;
5. generalization is not adaptation;
6. adaptation is not baseline superiority;
7. baseline superiority is not understanding;
8. absence of evidence is never `PASS`;
9. failures are retained;
10. complexity must earn its keep;
11. simple baselines stay first-class;
12. development and evaluation evidence stay separated;
13. hyperparameters are chosen without consulting held-out results;
14. every substantive experiment is reproducible from configuration, seeds,
    source identity and retained artifacts;
15. historical evidence is never rewritten to fit a new direction;
16. claims describe exactly what was measured.

## What this document is not

It is not a roadmap with dates, a product plan, or a claim about what AAA will
achieve. It is a statement of direction, written down so that the next
implementer does not mistake the current benchmark for the programme, and so
that nobody mistakes a 994-parameter network for a mind.
