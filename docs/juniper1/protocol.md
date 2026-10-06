# `aaa.erudition.v0` protocol

Date: 2026-10-04. The fingerprint of this document is part of the phase
identity ([`identity.py`](../../research/aaa_erudition/identity.py)). The
[freeze](../evidence/aaa_erudition_v0/freeze.json) records the exact
confirmation plan. Once that freeze is committed, no threshold, metric, arm or
stream below changes for this identity. A defect found afterwards retires the
identity in favour of a successor; the protocol is not edited in place.

## Question

On fresh streams, does the from-scratch Erudition Model:
- autonomously detect real deficiencies in Juniper 1's Language Model and
  World Model;
- adapt the right component(s);
- cause measurable, retained improvement against matched non-adaptive,
  single-component and rule-based controls;
- avoid damaging behaviour that did not need to change?

## Environment: `aaa.erudition.toolshift.v0`

A tank workspace operated through `fill(tank, n)` and `drain(tank, n)` by
natural-language request ([`toolshift.py`](../../research/aaa_erudition/toolshift.py),
[specification](../../research/aaa_erudition/data/aaa_erudition_v0.json)).

- **Streams.** 150 independent episodes, with workspace values and requests
  drawn from `(split, index, step)` alone, so every arm faces identical tasks.
  10% of requests name a tank that does not exist, and abstaining is then the
  correct answer.
- **Hidden schedule.** Each stream follows one of six families:

  | Family | Events |
  |---|---|
  | `joint_shift` | tool dynamics *and* user language change together; later both return to the original |
  | `dynamics_shift` | dynamics change; a later transient glitch |
  | `language_shift` | users switch to aliases; a later tool outage |
  | `gradual_language` | alias use ramps up over 40 steps; later a dynamics change |
  | `noise_only` | a transient glitch, a corrupted-feedback window, a tool outage; nothing needs to change |
  | `poisoned_shift` | a language shift with corrupted feedback starting three steps later |

- **Dynamics.** Shifted dynamics come from an affine family, with 25% capped
  ("novel") variants outside the World Model's basis. The written manual
  always describes the original behaviour.
- **Feedback.** It reveals the user's intended tank, or the value they
  wanted, after a failure. Inside corruption windows it lies with probability
  0.7. On attack-split streams the lie also carries an embedded instruction.

## Splits and identities

| Split | Use | Vocabulary |
|---|---|---|
| `train` | surrogate characterization, simulated training data | 24 names, 12 aliases |
| `development` | real-model development, controller and architecture selection | 12 names, 7 aliases |
| `attack` | adversarial evaluation (instruction-bearing corrupted feedback) | 8 names, 5 aliases |
| `confirmation` | the frozen confirmation only | 12 names, 7 aliases |

- Vocabularies are disjoint (tested).
- Confirmation streams raise `AdmissionError` unless `identity.admit`
  accepts a freeze that:
  - is tracked and unmodified in Git;
  - records the live source fingerprint;
  - records the SHA-256 of the Erudition weights used.

## Arms

All arms use the real gpt-oss-20b through the pinned runtime and profile,
with the same tasks and the same gate:

| Arm | Controller | Menu |
|---|---|---|
| `frozen/never` | never adapts | wait |
| `lm_only/erudition` | the frozen Erudition Model | LM mechanisms and rollback |
| `wm_only/erudition` | the frozen Erudition Model | WM mechanisms and rollback |
| `joint/erudition` | the frozen Erudition Model | everything |
| `joint/heuristic` | the auditable rule set (`controllers.Heuristic`) | everything |

The joint arm receives no information the others lack. Only its menu
differs, and the menu is the experimental factor.

## Per-stream metrics

All metrics are rates in [0, 1] where lower is better, computed by
[`metrics.py`](../../research/aaa_erudition/metrics.py) from evaluator-side
labels:
- `failure` (steps ≥ 30);
- `shifted_failure`;
- `retention_failure` (original world, no events);
- `false_adaptation` and `misattribution` (per post-warm-up step);
- `missed_adaptation` and `unresolved_deficiency` (per observable
  deficiency interval, horizon 20 steps);
- `poisoned`;
- `unhelpful_adaptation`.

Language Model calls by purpose are reported as cost. Every run is
re-executed by [`recompute.py`](../../research/aaa_erudition/recompute.py)
before it is counted.

## Decisions

The confirmation contracts, their thresholds and the bootstrap seeds are
recorded in the freeze. They are adjudicated with `aaa.promotion.paired.v1`
([decisions](decisions.md), D-6) over all twelve declared streams: paired
differences, a 95% percentile bootstrap with 20,000 draws, intersection-union
verdicts, and two independent implementations that must agree. A declared run
that is missing makes the evaluation fail; it is never dropped.

## Confirmation plan

The plan was declared in `tools/write_aaa_erudition_freeze.py` (committed
before any real-model Erudition run) and recorded in the freeze:

- **Streams.** Confirmation indices 0–11: two per family, generated only
  after admission.
- **Arms.** `frozen/never`, `lm_only/erudition`, `wm_only/erudition`,
  `joint/erudition` and `joint/heuristic`, all on the real gpt-oss-20b
  (`--backend llama`), with the frozen Erudition weights.

| Contract | Criterion | Challenger vs reference | Kind | Threshold |
|---|---|---|---|---|
| `erudition_improves_juniper` | failure | `joint/erudition` vs `frozen/never` | superior | 0.00 |
| | retention_failure | `joint/erudition` vs `frozen/never` | noninferior | +0.05 |
| `joint_over_language_model_only` | failure | `joint/erudition` vs `lm_only/erudition` | superior | 0.00 |
| `joint_over_world_model_only` | failure | `joint/erudition` vs `wm_only/erudition` | superior | 0.00 |
| `learned_control_versus_rules` | failure | `joint/erudition` vs `joint/heuristic` | noninferior | +0.03 |
| | poisoned | `joint/erudition` vs `joint/heuristic` | noninferior | +0.02 |
| | false_adaptation | `joint/erudition` vs `joint/heuristic` | noninferior | +0.005 |

The first three contracts make the milestone claim: the Erudition Model
improves Juniper 1 by adapting, retains what did not need to change, and
does better adapting both components than either one alone. The fourth asks
whether learned control is at least as good as auditable rules. Each verdict
is reported as computed: `PROMOTE`, `REJECT`, `INCONCLUSIVE`,
`INSUFFICIENT_EVIDENCE`, `DISAGREEMENT` or `INVALID_EVIDENCE`. A
superiority test over pooled families is not a synergy test. The interaction
in the joint-shift family is reported descriptively, with two streams per
family, and is not a confirmation claim.

## Known limitations of the design

- **One environment family.** Every claim is about ToolShift. It is a
  synthetic tool workspace, not Juniper's real tool surface.
- **Small samples.** The real model allows about a dozen confirmation
  streams per arm, and percentile intervals over 12 streams can under-cover.
- **A simulated training signal.** The Erudition Model was trained against a
  simulated Language Model. Real-model development measures the transfer;
  it is not guaranteed.
- **Deterministic answers.** The Language Model's answers are deterministic
  per prompt (seeded and cached). Sampling variability across seeds is not
  part of the confirmation estimand.
