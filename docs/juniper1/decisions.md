# Juniper 1 adaptation: design decisions

Date: 2026-10-04. Each entry gives:
- the problem;
- the candidates considered;
- the evidence;
- the choice and why it fits Juniper;
- what would falsify it;
- the main risks;
- what is deferred.

Evidence marked *development* comes from training- or development-split
runs. It informed design and is not a confirmation result.

## D-1. Erudition controls adaptation; it does not generate parameter updates

- **Problem.** The Erudition Model must choose whether, when, where and how
  to request adaptation of the Language Model and World Model from a fixed
  menu; host code owns candidate evaluation, retention and rollback.
- **Candidates.**
  - (a) a learned optimizer or hypernetwork that emits parameter deltas;
  - (b) a learned controller that chooses among inspectable mechanisms;
  - (c) a fixed rule set.
- **Evidence.** Learned optimizers fail out of distribution even at very large
  meta-training budgets ([literature](literature.md) §5). The base Language
  Model has no gradient access on FLOWBOX. Every mechanism in (b) can be
  evaluated, versioned and rolled back.
- **Choice.** (b), with (c) kept as a matched control (`Heuristic`).
- **Falsifier.** The Erudition Model fails to beat the rule set at matched
  evidence and menu, or fails to beat never-adapt.
- **Risks.** The controller's skill is bounded by the menu: a deficiency no
  mechanism can repair is invisible to it.
- **Deferred.** Learned mechanisms beyond the v0 menu.

## D-2. Language Model adaptation is a separate memory of typed edits

- **Problem.** The Language Model must improve from evidence while
  gpt-oss-20b stays immutable, hash-pinned and attributable to OpenAI.
- **Candidates.**
  - LoRA or full fine-tuning;
  - weight editing (ROME, MEMIT);
  - a learned soft prompt;
  - memory-based editing (SERAC and IKE family);
  - free-text reflections (Reflexion).
- **Evidence.**
  - FLOWBOX cannot fine-tune a 20B model, and llama.cpp exposes no gradients.
  - Weight editing violates immutability.
  - Memory-based editing keeps the base model intact and makes each edit
    removable ([literature](literature.md) §1).
  - Free-text lessons can hallucinate and poison memory (ExpeL).
  - *Development:* GPT-OSS followed an alias note and a dynamics note on every
    probe; the training-split characterization gives the rates.
- **Choice.** `AdapterState`: typed notes (alias, dynamics, observation)
  estimated from evidence and rendered by fixed templates. Mechanisms
  produce candidate note sets; the gate keeps or rejects them.
- **Why it counts as learned adaptation, not a prompt edit.** No human writes
  any note or chooses any text. The note contents are estimated from the
  workspace's own evidence: corrections extracted by the Language Model and
  aggregated with agreement rules, and coefficients from the World Model's
  posterior. Whether to create, retract or keep them is decided by the
  Erudition Model and the gate, and the measured behaviour change is the
  deliverable.
- **Falsifier.** LM-only adaptation does not reduce failure under language
  shift on the real model.
- **Risks.**
  - Memory grows; v0 workspaces are small, so this is not stressed.
  - Notes are only as good as the model's reading of them.
  - Corrupted feedback that is internally consistent can still plant a wrong
    alias; this is measured as `poisoned`.
- **Deferred.** A trained scope classifier: in v0 the scope rule is the
  phrase occurring in the request. Adapter-weight (LoRA) artifacts on a
  machine that can train them.

## D-WM. The World Model is a context library of Bayesian linear-Gaussian dynamics

- **Problem.** Juniper 1 needs an adaptable, uncertainty-aware,
  action-conditioned World Model that does not forget a version that returns.
  WM-S forgot library A when it adapted to B.
- **Candidates.** All have context timing given:
  - a library of Bayesian contexts;
  - one continually updated Bayesian model;
  - one online MLP ensemble;
  - a library of MLP ensembles.
- **Evidence (*development*, `wm_diagnostic`, 20 trials on training
  dynamics).**

  | Arm | Error on B after 2 / 6 / 10 transitions | Error on A when it returns | ±2 sd coverage after 10 |
  |---|---|---|---|
  | Library, Bayesian | 0.34 / 0.055 / 0.013 | 0.00 | 1.00 |
  | Single Bayesian | 0.93 / 0.90 / 0.90 | 0.74 | 0.83 |
  | Single MLP ensemble | 0.75 / 0.81 / 0.74 | 0.91 | 0.66 |
  | Library, MLP | 0.75 / 0.81 / 0.72 | 0.20 | 0.68 |

  On capped ("novel") dynamics outside the basis, the Bayesian library's
  error after 10 transitions is 0.26, but its coverage stays 0.94: the misfit
  shows up as uncertainty. Context separation removes most forgetting
  whichever learner is used; the Bayesian learner is what makes adaptation
  few-shot and exact. The MLP arms were trained with 300 Adam steps per
  update after a 30-step version was found unfairly weak.
- **Choice.** The library of Bayesian contexts (`aaa.world.cbl.v0`), with
  mechanisms `new_context`, `update_in_place` and `recall` invoked by the
  lifecycle.
- **Falsifier.**
  - Environments whose dynamics need a richer basis than the library can
    hold.
  - A neural learner that matches it in sample efficiency.
- **Risks.**
  - The basis is task-shaped: affine tool effects.
  - Context inference is done by the Erudition Model's recall decisions, not
    by the World Model.
- **Deferred.** A learned feature basis or neural contexts for richer
  environments, and automatic context inference (MOLe-style) as an
  alternative to controller-driven recall.

## D-3. The integration contract: WM revision and WM consultation

- **Problem.** How does World Model knowledge reach the Language Model's
  behaviour without the WM becoming a decision-maker?
- **Evidence (*development*, real model).**
  - Shown a single disagreeing prediction, GPT-OSS distrusted it and
    abstained.
  - Shown the predicted consequence for every `n`, it reasoned correctly.
  - After a dynamics shift, GPT-OSS abstains when the manual makes the target
    unreachable, so a check offered only on `act` proposals never helps (first
    pilot, training stream 0: 54 of 67 shifted steps abstained).
- **Choice.** Two fixed host rules, identical in every condition:
  - after an `act` proposal whose value the World Model (SD ≤ 1.5) predicts
    differently, one revision turn with the consequence table;
  - after an abstention, if the World Model's active context confidently
    departs from the manual, one consultation turn with its learned behaviour.
- **Consequence for the comparison.** Without LM adaptation, a WM-only system
  can still benefit from WM adaptation through these turns. Joint adaptation
  must earn its advantage over that, not over a crippled WM-only arm.
- **Risks.** A consultation could prompt the model to act on a nonexistent
  tank (fabrication); the infeasible requests and the gate's attack probe
  measure it.

## D-4. Erudition Model architecture and scale

- **Problem.** The input is a stream of evidence. Change, noise, corruption
  and recurrence are only separable over time, and adaptation history
  matters (oscillation, rollback).
- **Candidates.**
  - 4-layer transformer, 1,259,700 parameters;
  - 2-layer GRU, 1,020,692;
  - flat window MLP, 1,254,580;
  - transformer, 109,012;
  - transformer, 4,976,980.
- **Evidence (*development*, simulation, 36 streams; [report](development_report.md) §3).**
  Joint-arm failure by architecture:

  | Architecture | Parameters | Joint failure |
  |---|---|---|
  | Transformer | 1.26M | 0.323 |
  | GRU | 1.02M | 0.333 |
  | Transformer | 109K | 0.332 |
  | Transformer | 4.98M | 0.334 |
  | Flat MLP | 1.25M | 0.371 |

  Seeds 2 and 3 of the 1.26M transformer gave 0.346 and 0.325. The rule-based
  controller scored 0.344 on the same streams.
- **Choice.** The 1.26M transformer, which had the lowest failure. The
  one-million-parameter floor is met without padding: every parameter
  receives gradient (tested).
- **Recorded deviation.** The stated rule gives ties within 0.01 to the
  smaller model. That selects the 1.02M GRU (0.3329 against 0.3231, a gap of
  0.0098), not the transformer. The tie clause was applied only to the 109K
  model by mistake, and an independent review caught it after confirmation
  had begun. The confirmed artifact is the transformer.
- **What the evidence says about scale.** The 109K transformer is within the
  declared 0.01 tie band, and the 4.98M transformer is no better. On this
  environment, therefore, **the owner's floor, not the evidence, sets the
  size**. The sequence models beat the flat MLP. The gap between
  architectures is smaller than the gap between seeds.
- **Falsifier.** A richer environment in which the 109K model falls clearly
  behind would show that the capacity is needed. Nothing here shows it.
- **Deferred.** Richer inputs (text embeddings of feedback) that would
  naturally need more capacity. Online self-updating of the Erudition Model,
  deferred because it widens the poisoning surface.

## D-5. Training from consequences, in simulation, against a calibrated surrogate

- **Problem.** The controller must learn from consequences. The real model
  answers about one request every 4 seconds, and training needs hundreds of
  thousands of decisions.
- **Candidates.**
  - imitation of a hand-written oracle;
  - reinforcement learning on the real model;
  - counterfactual returns in simulation.
- **Choice.** At sampled decision points, copy the simulator once per
  allowed action, force the action, and score the next 25 steps (task
  success minus 0.002 per gate or extraction call and 0.01 per candidate).
  The hidden regime supplies an auxiliary diagnosis target. The simulated
  Language Model's rates are fitted to a real-model characterization on
  training-split situations; unmeasured parameters keep listed defaults.
- **Falsifier.** A large drop from simulated to real development
  performance (the sim-to-real gap), reported as such.
- **Risks.** The surrogate misses real-model behaviours the controller then
  never learns to handle.

## D-6. A new promotion contract, `aaa.promotion.paired.v1`

- **Problem.** Promotion must show target improvement *and* bounded
  regression. The endpoints are per-stream rates, can be zero, and the
  compared arms share streams.
- **Why not `crossed.v1`.** It estimates a ratio of positive means over an
  initialization × series grid. Zero rates would be refused, and the claim
  concerns one frozen artifact, not a population of initializations.
- **Choice.**
  - Paired mean differences over declared streams with a percentile bootstrap.
  - Superiority and noninferiority criteria; intersection-union verdicts.
  - Two structurally independent implementations and fail-closed
    adjudication.
  - `crossed.v1` is unchanged.
- **Risks.** With 12–18 streams, percentile intervals can under-cover. This is
  stated in the protocol and is one reason confirmation is not the last word.

## D-7. Research harness lives in AAA; Juniper-App is unchanged

- **Problem.** Where does the integration live during research?
- **Choice.** In AAA. The harness calls the same pinned runtime and artifact
  with the profile copied from Juniper-App. Juniper-App keeps its
  model-agnostic runtime, host-authoritative tools and constitution, and its
  product behaviour.
- **Revisit when.** Confirmed evidence justifies an experimental product
  boundary, which would need its own ADR.

## Rejected or deferred, briefly

- **A Decision Model under another name.** The host's gate and canary are
  fixed software, not learned, and no second learned controller exists.
- **Erudition learning language or code.** Language belongs to Juniper LM 1.1;
  the Erudition Model sees 40 numeric evidence features and no text.
- **Initializing from Champion 1, the Python learners, WM-S or any language
  model.** Forbidden by direction and unnecessary.
- **Training on JuniperBench, Cinqic History or any user data.** Not used; see
  the [data record](data.md).
