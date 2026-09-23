# AAA-1K v2 research brief: the final pass at about 1,000 parameters

Phase identity: `aaa.1k.v2`. Protocol: `aaa.1k.v2.protocol.v1`
([`research/aaa_1k_v2/plan.py`](../research/aaa_1k_v2/plan.py)). Reference
system: **Champion 1** (`aaa1k-champion-1`,
[`evidence/aaa1k_loop_0006/champion_1.json`](evidence/aaa1k_loop_0006/champion_1.json)).

This brief was written and committed before any development, attack or
confirmation identity of this phase was observed. It is part of the phase's
scientific fingerprint ([`research/aaa_1k_v2/identity.py`](../research/aaa_1k_v2/identity.py)).

## The question

> Given at most 1,000 trainable parameters and a tightly accounted
> adaptive-state budget, how capable, stable, general and computationally
> efficient can AAA's online predictive core become before additional
> parameter capacity is scientifically justified?

It is not a mandate to grow the model, to make CUDA the default, or to replace
Champion 1. **A no-change outcome is a valid result**: if no candidate earns
promotion under the rules below, Champion 1 remains the phase's best
defensible system and this phase reports why.

## What is fixed

| Constraint | Value |
|---|---|
| trainable parameters | at most 1,000 per candidate, recounted from the arrays (`arms.audit`) |
| adaptive state | reported separately: hidden state, TBPTT buffer, eligibility traces, optimizer state, normalization statistics, replay memory, adapter interaction state |
| optimizer | plain per-cell SGD for every candidate; an optimizer with state is a diagnostic, not a free improvement (§ Optimizer) |
| numerical platform | float64; confirmation on the CPU; CUDA qualified separately |
| causal boundary | predict, record, advance, reveal, score, update; no label, event, hidden coefficient or future value reaches a model |
| historical identity | `aaa.1k.v1`, Champion 0, Champion 1 and every retained artifact unchanged; `research/aaa_1k/` is imported, never edited |
| candidate budget | six candidates, precommitted (below) |

## Evaluation dimensions (frozen)

The final report shows every dimension separately. No composite "intelligence"
score is computed.

| Dimension | How it is measured |
|---|---|
| online learning | online versus frozen twin from a shared trunk (Q1-style branch) |
| adaptation after change | difference-of-differences on paired changed/unchanged streams |
| retention | fixed held-out regime-A probe bank after A, B and A again |
| persistent memory | versus the candidate's own state-reset ablation |
| long-range dependence | `long_gap_recall` (16-64 step gaps), `long_gap_recall_extended` (96-128), NARMA-20 |
| numerical stability | failed and diverged cells on 800-2000-step streams; clip rates |
| robustness to noise | `noise`, `noise_gaps`, `change_noise` (scored against latent truth) |
| robustness to missing observations | `random_gaps`, `wall_gaps`, `stationary_mixed`, `v1_occlusion` |
| robustness near boundaries | `wall_gaps`, `gravity_bounce`, `soft_wall`, `inelastic_wall` |
| robustness to dynamics changes | `accel_switch`, `change_gaps`, `oscillator_long`, `abcab`, `v1_aba` |
| uncertainty / error estimation | rank correlation, calibration slope and bias of the error head; under shift |
| within-family generalization | fresh streams of every development family |
| out-of-family generalization | held-out families first seen in confirmation |
| external benchmarks | NARMA-10, NARMA-20, Mackey-Glass 17, twelve dysts systems, four Monash datasets |
| parameter efficiency | trainable parameters |
| adaptive-state efficiency | total adaptive scalars |
| compute efficiency | cell-steps per second on CPU and CUDA |

## Measured weaknesses this phase must revisit

| Item | What is known | How this phase tests it |
|---|---|---|
| M1 / Q4 | the gated core loses to the ungated control at a fixed budget, entirely on `coarse_speed_v1`; the zero-bias operating point is implicated | candidate `gru_v1_keep-2`; ungated and minimal-gated candidates; a diagnostic of gate statistics, fixed-width versus fixed-parameter comparisons, initialization, quantization and speed-regime sensitivity |
| `AAA-162` | what `coarse_speed_v1` rewards is unresolved: the original decomposition used a model that cannot use memory there | a model-independent decomposition with analytic estimators (windowed fits of varying length, fixed versus switching speed, quantum sweep) |
| `AAA-163` | reduced SHA seeds can collide | 64-bit seeds and a fail-closed set-intersection proof against every historical identity (`identities.prove_disjoint`) |
| `AAA-172` | the v2.1 RLS learner unfolds around its own prediction; frame lock or stall untested | instrumentation on long quantized and smooth streams with predeclared lock and stall thresholds |
| online TBPTT semantics (`AAA-169`) | the live rule uses cached activations with current weights | `live` versus `replay` versus horizon 1/4/16 inside every TBPTT candidate's development grid, and exact untruncated RTRL for the diagonal candidate |
| R-10 missingness | zeros double as the missingness code | an explicit observed flag (`v2` features) with parameters reallocated to stay under the cap, plus a `no_observed_flag` ablation |
| R-11 error head | weakly informative, over-predicts | the `no_error_head_loss` ablation of challenger and champion; calibration under shift |
| R-05 independent benchmarks | every benchmark so far was designed by the implementer | the external suite |

## Candidates (budget: six, precommitted)

Champion 1 participates unchanged. Each candidate has a hypothesis tied to a
measured weakness or a literature mechanism (see
[`aaa_1k_v2_literature_review.md`](aaa_1k_v2_literature_review.md)).

| Candidate | Parameters | Hypothesis |
|---|---|---|
| `gru_v1_retuned` | 994 | Champion 1's architecture with v2-selected hyperparameters: separates architecture from hyperparameters, so no control is handicapped by Champion 1's learning rate |
| `gru_v1_keep-2` | 994 | the zero-bias operating point causes M1; a keep-gate bias of -2 (loop candidate c2) failed only with M2's signature, which Champion 1's rule removes |
| `gru_v2` | 932 | an explicit observed flag resolves the zero/missing ambiguity; 15 units pay for the extra input |
| `elman_v2` | 982 | ungated recurrence uses the budget for 28 units of state and was already competitive; with the flag |
| `mgu_v2` | 952 | one gate keeps most of the stability benefit and buys width (19 units) |
| `lru_v2` | 957 | a complex-diagonal linear recurrence with `|lambda| < 1` by construction is stable over long horizons, and its exact online gradient removes truncation bias |

Controls: `mlp_v2` (stateless, 947) and every candidate's one-mechanism
ablations. Baselines: persistence, reflected constant motion, dead reckoning,
windowed linear fit, the v2.1 RLS incumbent, and, externally, persistence,
online AR by recursive least squares, the seasonal naive forecast and the
published Monash baselines.

## Stages and decision

Development selects each candidate's hyperparameters by one rule and screens
each against Champion 1; at most one challenger proceeds. Attack tries to
break it on fresh identities, attack-only families, perturbed initializations
and a CUDA re-run. A frozen, committed manifest precedes one fresh
confirmation that includes the held-out families. Promotion requires every
criterion of [`aaa_1k_v2_benchmark_protocol.md`](aaa_1k_v2_benchmark_protocol.md)
to pass: stability, non-inferiority on every round-3 family, a **meaningful**
(at least 5%) and statistically resolved improvement over the stress suite, no
resolved regression beyond 5% on any stress family, and non-inferiority on the
external suite. Anything else leaves Champion 1 in place.

A separate, exploratory capacity diagnostic (about 0.5K, 1K, 2K and 4K) runs
only after confirmation, on its own identities, and cannot alter the decision.

## Optimizer

Adam carries two parameter-sized state vectors (about 2,000 scalars at 1K) and
momentum carries one. Neither is a candidate: the tournament is SGD-only so that
no candidate hides state. A development diagnostic compares SGD, momentum and
Adam on the development-selected challenger under full state accounting; an
optimizer with state would have to be a future candidate with its own
confirmation.

## What this phase will not claim

No generalization beyond the measured families and tasks; no claim of
intelligence, understanding, autonomy or agency from benchmark numbers; no
equality between arms whose difference is merely unresolved; no CPU/CUDA
numerical interchangeability beyond what the parity tests measure.
