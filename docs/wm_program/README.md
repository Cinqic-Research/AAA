# World-model and language program (`research/aaa_wm`)

Started 2026-09-26 from `main` at `dbee13c`. Branch: `opus/world-model-program`. Status: **two
confirmations observed; independent review of the world-model result done.** Nothing is merged or
promoted into the canonical line without owner review. Start with [handoff.md](handoff.md).

## Question

Can a world model give AAA a measurable, causal, generalizable decision advantage over the
strongest fair reference without it, under fresh held-out evidence? After that, does a
from-scratch American-English language model integrate usefully through explicit interfaces?

## Documents, in order

| Document | What it records |
|---|---|
| [diagnosis.md](diagnosis.md) | where a world model could help `aaa.python.v1` (only `repair` has action consequences; the free tool scores 0.953) |
| [wm0_v1_result.md](wm0_v1_result.md) | WM-0 on v1: +0.022 [-0.002, +0.049] over a compute- and information-matched direct scorer, which is **inconclusive** |
| [sequential_benchmark_design.md](sequential_benchmark_design.md) | the `seq.v0` draft, **rejected** by the red team (a RUN carried no information) |
| [opaque_benchmark_design.md](opaque_benchmark_design.md) | `aaa.python.opaque.v0`: debugging code that calls an opaque library |
| [opaque_tune_decisions.md](opaque_tune_decisions.md) | tune-stage results and the selection of the candidate and controls |
| [compute.md](compute.md) | measured compute and memory, the 2026-09-26 freeze incident and its fix |
| [opaque_evaluate_stage.md](opaque_evaluate_stage.md), [opaque_attack.md](opaque_attack.md), [opaque_adaptation_dev.md](opaque_adaptation_dev.md) | development evaluate stage, attack pool (used once), adaptation/retention/plasticity |
| [jepa_result.md](jepa_result.md) | J-4 action-conditioned JEPA: no decision gain; collapse diagnostics behave as designed |
| [opaque_confirmation.md](opaque_confirmation.md) | **world-model confirmation: 11/11 contracts PROMOTE, claim narrowed after review** |
| [independent_review_opaque.md](independent_review_opaque.md) | independent adversarial review of that confirmation |
| [pr30_independent_review.md](pr30_independent_review.md) | Codex GPT-6 AI review of the PR, repaired gates and remaining merge blockers |
| [language_track_design.md](language_track_design.md) | American-English corpus, tokenizer decision, size ladder, adapters v1/v2 (pre-registered) |
| [language_factorial_dev.md](language_factorial_dev.md) | AAA x WM x LM factorial (development) |
| [language_confirmation.md](language_confirmation.md) | **language confirmation: L2 PROMOTE; L1, F1, F2 INCONCLUSIVE; integration success not declared** |
| [mutation_results.json](mutation_results.json) | 12/12 deliberate boundary breaks caught |
| [handoff.md](handoff.md) | what is and is not established; next steps |
| [timeline.md](timeline.md) | reconstructed timeline and time accounting, including lost time and process lapses |
| `state.json` | machine-readable program state |

The literature and red-team notes are kept in the workspace (`/media/cinqic/Cinqic
Storage/AAA/wm/notes/`). They are working material, not canonical evidence.

## Code

| Path | Role |
|---|---|
| `research/aaa_wm/v1_wm0.py` | WM-0 diagnostic on v1 repair |
| `research/aaa_wm/seq/` | rejected `seq.v0` draft (kept for the record) |
| `research/aaa_wm/opaque/` | benchmark, environment, baselines, neural arms, WM-S, JEPA objective, freeze, confirmation, summaries |
| `research/aaa_wm/lm/` | American-English corpus pipeline, tokenizer, LM, training, evaluation |
| `research/aaa_wm/lang/` | language-conditioned statements, rule baseline, LM adapter, language-conditioned play |
| `research/aaa_wm/capped.sh`, `jobs.py`, `killjobs.sh` | memory-capped job control (see compute.md) |

## Environments

`requirements-lock.txt` (unchanged) covers everything except the neural arms and the LM.
Those use the isolated `requirements-torch-lock.txt` environment. Tests that need torch
skip without it.

## Protected identities

No fingerprinted file of `aaa.1k.v1`, `aaa.1k.v2`, the observation-noise protocol,
`aaa.python.v0` or `aaa.python.v1` is modified. The v1 confirmation stays spent.
`tools/check_protected_identities.py` passes.
