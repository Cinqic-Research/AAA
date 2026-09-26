# WM-0 on `aaa.python.v1` repair: development result

Date: 2026-09-26. Status: **development diagnostic, not confirmation, not a promotion
candidate.** Code: [`research/aaa_wm/v1_wm0.py`](../../research/aaa_wm/v1_wm0.py). The question
and the expected outcome were written in [diagnosis.md](diagnosis.md) before the first run.

## Setup

- **Identities.** v1 `train/repair` (2,400 tasks) for learning; v1 `development/repair` `tune`
  [0, 400) for selecting the rate and epoch count; `evaluate` [400, 1600) for the numbers
  below. The attack and confirmation pools were not touched.
- **Experience** (the same for every arm). Each training task passes through v1's `ToolEnvironment`:
  present, the logged visible-test tool, act, commit, reveal, learn. The learning signals are the
  bandit hidden-test feedback on the chosen candidate and the tool's per-candidate, per-visible-test
  pass/fail observations. The tool is never called at evaluation. Tool results were precomputed once
  with the real sandboxed tool and served by a cache subclass with the same boundary checks.
- **Model.** v1's `e2` candidate encoding (D = 256), one shared `tanh` core with 11 units (the ~4K
  size), and logistic heads. Every arm has between 3,543 and 3,555 trainable parameters.
- **Selection.** For each arm, the best 2-seed mean on `tune` over rates {0.03, 0.1, 0.3} and
  epochs {4, 8, 16}. Ties go to fewer epochs, then to the lower rate. Evaluation: 10 fresh
  initializations x 1,200 tasks. Intervals are crossed percentile bootstraps (seeds x tasks,
  4,000 draws).

## Results (frozen accuracy on `evaluate`, tool disabled)

| Arm | What it learns from / how it decides | Accuracy [95%] |
|---|---|---|
| `ref` | bandit hidden feedback only (v1 formulation) | 0.619 [0.588, 0.648] |
| `direct` | + scalar "all visible tests pass" per candidate (same data as `wm`) | 0.650 [0.623, 0.677] |
| `direct_t` | `direct` + pooled visible-test features at decision time (same information as `wm`) | 0.713 [0.689, 0.737] |
| `direct_t2` | `direct_t` with as many gradient steps per candidate as `wm` (compute-matched) | 0.716 |
| **`wm`** | per-(candidate, visible test) consequence head, action- and test-conditioned | **0.739 [0.715, 0.763]** |
| `wm` consequence disabled | trained `wm`, hidden-pass head only | 0.491 [0.440, 0.544] |
| `wm` shuffled actions | consequence of candidate k scored with candidate k+1's encoding | 0.129 [0.113, 0.146] |
| `wm` random weights | untrained | 0.226 [0.202, 0.249] |
| v1 baselines, same tasks | `rules` 0.423, `medoid` 0.320, **`visible_tests` tool 0.945** | from the v1 report |

Intrinsic diagnostic: the consequence head predicts per-test pass/fail at 0.835 accuracy
against the real tool.

| Contrast | Difference [95%] | Reading |
|---|---|---|
| wm - ref | +0.120 [+0.091, +0.151] | positive |
| wm - direct | +0.089 [+0.067, +0.110] | positive |
| direct_t - direct | +0.062 [+0.038, +0.088] | using the visible tests' values explains much of it |
| wm - direct_t | +0.026 [+0.008, +0.044] | positive, but compute-unmatched |
| **wm - direct_t2** | **+0.022 [-0.002, +0.049]** | **INCONCLUSIVE** (compute- and information-matched) |

## Verdict

- The gain of the "world model" over v1's formulation (+0.12) comes mostly from **more supervision
  (+0.03)** and from **using the visible tests' values (+0.06)**. v1's encoder does not read the
  visible tests at all. Neither effect needs a world model.
- Against the matched control, the action-conditioned, per-test consequence structure adds
  **+0.022 [-0.002, +0.049], which is unresolved**. WM-0 has **not** earned a place on v1.
- The causal ablations show that the trained model's decisions *depend on* its consequence
  predictions. Disabling them or shuffling actions destroys performance. That dependence is necessary
  for a world-model claim, but it is not sufficient: the matched direct scorer does almost as well.
- The real tool (0.945) dominates every no-tool arm. That is expected: it is an exact simulator of
  what it checks.

This matches the prediction written before the run and the literature expectation
(`notes/lit_world_models_jepa.md`, section 9.1). v1 does not have the causal structure a world model
needs. The program moves to [`aaa.python.opaque.v0`](opaque_benchmark_design.md).

## Limitations

- These are development identities. The tune/evaluate split is disjoint, but it is one pool
  family, and there are no attack-pool or confirmation claims.
- The matched-compute control repeats identical steps. It is not a different optimizer schedule.
- The "information-matched" arm pools the tests' features, which is one choice among several.
