# `aaa.python.opaque.v0`: tune-stage results and selections

Date: 2026-09-26. Status: **development, tune range only** (`development` indices 0-999, one
initialization). These numbers selected the candidate and the controls for the `evaluate`
stage. They were not used to set any confirmation threshold, and they support no claim.

## Tune results (seed 0, runs budget R = 2, steps 8; success = domain equivalence)

| Arm | Kind | All | in-dist | two_fault | novel_literals | novel_grammar | library_B |
|---|---|---:|---:|---:|---:|---:|---:|
| true-library planner, depth 2 | ceiling (not a baseline) | 0.794 | 0.840 | 0.700 | 0.845 | 0.770 | 0.815 |
| true-library planner, depth 1 | ceiling | 0.569 | 0.660 | 0.235 | 0.675 | 0.630 | 0.645 |
| **WM-S online** (exact interpreter + learned library, online) | world model | **0.507** | 0.540 | 0.465 | 0.600 | 0.530 | 0.400 |
| WM-S online, unknown prior 0.1 / 0.5 | sensitivity | 0.511 / 0.500 | | | | | |
| WM-S frozen | world model | 0.337 | 0.410 | 0.265 | 0.405 | 0.390 | 0.215 |
| black-box neural WM, depth 1 | world model | 0.191 | 0.295 | 0.110 | 0.235 | 0.110 | 0.205 |
| WM-S frozen, depth 1 | depth ablation | 0.183 | 0.270 | 0.025 | 0.275 | 0.205 | 0.140 |
| black-box neural WM, depth 2 | world model | 0.132 | 0.180 | 0.150 | 0.155 | 0.050 | 0.125 |
| policy_aux's auxiliary consequence head used as a world model, depth 2 | world model | 0.124 | 0.170 | 0.155 | 0.155 | 0.045 | 0.095 |
| neural policy + aux RUN-prediction head | model-free (supervision-matched) | 0.121 | 0.190 | 0.035 | 0.160 | 0.105 | 0.115 |
| WM-S, library values permuted | ablation | 0.110 | | | | | |
| WM-S, identity library | ablation | 0.102 | | | | | |
| neural policy | model-free reference | 0.067 | 0.115 | 0.020 | 0.110 | 0.030 | 0.060 |
| black-box WM, shuffled queries | ablation | 0.058 | | | | | |
| WM-S, empty library (interpreter only) | ablation | 0.043 | | | | | |
| neural value search (visible-pass target), depth 2 | value-equivalence control | 0.050 | 0.070 | 0.045 | 0.065 | 0.035 | 0.035 |
| neural value search, depth 2 | value-equivalence control | 0.042 | 0.050 | 0.030 | 0.070 | 0.035 | 0.025 |
| tool search, gap-reading | best non-learned | 0.029 | | | | | |
| WM-S online, stale predictions | action control | 0.029 | | | | | |
| black-box WM, output disabled / random weights | ablation | 0.029 / 0.020 | | | | | |
| WM-S online, shuffled plan-prediction pairing | action control | 0.021 | | | | | |
| edit-type prior / submit as-is | floors | 0.016 / 0.000 | | | | | |

Tool search with more runs (tune): R = 8 gives 0.069 to 0.096, R = 16 gives 0.123 to 0.166, and R = 32 gives 0.197 to 0.279.

Intrinsic numbers:

- The black-box WM predicts the exact result class with 0.364 accuracy on training programs and 0.187 on development programs.
- WM-S learned 6,845 library entries from 5.2M real observations of 200,000 training programs, with 0 wrong entries (checked on the evaluator side).

## Diagnosis

- **The black-box neural WM is limited by execution modeling (deficiency A).** It has to learn Python
  semantics *and* the library from end-to-end observations. Its depth-2 planner does worse than
  depth 1: more candidates means more confident false positives (objective mismatch).
- **WM-S splits the dynamics.** It executes the visible code exactly (public knowledge) and learns
  only the unknown component, the library. The learned library model is **causally necessary**:
  an empty, identity or permuted library drops it from 0.337 frozen / 0.507 online to between 0.04 and
  0.11. Its decisions depend on action-conditioned predictions: shuffled or stale predictions give
  0.021 and 0.029. Depth 2 matters (frozen 0.183 at depth 1 against 0.337 at depth 2, most of it
  on two_fault). Online updating from the agent's own real `RUN` observations adds +0.170
  [+0.134, +0.207], including on the changed library (0.215 to 0.400).

## Selections for `evaluate` (made here, before any evaluate-range run)

- **Candidate:** `wms:online`. Depth 2, unknown-entry prior 0.3 (the default; the sensitivity check
  above is flat), a table learned from 200,000 training programs sampled with the initialization seed,
  and online updates reset at the start of every stream.
- **Reference (strongest learned non-world-model arm on tune):** `policy_aux`. Also reported: `policy`,
  `value`.
- **Strongest non-learned:** `tool_gap` (R = 2).
- **Black-box world model representative:** `wm:d1` (better than depth 2 on tune).
- **Causal controls:** `wms` (frozen), `wms:empty`, `wms:identity`, `wms:random`,
  `wms:online+shuffle`, `wms:online+stale`, `wms:d1`.
- **Ceiling:** `ceiling` (true library, depth 2), reported separately.
- **Initializations:** seeds 0-4 for WM-S (the table subsample and the order of online learning);
  seeds 0-2 for the neural arms (training initialization and data order).

## Scope limits recorded now

- WM-S's interpreter is hand-written knowledge of Python semantics. The claim under test is
  therefore narrow: *given* exact knowledge of the visible language, a learned model of the
  unknown environment component, used for planning, improves AAA's decisions over model-free
  learners, value search, and the same agent without the learned model. It is not a claim that a
  generic neural world model learned Python.
- Compute is not matched. WM-S spends about 0.3-0.4 CPU seconds per episode imagining up to about
  1,000 programs. The neural policy spends about 0.06 seconds. The scarce resource in this benchmark
  is real runs (R), and every arm has the same number of them.
