# Where could a world model help AAA? A diagnosis of `aaa.python.v1`

Date: 2026-09-26. Status: **development diagnosis**, written before any world-model
code or experiment. Canonical base: `main` at `dbee13c` (PR #29 merged). Every
number below is quoted from retained v1 evidence
([development report](../aaa_python_v1_development_report.md), section 9 for
confirmation). Nothing here is new evidence.

## Definition used

A *world model* (WM) is a learned model conditioned on AAA's pre-action state
**and a candidate action** that predicts a legitimate consequence of that action,
meaning something a real run would reveal only after the action. A label guess that
is scored against an answer key has no consequence beyond being right or wrong, so
a model of it is the policy itself, not a world model.

## Per family

| Family | Visible before action | Action | Post-action feedback | Hidden | Do actions change future state? | Could a counterfactual change the action? | Strongest non-learned | Strongest tool | Reference (4K, confirmation) |
|---|---|---|---|---|---|---|---|---|---|
| `syntax` | source | valid / invalid | compile status, error line | parser result | no | no: the action is the prediction | lint rules 0.956 | none (running it is the oracle) | 0.942 |
| `outcome` | source | one of 6 categories | status, exception, error line | execution | no | no | fitted rules 0.709 | none | 0.680 |
| `output` | source | integer -50..50 | stdout | execution | no | no | majority 0.182 | none | 0.171 |
| `localize` | source (2 faults) | line | exception, error line | execution | no | no | first risky line 0.607 | none | 0.633 |
| `repair` | function, repair line, 4 candidates, 2 visible tests | candidate index | hidden-test results of the **chosen** candidate | other candidates' results, hidden tests | no (one step) | **yes**: predicting each candidate's test consequences ranks them | edit-type prior 0.435 | visible-test tool 0.953 | 0.657 (no tool); 4K+tool 0.953 (dev) |

## The deficiency classes

- **A. Representation.** This is dominant on `outcome` and `output`, which need values. `e2` deliberately
  never propagates values (constant propagation would compute the answer), and learners stay
  at or below fitted rules. This is a deficiency in *program-execution modeling*, not in
  modeling the consequences of AAA's actions.
- **B. Optimization.** This is minor. Budgets were tuned, a 64-epoch extension changed little, and clipping helped 1K.
- **C. Capacity.** Capacity helped through ~4K. From ~4K to ~10K, the gain is unresolved on confirmation.
- **D. Memory.** This is not measured as a limiter. Tasks are independent, and there is no recurrent state.
- **E. Counterfactual prediction.** This applies **only to `repair`**, and there it already exists in a
  degenerate form: v1's repair head *is* a one-step action-conditioned consequence
  model (a logistic scorer of P(chosen candidate passes hidden tests | view, candidate),
  trained by bandit feedback). A "WM-0" on v1 is structurally that head plus richer
  consequence targets.
- **F. Multi-step planning.** This does not apply. Every v1 episode is one action, and no action changes a later
  state.
- **G. Benchmark shortcuts.** These were repaired (`AAA-192`..`AAA-194`) and are attacked every run. Remaining known
  facts: one of four repair candidates equals the visible buggy line (`AAA-207`), and the
  medoid attack scores 0.328.
- **H. Tool use.** This is solved on `repair`: the real visible-test tool alone scores 0.953, and learners
  using it match it.
- **I. Language.** This does not apply. v1 has no natural-language input.

## Consequences for the world-model program

1. **v1 lacks the causal structure a world model needs to earn its place.** Four of five
   families have no action consequence beyond scoring. In the fifth, the real
   simulator (running visible tests) is free, exact on what it covers, and already
   near the ceiling (0.953). A learned simulator cannot beat a free, exact
   one on the task that tool solves. The honest v1 question is narrow: *without the tool*,
   does a model trained to predict per-candidate test consequences (from tool observations
   gathered in training) choose repairs better than the direct scorer trained on the same
   experience? The v1 dev pool can answer that as a development diagnostic (WM-0 below).
   It cannot support a world-model promotion, because the tool baseline dominates any
   no-tool arm and the v1 confirmation is spent.
2. **A successor benchmark with real sequential causal structure is required.** Its actions
   must change state (edits persist), its feedback must arrive over time (test runs), it
   must have a genuine reason not to run everything, and a strong tool-using baseline
   must keep the same budget. The reason not to run everything is that the candidate edit space
   is larger than an affordable number of test runs, as in real debugging. This does
   *not* mean crippling the tool. The tool stays exact; the question is where to spend it.
   Candidate: `aaa.python.seq.v0` ([design](sequential_benchmark_design.md), to be written).
3. **Execution modeling (deficiency A) is a separate hypothesis.** A model of program-state
   transitions (statement -> next variable state) is a world model *of the program*, not
   of AAA's actions. It is a candidate representation-learning objective (J-0/J-1 style
   JEPA), and it must earn its place through end-to-end task gains like anything else.
   It is recorded here so it is not confused with action-conditioned modeling.

## Declared first diagnostic (WM-0 on v1 development identities)

- **Question.** On v1 `repair` with the tool **disabled at decision time**, does a consequence model
  that predicts each candidate's per-visible-test outcome (action = candidate edit, conditioned
  on the test input), trained from logged tool observations on the *training* pool, pick
  better candidates than (a) the 4K reference repair scorer, and (b) a matched direct
  scorer trained on the same tool observations as a scalar target?
- **Why (b) matters.** The WM arm sees extra supervision (tool results on training tasks). Any gain
  over (a) could be data, not modeling. (b) receives exactly that data without the
  per-test, action-conditioned consequence structure.
- **Identities.** v1 `train` (learning) and `development` `tune`/`evaluate` ranges only. **No
  attack or confirmation pool.**
- **Expected outcome (stated before running).** Small or no gain over (b). If the gain over (a)
  vanishes against (b), the result is "more supervision helps", not "world model helps".
