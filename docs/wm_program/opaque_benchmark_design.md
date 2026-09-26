# `aaa.python.opaque.v0`: debugging against an opaque library (design)

Status: **design, prospective**, 2026-09-26. It supersedes the rejected
[`seq.v0` draft](sequential_benchmark_design.md), which remains in history. That draft's
red-team review is at `notes/redteam_seq_design.md` in the program workspace, and its key
numbers are quoted in the next section. No identities exist yet except the `pilot` split,
which is reserved for benchmark design and never reused.

## Why seq.v0 was rejected

The red team showed the following, and the pilot reproduced it:

- **`RUN` had no information value.** Its result is a pure function of the visible program and the
  visible inputs, so an agent that runs Python locally enumerates every program within two
  edits and scores about 0.93 with **zero** runs. The "perfect-consequence planner" (0.902 on the
  pilot) *is* that enumerator. A learned world model there is only a learned interpreter
  (deficiency A), and forbidding local execution would be an artificial handicap.
- **The generator leaked.** Whole-token mutations left thresholds and `range` bounds unmutable.
  Fixed template constants flagged faulty lines. The program universes were tiny. One fault
  often masked the other. In 23-41% of branching episodes, 9 tests accepted a semantically
  wrong fix.
- **The comparisons were unfair.** There was no supervision-matched `ref`, the planner had more
  inference compute, and the statistics were too coarse.

## The fix: the consequence depends on something the agent cannot run

The program calls **opaque library functions** `api0 .. api{M-1}`. Their implementations belong
to the environment, a *library version* the agent never sees. This is the everyday situation
of code that calls a remote service, a compiled extension, hardware, or a dependency whose
behavior is known only from experience. Local execution is impossible. The only way to
learn what a call does is to observe real runs, so `RUN` has genuine information value, and a
model of the library's consequences is a **world model in the proper sense**: a learned model
of environment dynamics that the agent's actions interact with.

The library is fixed within a *world version*. Training and development use version `A`. The
adaptation branch switches to version `B`, where some functions changed behavior, the way a
dependency upgrade does. A world model must adapt online from real observations, and a
retention probe returns to `A`.

## Programs

A random grammar, not fixed templates. It draws 3-7 statements over the parameter `p`
and a local `q`:

    q = E | q += E | q -= E | if C: S [else: S] | for i in range(n): S | return E

Expressions use `p`, `q`, integer literals, `+ - * // %`, and calls `apiK(E)`. Conditions use
`> >= < <= == !=`. Every program calls at least one API, and every literal is drawn from a
range. No constant or operator slot is fixed, so a line's text never identifies it as faulty.

## Faults and edits (one symmetric relation, over lexical tokens)

Edit sites come from `tokenize` tokens, so thresholds, `range` bounds and literals before `:`
are all sites. The symmetric mutation relation `mut`:

- integer literal `v -> v +- 1, v +- 2`. References draw literals from a range 2 inside the
  literal domain, so a mutation never clamps.
- `+ <-> -`, `* <-> +`, `// <-> %`, `> <-> >=`, `< <-> <=`, `== <-> !=`, `+= <-> -=`;
- `apiA -> apiB` for any other library name (API confusion).

A task injects `k` in {1, 2} faults at distinct sites. The generator accepts a task only if all
of the following hold:

- the buggy program fails at least one visible test;
- every fault is individually necessary: undoing any one of them is not equivalent to the
  reference on the full domain;
- no program within `k` edits of the buggy program is visible-test-equivalent and
  domain-equivalent to the reference other than through the true fix and its equivalents. That
  is, we do not reject alternate fixes; we count them as success.

## Success and hidden evaluation

**Success means domain equivalence.** The submitted program must produce the same result as the
reference, value or exception class, on **every** input of the declared domain `D` (for example,
`p` in [-6, 20], 27 inputs). There is no finite hidden test sample to overfit, so
"accepted-but-wrong" is zero by construction. Hidden evaluation happens once, at `SUBMIT`.

## Episode

- **Visible to the agent:** the program, `V = 3` visible tests `(p, expected)`, the history of its
  own actions and real `RUN` observations, and its remaining budgets.
- **Actions:** `EDIT(site, replacement)` (1 step); `RUN` (1 step and 1 run), which returns, for each
  visible test, `pass`, `wrong(got)` or `error(class)` for the *current* program; `SUBMIT`.
- **Budgets:** at most `S = 8` steps. The run budget `R` is swept over {0, 1, 2, 3, 4, 8}. The
  primary estimand is at `R = 2` (declared cost model: a run costs about as much as the whole
  reasoning step it verifies). The full curve is always reported.
- **Post-episode feedback** (for learning, only after `SUBMIT`): the success bit and the submitted
  program's per-input results on `D`. This is the analogue of v1 bandit feedback, delivered only
  at the end.

## Splits

`pilot` (design only), `train`, `development` (tune / evaluate / adapt), `attack`, and
`confirmation`, with disjoint indices and disjoint normalized program hashes. Every stream
contains a single slice (no mixed-slice streams). Slices:

- `in_distribution`
- `two_fault` (k = 2 only)
- `novel_literals` (literal range shifted)
- `novel_grammar` (held-out statement kinds, for example nested `if` in `for`)
- `library_B` (adaptation only)

## Arms

All learned arms use one token encoder family and one parameter budget, and they train on **one
shared behavior dataset**: the same training programs, the same executed runs, and the same
episode feedback.

- `policy` (model-free AAA reference): `Q(program, edit, visible tests, last RUN)`, the probability
  that the edit is part of a fix. It is used with verified tool search (RUN, keep or undo).
- `policy_aux`: `policy` with an auxiliary head that predicts `RUN` observations. It gets the same
  labels as the world model, and its decisions do not use them. This separates "more supervision"
  from "world-model use".
- `value_search`: `V(program', visible tests)`, the probability that `program'` passes all visible
  tests, used by the same planner as the world model (the value-equivalence control).
- `wm_search`: `C(program', p) -> distribution over the observation` (value or exception class) for
  each visible input. The planner scores plans (1 or 2 edits) by the predicted log-probability that
  every visible test passes, verifies the best plan with a real `RUN`, and re-plans on the
  observation.
- `wm_latent` (JEPA, J-2): `z' = T(z, a)` predicts the post-edit representation, with an EMA target
  encoder and a VICReg-style anti-collapse term. The consequence head reads `z'`. Compared against
  the exact-text transition.
- Controls: `wm` with random weights, with frozen initial weights, with its output disabled (the
  planner falls back to `policy`), with shuffled edit inputs, and with shuffled visible-test
  inputs. Depth 1 versus depth 2. A compute-matched `policy`, which ranks the same number of
  candidates with an ensemble or MC dropout.
- Non-learned: `submit_asis`, a fitted edit-type `prior`, `tool_search(prior)`, the
  `got`-gap hill-climber, `lookup` (exact program memory), and fixed-slot/mutable-site attacks.
- Ceiling (not a baseline): `true_library_planner`, the planner run with the real library.

## Adaptation

Development only until frozen:

- Online versus frozen world model on the `library_B` switch.
- Learning speed after the switch.
- Retention on `A` probes.
- Fresh versus experienced learners.
- The four frozen/online combinations of controller and world model.

## Statistics (to be frozen)

A `crossed` design with slice-pure streams: at least 30 streams of at least 40 episodes per
slice, and 10-20 fresh initializations. Error ratios use Jeffreys smoothing. Absolute-difference
non-inferiority applies against the best tool baseline. A gatekeeping order runs wm > policy,
then wm > value_search, then wm > policy_aux, then depth 2 > depth 1. The effect-size threshold
is fixed from development variance (MDE) before confirmation. Confirmation size follows from the
MDE and is expected to be several thousand episodes. Generation is cheap.
