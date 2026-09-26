# `aaa.python.seq.v0`: a sequential debugging benchmark (design draft)

Status: **draft for red-team review**, 2026-09-26. No identities exist yet. This is a new,
prospective benchmark. It does not modify `aaa.python.v0` or `v1`, whose modules it imports
read-only: the safe subset, the sandboxed oracle, the random stream and the symmetric
mutation relation `research.aaa_python_v1.generator.mutations`.

## Why a new benchmark

The [diagnosis](diagnosis.md) found that v1 has no multi-step causal structure. Its one
action-conditioned family (`repair`) is solved to 0.953 by a free, exact tool. A world
model can earn a place only where:

1. actions change the state that later actions act on;
2. the effects of an action are observed later, through a channel that is *not* free;
3. the space of possible actions is larger than what can be verified exhaustively;
4. a strong, fair tool-using baseline has exactly the same budget.

Real debugging has all four. Edits persist. Running tests takes time. A function
has many lines, each with many plausible edits. A careful engineer does not run the suite
after every conceivable edit.

## The task

A generated function `f(p)` of 4-8 lines in the v1 safe subset contains `k` injected
faults (`k` in {1, 2}, declared per slice). Each fault is one symmetric v1 mutation
on a distinct line. The agent sees the buggy source and `V = 3` visible tests
(`input -> expected`). It acts in steps:

| Action | Effect | Cost |
|---|---|---|
| `EDIT(line, m)` | replaces the *current* text of `line` with `m`, where `m` is in `mutations(current line)` (this includes undoing an earlier edit) | 1 step |
| `RUN` | executes the *current* program on the visible tests. The observation per test is `pass`, `wrong(got)` or `error(exception class)` | 1 step and 1 run |
| `SUBMIT` | ends the episode | none |

There are two budgets: at most `S = 6` steps and at most `R = 2` runs per episode (both declared in
the specification). Running out of steps submits automatically.

**Score.** The episode succeeds if the submitted program passes all `H = 6` hidden tests.
Any program that passes them counts, including an equivalent fix. Secondary metrics:
steps used, runs used, and the number of edits whose line was already correct (harm).

**What the agent never sees.** Hidden tests, the reference program, which lines are faulty,
the injected mutations, and hidden-test results. This holds even after `SUBMIT`, except for
the declared post-episode feedback. That feedback is one bit, success of the submission,
plus the hidden-test pass vector of the submitted program (the same kind of feedback as v1's
bandit repair feedback). It arrives only after `SUBMIT`.

## Why it has sequential causal structure

- With `k = 2`, no single edit can pass all tests. Credit for an edit arrives only after a second
  edit, or through the partial information a `RUN` gives (which tests pass, and the
  `got` values).
- `RUN` observations depend on the current program state, which the agent's own edits changed.
- The action space (about 4-8 lines x 2-8 mutations, roughly 20-50 edits) exceeds `R`. The
  agent must decide what to verify.

## Splits, slices and identities

Train 4,000 episodes, development 3,000 (tune 0-599, evaluate 600-2,399, adapt 2,400-2,999),
attack 600, confirmation 1,200, per the v1 separation rules. Confirmation is generated only
under a committed freeze. Slices: `in_distribution` (k = 1 or 2 on training templates),
`two_fault` (k = 2 only), `novel_literals`, `novel_names`, `novel_structure` (held-out templates),
and `novel_composition`. Normalized-source duplicates across splits are excluded, and near-duplicate
rates are reported.

## Baselines (all with the same step and run budgets)

1. `submit_asis`: submit without editing (floor).
2. `prior`: the edit-type prior fitted on training episodes (v1's `rules` analogue), `k_hat`
   edits on the most-likely lines, with no tool.
3. `tool_greedy`: in prior order, EDIT, RUN, keep the edit if the pass count improved, and undo
   it otherwise. Uses both runs.
4. `tool_analytic`: a hand-written debugger, deliberately strong. It reads the `got` versus
   `expected` pattern (constant offset, sign, off-by-one near a comparison threshold) and proposes
   the literal or operator edit that pattern implies, then verifies. This is the "careful human
   with a test runner" baseline, and the WM must beat it or honestly lose to it.
5. `oracle_planner` (ceiling, not a baseline): the same planner as the WM arm, but it queries the
   real sandbox for every imagined edit. It is the upper bound of model-based planning at this
   budget, and it shows how much headroom a perfect world model would have.
6. Attacks: majority edit, medoid-style geometry, a line-position prior, and fixed-output
   baselines.

## Learned arms (the factorial core)

All arms share one fixed encoder (v1 `e2` features of the *current* program and of each
candidate edit, plus a fixed hashed encoding of the visible tests and of the latest `RUN`
observation) and one parameter budget, targeting v1's ~4K reference size.

- `ref` (AAA reference, model-free): a scorer `Q(state, edit)` trained from episode outcomes
  (Monte-Carlo return, bandit-style), plus the declared submit rule. It uses `RUN` like
  `tool_greedy` does.
- `wm` (world model): a consequence model `C(state, edit) -> predicted RUN observation`
  (per-test pass, and the error class) of the program *after* the edit. It is trained only on
  `RUN` observations the agent actually made in training episodes, so the data is matched to
  `ref`'s episodes. The planner imagines edits, and pairs of edits for `k = 2`. Program-text
  transitions are applied exactly, since editing text is deterministic. Only the consequence is
  learned. The planner then chooses what to run and what to submit.
- `wm_latent` (JEPA-style, WM-4): the same, except the post-edit state representation is
  *predicted* from `z_t` and the edit embedding (`z_{t+1}_hat = T(z_t, a)`), with an
  EMA target encoder and a variance/covariance regularizer. It is compared against the exact
  text transition. When a text transition is exactly available, the latent transition must beat
  it to be kept.
- Controls: `wm` with random weights, with frozen initial weights, with the consequence output
  disabled (planner falls back to `ref`), with shuffled edit inputs, and with stale observations;
  a matched-parameter `ref` (same total parameters as `wm`); and a matched-compute `ref` (same
  number of gradient steps).

## Primary estimand (to be frozen)

Per `initialization x stream` cell, the Jeffreys-smoothed episode error ratio of `wm` against
`ref`, and of `wm` against the strongest non-learned tool baseline, under a versioned
`aaa.promotion.crossed.v1` contract over the declared slices as groups. The practical threshold
and every other rule are fixed in the freeze, before any confirmation identity exists.

## Open questions for the red team

- Can `tool_analytic` solve nearly everything? If it can, the benchmark does not require learning.
  That is a finding, not something to tune away. The response would be structure that
  pattern-reading cannot invert (loops, piecewise branches, two interacting faults).
- Does the `got` value in `RUN` make the task trivially invertible?
- Can line position or template identity reveal the faulty line?
- Is `R = 2` defensible, or does it look like a budget chosen to handicap the tool? The declared
  rationale is the action-space-to-run ratio; sensitivity to `R` in {1, 2, 3, 4} will be
  reported for every arm.
