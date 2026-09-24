# `aaa.python.v1` research brief: the evidence-gated pre-scale phase

Status: **development**. Written and committed before any v1 learner observed
a v1 development identity. Nothing here is a capability claim.

## The question

> Does roughly ten times Champion 1's trainable capacity (about 10K trainable
> parameters, against 994) solve a persistent, diagnosed limitation of AAA's
> Python learner that representation, output formulation, optimization, task
> construction or parameter-neutral mechanisms cannot solve adequately?

The question is *not* whether a 10K model can be built. The phase may end
with any of `SCALE_JUSTIFIED`, `REPRESENTATION_LIMITED`,
`OPTIMIZATION_LIMITED`, `TASK_OR_BENCHMARK_LIMITED`, `MIXED` or
`SCALE_NOT_JUSTIFIED`, and was designed without knowing which.

## Model lines this phase keeps apart

| Line | What it is | Status in v1 |
|---|---|---|
| Champion 1 | 994-parameter recurrent GRU dot-era reference with persistent state and online learning | frozen history; not modified; its parameter count is the "1K" reference point |
| `aaa.python.v0` learner | 153,600-parameter `OnlineLinear` pipeline instrument, two thirds of it the 101-way output head | frozen history; its negative result stands; re-run on v1 tasks only as an anchor arm |
| `aaa.python.v1` family | fixed encoder -> optional shared `tanh` core -> task heads; the core width is the capacity variable | new; its ~1K, ~2K, ~4K, ~10K and ~20K members are *diagnostic sizes of one family*, not product milestones |

The v1 family is not a widened Champion 1. Champion 1 consumes three
real-valued dot features; forcing program text through its GRU would test an
adapter. What v1 inherits is Champion 1's discipline: online learning from
post-action feedback, complete state accounting, cloning into matched online
and frozen twins, and development/attack/confirmation separation.

## Diagnostics before design

These were run on identities that v0 had already observed (v0 train and
development pools) with v0's own, unmodified learner, before the v1 design was
fixed. Scripts and outputs are retained under
[`evidence/aaa_python_v1/diagnostics/`](evidence/aaa_python_v1/diagnostics/).

1. **The v0 learner was under-trained (`AAA-197`).** Holding features and rate
   fixed and only adding passes over the same 600 training programs per
   family (3 initializations, all 400 development tasks, frozen):

   | epochs | syntax | outcome | output | localize | repair |
   |---:|---:|---:|---:|---:|---:|
   | 2 (v0) | 0.533 | 0.641 | 0.165 | 0.267 | 0.254 |
   | 6 | 0.816 | 0.677 | 0.159 | 0.256 | 0.325 |
   | 20 | 0.893 | 0.677 | 0.155 | 0.260 | 0.330 |

   Syntax, and partly outcome, were *optimization-budget* limited; output and
   localization are not. v0's statement that surface heuristics beat the
   learner on syntax remains true for v0's declared budget, and says little
   about the learner class.
2. **Hashing collisions are a small, D-dependent effect.** Signed hashing with
   a dedicated bias improved v0's learner at D = 256 (syntax +0.035, 5 of 5
   initializations positive) and was neutral at D >= 1024; D itself barely
   mattered for the linear learner (all family changes <= 0.03 between 256 and
   4096). v1 adopts signed hashing because a nonlinear core needs a small D to
   keep its parameter count honest.
3. **Benchmark shortcuts** (`AAA-192`..`AAA-194`): the repair medoid rule
   (0.90-0.93 without executing anything), v0's `outcome` "heuristic" being the
   majority baseline, and trivially localizable single-fault programs.
4. **Precision.** Stream variance dominates v0's contrasts; v0's 3 x 4
   design gave 95% half-widths of 0.04-0.15 and null-coverage of 0.84-0.90.

## What v1 changes in the exam

The generator `aaa.python.gen.v1` (see [protocol](aaa_python_v1_protocol.md)):
symmetric repair candidates (the medoid rule falls to 0.32, chance among the
three non-current candidates); always-two-fault localization; declared slices
`in_distribution`, `novel_literals`, `novel_names`, `novel_structure`,
`novel_composition`; larger pools; exclusion of every normalized v0 source hash
from v1's non-training splits; confirmation only under a committed freeze.

## Hypotheses and how each can fail

| # | Hypothesis | Mechanism it tests | Falsified by (development) |
|---|---|---|---|
| H1 | Structure the lexical encoder cannot express limits syntax, localization and outcome | `e2` channels (alpha roles, line structure, static flow) | `e2 - e1` at matched core is not a resolved gain >= 0.02 on any of those families |
| H2 | Formulation, not capacity, limits `localize` and `output` | pointer and ordinal heads | the alternative head does not reach one-hot accuracy - 0.01 |
| H3 | Capacity limits some family once representation and budget are fair | core width | `10K - 1K` not a resolved gain >= 0.03 on any family |
| H4 | A 4K core already captures any capacity gain | saturation | `10K - 4K` is a resolved gain >= 0.02 on at least two families |
| H5 | Online learning helps after a distribution switch | drift design | the adaptation difference-of-differences is not resolved positive |
| H6 | The learner keeps its ability to learn late in life | plasticity | the late/early permuted-label learning ratio has an upper bound below 0.9 |
| H7 | A learner can use a tool's evidence at least as well as the tool | tool channel | the tool-augmented learner is resolved below the visible-test baseline on repair |

## Development plan (declared)

All stages use [`experiment.DESIGN`](../research/aaa_python_v1/experiment.py):
D = 256; the full 2,400-program training pool per family; per-arm tuning of
learning rate {0.03, 0.1, 0.3} x epochs {1, 3, 8} on the development *tune*
range with its own initializations; evaluation on 10 initializations x 30
streams x 40 tasks of the *evaluate* range, frozen and online; crossed
bootstrap intervals (4,000 draws); Holm across the five families of each
primary contrast.
Development results guide engineering. They are never confirmation.

1. **Encoders at matched capacity.** `e0`, `e1`, `e2` behind (a) the linear
   core and (b) a 16-unit core, all with v0's one-hot heads, so encoders are
   compared at *identical* trainable parameter counts and head formulation.
   `e2` channel ablations (drop `alpha`, `line` or `flow`) at the 16-unit core.
   The v0 instrument, unmodified, is scored on the same streams as an anchor.
   **Encoder selection rule:** the encoder with the highest mean (over
   families) frozen evaluation accuracy at the 16-unit core.
2. **Heads, with the encoder held fixed.** At the selected encoder and the
   16-unit core: one-hot/one-hot, ordinal/one-hot, one-hot/pointer,
   ordinal/pointer. **Head selection rule, per family:** adopt the
   alternative head when its frozen accuracy is at least the one-hot head's
   minus 0.01 (a tie goes to the head with fewer parameters).
3. **Capacity.** Selected encoder and heads; core widths chosen by
   `hidden_for_budget` for budgets of about 1K, 2K, 4K, 10K and 20K trainable
   parameters; each size tuned separately. Data scaling: the 1K and 10K sizes
   also with 600 training programs per family.
4. **Optimization, one change at a time**, at 1K and 10K: weight decay 1e-4;
   momentum 0.9 with the rate scaled by 0.1 (velocity counted as state);
   gradient clipping at norm 1.
5. **Tool.** Repair with and without the visible-test tool inputs at the
   selected core sizes, against the `visible_tests` baseline.
6. **Adaptation and retention** (drift design) at 1K and 10K: 10 initializations x
   15 streams, branches of 20 tasks (the adapt range holds 300 `novel_structure`
   tasks per family), probe bank of 100.
7. **Plasticity** at 1K and 10K ([plasticity protocol](aaa_python_v1_protocol.md#plasticity)).

## Capacity decision rule (declared before any capacity arm ran)

Let `d1_f = acc(10K) - acc(1K)` and `d4_f = acc(10K) - acc(4K)` be frozen
evaluation-accuracy contrasts per family, with crossed 95% intervals and Holm
adjustment over the five families of each contrast.

- A family is **materially improved** by 10K if `d1_f >= 0.03` and its
  Holm-adjusted contrast is resolved positive.
- 10K **earns its size over 4K** on a family if `d4_f >= 0.02` and it is
  Holm-resolved positive.
- **`SCALE_JUSTIFIED`** (for development, i.e. "worth confirming") requires at
  least three materially improved families, at least two families where 10K
  earns its size over 4K, no family with `d1_f` resolved below -0.02, and no
  adaptation, forgetting or plasticity measure resolved worse at 10K than at 1K
  by more than 0.03.
- **`MIXED`**: one or two materially improved families, or three or more
  without two families earning over 4K (then the smaller size is preferred).
- **`SCALE_NOT_JUSTIFIED`**: no materially improved family.

The phase outcome then weighs capacity against the other measured effects:
`REPRESENTATION_LIMITED` when the best encoder or head change at matched
capacity exceeds the 1K-to-10K capacity gain on the families where either
matters; `OPTIMIZATION_LIMITED` when training budget or optimizer changes
dominate; `TASK_OR_BENCHMARK_LIMITED` when strong simple or tool baselines,
or shortcuts, make the capacity question unanswerable on a family.

## Confirmation

After development, the phase commits a freeze: source fingerprint,
specification, the selected arms, hypotheses, thresholds and
`aaa.promotion.crossed.v1` contracts. Only then are the 1,200-per-family
confirmation identities generated, once, from committed source. The freeze
and the confirmation result are reported whatever they say.
