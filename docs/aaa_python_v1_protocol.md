# `aaa.python.v1` benchmark protocol

The machine-readable protocol is the packaged specification
[`research/aaa_python_v1/data/aaa_python_v1.json`](../research/aaa_python_v1/data/aaa_python_v1.json)
(`python -m research.aaa_python_v1 spec-hash`). Where this document and the
specification differ, the specification is what runs, and the difference is a
defect. v1 is a successor to [`aaa.python.v0`](aaa_python_protocol.md); v0's
specification, source, golden keys and evidence are unchanged.

## What is shared with v0, byte for byte

The safe subset, the AST validator, the sandboxed CPython oracle and its
resource limits, the counter-mode SHA-256 stream, the five task families,
their label spaces, their post-action feedback fields and the causal episode
boundary (present, act, commit, reveal, score, learn). v1 imports these
modules; they are part of v1's fingerprint, and the v1 specification refuses
to load if its `subset`, `sandbox` or `interpreter` section differs from v0's.

## What v1 changes

| Change | Why | Issue |
|---|---|---|
| Repair candidates are four single mutations of a hidden base line under a symmetric mutation relation | In v0 the correct line was the edit-distance medoid; a non-executing rule scored 0.90-0.93 | `AAA-192` |
| Every `localize` program has two value-dependent faults | Half of v0's localization programs had one fault, which "the only risky line" found 97-99% of the time | `AAA-194` |
| Slices: `in_distribution`, `novel_literals`, `novel_names`, `novel_structure`, `novel_composition` | Separate interpolation, new constants, new identifiers, new structure and new compositions of familiar structure | -- |
| Pools: train 2,400; development 3,600; probe 100; attack 600; confirmation 1,200 per family | Precision (v0's 3 x 4 design gave 0.04-0.15 half-widths) and the tuning/evaluation/adaptation separation | -- |
| Non-training splits exclude every normalized source hash of v0's train, development and probe pools | Those programs were observed during v0 | -- |
| A declared, logged pre-action tool for `repair`: run the candidates on the visible tests | The next capability rung; a strong zero-parameter baseline | -- |
| Confirmation is generated only under an admitted freeze | v0 refused confirmation entirely; v1 admits it only prospectively | -- |

### Slices

Training and probe tasks are `in_distribution`. Development, attack and
confirmation cycle, by index, through `in_distribution`, `in_distribution`,
`novel_literals`, `novel_names`, `novel_structure`, `novel_composition`.

- `novel_literals`: constants 10-19 instead of 0-9.
- `novel_names`: every identifier from a disjoint pool: variables
  (`u v z k p2 cnt tot hold prev cur item sval`), the function (`g`), its
  parameter (`r`), loop variables (`t s`), the repair local (`h`) and the
  undefined names that raise `NameError` (`ghost nope w9 zz`).
- `novel_structure`: the `nested_loop` and `list_walk` templates and the
  `loop_sum` repair template, never used for training.
- `novel_composition`: two different training templates in one program; one
  literal assignment of the second becomes `<result of the first> % k`.

Exact normalized-source duplicates of earlier splits are excluded, but
`in_distribution` is interpolation, not memorization-free novelty: many
development programs share an alpha-renamed skeleton with a training
program. Near-duplicate rates are reported with results.

### Repair construction (`AAA-192`)

`mutations(line)` swaps one operator (`+`/`-`, `>`/`>=`, `<`/`<=`, `*`/`+`,
in both directions) or moves one literal by -2, -1, +1 or +2 (never below
zero, never clamped). The relation is symmetric: `m in mutations(x)` iff
`x in mutations(m)`. A hidden base `h` is drawn from `mutations(correct)`,
three other candidates from `mutations(h)`, and the buggy line shown in the
program is one of those three. All four candidates are one mutation from `h`,
and `h` is never shown. The oracle still accepts a task only if exactly the
correct candidate passes all four hidden tests. Measured on v1 development:
the medoid rule scores 0.322, chance among the non-current candidates is
0.333, and a fitted prior over edit types scores 0.423 (a genuine regularity
of which edits fix bugs; it is the bar a learner must beat).

## Development layout

The development pool is split by index, and no task has two roles:

| Range | Role |
|---|---|
| 0-399 | `tune`: learning rate and training budget selection only |
| 400-1599 | `evaluate`: 30 streams x 40 tasks |
| 1600-3399 | `adapt`: distribution-switch branches |
| 3400-3599 | plasticity test blocks |

## The tool channel

`ToolEnvironment.run_visible_tests(view)` is available after `present` and
before `commit`, only for `repair`, only to agents declared to use it. It runs
each candidate, substituted into the visible buggy function, on the **visible**
tests and returns pass/fail per candidate and test. It reads only fields of
the issued view and never runs a hidden test; the call is logged as a `tool`
event. For `syntax`, `outcome`, `output` and `localize`, running the task's own
program *is* the oracle, so no tool exists and the call refuses.

## Baselines

`uniform`, `majority` (repair explores uniformly during training, `AAA-193`),
`lookup`, `v0_heuristic` (unchanged), `rules` (fitted stupid rules: lint;
risky-construct signature -> training majority for `outcome`; program shape ->
training majority for `output`; first risky line for `localize`; edit-type
success rate for `repair`), `medoid` (the permanent `AAA-192` attack) and the
tool baseline `visible_tests`. CPython is the oracle and never a baseline.

## Statistics

The unit is an `initialization x stream` cell. Intervals are crossed
percentile bootstraps (4,000 draws, 95%). A sign is reported only when the
interval excludes zero; a zero-width interval from arms that never disagreed is
`DEGENERATE`. Each stage's primary contrasts are Holm-adjusted. Two-way
variance components are reported as a cross-check.

## Plasticity

Plasticity is the ability to keep learning, not the retention of old
performance. A learner lives through an extended continual stream (the
training pool, repeatedly, online). At declared checkpoints a clone is given a
*new* mapping it cannot already know: `syntax` and `outcome` tasks whose
labels are permuted by a fixed permutation. It learns online on a declared
block and is scored on a held-out block under the permuted mapping. The same
probe from a freshly initialized model is the fresh-learner control (Dohare et
al. 2024). Loss of plasticity is a late-life learning gain materially below
the early-life gain. Weight norms, saturated and dormant unit fractions,
activation effective rank and gradient norms are recorded at every checkpoint.

## Confirmation

`build("confirmation", ...)` refuses unless given an
[`Admission`](../research/aaa_python_v1/freeze.py) that re-verifies, at the
moment of use, that `docs/evidence/aaa_python_v1/freeze.json` is committed,
the tree is clean, and the frozen phase fingerprint and specification hash
equal the running source's. The freeze declares the design, the selected
arms, hypotheses and the `aaa.promotion.crossed.v1` contracts before any
confirmation identity exists.
