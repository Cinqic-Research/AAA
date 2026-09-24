# `aaa.python.v0` benchmark protocol

The machine-readable protocol is the packaged specification
[`research/aaa_python/data/aaa_python_v0.json`](../research/aaa_python/data/aaa_python_v0.json)
(`python -m research.aaa_python spec-hash`). This document explains it. Where
they differ, the specification is what runs, and the difference is a defect.

The exam is separate from the learner. Changing the learner
([architecture](aaa_python_architecture.md)) does not change anything in this
document. Changing anything in this document changes the specification hash.

## Task families

Each family is a separately reported capability. There is no combined
"Python score".

| Family | Ladder rung | The learner sees | The answer (from CPython) | Post-action feedback |
|---|---|---|---|---|
| `syntax` | 1 syntax validity | source (half deliberately corrupted) | `valid` / `invalid` from `compile()` | compile status, syntax-error line |
| `outcome` | 5-6 execution success, exception class | source (two thirds with value-dependent faults) | `ok` or the exception class | status, exception, error line, stdout |
| `output` | 2-4 semantics, state, output | fault-free source ending in `print(...)` | the printed integer in [-50, 50] | stdout |
| `localize` | 10 bug localization | a program that fails | the failing line number | exception, error line |
| `repair` | 11 candidate repair selection | a buggy function, the line to repair, four candidate lines, two visible input/output tests | the index of the only candidate passing the four hidden tests | hidden-test results of the *chosen* candidate only |

Faults are value-dependent: `v // (u - k)`, `[a, b, c][v - k]`, an undefined
name inside `if v > k:`, `v + (str(u) if u > k else u)`, and
`int("7" if v > k else "z")`. Whether one fires depends on program state, so
most faulty programs run cleanly. Half carry a second fault as a decoy, so a
risky-looking line does not identify the failing one. The assigned name is a
variable the program never uses, so the fault line has no fixed marker.

All answer keys come from executing or compiling the program with CPython.
They never come from the generator's intent. A corruption meant to break
syntax that happens to leave valid code is labelled `valid`.

## Safe subset and sandbox

**Allowed:** assignment and augmented assignment to one name or `name[index]`;
`if`/`else`; `for` over `range` with at most three small integer-literal
arguments, nested at most twice; top-level `def` with plain positional
parameters, calling only earlier functions (no recursion); `return`; `pass`;
integer, short string and boolean literals; lists of up to ten items;
`+ - * // %`, unary `-` and `not`; comparisons; `and`/`or`; conditional
expressions; the builtins `print range len abs min max int str bool sum`.

**Refused by the validator:** imports; attribute access of any kind (which
blocks every dunder reflection escape); every other builtin, including
`eval exec compile open input globals locals getattr`; `while`; `lambda`;
comprehensions; `try`, `with`, `raise`, `class`, `del`, `global`,
`nonlocal`, `async`; f-strings; `**` and `/`; keyword arguments; slices;
names outside `^[a-z][a-z0-9]{0,11}$` or in the reserved list; source over
2,000 bytes, 40 lines or 400 AST nodes; integer literals above 999.

**Sandbox:** nothing unvalidated is executed; syntax checks only compile.
Each program runs in its own CPython process with `-I -S`, an empty
environment and a fresh temporary directory. The child sets CPU (2 s),
address-space (512 MiB), file-size (0) and process-count (0) limits where the
platform provides them, and records which it applied. Only the allowed
builtins exist in the program's namespace. A 5 s wall timeout kills it, and
stdout is capped at 4,096 characters. A child that dies or returns garbage is
`sandbox_failure`, `timeout` or `cpu_limit`, which is never mistaken for a
program outcome. `python -m research.aaa_python safety` runs 68 checks
against this list.

## Identity, splits and slices

A task is a pure function of `aaa.python.v0:<split>:<family>:<index>` and a
generator version. Drawing uses a counter-mode SHA-256 stream, not
`random`, so sources are byte-identical on every supported interpreter.
Oracle-dependent acceptance (output range, actually failing, exactly one
passing repair) retries the same identity with the next attempt counter.

| Split | Pool per family | Use |
|---|---:|---|
| `train` | 600 | learner and baseline training |
| `development` | 400 | evaluation streams and adaptation branches |
| `probe` | 60 | the retention bank (first 40), never trained on |
| `attack` | 200 | reserved for a later attack stage; unused in v0 |
| `confirmation` | none | **reserved and refused**: the generator raises `ConfirmationNotAdmitted` |

A split excludes every normalized source hash that occurs in the declared
pools of all earlier splits. The exact-match lookup baseline therefore has a
0% hit rate on development tasks, which is measured rather than assumed.
Training and probe tasks are `in_distribution`. Development and attack tasks
cycle `in_distribution`, `in_distribution`, `novel_literals` (constants
10-19 instead of 0-9) and `novel_structure` (nested loops and list walks,
templates never used in training).

## The causal boundary

    present -> act -> commit -> reveal -> score -> learn (only where permitted)

- The learner receives a `TaskView` built field by field. It has exactly the
  fields `task_ref, family, source, labels, repair_line, candidates,
  visible_tests`. The evaluator's task object, answer key, oracle observation,
  hidden tests and per-candidate results are not reachable from it
  (`leakage` walks the reachable object graph to check).
- `commit` validates the action against the label space. `reveal` before a
  commit, a second commit, and feedback for another task are refused.
- Feedback holds only the family's declared fields. For repair it is the
  chosen candidate's hidden-test results, never the index of the right one.
  Arms with feedback disabled receive nothing.
- Every event carries a sequence number, so the order can be audited.
- What leaks by design, after the action: for `syntax`, `outcome`, `output`
  and `localize`, the revealed feedback implies the answer. That is exactly
  what running the program shows, and it is why learning from it is permitted.

The boundary protects against accidental leakage by trusted research code.
It is not a sandbox against a learner written to subvert the Python runtime.

## Arms

From one trained state per initialization: `online` (updates from feedback),
`frozen`, `feedback_disabled` (updates allowed, feedback withheld) and
`reset_each_task` (restored to the untrained state before every task, so
memory is disabled). Each alternative representation has an online/frozen
pair (`online@bytes`, `frozen@bytes`, and so on). The baselines are `uniform`
(chance over valid labels), `majority` (most frequent label in training
feedback), `lookup` (exact-match memory, majority fallback, hit rate
reported) and `heuristic` (bracket/colon/indent lint for syntax, first risky
line for localization, last literal for output, first candidate that differs
from the current line for repair). Baselines do not adapt during evaluation.
CPython is the oracle and is never a baseline.

## Designs and estimands

- **Evaluation.** 3 initializations x 4 streams x 25 development tasks per
  family (streams are consecutive index ranges, mixing slices).
- **Adaptation (matched).** Per `initialization x stream` cell, a learner
  runs a 10-task in-distribution prefix online; its complete state is cloned
  into online and frozen copies on a *changed* branch (15 `novel_structure`
  tasks) and on a *control* branch (15 in-distribution tasks). The estimand is

      DiD = (err_frozen - err_online)_changed - (err_frozen - err_online)_control

  so the ordinary benefit of continued learning cancels. That is the lesson of
  AAA-1K's `paired_change_v1`, applied to a different domain.
- **Retention.** A fixed 40-task probe bank is scored read-only from the
  post-prefix state and from each branch's online end state. Forgetting is
  `err_after - err_before`.

## Metrics and statistics

Per family and arm: accuracy, balanced accuracy, coverage (1 - abstention
rate), Brier score of the reported confidence, expected calibration error
(10 bins), and for `lookup` its hit rate. Paired contrasts: online minus each
control and baseline, and each alternative representation's online arm minus
the default representation's.

The **unit** is the `initialization x stream` cell. Tasks within a stream,
and tokens within a task, are never resampled as independent. Intervals are
crossed percentile bootstraps (4,000 draws, 95%, seed 20260923) that resample
initializations and streams independently. A sign is reported as `POSITIVE`
or `NEGATIVE` only when the interval excludes zero; otherwise it is
`INCONCLUSIVE`, and with fewer than two of either factor it is
`INSUFFICIENT_EVIDENCE`. Ragged or non-finite grids are refused.

Abstention is part of the interface (an action may abstain), and calibration
is reported separately from correctness. The v0 learner's declared abstention
threshold is 0, so it never abstains. Rewarding calibrated abstention is a
later rung.

## Cross-version answer keys

The canonical benchmark must mean the same thing on CPython 3.10-3.13. The
answers are chosen to be interpreter-independent: exception *class* rather
than message; syntax *validity* rather than the reported syntax-error line,
which moves between versions; one statement per line, so error lines are
stable. The packaged golden keys
([`golden_answers_v0.json`](../research/aaa_python/data/golden_answers_v0.json))
hold the source hash and answer of the first 20 development tasks of every
family. `python -m research.aaa_python golden` fails on any interpreter that
disagrees, and the unit suite runs it on every version in the CI matrix. A
task whose semantics differed by version would have to be excluded or
versioned by interpreter; none has been found.

Derived *floats* are a separate matter. CPython 3.12 changed the built-in float
`sum()` to compensated summation, so a summary recomputed on 3.10 or 3.11 can
differ from one made on 3.12 in its final bits (about 1e-16 relative).
Recomputation therefore compares summary floats within a relative 1e-12 and
every count, status and resolved sign exactly (`AAA-186`).

## Development, attack and confirmation

v0 is a development phase. Development evidence is retained with its
provenance and is labelled as development evidence. The attack pool is
reserved. **Formal confirmation is not admitted**: v0 declares no candidate,
no promotion thresholds and no confirmation identities, and `confirm` and
confirmation generation both refuse with exit 2. Before any confirmation, a
successor protocol must freeze the generator, metrics, baselines,
candidate-selection rule, confirmation identities, statistical estimands and
promotion criteria, and commit its source identity before observation. Its
promotion decision is declared to use the `aaa.promotion.crossed.v1` contract
([promotion contract](promotion_contract.md)), whose primary and independent
implementations are proved to agree, including on single-series groups.
