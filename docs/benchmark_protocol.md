# AAA benchmark protocol v2.1

The canonical, executable specification is
[`aaa/benchmark/data/benchmark_v2_1.json`](../aaa/benchmark/data/benchmark_v2_1.json).
It ships inside the installed package, and every value in it is read by the
runner. Print its identity with:

```bash
python -m aaa.cli spec-hash
```

This document explains the protocol. Where the two disagree, the JSON wins —
and `tests/test_spec.py` enforces that every declared leaf is actually consumed
by the implementation, so a JSON value that nothing reads is a test failure
rather than a comfortable fiction.

## Lineage

| Version | Status | Where |
|---|---|---|
| v1 | historical | [`results/final/`](../results/final/) — the original moving-dot evaluation, preserved unchanged |
| v2 | **superseded** | [`benchmarks/benchmark_v2.json`](../benchmarks/benchmark_v2.json) and [`results/benchmark_v2/`](../results/benchmark_v2/) |
| v2.1 | active | this document |

v2's recorded confirmations are retained as historical evidence. They are **not**
acceptance evidence: the methodology that produced them had defects that changed
what several of its gates meant. See [`errata.md`](errata.md) for exactly which
claims were affected and why.

## What the learner may and may not see

For every scored transition the runner does exactly this:

1. the learner receives only the causally available observation history;
2. every predictor produces a prediction;
3. the predictions are recorded;
4. the environment advances;
5. the actual target is revealed;
6. the prediction error is scored;
7. only then may an enabled learner update;
8. the newly revealed observation enters history.

A predictor never receives the scenario name, an event flag, the velocity, the
change schedule, a hidden coefficient, or a future observation. The evaluator
keeps event labels for metrics only. `tests/test_experiment.py` pins this.

Two things *are* public, programmed knowledge of the observation format rather
than learned capability, and the protocol is explicit about both:

- the coordinate interval `[lower_bound, upper_bound]` and the time step;
- the reflection map at the boundaries.

Because reflection is public, the baseline suite includes a **reflected
constant-motion** predictor that uses the identical map. That is the
like-for-like comparison for any candidate allowed to reflect its own
prediction.

## Units

Every benchmark aggregate is in normalized units:

```text
normalized_absolute_error = |prediction - actual| / (upper_bound - lower_bound)
```

All of them route through one accessor in `aaa/metrics.py`. Raw-unit values are
retained in the step logs and are always labelled raw. `tests/test_metrics.py`
runs the unit contract on `[0, 1]`, `[0, 10]`, `[-5, 5]` and shifted intervals,
because an interval of width 1 hides exactly this class of mistake.

## Baselines, and what each one isolates

| Predictor | What it isolates |
|---|---|
| `persistence` | whether any motion model helps at all |
| `constant_motion` | analytic first-order extrapolation, no boundary knowledge |
| `constant_motion_reflected` | the same, plus the identical public boundary policy — **the like-for-like baseline** |
| `zero_control` | the candidate architecture with zero-initialized parameters: what it predicts before learning anything |
| `legacy_linear_sgd` | the historical v1 learner, reported as a diagnostic only |
| `candidate_frozen` | the selected candidate, updates disabled: what training produced |
| `candidate_online` | the selected candidate, updates enabled: what continued updating adds |
| `candidate_no_reflect` | the candidate with the boundary policy removed: how much of any bounce advantage is the transform |

## Families

| Family | What it asks | Candidate mode |
|---|---|---|
| `constant_velocity` | did the candidate identify the constant-velocity law, in every declared stratum? | frozen |
| `bouncing` | frozen generalization to an unfamiliar regime, and bounce-transition accuracy | frozen |
| `speed_change` | robustness of a *frozen* model to an unannounced speed jump | frozen (an online arm is reported) |
| `speed_extrapolation` | accuracy above the training speed range — labelled out-of-distribution | frozen |
| `always_online` | one continuously updating instance across a rotation of regimes, with no evaluator mode switching | online throughout |
| `changed_law` | the matched frozen-versus-updating adaptation experiment | both branches |

### Deliberate freezing versus deliberate autonomy

The `speed_change` family freezes the candidate on purpose. That is a
*robustness control*, not evidence of online adaptation, and its gate is named
`speed_change_frozen_robustness` to say so.

The `always_online` family is the autonomy track. One candidate instance per
replica keeps predicting and updating across straight, bouncing and changed
segments. The evaluator never tells it the scenario, never freezes it, never
resets it and never signals an event. The two families answer different
questions and both are retained.

### The changed-law experiment

A stable second-order damped oscillator, integrated with semi-implicit Euler
around the interval midpoint `m`:

```text
a[t]   = -2 * zeta * omega * v[t] - omega^2 * (x[t] - m)
v[t+1] = v[t] + a[t] * dt
x[t+1] = reflect(x[t] + v[t+1] * dt)
```

Each episode runs a common pre-change prefix, then clones the complete learner
state and observation history at the intervention boundary and branches into
four arms: changed-law frozen, changed-law updating, unchanged frozen, and
unchanged updating. The unchanged branch keeps the pre-change coefficients for
the full horizon and is the control for "does continuing to update hurt when
nothing changed?".

This matched structure is the soundest part of the design and is protected by
regression tests. The branch state is hashed into
`metrics/changed_law_interventions.json`, so a reviewer can confirm that the
frozen and updating copies really did start from the same state without
trusting the runtime.

**What is and is not held out.** The candidate is allowed to learn the
*pre-change* oscillator dynamics from its own observations during the prefix.
The post-change coefficients and the intervention itself are held out. Saying
"the oscillator family is never seen" would be false and the protocol does not
say it.

Intervention timing and the post-change coefficients are drawn from predeclared
ranges in the specification, and the full recovery horizon always fits inside
the branch. No event value is chosen by looking at candidate error.

## Stratification: coverage is planned, not hoped for

Episodes are allocated deliberately across the full declared Cartesian product
of direction x position band x speed band. The expected strata are known before
any result is observed, and the acceptance system checks that:

- every required stratum exists;
- each meets its minimum episode count;
- each is represented across replicas rather than concentrated in one;
- both walls are observed;
- bounce and eligible change events meet their minimums.

A missing or thin stratum is `INSUFFICIENT_EVIDENCE`, never `PASS`.

## Gates

Every gate reports one of `PASS`, `FAIL`, `NOT_VERIFIED` or
`INSUFFICIENT_EVIDENCE`. Only `PASS` satisfies a required gate. Absence of
evidence is never converted into success.

| Gate | Required | What it actually measures |
|---|---|---|
| `stratum_coverage` | yes | planned coverage is complete, across replicas, with both walls and the declared event minimums |
| `constant_velocity_identification` | yes | absolute straight-motion accuracy, per stratum |
| `learning_progress` | yes | frozen checkpoints at increasing cumulative update budgets, scored on one fixed probe bank |
| `frozen_prediction_accuracy` | yes | absolute accuracy generalizing to the unfamiliar bouncing regime |
| `bounce_event_accuracy` | yes | absolute bounce-transition accuracy **and** non-regression against the reflected baseline |
| `speed_change_frozen_robustness` | yes | a frozen model's post-event error does not regress against the reflected baseline |
| `changed_law_adaptation` | yes | updating beats its identical frozen copy and persistence by the declared margin, and does not regress against constant motion |
| `unchanged_control` | yes | updating in an unchanged world does not degrade the model |
| `recovery` | yes | eligible events recover within the horizon; unrecovered events are counted |
| `always_online_stability` | yes | a continuously updating instance does not regress against the reflected baseline |
| `speed_extrapolation_report` | no | reported out-of-distribution accuracy |
| `correctness` | yes | the evidence itself is complete, consistent, schema-valid and recomputes |
| `reproducibility` | yes | a rerun, a save/resume comparison and a golden seed fixture were actually executed |
| `cpu_usability` | yes | the selected candidate's own predict/update latency |

### Why `straight_learning` is gone

The v2 gate named `straight_learning` asked only whether the candidate beat a
zero-initialized control on constant-velocity motion. A parameterless analytic
rule satisfies that trivially, so the gate did not measure learning. It is
replaced by three separately named claims that do not imply one another:

- `constant_velocity_identification` — did it get the law right?
- `learning_progress` — did error on a **fixed** probe bank fall as the
  cumulative update budget grew?
- `frozen_prediction_accuracy` — does what it learned generalize?

The learning curve is the measurement that distinguishes learning from a
formula: a rule with no parameters has no budgets and produces a flat curve.
`tests/test_gates.py` includes a case proving a parameterless rule cannot
satisfy the learning gate.

### Why the bounce gate changed shape

The v2 gate demanded a >=20% event-error improvement over **raw** constant
motion. Development measurement showed that essentially all of that margin was
the reflection transform, not learned parameters: the candidate with reflection
removed is indistinguishable from raw constant motion at bounce transitions,
and the reflected constant-motion baseline is exact at machine precision in
this deterministic, noiseless world.

A >=20% relative improvement over a baseline that is already at ~1e-17 is not
attainable by anything, and the old gate could only be passed by comparing
against a baseline denied information the candidate was given. A criterion that
no correct system can satisfy, and that an unfair comparison does satisfy, is
not measuring anything useful.

The replacement asks two questions that are both answerable and both honest:

1. an **absolute** accuracy requirement on bounce-transition normalized error;
2. **non-regression** against the like-for-like reflected baseline, with an
   absolute floor, because ratios between two quantities near 1e-17 are noise.

The report additionally decomposes where any apparent advantage comes from —
analytic extrapolation, the public boundary policy, and learned parameters —
under separate names.

This change was made on development evidence and frozen before any confirmation
batch was generated. **No threshold was tuned to a confirmation outcome.** The
scientific reasoning is recorded here and in [`errata.md`](errata.md) precisely
so that this is checkable rather than asserted.

## Statistics

Method: paired hierarchical bootstrap, 4,000 draws, 95% percentile intervals,
Holm-Bonferroni across primary comparisons.

Each draw:

1. resamples top-level training replicas with replacement;
2. within each selected replica, resamples that replica's episode summaries
   with replacement, preserving that replica's observation count;
3. preserves model pairing where the comparison is matched;
4. recomputes **the exact statistic the gate uses**.

Consequences the implementation is held to:

- if the point estimate is `1 - mean(candidate) / mean(baseline)`, every draw
  recomputes that ratio of means. The interval is never a mean of episode
  ratios while the headline is a ratio of means;
- adjacent time steps are never treated as independent observations;
- event weighting is declared once and used for both the point estimate and the
  interval, under one name;
- an episode with zero bounce events does not contribute its whole-episode
  error to the event resample. It is excluded and counted separately;
- a gate about the post-change window resamples the post-change statistic, not
  whole-episode error;
- non-regression claims get intervals too, not only superiority claims;
- an interval that could not be computed is reported as
  `INSUFFICIENT_EVIDENCE` with its reason, never as `PASS`.

Multiplicity counts repeated confirmation attempts against the same
specification, so re-rolling until something passes costs statistical power
rather than being free.

Five training replicas is the routine engineering minimum and is labelled as
such. A `high_replication` role runs 20 independent lineages for stronger
claims. Five checkpoints evaluated on 500 episodes is 500 episodes from five
learners; it is never described as 500 independent learners.

## Recovery

Eligibility and recovery are defined independently.

- the pre-event reference is a recent, bounded window before the intervention,
  not the entire early training prefix;
- **eligibility is driven by the peak shock**, not by the post-event average. A
  single large shock followed by fast recovery stays eligible; under the v2
  rule a 50-step average diluted it away and the event vanished from the
  denominator;
- each predictor's recovery uses **that predictor's own** pre-event errors. One
  learner's error sequence is never reused as the reference for persistence and
  constant motion;
- every episode lands in exactly one status: no intervention, insufficient
  pre-event evidence, insufficient post-event evidence, no measured shock,
  eligible-and-recovered, or eligible-and-unrecovered;
- unrecovered events are counted. They never disappear.

The report gives the full distribution — eligible, recovered, unrecovered,
ineligible reasons, per-replica rates, median, p90, p95, max — not a bare
`475 / 475`.

## Confirmation discipline

A confirmation batch is a predeclared, immutable identity, and the **stream
seed is derived from that identity**. Under v2, "freshness" came from a
user-editable `--attempt-id` that never entered the seed, so two confirmations
differing only by label re-ran identical numbers.

```bash
python -m aaa.cli declare-batch aaa-v2_1-confirmation-a-0001 --role confirmation_a
python -m aaa.cli freeze --batch aaa-v2_1-confirmation-a-0001 --batch aaa-v2_1-confirmation-b-0001
# commit the freeze manifest and the batch registry, then:
python -m aaa.cli benchmark --role confirmation_a --batch-id aaa-v2_1-confirmation-a-0001
```

The runner — not only the CLI — enforces every confirmation invariant, so a
Python caller cannot walk around them:

- the canonical committed specification, verified by hash; custom
  specifications are development experiments only;
- a predeclared, unconsumed batch;
- the declared replica and episode minimums;
- a clean source tree;
- agreement with the committed freeze manifest, covering the specification
  hash, checkpoint hashes, training stream identities, dependency lock hash and
  the planned batch list.

A consumed batch cannot be reused as a fresh confirmation; re-running one
requires explicit `--reproduce`. A **failed batch is retired permanently** and
its result stays recorded.

### A and B

`ab_relationship` is declared explicitly rather than left to an ignored
parameter. The default, `shared_frozen_checkpoints`, trains and freezes the
selected checkpoints once and evaluates those exact checkpoints on two
independent confirmation streams: A and B together measure generalization of
one selected model set. The `high_replication` role instead uses independent
training lineages, which additionally measures sensitivity to training
randomness. The manifest records which relationship was used and the checkpoint
hashes, so checkpoint sharing is visible rather than implied.

## Exit codes

| Role | Behaviour |
|---|---|
| `development` | returns 0 even when gates fail — exploration may record a failure |
| `confirmation_a`, `confirmation_b`, `high_replication` | returns non-zero if any required gate is not `PASS`, including `NOT_VERIFIED` and `INSUFFICIENT_EVIDENCE` |
| any role, refused before running | returns 2 |

CI checks the status. `tools/check_exit_codes.py` proves the contract from real
subprocesses.

## Evidence

Each attempt directory contains its resolved specification, provenance
(commit, tree hash, dirty flag, dependency lock hash, hardware), per-replica
checkpoints and budget checkpoints, compressed per-step records, per-family
metrics, the intervention record, the experiment registry, verification
results, checksums, plots, a machine-readable summary and a report.

Every metric and gate can be recomputed from the retained raw evidence without
retraining or re-simulating:

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/<attempt>
```

It verifies checksums and the specification hash first, and exits non-zero if
any stored gate status fails to reproduce.
