# Observation-noise phase: `aaa.observation_noise.v1`

This is a separately versioned study of one bounded question: how well does
AAA predict a moving dot's next latent position when only the observation
channel is noisy? It is not a revision of benchmark v2.1 and it does not add
language, vision, goals, actions, process noise, missingness, or variable
sampling intervals.

The executable protocol is
[`aaa/noise/data/observation_noise_v1.json`](../aaa/noise/data/observation_noise_v1.json).
It is design-frozen before comparative candidate work. The protocol hash is
printed by `python -m aaa.cli observation-noise-protocol-hash` and is recorded
in every attempt.

## Reference boundary

The v2.1 specification remains byte-for-byte unchanged. The phase records both
its raw SHA-256 (`4993c5e6e173f9dd5ef002bc84ff4c484da853b066ea45662d41ad826d10d48e`)
and its resolved specification hash
(`f8e1090bf5b1aeb02cb8a129fec0ec9c83ab1b50cb2c500496b862fe6a1a5e37`). The
hashes address different representations. A zero-noise equivalence fixture
compares latent trajectories, observations, forecasts, targets, updates,
reflection handling, counters, and metrics against an isolated v2.1 reference
checkout before any confirmation claim.

The old v2.1 source fingerprint is not weakened to accommodate this phase.
Adding a new namespace legitimately changes the current checkout. Historical
reference checks use the pinned commit and the repository lock in an isolated
checkout.

## Causal observation model

For latent position `x_t`, interval width `L`, and sensor innovation `eta_t`,
the only predictor-visible value is

```text
y_t = x_t + L * eta_t
```

The runner creates and caches exactly one sensor sample for every timestamp,
including warmup. Repeated observation calls return the same cached value. A
primitive record keeps `latent_position`, `raw_observation`, and
`observed_history` as separate fields. Noise is never clipped or reflected at
the physical boundary.

The order is:

```text
predict from observed history
record the forecast
advance the unchanged latent simulator
reveal latent truth and the new noisy observation
score latent and noisy-observation tasks
update from the new noisy observation only
append the noisy observation to history
```

Predictors never receive latent truth, latent velocity, simulator parameters,
intervention labels, noise innovations, event times, or future samples. The
evaluator may use latent truth privately for scoring.

## Why differencing needs an explicit diagnostic

The incumbent uses a displacement feature from observations. With

```text
y_t = x_t + L eta_t
```

the observed displacement is

```text
y_t - y_(t-1) = (x_t - x_(t-1)) + L(eta_t - eta_(t-1))
```

and the next noisy displacement target is

```text
y_(t+1) - y_t = (x_(t+1) - x_t) + L(eta_(t+1) - eta_t)
```

Thus the current noise term `-L eta_t` occurs in both the regressor and the
target. For independent innovations their covariance contribution is
`Cov(L(eta_t-eta_(t-1)), L(eta_(t+1)-eta_t)) = -L^2 s^2`, before any physical
signal covariance is considered. The predictor is therefore exposed to a
shared-noise errors-in-variables problem. For the AR(1) channel, the same
calculation must retain the declared lag covariance instead of treating the
terms as independent. Numerical square-root RLS stability does not establish
unbiased estimation or noise robustness.

This derivation motivates the generator tests, clean/noise-trained comparison,
causal smoothing baseline, and the separate noise-shift controls. It is not a
claim that a refinement succeeds.

## Frozen channels and cells

The dimensionless RMS scale set is `0`, `0.0005`, `0.002`, and `0.01`. Gaussian
and uniform at the two lower nonzero levels are primary cells. The highest
level, correlated AR(1), and impulsive channels are mandatory stress cells.
The exact definitions and stream derivation are in the machine-readable
protocol. Actual schedule arrays are saved and hashed before a cell runs.

The stationary study covers constant velocity, bouncing, speed change, and the
changed-law oscillator. A separate factorial records unchanged/changed
dynamics crossed with unchanged/changed noise. Law changes occur at transition
300 with a 100-transition branch. Sensor shifts are `0.0005 -> 0.002` and
`0.002 -> 0.0005`, aligned at 300 or staggered at 340. A noise shift is never
counted as successful adaptation to a physical-law change.

## Models and matched comparisons

The mandatory comparator set is persistence, raw constant motion, reflected
constant motion, causal smoothing-plus-motion, a fixed-parameter alpha-beta
filter, the unchanged square-root RLS incumbent, and a no-learning control.
The reflected constant-motion model receives the same public boundary map as
the incumbent; boundary handling is not counted as learned capability.

There are two fixed training conditions: clean-trained and noise-trained. The
noise mixture is fixed to Gaussian and uniform at the two primary lower levels.
The complete learner/filter state is cloned at interventions. A frozen arm
freezes learned parameters and adaptive predictor state, while ordinary
observation history and fixed-parameter filter state continue to evolve in both
arms. Initial forecasts must agree. Point-prediction adaptation and interval
calibration adaptation are separate analyses.

Candidate work is bounded at twelve configurations and three substantive
mechanism changes. The ledger includes unsuccessful configurations. A
refinement is not promoted merely because it is more complex or because a
confirmation outcome was unfavorable.

## Replication and uncertainty

The fixed confirmation plan is ten independently trained learner lineages,
32 latent episodes per family and stationary cell per lineage, and three
independent sensor realizations per latent episode. The existing 32 strata are
checked directly. Confirmation A and B share the frozen training lineages but
have disjoint evaluation streams; they are replication across schedules, not
independent retraining studies.

Aggregate uncertainty resamples training lineage, latent episode within
lineage, and sensor realization within episode, preserving paired models and
branches. The protocol declares 4,000 draws, 90% and 95% intervals, fixed
balanced cell weighting, and one Holm family across primary endpoint/cell/batch
claims. Undefined ratios remain undefined.

## Evidence and outcomes

Every attempt has an immutable ID and atomic lifecycle records. Full primitive
records, schedule arrays, checkpoint state, source identity, lock identity,
exact invocation, diagnostics, summaries, plots, exclusions, and checksums are
retained. Failed and cancelled attempts remain visible. Independent
recomputation reads primitive artifacts without importing the production metric
collector or gate evaluator. Reproduction reruns generation from declared
inputs; recomputation verifies retained evidence. They are distinct verdicts.

The current primitive record schema is `aaa.observation_noise_step.v3`. In
addition to latent and noisy point-prediction fields, it stores the causal
prediction intervals available before each reveal. Intervals are calibrated
from past noisy residuals only; warmup rows carry an explicit unavailable
value. The statistical artifact reports coverage, width, and interval score
separately from point-prediction adaptation. Completed trials are first written
to atomic `trial_records/` shards, which makes an interrupted attempt safely
resumable without replacing retained evidence.

The phase uses five separate outcomes: engineering complete, scientifically
supported, negative, inconclusive, and blocked by missing evidence. Only an
explicit `PASS` satisfies a required engineering gate. A valid negative result
may be merged as evidence but cannot promote an unsupported refinement or
authorize the next capability stage.
