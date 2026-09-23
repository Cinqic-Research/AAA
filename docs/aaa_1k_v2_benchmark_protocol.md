# AAA-1K v2 benchmark protocol (`aaa.1k.v2.protocol.v1`)

The executable form of this protocol is
[`research/aaa_1k_v2/plan.py`](../research/aaa_1k_v2/plan.py); where the two
disagree, the code is authoritative and the disagreement is a defect. Both were
committed before any development, attack or confirmation identity of this
phase was observed. A value changed after an identity is observed creates a new
stage with new identities.

## 1. Suites

**Regression suite** (round-3 families through the unmodified historical
generators, fresh seeds): `v1_motion_straight`, `v1_motion_bouncing`,
`v1_motion_changed`, `v1_motion_dynamics`, `v1_occlusion`, `v1_coarse_speed`,
`v1_aba`.

**Stress suite `aaa.1k.stress.v1`** ([`research/aaa_1k_v2/stress.py`](../research/aaa_1k_v2/stress.py)):

| Family | Role | What it stresses |
|---|---|---|
| `long_bouncing` | development | 2,000-step reflecting motion |
| `long_coarse` | development | 2,000-step coarse construction (the M2 condition) |
| `random_gaps` | development | random gap schedule, lengths 1-12 |
| `wall_gaps` | development | fast motion, gaps before, during and after wall contacts |
| `long_gap_recall` | development | occlusions of 16, 32 and 64 steps |
| `stationary_mixed` | development | genuine stationary periods mixed with motion and gaps |
| `noise` | development | observation noise sigma 0.002, scored on latent truth |
| `change_gaps` | development | unannounced speed changes plus gaps |
| `quantized` | development | quanta 0.0025-0.008 at fixed speed |
| `oscillator_long` | development | damped oscillator with coefficient changes every 300 steps |
| `noise_gaps` | attack | noise plus gaps |
| `change_noise` | attack | speed changes plus noise |
| `accel_switch` | attack | switching constant acceleration with drag |
| `coarse_near` | attack | coarse construction at nearby speeds and quanta |
| `long_gap_recall_extended` | attack | occlusions of 96 and 128 steps |
| `gravity_bounce` | held out | constant gravity, acceleration at the wall |
| `soft_wall` | held out | spring walls: the public boundary map is wrong |
| `inelastic_wall` | held out | restitution 0.6-0.9 |
| `abcab` | held out | A (constant velocity), B (oscillator), C (gravity), A, B |

Development families may inform selection. Attack-only families appear first
at attack. Held-out families appear first in confirmation and measure
out-of-family generalization. The observation format is AAA-1K's public one
everywhere, so agents apply the reflection map even where the dynamics break
it; those families measure what that costs.

**External suite** ([`aaa_1k_v2_external_benchmarks.md`](aaa_1k_v2_external_benchmarks.md)):
NARMA-10, NARMA-20, Mackey-Glass 17, twelve rule-selected dysts systems, four
rule-selected Monash datasets.

## 2. Identities

`benchmarks/aaa1k_v2_identity_registry.json`, built from
`plan.declared_blocks()`: 64-bit SHA-256 seeds per role and namespace, proven
disjoint by set intersection from each other and from every AAA-1K seed below
index 100,000, every seed in AAA-1K and loop evidence, and every loop ledger
block (`identities.prove_disjoint`, fail-closed; `AAA-163`).

| Role | Inits | Streams per dot family | External realizations |
|---|---|---|---|
| development | 4 | 12 | NARMA-10 3, NARMA-20 3, Mackey-Glass 3, dysts 2 per system |
| attack | 5 | 8 | same as development |
| confirmation | 8 | 16 | NARMA-10 8, NARMA-20 8, Mackey-Glass 8, dysts 4 per system |
| diagnostic | 5 | 8 | none |
| capacity | 5 | 8 | NARMA-10 3, NARMA-20 3, Mackey-Glass 3 |

Monash data are fixed: development evaluates on the archive's *training*
split with its own last `h` observations held out; the archive's test
observations are read only in confirmation. Scratch and qualification blocks
serve shakedowns and compute measurement and can satisfy no criterion.

## 3. Development

**Grid.** Learning rate {0.001, 0.003, 0.01, 0.03, 0.1, 0.3} x TBPTT
{(1, live), (4, live), (4, replay), (16, live), (16, replay)} x clip {none, 1,
10}: 90 configurations per TBPTT candidate, 18 for the RTRL candidate.
`lambda = 0.25`. Champion 1 runs its frozen configuration.

**Eligibility.** A configuration is eliminated by any failed or diverged
development cell, dot or external. A dot cell diverged if its MAE exceeds twice
persistence's on the same stream (the loop's rule). An external cell diverged
if it failed, its forecast is non-finite, or its prequential error exceeds
twice persistence's. The stability margin keeps only learning rates at least
two grid positions below the lowest rate at which the unclipped reference
configuration (T=4 live, or RTRL) diverged anywhere. If it never diverged, the
margin was not demonstrated, and the artifact says so.

**Selection score.** The geometric mean, over the 17 development dot families
and NARMA-10, NARMA-20 and Mackey-Glass, of mean error divided by Champion 1's
on the same cells. Ties within 2% prefer, in order: the lower learning rate,
the live rule, the shorter horizon, the larger clip threshold.

**Screen** (each candidate's selected configuration against Champion 1, same
development cells, crossed initialization x stream bootstrap, 4,000 draws):

| | Criterion |
|---|---|
| S1 | stability (eligibility already requires it) |
| S2 | every v1 family: relative-MAE upper bound <= +2% |
| S3 | development stress families: geometric-mean relative MAE upper bound < 1.0 |
| S4 | NARMA-10, NARMA-20, Mackey-Glass: geometric-mean relative error upper bound <= 1.02 |

The passing candidate with the lowest S3 point estimate becomes the
**challenger**. If none passes, there is no challenger and Champion 1 remains.

## 4. Attack

On attack identities (5 fresh inits, 8 fresh streams for every development and
attack-only family, fresh external realizations):

| | Criterion |
|---|---|
| A1 | challenger has zero failed cells and no more diverged cells than Champion 1 |
| A2 | every v1 family non-inferior at +2% |
| A3 | stress families (development and attack-only): geometric-mean upper bound < 1.0 |
| A4 | no stress family with relative-MAE lower bound above +5% |
| A5 | NARMA-10, NARMA-20, Mackey-Glass and the twelve dysts systems: geometric-mean upper bound <= 1.02 |
| A6 | A1-A3 recomputed from a CUDA re-run of the attack v1 families give identical verdicts |
| A7 | the challenger with its input and recurrent initial weights scaled x1.5 still passes A1 |

Every criterion must pass, or the challenger is rejected.

## 5. Freeze and confirmation

The freeze manifest records the challenger's complete arm specification,
Champion 1's, the v2 source fingerprint, both dependency-lock hashes, the
backend (`cpu`), the dtype, the hardware standard, the suite and family lists,
the metrics, every threshold, the confirmation identity blocks with their
seed-list hashes, the bootstrap design and seeds, and the arms to be observed.
It is committed before confirmation. Confirmation refuses a manifest that is
untracked, differs from `HEAD`, or disagrees with the live fingerprint, and it
marks every confirmation block spent before the first cell runs.

**Confirmation** observes every family, including the held-out ones (8 inits x
16 streams), the external suite including the Monash test horizons, and the
capability designs: online versus frozen twins, paired-change adaptation, and
the retention probe bank. Arms: Champion 1, the challenger, the other
candidates (descriptive only, never eligible for promotion), `mlp_v2`, the
one-mechanism ablations of challenger and champion, and the analytic
baselines.

| | Criterion (challenger versus Champion 1) |
|---|---|
| K1 | zero failed cells; diverged cells no more than Champion 1's |
| K2 | every v1 family non-inferior at +2% |
| K3 | geometric-mean relative MAE over all 19 stress families <= 0.95, with upper bound < 1.0 |
| K4 | no stress family with relative-MAE lower bound above +5% |
| K5 | external suite geometric-mean relative error upper bound <= 1.02 |

**Decision.** PROMOTE if every criterion passes; REJECT if any fails; otherwise
INCONCLUSIVE. Promotion needs every criterion (intersection-union), so no alpha
is spent per criterion. Claims outside K1-K5 (the capability vector, the other
candidates, ablations) are exploratory and are labelled so; where several are
tested together, Holm-adjusted p-values are reported alongside.

The decision is recomputed by an independent routine from the retained
per-cell primitives, with thresholds read from the freeze, before it is
recorded.

## 6. Statistics

Every comparison is paired and crossed. `stats.relative_crossed` resamples
initializations and streams independently. `stats.geometric_relative`
resamples the shared initializations once per draw and each family's streams
separately, then takes the geometric mean of the per-family ratios of means.
Families are the declared suite and are not resampled. An interval crossing a
bound is INCONCLUSIVE, never a pass, and "not statistically resolved" is never
reported as "equal".

## 7. Reproduction standard

Confirmation evidence is produced on the CPU (NumPy float64). On the Zen 3
evidence platform a rerun must reproduce every cell bitwise. Elsewhere the
standard is verdict-level: identical identities, structure and adjudicated
verdicts, with numerical drift reported (`AAA-173`). Development evidence
records its backend and reproduces at the same standard on the same backend.
