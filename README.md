# AAA — Accurate Autonomous Adaptation

An experimental AI research project exploring Accurate Autonomous Adaptation: learning from observations, predicting what happens next, and adapting when conditions change.

AAA is a Cinqic research prototype, not a claim that the name itself demonstrates those capabilities.

## Research question

Can an autonomous system observe an environment, learn to predict what happens next, and adapt when the environment changes?

This first version makes the question small and inspectable. A one-dimensional world contains one moving dot. The system sees only numerical positions, predicts the next position before it is revealed, scores the prediction, and optionally updates a small online linear model. “Autonomous” here means that this observation, prediction, scoring, and learning loop runs unattended after launch. It does not mean general intelligence, independent goal formation, or self-directed research.

The project description is: “An experimental AI research project exploring Accurate Autonomous Adaptation: learning from observations, predicting what happens next, and adapting when conditions change.”

## What is implemented

- A seeded, bounded CPU simulator with straight, bouncing, and changed-motion scenarios.
- Correct boundary overshoot reflection; positions remain in the configured interval.
- Persistence baseline: `x[t+1] = x[t]`.
- Constant-motion baseline: `x[t+1] = x[t] + (x[t] - x[t-1])`.
- An online linear predictor using the last four positions and a bias term.
- A separate normalized RLS candidate and a predeclared benchmark v2; the legacy SGD learner remains a named diagnostic track.
- Strict temporal ordering: predict and record, advance, reveal and score, update, then append the new observation.
- Learning from scratch across multiple seeds, development-only learning-rate selection, frozen generalization on final seeds, and frozen-versus-online adaptation after an unannounced speed change.
- JSONL and CSV per-step logs, aggregate metrics, checkpoints, package/git metadata, four static plots, an experiment report, and an optional keyboard-controlled animation.

The learner receives four positions ordered oldest to newest: `[x[t-3], x[t-2], x[t-1], x[t]]`. The actual feature vector is `[1, x[t-3], x[t-2], x[t-1], x[t]]`. The model predicts a displacement `d_hat = w · features`, and its next-position prediction is `x[t] + d_hat`. The target is the observed displacement `x[t+1] - x[t]`. For loss `0.5 * (d_hat - d)^2`, the update is:

```text
features = [1, x[t-3], x[t-2], x[t-1], x[t]]
gradient = (d_hat - d) * features
w <- w - learning_rate * gradient
```

Weights start at zero, so the initial linear prediction is persistence. The prediction method does not update weights. During frozen evaluation no update is called; during online adaptation only the explicitly enabled copy updates. Model state is saved as readable JSON with a format version and required metadata, not unsafe object serialization.

## What the learner does and does not observe

The learner receives only the current four-position history and, after scoring, the actual next position for the update. It is not given velocity, the movement equation, scenario name, boundary events, the change schedule, the change factor, or future observations. The evaluator may retain bounce/change labels for metrics, but those labels never enter predictor inputs or updates. The coordinate interval `[0, 1]` and fixed time step are public observation-format configuration.

All experiments use deterministic motion without observation noise or random disturbances. The three scenarios are:

1. `straight`: constant velocity, with sampled episodes ending before a boundary.
2. `bouncing`: constant speed with reflection at the boundaries, including overshoot handling.
3. `changed`: bouncing motion with an unannounced speed-magnitude factor change during the episode. The default event is at transition 60 and is sampled away from a boundary where practical.

## Installation

From the project directory:

```bash
cd /home/cinqic/Documents/AAA
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The experiment is CPU-only. NumPy and Matplotlib are the runtime dependencies. A GPU is not required, and no external AI API, pretrained model, web service, or frontend is used.

## Commands

Run focused correctness tests:

```bash
python -m unittest discover -s tests -v
```

Run a small smoke experiment. It creates a unique directory under `runs/` and exercises development selection, training, checkpointing, frozen evaluation, adaptation, plots, and report generation:

```bash
python -m aaa.cli smoke --output-root runs
```

Run the full evaluation. The default final split has ten seeds:

```bash
python -m aaa.cli full --output-root runs
```

Run the separately versioned benchmark v2. The specification is frozen in
[`benchmarks/benchmark_v2.json`](benchmarks/benchmark_v2.json) before candidate
confirmation. Use a distinct attempt ID for each confirmation batch:

```bash
.venv/bin/python -m aaa.cli benchmark-v2 \
  --role confirmation_a --attempt-id confirmation-a --output-root runs
```

The command records compressed per-step predictions, evaluator metadata,
replica checkpoints, source-tree identity, checksums, gate decisions, and a
generated report. `confirmation_b` must use a fresh attempt ID and the same
committed source, configuration, model, and requirements. Development runs
may use `--replicas` and `--episodes`; those overrides are rejected for a
confirmation run when they would weaken the predeclared minimums.

Launch the optional animation from a trained checkpoint:

```bash
python -m aaa.cli animate \
  --checkpoint runs/<run-id>/checkpoints/trained_linear.json \
  --scenario changed \
  --seed 201
```

In the animation window, Space pauses/resumes and `r` restarts. A graphical display is required for this command. The headless evaluation and plots do not require one.

## Reproducing and inspecting a run

Every invocation uses a unique UTC run directory such as `runs/20260909T120000Z-full/`; old results are not overwritten. The run contains:

```text
config.json                         complete resolved configuration
metadata.json                       Python/package/platform/git metadata
dev_selection.json                  development-only learning-rate selection
checkpoints/trained_linear.json     inspectable model parameters
learning_from_scratch/steps.jsonl   complete nested per-step log
learning_from_scratch/steps.csv     flat per-step view
learning_from_scratch/metrics.json  aggregate, per-seed, and per-episode metrics
frozen_generalization/...           final-seed frozen comparison and integrity evidence
online_adaptation/...               same-trajectory frozen/online change test
plots/*.png                         static plots
summary.json                        compact index of all measured outputs
experiment_report.md                report generated from those measurements
```

The full run uses separate seeds: development `(101, 102, 103)`, training `(11, 12, 13, 14, 15)`, and final `(201, ..., 210)`. Development seeds select the learning rate using held-out development episodes. That choice is written once to `config.json` before final evaluation. Final seeds are not used for tuning. The final generalization distribution uses new straight and bouncing episodes with sampled positions, directions, and speeds; history resets at each episode while the trained weights remain frozen.

The benchmark v2 stream allocation uses `numpy.random.SeedSequence` from the
role, family, replica, and episode IDs. It is independent of worker scheduling.
Unchanged dynamics controls branch from the same pre-event position, velocity,
history, and complete learner state as the changed-law branch. The evaluator
retains event labels, but predictors receive only observations and past scored
targets.

The adaptation experiment starts a frozen and an updating copy from the same saved checkpoint. Both copies, the baselines, and the evaluator share one realized changed-motion trajectory. Recovery is defined before final evaluation: tolerance is `max(1.5 * pre-change MAE, 0.01)`, with a five-transition rolling mean and three sustained qualifying windows. If the post-change error does not increase meaningfully, recovery is not applicable. If sustained recovery is not observed before the horizon ends, the result says `not recovered within evaluation horizon`.

## Interpreting results

The report keeps these conclusions separate:

- The implementation runs correctly.
- The model improves with experience.
- The model generalizes to unfamiliar episodes.
- Continued updates help after a change.
- The model beats a baseline.

None automatically proves the others. Constant-motion extrapolation is a strong baseline for this deterministic world and may recover quickly after a speed change. A learned model losing to a baseline is a valid result. Four positions provide a short observation history, not a general long-term memory system. The programmer supplies the world mechanics, event schedule, baselines, and evaluation protocol; parameter updates supply only the learned linear relationship from recent positions to the next displacement.

AAA does not claim general intelligence, understanding of physics, a novel scientific architecture, or independent goal formation. The report is evidence about this particular local prototype and its documented seeds and configuration.

## Limitations and next experiment

The world is one-dimensional, deterministic, and noise-free. The model has no explicit bounce detector or change detector, and a finite episode can end before recovery is observed. Predictions are not clipped by default; raw predictions are retained in the logs. If a user enables clipping in a future configuration, the same policy must be applied to every predictor and the raw values must remain visible.

The single most useful next experiment is controlled observation noise: add fixed noise levels, freeze the noise schedule and final split before evaluation, and repeat the same frozen-versus-online protocol to test whether updates contribute under imperfect observations.
