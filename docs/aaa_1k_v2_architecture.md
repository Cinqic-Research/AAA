# AAA-1K v2 architecture specification

Phase `aaa.1k.v2`. Model format `aaa.1k.v2.batched.v1`. This document
specifies the batched, backend-generic implementation of the online predictive
core and the six candidates. It is fingerprinted with the phase
([`research/aaa_1k_v2/identity.py`](../research/aaa_1k_v2/identity.py)).

`aaa.1k.v1` ([`aaa_1k_architecture.md`](aaa_1k_architecture.md)) is unchanged.
Champion 1 remains defined by `research/aaa_1k` plus the loop's
`ReachGatedUnfoldAgent`; v2 re-implements the same arithmetic in batched form
and proves the correspondence by test.

## 1. Cells, batches and backends

A **cell** is one model with its own parameters, hidden state, TBPTT window or
eligibility traces, hyperparameters, adapter state and counters. A
`BatchedLearner` advances `B` cells in lockstep with one array program. Cells
never exchange information: every operation is elementwise across the batch
or a per-cell matrix product. Consequently a cell's result does not depend on
which other cells share its batch or chunk (tested bitwise), and a
hyperparameter sweep is simply a wider batch.

The array module `xp` comes from `aaa.compute.resolve_backend`: NumPy on the
CPU, CuPy on CUDA. Scientific code never branches on the device. Initial
parameters are always drawn on the host with NumPy's `default_rng(seed)` and
transferred, so initial weights never depend on the backend. Everything is
float64.

## 2. Cores (`research/aaa_1k_v2/cores.py`)

| Core | State update | Online rule |
|---|---|---|
| `gru` | AAA-1K's convention exactly: `z` keep gate, `r` reset gate, one bias per gate | TBPTT |
| `elman` | `h' = tanh(W x + U h + b)` | TBPTT |
| `mgu` | `f = sigma(W_f x + U_f h + b_f)`, `n = tanh(W_n x + U_n (f*h) + b_n)`, `h' = (1-f) h + f n` | TBPTT |
| `mlp` | stateless `x -> tanh -> tanh` | one-step gradient |
| `lru` | complex diagonal `h' = lambda * h + B x`, `lambda = exp(-exp(nu) + i theta)`, read out by `tanh(W1 [Re h, Im h] + V x + b1)` | exact RTRL |

All cores end in a linear readout with two outputs (predicted normalized
target, raw error-magnitude score) initialized to exactly zero, so an untrained
cell predicts persistence (dot and forecasting tasks) or the target centre
(NARMA). With `inputs=3, hidden=16` the GRU's initial parameters are bitwise
those of `research.aaa_1k.model.AAA1KGRU` for the same seed, and the Elman
core's at `inputs=3, hidden=28` are those of `VanillaRNNControl`.

The LRU initializes each mode's time constant log-uniformly in [2, 200] steps
and its phase uniformly in [0, pi/2]; input weights are Glorot-uniform scaled
by `sqrt(1 - |lambda|^2)` so each mode's stationary variance is of order one.
`|lambda| < 1` holds for every finite `nu`, so the linear recurrence cannot
become unstable however the parameters move.

## 3. Loss and learning (`research/aaa_1k_v2/engine.py`)

`L = (o0 - d*)^2 + lambda (softplus(o1) - stopgrad(|o0 - d*|))^2` with
`lambda = 0.25`, exactly v1's loss. Rules:

* **`live`** -- AAA-1K's rule: backpropagate the latest loss through the last
  `T` cached transitions using current matrices (`AAA-169`).
* **`replay`** -- recompute the window forward from its realized start state
  under current parameters, then backpropagate: the exact gradient of the
  latest loss at the current parameters.
* **`rtrl`** -- for the LRU only: traces `E_lambda = dh/dlambda` and
  `E_B = dh/dB` are advanced with the state; the gradient of the current loss is
  exact and untruncated.

Every rule applies plain per-cell SGD, `p <- p - lr * scale * g`, with an
optional per-cell global-norm clip (`scale = clip / ||g||` when `||g|| > clip`).
There is no optimizer state. Expressions follow `research.aaa_1k.model`
operation by operation so the CPU path can be compared with it cell by cell.

**Failure.** A cell whose forward pass, gradient norm or updated parameters are
non-finite is marked failed at that step. It stops updating and has no
primitive from that step on. Detection needs no host synchronization, and a
non-finite value cannot leave its cell.

## 4. Adapters (`research/aaa_1k_v2/adapters.py`)

The adapter owns the interaction history. Its only input is
`accept(values, observed)`.

**Dot adapter.** The batched counterpart of `research.aaa_1k.agents.NeuralAgent`:
zero-order hold with per-step velocity (`(x - x_prev) / gap`), previous signed
error, pending raw and scored predictions, and target construction. Feature
sets: `v1` (Champion 1: centred position, displacement, previous error) and `v2`
(adds the observed flag). Target rules: `folded`, `own` (Champion 0) and
`reach_gated` (Champion 1's c10). Predictions are reflected with the public
map; the first four folds reproduce `aaa.predictors.reflect_prediction`
bit for bit, and a prediction still outside after four folds (only a wildly
diverged one) is folded in closed form. A cell trains only on a transition
whose both ends it was shown, as in v1.

**Series adapter.** For external benchmarks. Input channel `s_t`, target `y_t`,
either the change `y - s` (forecasting) or the value itself (NARMA). Features
occupy the same slots as the dot's: centred level, change, previous error
(and the flag). Every scale is passed in from training-only information.

## 5. Candidates and accounting

| Arm | Core | Features | Trainable | Hidden | TBPTT buffer at T=4 | Traces | Total adaptive |
|---|---|---|---|---|---|---|---|
| `c1_champion1` | GRU 3x16x2 | v1, reach-gated | 994 | 16 | 404 | 0 | 1,414 |
| `gru_v1_retuned` | GRU 3x16x2 | v1, reach-gated | 994 | 16 | depends on selected T | 0 | |
| `gru_v1_keep-2` | GRU 3x16x2, `b_z = -2` at init | v1, reach-gated | 994 | 16 | | 0 | |
| `gru_v2` | GRU 4x15x2 | v2, reach-gated | 932 | 15 | 384 | 0 | 1,331 |
| `elman_v2` | Elman 4x28x2 | v2, reach-gated | 982 | 28 | 248 | 0 | 1,258 |
| `mgu_v2` | MGU 4x19x2 | v2, reach-gated | 952 | 19 | 404 | 0 | 1,375 |
| `lru_v2` | LRU 24 modes, readout 13 | v2, reach-gated | 957 | 48 | 0 | 240 | 1,245 |
| `mlp_v2` (control) | MLP 4x27x27x2 | v2 | 947 | 0 | 62 | 0 | |

Counts are recomputed from arrays by `arms.audit`, which also reports the dot
adapter's 14 per-cell interaction scalars. The TBPTT buffer depends on the
horizon each candidate's development selection chooses; the final report uses
the selected horizon. Ablations share their parent's configuration exactly:
`state_reset`, `no_error_input`, `no_observed_flag` (v2 only) and
`no_error_head_loss` (`lambda = 0`, which leaves the head untrained and
functionally removes it).

## 6. Parity with the historical implementation

`tests/test_aaa_1k_v2.py` requires batched Champion 1 on the CPU to agree with
`research.aaa_1k` + `ReachGatedUnfoldAgent` to within 1e-12 per step on
historical streams. In the qualification run most cells are bitwise identical;
the remainder differ in the last bit because batched matrix products sum in a
different order from OpenBLAS's matrix-vector kernel. The CUDA path agrees to
about 1e-15 relative on the same cells. Cross-backend numerical tolerances and
verdict-level equivalence are defined and measured in
[`aaa_1k_v2_compute_report.md`](aaa_1k_v2_compute_report.md); paired scientific
comparisons always use one backend.

## 7. Serialization

`BatchedLearner.state_dict` and `adapters.adapter_state` capture parameters,
hidden state, TBPTT window, traces, counters and interaction state. Loading
validates core, rule, cells, shapes and finiteness, and a run interrupted and
resumed is bitwise identical to an uninterrupted one (tested).
