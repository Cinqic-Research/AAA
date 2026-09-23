# AAA-1K v2 external benchmarks

The purpose of this suite is to stop measuring AAA only with tasks Cinqic
invented. Every entry below is defined by an external source, selected by a
rule written before any AAA model saw it, and fed to the models through a
causal adapter that owns no benchmark knowledge:

```text
external benchmark -> causal observation adapter -> generic AAA interface -> model
```

The benchmark owns truth. A model sees `s_t`, predicts `y_t`, and only then is
`y_t` revealed. No statistic of scored data enters normalization.

Code: [`research/aaa_1k_v2/external/`](../research/aaa_1k_v2/external/).
Metrics: [`metrics.py`](../research/aaa_1k_v2/external/metrics.py).

## Common protocol

* **Interface.** Every candidate and Champion 1 participate unchanged: the
  series adapter fills the same input slots as the dot's (centred level, change,
  previous error, and the observed flag for `v2` candidates, always 1 here). No
  wall map and no unfolding are used.
* **Allowed adaptation.** Online SGD throughout the online segment (the
  models' own rule); weights frozen during recursive forecasts.
* **Recursive forecasts** feed each prediction back as the next input with the
  previous-error input at zero, because nothing is revealed.
* **Baselines** (no initialization): persistence; online AR by recursive least
  squares (order 10, forgetting 1, ridge 100 initial covariance; its `d + d^2`
  adaptive scalars are reported); for Monash also the seasonal naive forecast
  and the archive's published results.
* **Dev / confirmation separation.** Synthetic realizations come from disjoint
  identity blocks per role. Monash development evaluates only inside the
  archive's training split.

## Entries

### NARMA-10 (`narma10`)

| Field | Value |
|---|---|
| Source | Atiya & Parlos (2000), *IEEE TNN* 11(3):697-709 |
| Definition | `y(t+1) = 0.3 y(t) + 0.05 y(t) sum_{i=0}^{9} y(t-i) + 1.5 u(t-9) u(t) + 0.1`, `u ~ U[0, 0.5]` i.i.d. |
| License | a published equation; no data redistributed |
| Generator | `synthetic.narma("narma10", seed, 6000, washout=200)` |
| Dimensionality | input 1, target 1 |
| Task | at step `t` see `u(t)`, predict `y(t)`; target mode `absolute` |
| Normalization | input: exact moments of `U[0,0.5]`; target: mean and s.d. of 4 calibration realizations (identity `calibration.narma10`) |
| Split | online over 6,000 steps; scored window: the last 2,000 |
| Metric | NMSE `mean((yhat - y)^2) / var(y)` (e.g. Rodan & Tino 2011); NRMSE also reported |
| Deviation | online prequential learning instead of an offline readout; the previous-error input reveals `y(t-1)` after it was scored, which the input-only reservoir protocol does not. The `no_error_input` ablation recovers the input-only setting. A realization with a non-finite or `|y| > 10` value (NARMA-10 occasionally diverges) is redrawn from the next sub-seed of the same identity; redraws are recorded |

### NARMA-20 (`narma20`)

As NARMA-10 with `y(t+1) = tanh(0.3 y(t) + 0.05 y(t) sum_{i=0}^{19} y(t-i) +
1.5 u(t-19) u(t) + 0.01)` (Rodan & Tino 2011, with the tanh saturation that
paper introduced). **NARMA-30 is not used**: the literature gives at least two
incompatible parameter sets for it, and one name must not cover two systems
(Wringe et al. 2024 review, arXiv:2405.06561).

### Mackey-Glass 17 (`mackey_glass17`)

| Field | Value |
|---|---|
| Source | Mackey & Glass (1977), *Science* 197:287-289; protocol per Jaeger (2010) as summarised in arXiv:2405.06561 |
| Definition | `dx/dt = 0.2 x(t-17) / (1 + x(t-17)^10) - 0.1 x(t)`; forward Euler, `dt = 0.1`; subsample every 10; `y = tanh(x - 1)`; washout 1,000 |
| Realizations | constant initial history `x0 ~ U[1.1, 1.3]` per seed |
| Task | one-step forecasting over 4,000 steps (scored window: last 2,000), then an 84-step recursive forecast |
| Metrics | prequential NRMSE; NRMSE of the recursive forecast; the error at step 84 (for NRMSE84 across realizations) |
| Normalization | 4 calibration realizations (`calibration.mackey_glass17`) |
| Deviation | online learning instead of an offline ridge readout on 3,000 or 21,000 points |

### dysts subset (`dysts:<system>`)

| Field | Value |
|---|---|
| Source | Gilpin (2021), *Chaos as an interpretable benchmark for forecasting and data-driven modelling*, NeurIPS Datasets & Benchmarks; github.com/williamgilpin/dysts |
| Version | dysts 0.96; metadata `chaotic_attractors.json` vendored unchanged, sha256 `f750c02d...f73174` |
| License | Apache-2.0 (equations transcribed from `dysts/flows.py` with attribution) |
| Selection | eligible: autonomous, not delay, dimension 3-4, bounded, period and positive Lyapunov exponent known (106 of 135). Rank by `lambda_max * period`, quartiles, three per quartile drawn with `derive_seed("development", "external.dysts-selection", 0)` |
| Selected | Hadley, SprottD, SprottK; KawczynskiStrizhak, SprottE, ZhouChen; Chen, Halvorsen, SprottJerk; Colpitts, HastingsPowell, SprottMore (all three-dimensional, a property of the draw) |
| Sampling | 100 points per dominant period, as dysts' `make_trajectory(resample=True, pts_per_period=100)` |
| Integration | fixed-step RK4 with step `period/100/k`, `k` the smallest integer that keeps the step at or below the metadata `dt`. **Deviation** from dysts' Radau at 1e-12, validated in [`evidence/aaa_1k_v2/dysts_validation.json`](evidence/aaa_1k_v2/dysts_validation.json): normalized RMS difference from dysts over the first quarter period about 1e-4 to 2e-4; attractor s.d. within 4%; ZhouChen's mean differs by 0.5 s.d. over 3,000 samples, consistent with slow switching between wings |
| Realizations | metadata initial condition plus 1% of each coordinate's attractor s.d., seeded; 20-period transient discarded |
| Observation | coordinate 0 standardized with dysts' published mean and s.d. (public constants) |
| Task | one-step forecasting over 3,000 steps (scored: last 1,000), then a 100-step (one period) recursive forecast |
| Metrics | prequential NRMSE; recursive-forecast NRMSE; sMAPE (percent, dysts' headline metric) on the raw coordinate |

### Monash Time Series Forecasting Archive (`monash:<dataset>`)

| Field | Value |
|---|---|
| Source | Godahewa et al. (2021), NeurIPS Datasets and Benchmarks; forecastingdata.org; Zenodo |
| License | CC BY 4.0; files downloaded on demand into `$AAA_DATA_ROOT` (default `~/.cache/aaa/external`), never committed |
| Selection | from the forecastingdata.org dataset table (read 2026-09-23): a no-missing variant exists; shortest series >= 700; total <= 2.5M observations; one per frequency, fewest series, ties alphabetical |
| Selected | Saugeen River Flow (daily, 1 series), M4 Hourly (414), Australian Electricity Demand (half-hourly, 5), FRED-MD (monthly, 107); all series used |
| Checksums | Zenodo MD5 plus recorded SHA-256 of each archive and extracted `.tsf`; a mismatch refuses the data |
| Horizon | M4 Hourly 48 (`.tsf` header); Saugeen 30, FRED-MD 12, Australian Electricity Demand 336 (the paper's rule for non-competition datasets) |
| Split | test = last `h` of each series (confirmation); development holds out the last `h` of the training split instead |
| Metric | MASE exactly as the archive's `utils/error_calculator.R`: in-sample seasonal naive scale at lag `min(seasonality)` (24 hourly, 48 half-hourly, 7 daily, 12 monthly), NA retried at lag 1, infinite values dropped; sMAPE as `calculate_smape` |
| Normalization | each series' own warm-up window (first max(50, 10%) training points); prequential scoring starts after it |
| Published baselines | the archive's mean MASE for SES, Theta, TBATS, ETS, (DHR-)ARIMA, PR, CatBoost, FFNN, DeepAR, N-BEATS, WaveNet and Transformer, parsed from the results table (page sha256 `ef7a4916...f8d8`) and stored in `monash.PUBLISHED_MASE` |
| Comparability | same training data, horizon and MASE; the published methods fit offline, AAA models learn online and forecast recursively. The results row labelled "Aus. Elecdemand" is matched to Australian Electricity Demand by name only; that mapping is not verified |

## Streaming concept-drift track: River (assessed, not integrated)

River 0.26.1 (BSD-3-Clause) was installed in a scratch environment and every
generator in `river.datasets.synth` was instantiated
([`evidence/aaa_1k_v2/river_assessment.json`](evidence/aaa_1k_v2/river_assessment.json)).
Its four regression generators -- `FriedmanDrift` (the one with concept drift)
and the drift-free `Friedman`, `Mv` and `Planes2D` -- each produce i.i.d.
tabular samples with 10 exogenous features and no temporal dependence. The
other fifteen (SEA, STAGGER, Hyperplane, RandomRBFDrift, Sine, Agrawal, LED and
the rest) are classification streams.

AAA's 1K core is a scalar next-step predictor whose inputs are one observed
channel, its change and its previous error. Taking part in a 10-feature
tabular regression would need a different, task-specific input model, and
converting the classification streams into regression would change the
problem. Neither is compatible without faking compatibility, so the River
track is **rejected for this phase** with this justification. External
distribution shift is still measured: the Monash series are non-stationary,
NARMA and Mackey-Glass realizations differ by seed, and the dot suite's change
families are unannounced.

## Prestige benchmarks deliberately not used

* **Gymnasium Classic Control.** Its tasks score control policies. The 1K core
  chooses no actions, and prediction is not reinforcement-learning control.
  It belongs on the roadmap for when AAA acts.
* **MLPerf Tiny.** It standardizes embedded inference of fixed image and audio
  models on hardware. It says nothing about whether AAA's online recurrent
  adaptation improves, and its models are not a scorecard for this core.

A benchmark is valuable because it measures the right thing.
