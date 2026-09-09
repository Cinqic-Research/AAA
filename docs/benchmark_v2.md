# Benchmark v2 specification

This file describes the predeclared engineering benchmark in `benchmarks/benchmark_v2.json`. It is separate from `results/final`, which remains the historical v1 snapshot and regression track.

The required normalized error is absolute position error divided by the configured interval width `L = upper_bound - lower_bound`. A prediction is made before the target transition is advanced and revealed. The first post-event prediction is the unavoidable surprise step and remains in every total and 50-transition response metric.

The constant-velocity and bouncing families use 100 confirmation episodes per independent training replica, four fixed position bands, four fixed speed bands, and both velocity directions. Constant-velocity episodes run for 40 transitions so the full declared speed range remains inside bounds; bouncing episodes run for 1,200 transitions. Event counts are reported and the benchmark requires at least 500 scored bounce events across at least 100 episodes. Speed-change episodes use the 1,200-transition horizon and unannounced factors of 0.45 or 1.80 at transition 600. The learner sees positions only; scenario, direction, speed, event time, and event labels remain evaluator-side metadata.

The changed-law family is a stable second-order damped oscillator. With midpoint `m`, position `x`, velocity `v`, natural frequency `ω`, and damping coefficient `ζ`, the simulator uses semi-implicit Euler:

```text
a[t] = -2 ζ ω v[t] - ω² (x[t] - m)
v[t+1] = v[t] + a[t] dt
x[t+1] = reflect(x[t] + v[t+1] dt)
```

The pre-event coefficients are `ω=1.5, ζ=0.10`; the changed-law coefficients are `ω=8.0, ζ=0.15`; the event is at transition 600. The larger coefficient change is intentional: it prevents persistence from making the adaptation question vacuous while remaining a stable, bounded simulator under the declared time step. Each evaluator first runs a common prefix, clones the complete candidate state and observation history immediately before the intervention, and then branches: changed-law frozen, changed-law updating, unchanged frozen, and unchanged updating. The unchanged branch uses the pre-event coefficients for the full horizon. The evaluator does not notify the learner or reset its optimizer.

The declared candidate is the zero-initialized normalized RLS predictor with observation-derived features `[1, Δx[t]/(dt*speed_max), (x[t]-m)/L]` and deterministic observation-only reflection of its position prediction. Its declared forgetting factor is `0.90` and ridge is `1e-4`; the full covariance is serialized. The legacy global position-feature SGD model remains a named diagnostic comparison. The candidate is selected from development evidence before confirmation; no confirmation result selects a model or threshold.

Confirmation requires at least five independently trained replicas and 100 episodes per required family. Results include episode and replica counts, stratum summaries, event counts, worst-replica values, p95 errors, signed bias where available, failures/censoring, and paired hierarchical bootstrap 95% intervals. Bootstrap units are replicas and then episodes within replica; adjacent transitions are never treated as independent training replicas.

Run a fresh attempt with:

```bash
.venv/bin/python -m aaa.cli benchmark-v2 \
  --role confirmation_a \
  --attempt-id confirmation-a \
  --output-root runs
```

Confirmation B uses a new attempt ID and the same committed source, thresholds, model, budgets, and specification. A failed attempt is retained and its seeds are retired from later confirmation. The result directory contains compressed raw step records, checkpoints, metadata, checksums, metrics, gate decisions, and a generated report.
