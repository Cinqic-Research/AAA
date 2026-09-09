# Reproduction and evidence handling

The historical v1 regression track is the committed `results/final` snapshot. It is not overwritten by benchmark v2. The exact reviewed baseline was commit `773ff195c3bece781fac613ac0d26f56cfc9625e`; its original results remain historical observations.

Use the local virtual environment and pinned runtime set:

```bash
cd /home/cinqic/Documents/AAA
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m aaa.cli diagnose --output diagnosis
```

For development-only work, small overrides are allowed:

```bash
.venv/bin/python -m aaa.cli benchmark-v2 --role development --attempt-id dev-001 --replicas 1 --episodes 3 --output-root runs
```

For confirmation, do not override the five-replica/100-episode minimums:

```bash
.venv/bin/python -m aaa.cli benchmark-v2 --role confirmation_a --attempt-id confirmation-a-001 --output-root runs
.venv/bin/python -m aaa.cli benchmark-v2 --role confirmation_b --attempt-id confirmation-b-001 --output-root runs
```

Each attempt records its source commit, a content tree hash, dirty-state flag, resolved specification, dependency versions, hardware snapshot, stable RNG mapping, serialized learner state, raw per-step predictions, metrics, checksums, and gate outcomes. The compressed raw directory is generated under ignored `runs/`; attach it to CI or retain it in another durable location before using the attempt as evidence. The workflow `.github/workflows/benchmark.yml` is the explicit hosted full-benchmark route and advertises a 30-day artifact retention.

No GPU is required. The core model is CPU-only. GUI animation is an optional check; static plots use Agg only when rendering is explicitly requested, while the animation command leaves GUI backend selection to the user environment.
