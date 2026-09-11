# Reproduction

Every command below is run from a clean checkout. No private filesystem path is
required.

```bash
git clone https://github.com/Cinqic/AAA.git
cd AAA
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python tools/check_lock.py
```

`tools/check_lock.py` prints the dependency-lock hash and fails if the installed
set drifts from the pins. CI runs the same check, so "the locked environment"
means the environment that was actually exercised.

## Verification suite

```bash
python -m unittest discover -s tests -t . -v     # 377 tests
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m coverage run -m unittest discover -s tests -t . && python -m coverage report
python tools/check_exit_codes.py
```

Set `MPLBACKEND=Agg` in a headless environment.

## Identity

```bash
python -m aaa.cli spec-hash          # canonical specification path, version, hash
python -m aaa.cli batches            # declared confirmation batches and their status
```

## Development work

Development runs may use small overrides and may record failed gates. They exit
0 regardless, because exploration is allowed to fail.

```bash
python -m aaa.cli benchmark --role development --attempt-label dev-001 \
  --replicas 2 --episodes 3 --output-root runs
```

Development-only evidence:

```bash
python -m aaa.cli diagnose --output docs/evidence/diagnosis
python -m aaa.cli select-candidate --output docs/evidence/candidate_selection.json
```

## Formal confirmation

Confirmation refuses a custom specification, an undeclared or already-consumed
batch, weakened minimums, a dirty source tree, or drift from the committed
freeze manifest. It exits non-zero if any required gate is not `PASS`.

```bash
python -m aaa.cli declare-batch aaa-v2_1-confirmation-a-0001 --role confirmation_a
python -m aaa.cli declare-batch aaa-v2_1-confirmation-b-0001 --role confirmation_b
python -m aaa.cli freeze \
  --batch aaa-v2_1-confirmation-a-0001 \
  --batch aaa-v2_1-confirmation-b-0001
git add benchmarks/freeze_manifest.json benchmarks/confirmation_batches.json
git commit -m "Freeze the confirmation plan"

python -m aaa.cli benchmark --role confirmation_a \
  --batch-id aaa-v2_1-confirmation-a-0001 --output-root runs
python -m aaa.cli benchmark --role confirmation_b \
  --batch-id aaa-v2_1-confirmation-b-0001 --output-root runs
```

Nothing may be tuned between A and B. A failed attempt is kept, its batch is
retired permanently, and a new predeclared batch is required for a fresh
attempt.

This has already happened. The round-1 pair failed and was retired; the
candidate was repaired on development evidence, which changed the specification
hash, which in turn meant round 2 needed newly declared batches. A batch
declared against one specification hash is refused under another. Both rounds
are in `benchmarks/confirmation_batches.json` and `results/benchmark_v2_1/`.

A higher-replication track with 20 independent training lineages:

```bash
python -m aaa.cli benchmark --role high_replication --output-root runs
```

## Recomputing evidence without rerunning anything

```bash
python -m aaa.cli recompute runs/benchmark-v2_1/<attempt>
```

This verifies checksums and the specification hash, rebuilds every per-step
metric, episode and replica summary, interval and gate from the retained raw
records, and exits non-zero if a stored gate status fails to reproduce. It never
retrains and never re-simulates.

## Regenerating raw evidence for a recorded attempt

```bash
git checkout <recorded commit>
python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <recorded batch id> --reproduce --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/<recorded batch id>
```

The reproduction must match the recorded checksums.

## Historical v1 track

```bash
python -m aaa.cli smoke --output-root runs
python -m aaa.cli full --output-root runs
```

Preserved for regression and provenance. Not acceptance evidence. Nothing is
written outside the requested output root.

## Optional animation

```bash
python -m aaa.cli animate --checkpoint runs/<attempt>/checkpoints/replica-00.json \
  --scenario changed --seed 201
```

Space pauses and resumes, `r` restarts. A display is required for this command
and for nothing else; its temporal semantics are tested headlessly.

## Hardware

No GPU is required and none is used. The model has three parameters. Every
attempt records the CPU model, core count, memory, OS, Python, NumPy, BLAS
build metadata and available disk space, so a latency number can be read in
context.
