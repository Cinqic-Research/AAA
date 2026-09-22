# Dependencies and licenses

AAA runs on CPU with public open-source Python packages. Direct dependencies are
declared in `pyproject.toml`; the validated environment is pinned in
`requirements-lock.txt`, which CI installs and then verifies against the
installed set with `tools/check_lock.py`.

| Dependency | Use | License |
|---|---|---|
| NumPy | seeded simulation, linear algebra, SVD reference solutions, resampling | BSD-3-Clause |
| Matplotlib | static plots and the optional animation | Matplotlib license (PSF-compatible) |
| ruff | formatter and linter | MIT |
| mypy | static type checking | MIT |
| coverage | coverage measurement | Apache-2.0 |
| build | wheel build frontend | MIT |
| setuptools | pinned PEP 517 build backend | MIT |
| Python standard library | JSON/JSONL, gzip, hashing, timing, CLI, platform metadata | PSF License |

`psutil` is optional metadata enrichment only. When it is absent the run records
that the field was not measured rather than guessing.

The repository is Apache License 2.0. No proprietary model API, paid service,
pretrained model, cloud compute or GPU-only dependency is required, and none is
used. The development workstation happens to contain a discrete GPU; no
dependency here uses it, and [`hardware.md`](hardware.md) explains why that
changes nothing about this list.

## Regenerating the lock

```bash
python -m pip install -e '.[dev]'
python -m pip list --format=freeze
python tools/check_lock.py
```

The lock is also inside the `aaa.1k.v1` phase fingerprint
(`research/aaa_1k/identity.py`). Changing one pin changes that fingerprint, so
Champion 0 and Champion 1 verification fail until the change is recorded as a
new phase identity; see `AAA-152` and the fingerprinted-files rule in
[`CONTRIBUTING.md`](../CONTRIBUTING.md). For that reason
`.github/dependabot.yml` ignores every `pip` update, security updates
included, and Dependabot alerts remain the notification path for a vulnerable
pin (`AAA-175`). Regenerating the lock is a maintainer decision, never a bot
pull request.

`pyproject.toml` pins the PEP 517 backend itself to `setuptools==84.0.0`.
The locked CI job installs the matching lock and builds with
`python -m build --no-isolation`; `--no-deps` applies only when installing the
already-built AAA wheel and is not presented as build-backend pinning.
