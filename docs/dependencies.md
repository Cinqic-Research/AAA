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
| Python standard library | JSON/JSONL, gzip, hashing, timing, CLI, platform metadata | PSF License |

`psutil` is optional metadata enrichment only. When it is absent the run records
that the field was not measured rather than guessing.

The repository is Apache License 2.0. No proprietary model API, paid service,
pretrained model, cloud compute or GPU-only dependency is required, and none is
used.

## Regenerating the lock

```bash
python -m pip install -e '.[dev]'
python -m pip list --format=freeze
python tools/check_lock.py
```
