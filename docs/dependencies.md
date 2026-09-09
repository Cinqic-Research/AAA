# Dependencies and licenses

AAA keeps a CPU route with public open-source Python dependencies. The direct runtime dependencies are declared in `pyproject.toml` and pinned for the validated environment in `requirements-lock.txt`.

| Dependency | Use | License/provenance |
|---|---|---|
| NumPy | seeded simulation, vector math, SVD/RLS diagnostics | BSD-3-Clause; PyPI/public upstream project |
| Matplotlib | optional static plots and GUI animation | Matplotlib license, PSF-compatible; PyPI/public upstream project |
| pytest | optional development test runner | MIT; PyPI/public upstream project |
| Python standard library | JSON/JSONL, gzip, hashing, timing, CLI, platform metadata | Python Software Foundation License |

The repository itself is Apache License 2.0. No proprietary model API, paid service, pretrained model, cloud compute, or GPU-only dependency is required. `psutil` is optional metadata enrichment only; if unavailable, the run records that the field was not measured instead of guessing.
