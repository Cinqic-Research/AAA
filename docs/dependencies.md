# Dependencies and licenses

AAA runs on CPU with public open-source Python packages; new work can optionally use an NVIDIA GPU through CuPy. Direct dependencies are
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

### Optional CUDA environment (new work from `aaa.1k.v2`)

| Dependency | Use | License |
|---|---|---|
| CuPy 14.2.0 (`cupy-cuda13x`) | the CUDA backend of `aaa.compute`: the same array program as NumPy on an NVIDIA GPU | MIT |
| `cuda-toolkit` 13.2.2 metapackage, `nvidia-cuda-runtime` 13.2.86, `nvidia-cuda-nvrtc` 13.2.86, `nvidia-cublas` 13.4.1.3, `cuda-pathfinder` 1.8.2 | NVIDIA's CUDA runtime, runtime compiler and BLAS, installed from PyPI so no system CUDA toolkit is needed | NVIDIA proprietary (redistributable wheels) |

They are pinned, together with every base pin, in `requirements-cuda-lock.txt` and
verified with `python tools/check_lock.py --lock requirements-cuda-lock.txt`.
Nothing in the base installation, CI or any historical phase needs them, and
CUDA is never selected implicitly: a requested but unavailable device raises.
See [`aaa_1k_v2_compute_strategy.md`](aaa_1k_v2_compute_strategy.md) for why
CuPy was chosen over PyTorch (measured on the RTX 2060).

External benchmark data (Monash archive, CC BY 4.0) is downloaded on demand
into `$AAA_DATA_ROOT` and checksum-verified; it is not a Python dependency and
is never committed. The dysts 0.96 metadata file (Apache-2.0) is vendored with
attribution in `research/aaa_1k_v2/external/data/`.

`psutil` is optional metadata enrichment only. When it is absent the run records
that the field was not measured rather than guessing.

The repository is Apache License 2.0. No proprietary model API, paid service,
pretrained model, cloud compute or GPU-only dependency is required. The
optional CUDA environment above is the only GPU-related dependency, and no
historical result depends on it ([`hardware.md`](hardware.md)).

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
