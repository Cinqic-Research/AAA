"""The execution platform of an experiment, recorded as scientific environment.

:func:`backend_provenance` returns everything a reader needs to know which
numerical platform produced a number: the requested and resolved device, the
array library and its version, the dtype, the BLAS or cuBLAS build, the GPU,
its memory, the NVIDIA driver, the CUDA runtime CuPy is actually using (which
is *not* the same thing as the driver's supported CUDA version or an installed
toolkit), the CPU, the dependency-lock hashes and the machine profile hash.

Nothing here is guessed. A field that cannot be measured is ``None`` and the
reason is recorded next to it.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .device import Backend

PROVENANCE_SCHEMA = "aaa.compute.provenance.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_LOCK = "requirements-lock.txt"
CUDA_LOCK = "requirements-cuda-lock.txt"
MACHINE_PROFILE = "benchmarks/hardware/flowbox.json"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def numpy_blas() -> dict[str, Any]:
    """NumPy's BLAS/LAPACK build, from NumPy's own build configuration."""

    try:
        dependencies = np.__config__.CONFIG.get("Build Dependencies", {})
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}
    blas = dependencies.get("blas", {}) if isinstance(dependencies, dict) else {}
    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "openblas_configuration": blas.get("openblas configuration"),
    }


def _nvidia_smi(fields: str) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    return [value.strip() for value in completed.stdout.strip().splitlines()[0].split(",")]


def cuda_details(backend: Backend) -> dict[str, Any]:
    """Device, driver and runtime facts for a CUDA backend, from CuPy itself."""

    cupy = backend.xp
    runtime = cupy.cuda.runtime
    props = runtime.getDeviceProperties(backend.index)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    driver = runtime.driverGetVersion()
    runtime_version = runtime.runtimeGetVersion()
    try:
        nvrtc = ".".join(str(part) for part in cupy.cuda.nvrtc.getVersion())
    except Exception:
        nvrtc = None
    try:
        from cupy.cuda import cublas

        cublas_version: int | None = int(cublas.getVersion(cupy.cuda.device.get_cublas_handle()))
    except Exception:
        cublas_version = None
    smi = _nvidia_smi("driver_version,power.limit")
    return {
        "gpu_name": name,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "total_memory_bytes": int(props["totalGlobalMem"]),
        "multiprocessors": int(props["multiProcessorCount"]),
        "nvidia_driver_version": smi[0] if smi else None,
        "driver_supported_cuda": f"{driver // 1000}.{(driver % 1000) // 10}",
        "cupy_cuda_runtime": f"{runtime_version // 1000}.{(runtime_version % 1000) // 10}",
        "nvrtc_version": nvrtc,
        "cublas_version": cublas_version,
        "cuda_toolkit_wheels": {
            name: _version(name)
            for name in ("cuda-toolkit", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc", "nvidia-cublas")
        },
        "power_limit_watts": smi[1] if smi and len(smi) > 1 else None,
    }


def backend_provenance(backend: Backend, *, root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Complete platform record for one resolved backend."""

    record: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "requested_device": str(backend.requested),
        "resolved_device": backend.resolved,
        "processor": "GPU" if backend.is_cuda else "CPU",
        "array_library": backend.library,
        "array_library_version": _version("numpy")
        if backend.library == "numpy"
        else _version("cupy-cuda13x") or _version("cupy-cuda12x") or _version("cupy"),
        "dtype": np.dtype(backend.dtype).name,
        "numpy_version": np.__version__,
        "numpy_blas": numpy_blas(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "deterministic_settings": {
            "note": (
                "no atomics or nondeterministic reductions are used by AAA's array programs; "
                "run-to-run determinism on each backend is tested, cross-backend equality is not assumed"
            ),
            "cupy_cub_accelerators": None,
        },
        "base_lock_sha256": _sha256(root / BASE_LOCK),
        "cuda_lock_sha256": _sha256(root / CUDA_LOCK),
        "machine_profile_sha256": _sha256(root / MACHINE_PROFILE),
        "cuda": None,
    }
    if backend.is_cuda:
        record["cuda"] = cuda_details(backend)
        try:
            from cupy._core import _accelerator

            record["deterministic_settings"]["cupy_cub_accelerators"] = [
                str(value) for value in _accelerator.get_routine_accelerators()
            ]
        except Exception:
            record["deterministic_settings"]["cupy_cub_accelerators"] = "unavailable"
    return record
