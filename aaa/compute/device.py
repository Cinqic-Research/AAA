"""Device specifications, CUDA availability, and backend resolution.

Grammar
-------
``cpu``
    NumPy on the host CPU. Always available.
``cuda`` / ``cuda:<index>``
    CuPy on an NVIDIA device. ``cuda`` means ``cuda:0``. Requesting CUDA when
    CuPy is not installed, no device is visible, or the index is out of range
    raises :class:`DeviceUnavailableError`; it never silently falls back to the
    CPU, because a silent fallback would change the numerical platform of an
    experiment without saying so.
``auto``
    Resolve to CPU or CUDA from a *workload hint* (the number of independent
    cells run in lockstep) using the measured crossover in
    :data:`AUTO_CUDA_MIN_CELLS`. Without a hint, ``auto`` resolves to the CPU.
    ``auto`` is a convenience for exploratory work only. A formal experiment
    records and freezes the device it resolved to *before* it observes
    anything, and confirmation code refuses an unresolved ``auto``
    (:func:`require_resolved`).

The same array program runs on either backend through ``Backend.xp``: the
``numpy`` module on the CPU and the ``cupy`` module on CUDA. Scientific code
takes a :class:`Backend` and never branches on ``if cuda``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np

DEVICE_SPEC_GRAMMAR = "cpu | cuda | cuda:<index> | auto"

AUTO_CUDA_MIN_CELLS = 1024
"""Smallest lockstep cell count at which ``auto`` selects CUDA.

Measured on FLOWBOX for the AAA-1K online GRU step against a *single* CPU
process (``docs/aaa_1k_v2_compute_report.md``): at 512 cells one CPU core ran
about 96k cell-steps/s against the RTX 2060's 77k, and from 1,024 cells the GPU
led (155k against 88k). Below this many independent cells kernel-dispatch
overhead dominates a 1K-parameter model's arithmetic. It is a performance
policy, not a scientific parameter, and it never applies to a frozen run.
Multi-process CPU work (``--workers``) moves the crossover higher; the compute
report gives both.
"""

_SPEC = re.compile(r"^(cpu|auto|cuda)(?::(\d+))?$")

DeviceKind = Literal["cpu", "cuda", "auto"]


class InvalidDeviceError(ValueError):
    """The device string does not follow :data:`DEVICE_SPEC_GRAMMAR`."""


class DeviceUnavailableError(RuntimeError):
    """A well-formed device was requested but cannot be used on this machine."""


@dataclass(frozen=True)
class DeviceSpec:
    """A parsed device request. ``index`` is set only for CUDA."""

    kind: DeviceKind
    index: int | None = None

    def __str__(self) -> str:
        return f"cuda:{self.index}" if self.kind == "cuda" else self.kind


def parse_device(text: str) -> DeviceSpec:
    """Parse ``text`` or raise :class:`InvalidDeviceError`."""

    if not isinstance(text, str):
        raise InvalidDeviceError(f"device must be a string ({DEVICE_SPEC_GRAMMAR}), got {text!r}")
    match = _SPEC.match(text.strip().lower())
    if match is None:
        raise InvalidDeviceError(f"invalid device {text!r}; expected {DEVICE_SPEC_GRAMMAR}")
    kind, index = match.group(1), match.group(2)
    if kind != "cuda" and index is not None:
        raise InvalidDeviceError(f"only cuda takes a device index, got {text!r}")
    if kind == "cuda":
        return DeviceSpec("cuda", int(index) if index is not None else 0)
    return DeviceSpec("cpu" if kind == "cpu" else "auto")


def _default_cupy_loader() -> ModuleType:
    return importlib.import_module("cupy")


CupyLoader = Callable[[], ModuleType]


def cuda_available(loader: CupyLoader = _default_cupy_loader) -> tuple[bool, str, int]:
    """Return ``(available, reason, device_count)`` without raising."""

    try:
        cupy = loader()
    except ImportError as error:
        return False, f"CuPy is not installed ({error})", 0
    except Exception as error:  # a broken CUDA install surfaces here, not as ImportError
        return False, f"CuPy failed to import ({type(error).__name__}: {error})", 0
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as error:
        return False, f"no usable CUDA driver/device ({type(error).__name__}: {error})", 0
    if count < 1:
        return False, "CUDA reports zero devices", 0
    return True, "ok", count


def auto_policy(cells: int | None, *, cuda_ok: bool) -> Literal["cpu", "cuda"]:
    """The ``auto`` decision for a workload of ``cells`` lockstep cells."""

    if not cuda_ok or cells is None:
        return "cpu"
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError("cells must be a positive integer workload hint")
    return "cuda" if cells >= AUTO_CUDA_MIN_CELLS else "cpu"


class Backend:
    """One resolved compute backend: an array module plus where it runs.

    ``requested`` is what the caller asked for (possibly ``auto``);
    ``resolved`` is always ``cpu`` or ``cuda:<index>``.
    """

    dtype = np.float64

    def __init__(self, *, requested: DeviceSpec, kind: Literal["cpu", "cuda"], index: int | None) -> None:
        self.requested = requested
        self.kind: Literal["cpu", "cuda"] = kind
        self.index = index
        if kind == "cpu":
            self.xp: Any = np
            self.library = "numpy"
        else:
            self.xp = importlib.import_module("cupy")
            self.library = "cupy"
            self.xp.cuda.Device(index).use()

    @property
    def resolved(self) -> str:
        return "cpu" if self.kind == "cpu" else f"cuda:{self.index}"

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    def asarray(self, value: Any, dtype: Any = None) -> Any:
        return self.xp.asarray(value, dtype=self.dtype if dtype is None else dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> Any:
        return self.xp.zeros(shape, dtype=self.dtype if dtype is None else dtype)

    def to_host(self, value: Any) -> np.ndarray:
        """Copy to a NumPy array (a no-op view on the CPU)."""

        if self.kind == "cpu":
            return np.asarray(value)
        return self.xp.asnumpy(value)

    def synchronize(self) -> None:
        """Block until queued device work is complete (timing must call this)."""

        if self.kind == "cuda":
            self.xp.cuda.Device(self.index).synchronize()

    def __repr__(self) -> str:
        return f"Backend(requested={self.requested!s}, resolved={self.resolved}, library={self.library})"


def resolve_backend(
    device: str | DeviceSpec = "cpu",
    *,
    cells: int | None = None,
    loader: CupyLoader = _default_cupy_loader,
) -> Backend:
    """Resolve a device request to a :class:`Backend`, or raise."""

    spec = parse_device(device) if isinstance(device, str) else device
    if spec.kind == "cpu":
        return Backend(requested=spec, kind="cpu", index=None)
    available, reason, count = cuda_available(loader)
    if spec.kind == "auto":
        choice = auto_policy(cells, cuda_ok=available)
        if choice == "cpu":
            return Backend(requested=spec, kind="cpu", index=None)
        return Backend(requested=spec, kind="cuda", index=0)
    if not available:
        raise DeviceUnavailableError(f"device {spec} requested but CUDA is unavailable: {reason}")
    assert spec.index is not None
    if spec.index >= count:
        raise DeviceUnavailableError(f"device {spec} requested but only {count} CUDA device(s) are visible")
    return Backend(requested=spec, kind="cuda", index=spec.index)


def require_resolved(device: str | DeviceSpec) -> DeviceSpec:
    """Refuse ``auto``: a frozen experiment must name the device it will run on."""

    spec = parse_device(device) if isinstance(device, str) else device
    if spec.kind == "auto":
        raise InvalidDeviceError("a frozen or confirmatory run must name a resolved device, not 'auto'")
    return spec
