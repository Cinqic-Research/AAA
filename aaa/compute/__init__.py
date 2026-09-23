"""Explicit compute-device selection and backend provenance for new AAA work.

Historical AAA evidence (v1, v2.1, observation noise, ``aaa.1k.v1``, the loop
iterations and both champions) was produced by CPU/NumPy code that does not
import this package, and nothing here changes it. This package exists so that
*new* phases can run the same array program on the CPU or on a CUDA device,
choose between them explicitly, and record which one they actually used.

The device is part of the scientific environment. A result produced on a GPU is
not the same number as a result produced on a CPU, and nothing in this package
pretends otherwise: every resolved backend carries its provenance, and formal
experiments must freeze a resolved device rather than ``auto``.
"""

from __future__ import annotations

from .device import (
    DEVICE_SPEC_GRAMMAR,
    Backend,
    DeviceSpec,
    DeviceUnavailableError,
    InvalidDeviceError,
    auto_policy,
    cuda_available,
    parse_device,
    resolve_backend,
)
from .provenance import backend_provenance

__all__ = [
    "DEVICE_SPEC_GRAMMAR",
    "Backend",
    "DeviceSpec",
    "DeviceUnavailableError",
    "InvalidDeviceError",
    "auto_policy",
    "backend_provenance",
    "cuda_available",
    "parse_device",
    "resolve_backend",
]
