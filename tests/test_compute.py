"""Device selection, backend resolution, provenance and the hardware probe.

CUDA itself is exercised only where CuPy and a device exist (FLOWBOX); ordinary
CI has neither, so the absence paths are tested with injected loaders and the
parity tests skip. GitHub-hosted CI does not test CUDA, and nothing claims it does.
"""

from __future__ import annotations

import getpass
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aaa.compute import (
    DeviceUnavailableError,
    InvalidDeviceError,
    auto_policy,
    backend_provenance,
    cuda_available,
    parse_device,
    resolve_backend,
)
from aaa.compute.device import AUTO_CUDA_MIN_CELLS, require_resolved
from aaa.compute.hardware import MACHINE, probe, provenance_counts, redact


def _missing_cupy() -> object:
    raise ImportError("No module named 'cupy'")


def _zero_devices() -> object:
    return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0)))


def _broken_driver() -> object:
    def fail() -> int:
        raise RuntimeError("cudaErrorInsufficientDriver")

    return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=fail)))


def _one_device() -> object:
    return SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 1)))


def cuda_ready() -> bool:
    return cuda_available()[0]


class DeviceParsingTests(unittest.TestCase):
    def test_grammar(self) -> None:
        self.assertEqual(str(parse_device("cpu")), "cpu")
        self.assertEqual(str(parse_device("cuda")), "cuda:0")
        self.assertEqual(str(parse_device("CUDA:3")), "cuda:3")
        self.assertEqual(str(parse_device("auto")), "auto")

    def test_invalid_devices_are_refused(self) -> None:
        for text in ("gpu", "cuda:", "cuda:-1", "cpu:0", "auto:1", "cuda:x", "", "tpu"):
            with self.subTest(text=text), self.assertRaises(InvalidDeviceError):
                parse_device(text)
        with self.assertRaises(InvalidDeviceError):
            parse_device(0)  # type: ignore[arg-type]

    def test_frozen_runs_refuse_auto(self) -> None:
        with self.assertRaises(InvalidDeviceError):
            require_resolved("auto")
        self.assertEqual(str(require_resolved("cuda:0")), "cuda:0")


class ResolutionTests(unittest.TestCase):
    def test_cpu_always_resolves_to_numpy(self) -> None:
        backend = resolve_backend("cpu")
        self.assertEqual(backend.resolved, "cpu")
        self.assertIs(backend.xp, np)
        self.assertEqual(backend.to_host(backend.asarray([1.0, 2.0])).tolist(), [1.0, 2.0])

    def test_cuda_without_cupy_raises_instead_of_falling_back(self) -> None:
        for loader in (_missing_cupy, _zero_devices, _broken_driver):
            with self.subTest(loader=loader.__name__), self.assertRaises(DeviceUnavailableError):
                resolve_backend("cuda", loader=loader)

    def test_invalid_index_is_refused(self) -> None:
        with self.assertRaises(DeviceUnavailableError):
            resolve_backend("cuda:1", loader=_one_device)

    def test_availability_reports_reasons_without_raising(self) -> None:
        self.assertEqual(cuda_available(_missing_cupy)[0], False)
        self.assertIn("not installed", cuda_available(_missing_cupy)[1])
        self.assertEqual(cuda_available(_zero_devices)[:2], (False, "CUDA reports zero devices"))

    def test_auto_policy(self) -> None:
        self.assertEqual(auto_policy(None, cuda_ok=True), "cpu")
        self.assertEqual(auto_policy(10**6, cuda_ok=False), "cpu")
        self.assertEqual(auto_policy(AUTO_CUDA_MIN_CELLS - 1, cuda_ok=True), "cpu")
        self.assertEqual(auto_policy(AUTO_CUDA_MIN_CELLS, cuda_ok=True), "cuda")
        with self.assertRaises(ValueError):
            auto_policy(0, cuda_ok=True)

    def test_auto_without_cuda_is_cpu(self) -> None:
        self.assertEqual(resolve_backend("auto", cells=10**6, loader=_missing_cupy).resolved, "cpu")


class ProvenanceTests(unittest.TestCase):
    def test_cpu_provenance_records_the_platform(self) -> None:
        record = backend_provenance(resolve_backend("cpu"))
        self.assertEqual(record["resolved_device"], "cpu")
        self.assertEqual(record["processor"], "CPU")
        self.assertEqual(record["dtype"], "float64")
        self.assertIsNone(record["cuda"])
        self.assertEqual(len(record["base_lock_sha256"]), 64)
        json.dumps(record, allow_nan=False)


class HardwareProbeTests(unittest.TestCase):
    def test_redaction_removes_home_and_username(self) -> None:
        home = str(Path.home())
        self.assertEqual(redact(f"{home}/x"), "<home>/x")
        user = getpass.getuser()
        self.assertNotIn(user, redact(f"/media/{user}/Cinqic Storage"))

    def test_probe_is_strict_json_and_private(self) -> None:
        profile = probe(commit=None)
        text = json.dumps(profile, allow_nan=False)
        self.assertNotIn(str(Path.home()), text)
        self.assertNotIn("GPU-", text, "a GPU UUID must never be recorded")
        counts = provenance_counts(profile)
        self.assertGreater(counts[MACHINE], 10)
        for section in ("cpu", "memory", "gpu", "cuda_stack", "storage", "platform"):
            self.assertIn(section, profile)

    def test_committed_profile_is_private(self) -> None:
        path = Path(__file__).resolve().parents[1] / "benchmarks" / "hardware" / "flowbox.json"
        if not path.exists():
            self.skipTest("no committed profile")
        text = path.read_text(encoding="utf-8")
        profile = json.loads(text)
        self.assertEqual(profile["schema"], "aaa.compute.hardware_profile.v1")
        self.assertNotIn("/home/", text)
        self.assertNotIn("GPU-", text)


@unittest.skipUnless(cuda_ready(), "CUDA is not available on this machine")
class CudaBackendTests(unittest.TestCase):
    """Run on FLOWBOX only; hosted CI has no NVIDIA device."""

    def test_cuda_resolution_and_provenance(self) -> None:
        backend = resolve_backend("cuda:0")
        record = backend_provenance(backend)
        self.assertEqual(record["processor"], "GPU")
        self.assertIn("gpu_name", record["cuda"])
        values = backend.to_host(backend.xp.arange(4.0))
        self.assertEqual(values.tolist(), [0.0, 1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
