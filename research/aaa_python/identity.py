"""Scientific source identity and run provenance for ``aaa.python.v0``.

The fingerprint hashes every tracked file whose bytes can change a v0 number:
this package (including its packaged specification and golden answer keys),
the promotion contract it declares (``aaa/promotion/``), the phase tests and
protocol documents, and the dependency lock. Generated evidence and reports are
excluded, so committing a result cannot change the identity it records.

Provenance is captured *before* any stage runs (the ``AAA-179`` lesson): the
commit, dirty flag and fingerprint describe the code that is about to produce
the evidence, not code edited while it ran.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from . import GENERATOR_VERSION, PROTOCOL_VERSION
from . import spec as spec_module
from .oracle import interpreter_provenance

IDENTITY_SCHEMA = "aaa.python.v0.scientific_source.v1"
ROOT = Path(__file__).resolve().parents[2]
PREFIXES = ("research/aaa_python/", "aaa/promotion/")
TESTS_PREFIX = "tests/test_aaa_python"
FILES = (
    "tests/test_promotion.py",
    "docs/aaa_python_research_brief.md",
    "docs/aaa_python_protocol.md",
    "docs/aaa_python_architecture.md",
    "docs/promotion_contract.md",
    "requirements-lock.txt",
)
GENERATED_PREFIXES = ("docs/evidence/aaa_python_v0/",)


class IdentityError(RuntimeError):
    pass


def _tracked(root: Path) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached"], stderr=subprocess.PIPE
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise IdentityError("the aaa.python.v0 fingerprint requires a Git checkout") from error
    return {name for name in output.decode("utf-8").split("\0") if name}


def phase_files(root: Path) -> list[str]:
    tracked = _tracked(root)
    selected = {
        name
        for name in tracked
        if (name.startswith(PREFIXES) or name.startswith(TESTS_PREFIX))
        and not name.startswith(GENERATED_PREFIXES)
    }
    selected.update(name for name in FILES if name in tracked)
    return sorted(selected)


def fingerprint(root: Path = ROOT) -> dict[str, Any]:
    names = phase_files(root)
    missing = [name for name in FILES if name not in names]
    if missing:
        raise IdentityError(f"fingerprint is missing required files: {missing}")
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise IdentityError(f"{name} is missing, not a regular file, or a symlink")
        data = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": IDENTITY_SCHEMA,
        "phase_version": PROTOCOL_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def provenance(device: str = "cpu") -> dict[str, Any]:
    """Everything needed to identify the producing code and platform, captured before observation."""

    from aaa.compute.device import resolve_backend
    from aaa.compute.provenance import backend_provenance

    if device != "cpu":
        raise ValueError("aaa.python.v0 runs its NumPy learner on the CPU only; no other device is accepted")
    # An installed package can sit inside a checkout (a venv in the repository);
    # only a Git top level that *is* this package root counts as the source.
    top = _git("rev-parse", "--show-toplevel")
    in_checkout = top is not None and Path(top).resolve() == ROOT
    record: dict[str, Any] = {
        "captured_before_run": True,
        "captured_at_unix": time.time(),
        "protocol": PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
        "spec_sha256": spec_module.spec_hash(),
        "interpreter": interpreter_provenance(),
        "compute": backend_provenance(resolve_backend("cpu")),
    }
    if in_checkout:
        record["source"] = {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
            "phase_fingerprint": fingerprint(ROOT)["sha256"],
        }
    else:
        record["source"] = {
            "commit": None,
            "dirty": None,
            "phase_fingerprint": "NOT_AVAILABLE: installed package",
        }
    return record
