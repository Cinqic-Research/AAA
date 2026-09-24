"""CPython as the external ground-truth oracle for ``aaa.python.v0``.

The oracle is part of the *evaluator*, never of the learner, and it is not a
baseline: an interpreter's perfect answer is not learned intelligence.

Containment, in layers:

1. :func:`execute` refuses any program the subset validator rejects. Nothing
   unvalidated is ever executed; :func:`check_syntax` only *compiles*.
2. The program runs in a separate CPython process started with ``-I -S`` (no
   user site, no ``PYTHON*`` environment variables, no script directory on the
   path), an empty environment, and a fresh temporary working directory that is
   deleted afterwards.
3. The child applies CPU, address-space, file-size and process-count limits
   where the platform provides them, and records which it applied.
4. Validated programs see only the allowed builtins. The AST validator blocks
   attributes and reflective routes to excluded builtins; the child builtins
   restriction alone does not contain arbitrary unvalidated Python.
5. A hard wall-clock timeout kills the child, and output is capped.

A child that dies, is killed or returns garbage is ``sandbox_failure`` (or
``timeout`` / ``cpu_limit``), which is never confused with a program outcome.
"""

from __future__ import annotations

import json
import platform
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import spec as spec_module
from .subset import validate

CHILD = Path(__file__).with_name("_sandbox_child.py")
PROGRAM_STATUSES = ("ok", "exception", "output_limit", "valid", "syntax_error")


@dataclass(frozen=True)
class Outcome:
    status: str
    exception: str | None
    line: int | None
    stdout: str
    truncated: bool
    limits_applied: dict[str, int]
    seconds: float

    @property
    def is_program_outcome(self) -> bool:
        return self.status in PROGRAM_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _request(source: str, mode: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    sandbox = spec["sandbox"]
    return {
        "mode": mode,
        "source": source,
        "builtins": list(spec["subset"]["builtins"]),
        "cpu_seconds": sandbox["cpu_seconds"],
        "address_space_bytes": sandbox["address_space_bytes"],
        "max_stdout_chars": sandbox["max_stdout_chars"],
    }


def _failure(status: str, started: float) -> Outcome:
    return Outcome(status, None, None, "", False, {}, time.perf_counter() - started)


def _run_child(request: Mapping[str, Any], timeout: float) -> Outcome:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="aaa-python-oracle-") as workdir:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", str(CHILD)],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                cwd=workdir,
                env={},
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _failure("timeout", started)
    if completed.returncode != 0:
        killed_by_cpu = hasattr(signal, "SIGXCPU") and completed.returncode == -signal.SIGXCPU
        return _failure("cpu_limit" if killed_by_cpu else "sandbox_failure", started)
    try:
        lines = completed.stdout.strip().splitlines()
        payload = json.loads(lines[-1])
        return Outcome(
            status=payload["status"],
            exception=payload.get("exception"),
            line=payload.get("line"),
            stdout=payload.get("stdout", ""),
            truncated=bool(payload.get("truncated", False)),
            limits_applied=payload.get("limits_applied", {}),
            seconds=float(payload.get("seconds", 0.0)),
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return _failure("sandbox_failure", started)


def execute(source: str, spec: Mapping[str, Any] | None = None) -> Outcome:
    """Run a validated subset program. Raises :class:`~.subset.SubsetError` before running anything else."""

    spec = spec or spec_module.load()
    validate(source, spec)
    return _run_child(_request(source, "exec", spec), spec["sandbox"]["wall_timeout_seconds"])


def check_syntax(source: str, spec: Mapping[str, Any] | None = None) -> Outcome:
    """Compile (never execute) ``source`` in the sandbox: ``valid`` or ``syntax_error``."""

    spec = spec or spec_module.load()
    if len(source.encode("utf-8")) > spec["subset"]["limits"]["max_source_bytes"]:
        raise ValueError("source exceeds the byte limit")
    return _run_child(_request(source, "compile", spec), spec["sandbox"]["wall_timeout_seconds"])


def run_many(jobs: Sequence[tuple[str, str]], spec: Mapping[str, Any] | None = None) -> list[Outcome]:
    """``[(mode, source)]`` -> outcomes in order; each job is its own sandboxed process."""

    spec = spec or spec_module.load()
    workers = spec["sandbox"]["parallel_workers"]

    def one(job: tuple[str, str]) -> Outcome:
        mode, source = job
        return execute(source, spec) if mode == "exec" else check_syntax(source, spec)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, jobs))


def interpreter_provenance() -> dict[str, Any]:
    """The interpreter is the experimental environment, so its identity is scientific provenance."""

    return {
        "implementation": sys.implementation.name,
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:3]),
        "cache_tag": sys.implementation.cache_tag,
        "compiler": platform.python_compiler(),
        "build": list(platform.python_build()),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
