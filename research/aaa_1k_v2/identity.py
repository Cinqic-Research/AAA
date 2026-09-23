"""Scientific source identity of ``aaa.1k.v2`` (non-self-referential, fail-closed).

The fingerprint hashes every file whose bytes can change a v2 number:

* this phase's package ``research/aaa_1k_v2/`` (including the vendored dysts
  metadata) and the compute package ``aaa/compute/``;
* the historical code v2 *executes*: the whole ``research/aaa_1k/`` package
  (stream generators and baseline agents) and the shared ``aaa`` modules it
  imports -- all unchanged, and still covered by the ``aaa.1k.v1`` fingerprint;
* this phase's tests and protocol documents;
* both dependency locks.

Generated outputs -- evidence, reports, the handoff, the self-review, the
tracked-file audit -- are excluded, so committing a result cannot change the
identity that result records.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import PHASE_VERSION

IDENTITY_SCHEMA = "aaa.1k.v2.scientific_source.v1"
PREFIXES = ("research/aaa_1k_v2/", "research/aaa_1k/", "aaa/compute/")
SHARED = (
    "research/__init__.py",
    "aaa/__init__.py",
    "aaa/config.py",
    "aaa/environment.py",
    "aaa/predictors.py",
)
TESTS = ("tests/test_aaa_1k_v2.py", "tests/test_compute.py")
DOCS = (
    "docs/aaa_1k_v2_research_brief.md",
    "docs/aaa_1k_v2_architecture.md",
    "docs/aaa_1k_v2_benchmark_protocol.md",
    "docs/aaa_1k_v2_external_benchmarks.md",
)
LOCKS = ("requirements-lock.txt", "requirements-cuda-lock.txt")
GENERATED = (
    "docs/aaa_1k_report.md",
    "docs/aaa_1k_handoff.md",
    "docs/aaa_1k_self_review.md",
)


class IdentityError(RuntimeError):
    pass


def _tracked(root: Path) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z", "--cached"], stderr=subprocess.PIPE
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise IdentityError("the v2 fingerprint requires a readable Git repository") from error
    return {name for name in output.decode("utf-8").split("\0") if name}


def phase_files(root: Path) -> list[str]:
    tracked = _tracked(root)
    selected = {name for name in tracked if name.startswith(PREFIXES) and name not in GENERATED}
    selected.update(name for name in (*SHARED, *TESTS, *DOCS, *LOCKS) if name in tracked)
    return sorted(selected)


def fingerprint(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    names = phase_files(root)
    missing = [name for name in (*SHARED, *DOCS, *LOCKS) if name not in names]
    if missing:
        raise IdentityError(f"v2 fingerprint is missing required files: {missing}")
    files: dict[str, dict[str, Any]] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise IdentityError(f"{name} is missing, not a regular file, or a symlink")
        content = path.read_bytes()
        files[name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": IDENTITY_SCHEMA,
        "phase_version": PHASE_VERSION,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "files": files,
    }
