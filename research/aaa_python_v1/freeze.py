"""The ``aaa.python.v1`` confirmation freeze and its admission check.

Confirmation identities may be generated only when **all** of these hold:

1. a freeze manifest exists at :data:`FREEZE_PATH`, is tracked by Git and is
   committed (it is part of ``HEAD``, not merely present on disk);
2. the working tree is clean, so the running source is the committed source;
3. the manifest's recorded phase fingerprint equals the fingerprint of the
   running source (:func:`.identity.fingerprint`), which covers the generator,
   encoders, models, experiment, statistics and the frozen v0 modules v1
   imports;
4. the manifest declares the confirmation design and decision contracts that
   will read the evidence, so nothing about the decision can be chosen after
   observation.

The freeze file itself is excluded from the fingerprint it records (it would
otherwise be self-referential), as are generated evidence and reports.
:class:`Admission` re-verifies at the moment of use, so an admission obtained
earlier cannot outlive a later edit to the source.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION

ROOT = Path(__file__).resolve().parents[2]
FREEZE_PATH = "docs/evidence/aaa_python_v1/freeze.json"
FREEZE_SCHEMA = "aaa.python.v1.freeze.v1"
REQUIRED = ("schema", "protocol", "phase_fingerprint", "spec_sha256", "design", "contracts", "hypotheses")


class FreezeError(RuntimeError):
    pass


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout


def load_manifest(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_PATH
    if not path.is_file() or path.is_symlink():
        raise FreezeError(f"no freeze manifest at {FREEZE_PATH}")

    def refuse(token: str) -> Any:
        raise FreezeError(f"non-standard JSON constant {token} in the freeze manifest")

    manifest = json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)
    if not isinstance(manifest, dict):
        raise FreezeError("the freeze manifest must be a JSON object")
    missing = [key for key in REQUIRED if key not in manifest]
    if missing:
        raise FreezeError(f"freeze manifest is missing {missing}")
    if manifest["schema"] != FREEZE_SCHEMA or manifest["protocol"] != PROTOCOL_VERSION:
        raise FreezeError("freeze manifest schema or protocol does not match this package")
    return manifest


def check(root: Path = ROOT) -> dict[str, Any]:
    """The verified manifest, or :class:`FreezeError` naming the first failed condition."""

    from . import spec as spec_module
    from .identity import fingerprint

    manifest = load_manifest(root)
    committed = _git(root, "show", f"HEAD:{FREEZE_PATH}")
    if committed != (root / FREEZE_PATH).read_text(encoding="utf-8"):
        raise FreezeError("the freeze manifest on disk is not the committed one")
    if _git(root, "status", "--porcelain").strip():
        raise FreezeError("the working tree is dirty; confirmation runs only from committed source")
    live = fingerprint(root)["sha256"]
    if manifest["phase_fingerprint"] != live:
        raise FreezeError(
            f"source fingerprint {live} differs from the frozen {manifest['phase_fingerprint']}"
        )
    if manifest["spec_sha256"] != spec_module.spec_hash():
        raise FreezeError("the packaged specification differs from the frozen one")
    return manifest


@dataclass(frozen=True)
class Admission:
    """Proof, re-checked at use, that confirmation may be generated under a committed freeze."""

    root: Path = ROOT
    manifest: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def obtain(cls, root: Path = ROOT) -> Admission:
        return cls(root, check(root))

    def verify(self) -> bool:
        try:
            return check(self.root) == self.manifest
        except (FreezeError, OSError, ValueError):
            return False
