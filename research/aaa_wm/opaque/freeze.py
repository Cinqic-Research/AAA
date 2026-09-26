"""The ``aaa.python.opaque.v0`` confirmation freeze and admission (mirrors ``aaa.python.v1``).

Confirmation identities may be generated only when all of these hold:

1. the freeze manifest exists at :data:`FREEZE_PATH` and is committed (on disk == ``HEAD``);
2. the working tree is clean;
3. the manifest's ``source_fingerprint`` equals :func:`fingerprint` of the running source (every
   file under ``research/aaa_wm/opaque`` except the freeze manifest's own evidence, the
   specification, and the v0/v1 modules those files import);
4. the manifest declares the confirmation design, arms, initializations and every decision
   contract before any confirmation identity exists.

:class:`Admission` re-verifies at use, so a later edit invalidates an earlier admission.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import generator as gen

ROOT = Path(__file__).resolve().parents[3]
FREEZE_PATH = "docs/evidence/aaa_wm_opaque_v0/freeze.json"
SCHEMA = "aaa.wm.opaque.freeze.v1"
REQUIRED = ("schema", "protocol", "source_fingerprint", "spec_sha256", "design", "arms", "contracts", "hypotheses")
IMPORTED = (
    "research/aaa_python/subset.py",
    "research/aaa_python/rng.py",
    "research/aaa_python/oracle.py",
    "research/aaa_python/_sandbox_child.py",
    "research/aaa_python_v1/spec.py",
    "research/aaa_python_v1/data/aaa_python_v1.json",
    "aaa/promotion/contract.py",
    "aaa/promotion/primary.py",
    "aaa/promotion/independent.py",
    "aaa/promotion/adjudicate.py",
)


class FreezeError(RuntimeError):
    pass


def source_files(root: Path = ROOT) -> list[str]:
    own = sorted(str(p.relative_to(root)) for p in (root / "research/aaa_wm/opaque").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return own + list(IMPORTED)


def fingerprint(root: Path = ROOT) -> dict[str, Any]:
    h = hashlib.sha256()
    files = {}
    for rel in source_files(root):
        data = (root / rel).read_bytes()
        files[rel] = hashlib.sha256(data).hexdigest()
        h.update(rel.encode() + b"\0" + data + b"\0")
    return {"sha256": h.hexdigest(), "files": files}


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        raise FreezeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def load_manifest(root: Path = ROOT) -> dict[str, Any]:
    path = root / FREEZE_PATH
    if not path.is_file() or path.is_symlink():
        raise FreezeError(f"no freeze manifest at {FREEZE_PATH}")

    def refuse(token: str) -> Any:
        raise FreezeError(f"non-standard JSON constant {token}")

    m = json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)
    if not isinstance(m, dict) or any(k not in m for k in REQUIRED):
        raise FreezeError("freeze manifest is malformed or incomplete")
    if m["schema"] != SCHEMA or m["protocol"] != gen.PROTOCOL:
        raise FreezeError("schema or protocol mismatch")
    return m


def check(root: Path = ROOT) -> dict[str, Any]:
    m = load_manifest(root)
    if _git(root, "show", f"HEAD:{FREEZE_PATH}") != (root / FREEZE_PATH).read_text(encoding="utf-8"):
        raise FreezeError("the freeze manifest on disk is not the committed one")
    if _git(root, "status", "--porcelain").strip():
        raise FreezeError("the working tree is dirty")
    live = fingerprint(root)["sha256"]
    if m["source_fingerprint"] != live:
        raise FreezeError(f"source fingerprint {live} differs from the frozen {m['source_fingerprint']}")
    if m["spec_sha256"] != gen.spec_hash():
        raise FreezeError("specification differs from the frozen one")
    return m


@dataclass(frozen=True)
class Admission:
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
