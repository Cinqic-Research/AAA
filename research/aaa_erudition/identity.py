"""Scientific source identity, freeze and confirmation admission for ``aaa.erudition.v0``.

The fingerprint hashes every tracked file whose bytes can change a number:
this package (specification, surrogate parameters and backend profile
included), its tests, the protocol and architecture documents, and the
dependency locks. Generated evidence is excluded, so committing a result
cannot change the identity it records.

Admission to the confirmation split requires a freeze manifest that is
tracked and unmodified in Git, whose recorded fingerprint equals the live
one, and whose frozen Erudition weights hash to the recorded digest.
Anything else raises ``AdmissionError``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import PROTOCOL_VERSION
from .toolshift import AdmissionError

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = ("research/aaa_erudition/", "tests/test_aaa_erudition")
FILES = (
    "docs/juniper1/protocol.md",
    "docs/juniper1/architecture.md",
    "requirements-lock.txt",
    "requirements-torch-lock.txt",
)
GENERATED_PREFIXES = ("docs/evidence/aaa_erudition_v0/",)
FREEZE_SCHEMA = "aaa.erudition.freeze.v1"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise AdmissionError(f"git {' '.join(args)} failed; a Git checkout is required") from error


def phase_files() -> list[str]:
    tracked = [name for name in _git("ls-files", "-z", "--cached").split("\0") if name]
    selected = {n for n in tracked if n.startswith(PREFIXES) and not n.startswith(GENERATED_PREFIXES)}
    selected.update(n for n in FILES if n in tracked)
    return sorted(selected)


def fingerprint() -> dict[str, Any]:
    files = {}
    for name in phase_files():
        files[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"protocol": PROTOCOL_VERSION, "files": len(files), "sha256": digest}


def committed_freeze(freeze_path: Path) -> dict[str, Any]:
    """A freeze that is tracked and unmodified in Git. Enough to recompute committed evidence."""

    relative = freeze_path.resolve().relative_to(ROOT)
    if str(relative) not in _git("ls-files", "--cached", str(relative)).splitlines():
        raise AdmissionError(f"{relative} is not tracked; commit the freeze before confirmation")
    if _git("status", "--porcelain", "--", str(relative)).strip():
        raise AdmissionError(f"{relative} differs from the committed freeze")
    freeze: dict[str, Any] = json.loads(freeze_path.read_text("utf-8"))
    if freeze.get("schema") != FREEZE_SCHEMA:
        raise AdmissionError("not an aaa.erudition freeze")
    return freeze


def admit(freeze_path: Path, model_path: Path) -> dict[str, Any]:
    """Check a freeze before any confirmation identity is generated. Returns the freeze.

    Generating confirmation streams additionally requires the live source to be
    exactly the frozen one and the Erudition weights to be the frozen weights.
    """

    freeze = committed_freeze(freeze_path)
    live = fingerprint()
    if freeze["fingerprint"]["sha256"] != live["sha256"]:
        raise AdmissionError("the phase source has changed since the freeze")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != freeze["erudition_model"]["weights_sha256"]:
        raise AdmissionError("the Erudition Model is not the frozen one")
    if _git("status", "--porcelain", "--", *phase_files()).strip():
        raise AdmissionError("tracked phase files have uncommitted changes")
    return freeze
