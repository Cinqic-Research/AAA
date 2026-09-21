"""Freeze: fix everything confirmation depends on, then prove nothing moved.

The freeze manifest records, before any confirmation identity is observed:

* the champion's phase fingerprint and the champion record's hash;
* the *confirmation source fingerprint*: a hash over exactly the files that
  can change a confirmation number -- the whole AAA-1K phase source set plus
  the loop modules the confirmation path imports (:data:`CONFIRMATION_SOURCES`).
  Reports, documentation and the recomputation tool are deliberately outside
  it, so they can be written after the result without breaking the freeze;
  a test asserts the confirmation path imports nothing outside the set;
* the claim under test, the design, every threshold and every criterion text;
* the confirmation identity blocks and a hash of their seed lists, with the
  disjointness proof.

:func:`verify_freeze` recomputes all of it and lists every disagreement. The
``confirm`` command runs only when that list is empty *and* the manifest is
committed and unmodified in Git, so a result can never be observed under a
freeze that exists only in a working tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.aaa_1k.identity import phase_files, phase_fingerprint

from .evidence import payload_sha256, read_strict_json
from .identities import block_seeds, find_block

FREEZE_SCHEMA = "aaa.loop.freeze.v1"

CONFIRMATION_SOURCES = (
    "research/__init__.py",
    "research/aaa_1k_loop/__init__.py",
    "research/aaa_1k_loop/arms.py",
    "research/aaa_1k_loop/bounded.py",
    "research/aaa_1k_loop/decision.py",
    "research/aaa_1k_loop/develop.py",
    "research/aaa_1k_loop/dynamics.py",
    "research/aaa_1k_loop/evidence.py",
    "research/aaa_1k_loop/harness.py",
    "research/aaa_1k_loop/identities.py",
    "research/aaa_1k_loop/iteration3.py",
    "research/aaa_1k_loop/challengers.py",
)
"""Loop modules on the confirmation path, in addition to every AAA-1K phase file."""


class FreezeError(RuntimeError):
    """Raised when a freeze is missing, uncommitted, stale or contradicted."""


def confirmation_source_fingerprint(root: Path) -> dict[str, Any]:
    names = sorted(set(phase_files(root)) | set(CONFIRMATION_SOURCES))
    files: dict[str, str] = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise FreezeError(f"confirmation source {name} is missing, not a file, or a symlink")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "file_count": len(files), "files": files}


def seed_list_sha256(ledger: Mapping[str, Any], block_id: str) -> str:
    return payload_sha256(block_seeds(find_block(ledger, block_id)))


def build_freeze(
    root: Path,
    *,
    ledger: Mapping[str, Any],
    freshness: Mapping[str, Any],
    champion_path: str,
    attack_path: str,
    blocks: tuple[str, ...],
    frozen_content: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": FREEZE_SCHEMA,
        "champion_phase_fingerprint": phase_fingerprint(root)["sha256"],
        "champion_record": {
            "path": champion_path,
            "sha256": hashlib.sha256((root / champion_path).read_bytes()).hexdigest(),
        },
        "attack_evidence": {
            "path": attack_path,
            "sha256": hashlib.sha256((root / attack_path).read_bytes()).hexdigest(),
        },
        "confirmation_source_fingerprint": confirmation_source_fingerprint(root),
        "confirmation_blocks": {
            block_id: {
                **{
                    key: find_block(ledger, block_id)[key]
                    for key in ("iteration", "role", "namespace", "start", "count")
                },
                "seed_list_sha256": seed_list_sha256(ledger, block_id),
            }
            for block_id in blocks
        },
        "freshness_proof": dict(freshness),
        "frozen": dict(frozen_content),
        "frozen_sha256": payload_sha256(dict(frozen_content)),
    }


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def require_committed(root: Path, relative: str) -> str:
    """The manifest must be tracked and identical to HEAD. Returns HEAD's commit."""

    if _git(root, "ls-files", "--error-unmatch", relative).returncode != 0:
        raise FreezeError(f"{relative} is not tracked: commit the freeze before observing confirmation")
    if _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise FreezeError(f"{relative} differs from HEAD: the observed freeze must be the committed one")
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise FreezeError("cannot resolve HEAD")
    return head.stdout.strip()


def verify_freeze(
    manifest: Mapping[str, Any],
    root: Path,
    *,
    ledger: Mapping[str, Any],
    frozen_content: Mapping[str, Any],
) -> list[str]:
    """Every way the live repository disagrees with the freeze."""

    problems: list[str] = []
    if manifest.get("schema") != FREEZE_SCHEMA:
        return ["unknown freeze schema"]
    if phase_fingerprint(root)["sha256"] != manifest["champion_phase_fingerprint"]:
        problems.append("champion phase fingerprint changed since the freeze")
    live = confirmation_source_fingerprint(root)
    frozen_source = manifest["confirmation_source_fingerprint"]
    if live["sha256"] != frozen_source["sha256"]:
        changed = sorted(
            name
            for name in set(live["files"]) | set(frozen_source["files"])
            if live["files"].get(name) != frozen_source["files"].get(name)
        )
        problems.append(f"confirmation source changed since the freeze: {changed}")
    for key in ("champion_record", "attack_evidence"):
        entry = manifest[key]
        path = root / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            problems.append(f"{key} {entry['path']} changed or vanished since the freeze")
    if (
        payload_sha256(dict(frozen_content)) != manifest["frozen_sha256"]
        or dict(frozen_content) != manifest["frozen"]
    ):
        problems.append("claim, design, thresholds or criteria differ from the frozen ones")
    for block_id, frozen_block in manifest["confirmation_blocks"].items():
        try:
            block = find_block(ledger, block_id)
        except Exception as error:
            problems.append(f"confirmation block {block_id}: {error}")
            continue
        if block["role"] != "confirmation":
            problems.append(f"block {block_id} is not a confirmation block")
        if seed_list_sha256(ledger, block_id) != frozen_block["seed_list_sha256"]:
            problems.append(f"block {block_id} seeds differ from the frozen ones")
    return problems


def load_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FreezeError(f"no freeze manifest at {path}")
    manifest = read_strict_json(path)
    if not isinstance(manifest, dict):
        raise FreezeError("freeze manifest must be an object")
    return manifest
