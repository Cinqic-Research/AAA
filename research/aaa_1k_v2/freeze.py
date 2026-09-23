"""Freeze: fix everything confirmation depends on, commit it, then prove nothing moved.

The manifest records the challenger's complete arm specification and Champion
1's, every descriptive arm, the v2 source fingerprint (with its per-file
hashes), the aaa.1k.v1 fingerprint, both lock hashes, the machine profile hash,
the backend and dtype, the reproduction standard, the families and external
tasks, every threshold and criterion text, the confirmation identity blocks
with the SHA-256 of their seed lists, the bootstrap design, and the SHA-256 of
the development and attack artifacts it rests on.

``confirm`` runs only if the manifest is tracked and identical to ``HEAD``, the
working tree is clean, and :func:`verify` finds no disagreement with the live
repository.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import PHASE_VERSION, plan
from .arms import ArmSpec, core_spec
from .engine import CellConfig
from .identities import find, seeds_of

FREEZE_SCHEMA = "aaa.1k.v2.freeze.v1"
REPRODUCTION_STANDARD = (
    "CPU (NumPy float64). Bitwise reproduction of every cell on the Zen 3 evidence platform; elsewhere "
    "verdict-level (identical identities, structure and adjudicated verdicts, numerical drift reported; AAA-173)."
)


class FreezeError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _payload_sha(payload: Any) -> str:
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def arm_from_dict(record: Mapping[str, Any]) -> ArmSpec:
    """Rebuild an :class:`ArmSpec` from :meth:`ArmSpec.to_dict` output (round-trip tested)."""

    core = dict(record["core"])
    extra = {"readout_hidden": core["readout_hidden"]} if "readout_hidden" in core else {}
    config = dict(record["config"])
    return ArmSpec(
        name=record["name"],
        kind=record["kind"],
        core=core_spec(core["kind"], core["inputs"], core["hidden"], core["outputs"], **extra),
        features=record["features"],
        target_rule=record["target_rule"],
        rule=record["rule"],
        tbptt_steps=record["tbptt_steps"] if record["tbptt_steps"] is not None else 0,
        config=CellConfig(
            learning_rate=config["learning_rate"],
            gradient_clip=config["gradient_clip"],
            error_loss_weight=config["error_loss_weight"],
            freeze_recurrent=config["freeze_recurrent"],
            reset_state_every_step=config["reset_state_every_step"],
            zero_input=tuple(config["zero_input"]),
            optimizer=config.get("optimizer", "sgd"),
        ),
        init=tuple(sorted(record["init"].items())),
        update_enabled=record["update_enabled"],
        description=record["description"],
    )


def confirmation_block_ids(registry: Mapping[str, Any]) -> list[str]:
    return sorted(block["block_id"] for block in registry["blocks"] if block["role"] == "confirmation")


def frozen_content(
    *,
    challenger: ArmSpec | None,
    arms: Mapping[str, tuple[ArmSpec, bool]],
    capability_arms: Sequence[str],
    monash_arms: Sequence[str],
) -> dict[str, Any]:
    from .confirmation import EXTERNAL, K3_RULE, MONASH

    return {
        "challenger": None if challenger is None else {"name": challenger.name, "spec": challenger.to_dict()},
        "arms": {
            name: {"spec": arm.to_dict(), "external": external} for name, (arm, external) in arms.items()
        },
        "capability_arms": list(capability_arms),
        "monash_arms": list(monash_arms),
        "families": list(plan.STAGE_FAMILIES["confirmation"]),
        "held_out_families": list(plan.HELD_OUT_FAMILIES),
        "external_tasks": list(EXTERNAL),
        "monash_tasks": list(MONASH),
        "backend": plan.CONFIRMATION_BACKEND,
        "dtype": "float64",
        "reproduction_standard": REPRODUCTION_STANDARD,
        "thresholds": {
            "noninferiority_margin": plan.NONINFERIORITY_MARGIN,
            "stress_improvement": plan.STRESS_IMPROVEMENT,
            "regression_guard": plan.REGRESSION_GUARD,
            "external_margin": plan.EXTERNAL_MARGIN,
            "divergence_factor": plan.DIVERGENCE_FACTOR,
        },
        "criteria": {**plan.CONFIRMATION, "K3_status_rule": K3_RULE},
        "statistics": {
            "bootstrap_draws": plan.BOOTSTRAP_DRAWS,
            "confidence": plan.CONFIDENCE,
            "bootstrap_seed_block": "v2-confirmation-bootstrap",
            "designs": "crossed initialization x stream percentile bootstrap; geometric mean over families with "
            "shared-initialization resampling (research/aaa_1k_v2/stats.py)",
        },
        "protocol": plan.PROTOCOL_VERSION,
    }


def build_manifest(
    root: Path,
    registry: Mapping[str, Any],
    content: Mapping[str, Any],
    *,
    development: str,
    attack: str | None,
) -> dict[str, Any]:
    from research.aaa_1k.identity import phase_fingerprint

    from .identity import fingerprint

    blocks = {}
    for block_id in confirmation_block_ids(registry):
        block = find(registry, block_id)
        if block["status"] != "reserved":
            raise FreezeError(f"confirmation block {block_id} is {block['status']}, not fresh")
        blocks[block_id] = {
            **{key: block[key] for key in ("namespace", "start", "count")},
            "seed_list_sha256": _payload_sha(seeds_of(dict(registry), block_id)),
        }
    v2 = fingerprint(root)
    return {
        "schema": FREEZE_SCHEMA,
        "phase": PHASE_VERSION,
        "source_commit": _git(root, "rev-parse", "HEAD"),
        "v2_fingerprint": v2["sha256"],
        "v2_fingerprint_files": v2["files"],
        "aaa_1k_v1_fingerprint": phase_fingerprint(root)["sha256"],
        "locks": {
            name: _sha((root / name).read_bytes())
            for name in ("requirements-lock.txt", "requirements-cuda-lock.txt")
        },
        "machine_profile_sha256": _sha((root / "benchmarks/hardware/flowbox.json").read_bytes()),
        "evidence": {
            "development": {"path": development, "sha256": _sha((root / development).read_bytes())},
            "attack": None
            if attack is None
            else {"path": attack, "sha256": _sha((root / attack).read_bytes())},
        },
        "confirmation_blocks": blocks,
        "frozen": dict(content),
        "frozen_sha256": _payload_sha(dict(content)),
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise FreezeError(completed.stderr.strip() or f"git {' '.join(args)} failed")
    return completed.stdout.strip()


def require_committed(root: Path, relative: str) -> str:
    """The manifest must be tracked and identical to ``HEAD``, and the tree clean."""

    if subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", relative], capture_output=True, check=False
    ).returncode:
        raise FreezeError(f"{relative} is not tracked: commit the freeze before confirmation")
    if subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", relative], check=False
    ).returncode:
        raise FreezeError(f"{relative} differs from HEAD")
    if _git(root, "status", "--porcelain"):
        raise FreezeError("the working tree is not clean; confirmation must run from a committed tree")
    return _git(root, "rev-parse", "HEAD")


def verify(manifest: Mapping[str, Any], root: Path, registry: Mapping[str, Any]) -> list[str]:
    """Every disagreement between the manifest and the live repository."""

    from research.aaa_1k.identity import phase_fingerprint

    from .identity import fingerprint

    problems: list[str] = []
    if manifest.get("schema") != FREEZE_SCHEMA:
        return ["unknown freeze schema"]
    live = fingerprint(root)
    if live["sha256"] != manifest["v2_fingerprint"]:
        changed = sorted(
            n
            for n in set(live["files"]) | set(manifest["v2_fingerprint_files"])
            if live["files"].get(n) != manifest["v2_fingerprint_files"].get(n)
        )
        problems.append(f"v2 source changed since the freeze: {changed}")
    if phase_fingerprint(root)["sha256"] != manifest["aaa_1k_v1_fingerprint"]:
        problems.append("aaa.1k.v1 fingerprint changed")
    for name, digest in manifest["locks"].items():
        if _sha((root / name).read_bytes()) != digest:
            problems.append(f"{name} changed")
    for role, entry in manifest["evidence"].items():
        if entry is not None and _sha((root / entry["path"]).read_bytes()) != entry["sha256"]:
            problems.append(f"{role} evidence changed")
    if _payload_sha(manifest["frozen"]) != manifest["frozen_sha256"]:
        problems.append("frozen content does not match its hash")
    for name, entry in manifest["frozen"]["arms"].items():
        if arm_from_dict(entry["spec"]).to_dict() != entry["spec"]:
            problems.append(f"arm {name} does not round-trip")
    for block_id, frozen_block in manifest["confirmation_blocks"].items():
        block = find(registry, block_id)
        if block["role"] != "confirmation":
            problems.append(f"{block_id} is not a confirmation block")
        if _payload_sha(seeds_of(dict(registry), block_id)) != frozen_block["seed_list_sha256"]:
            problems.append(f"{block_id} seeds differ from the frozen seeds")
    if set(manifest["confirmation_blocks"]) != set(confirmation_block_ids(registry)):
        problems.append("the set of confirmation blocks differs from the registry's")
    return problems
