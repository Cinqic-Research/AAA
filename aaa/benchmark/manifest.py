"""The confirmation freeze manifest.

The manifest is written and committed **before** any fresh confirmation result
is viewed. Every confirmation attempt re-derives the same values and refuses to
run on a mismatch, so tuning between Confirmation A and Confirmation B is a
mechanical error rather than a matter of discipline.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evidence import dependency_lock, git_metadata, json_dump, sha256_text
from .spec import BenchmarkSpec, spec_hash

MANIFEST_SCHEMA = "aaa.freeze_manifest.v1"


class FreezeMismatch(RuntimeError):
    """Raised when a confirmation attempt disagrees with the frozen manifest."""


@dataclass(frozen=True)
class FreezeManifest:
    payload: dict[str, Any]

    @property
    def spec_hash(self) -> str:
        return str(self.payload["spec_hash"])

    @property
    def checkpoint_hashes(self) -> list[str]:
        return [str(item) for item in self.payload["checkpoints"]["hashes"]]

    @property
    def planned_batches(self) -> list[str]:
        return [str(item) for item in self.payload["planned_confirmation_batches"]]

    def identity_hash(self) -> str:
        comparable = {
            key: value for key, value in self.payload.items() if key not in ("frozen_at_utc", "notes")
        }
        return sha256_text(json.dumps(comparable, sort_keys=True, separators=(",", ":")))


def build_manifest(
    spec: BenchmarkSpec,
    *,
    project_root: Path,
    checkpoint_hashes: Sequence[str],
    training_seeds: Mapping[int, Sequence[int]],
    planned_batches: Sequence[str],
    notes: str = "",
) -> FreezeManifest:
    git = git_metadata(project_root)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": spec.spec_version,
        "protocol_status": spec.status,
        "spec_hash": spec_hash(spec),
        "source": {"commit": git["commit"], "tree_hash": git["tree_hash"], "dirty": git["dirty"]},
        "dependency_lock": dependency_lock(project_root),
        "candidate": {
            "model": spec.candidate.model,
            "feature_set": spec.candidate.feature_set,
            "features": list(spec.candidate.features),
            "forgetting": spec.candidate.forgetting,
            "forgetting_mode": spec.candidate.forgetting_mode,
            "ridge": spec.candidate.ridge,
            "trace_bound": spec.candidate.trace_bound,
            "reflect": spec.candidate.reflect,
            "unfold_target": spec.candidate.unfold_target,
            "dead_zone": spec.candidate.dead_zone,
            "detector_multiplier": spec.candidate.detector_multiplier,
            "detector_floor": spec.candidate.detector_floor,
            "detector_decay": spec.candidate.detector_decay,
            "displacement_scale": spec.displacement_scale,
        },
        "training": {
            "episodes_per_replica": spec.training.episodes_per_replica,
            "steps_per_episode": spec.training.steps_per_episode,
            "speed_min": spec.training.speed_min,
            "speed_max": spec.training.speed_max,
            "distribution_relationship": spec.training.distribution_relationship,
            "seeds": {str(replica): list(seeds) for replica, seeds in sorted(training_seeds.items())},
        },
        "checkpoints": {
            "relationship": spec.confirmation.ab_relationship,
            "hashes": list(checkpoint_hashes),
        },
        "baselines": dict(spec.baselines),
        "gates": [
            {
                "name": gate.name,
                "evaluator": gate.evaluator,
                "required": gate.required,
                "threshold": gate.threshold,
            }
            for gate in spec.gates
        ],
        "sample_counts": {
            "replicas": spec.confirmation.replicas,
            "episodes_per_family": spec.confirmation.episodes_per_family,
            "minimum_bounce_events": spec.confirmation.minimum_bounce_events,
            "minimum_eligible_change_events": spec.confirmation.minimum_eligible_change_events,
            "minimum_episodes_per_stratum": spec.stratification.minimum_episodes_per_stratum,
            "minimum_replicas_per_stratum": spec.stratification.minimum_replicas_per_stratum,
        },
        "event_distributions": {
            "speed_change": {
                "change_step": spec.motion_families["speed_change"].change_step,
                "factor_low": spec.motion_families["speed_change"].change_factor_low,
                "factor_high": spec.motion_families["speed_change"].change_factor_high,
            },
            "changed_law": {
                "prefix_steps": spec.changed_law.prefix_steps,
                "branch_steps": spec.changed_law.branch_steps,
                "pre_omega": spec.changed_law.pre_omega,
                "pre_damping": spec.changed_law.pre_damping,
                "post_omega_range": [spec.changed_law.post_omega_low, spec.changed_law.post_omega_high],
                "post_damping_range": [spec.changed_law.post_damping_low, spec.changed_law.post_damping_high],
                "timing": "fixed at the end of the common prefix (not randomized)",
            },
        },
        "recovery": {
            "pre_event_reference_length": spec.recovery.pre_event_reference_length,
            "post_event_horizon": spec.recovery.post_event_horizon,
            "shock_window": spec.recovery.shock_window,
            "shock_multiplier": spec.recovery.shock_multiplier,
            "shock_floor": spec.recovery.shock_floor,
            "tolerance_multiplier": spec.recovery.tolerance_multiplier,
            "tolerance_floor": spec.recovery.tolerance_floor,
            "rolling_window": spec.recovery.rolling_window,
            "sustain_windows": spec.recovery.sustain_windows,
        },
        "statistics": {
            "method": spec.statistics.method,
            "draws": spec.statistics.draws,
            "interval": spec.statistics.interval,
            "multiplicity": spec.statistics.multiplicity,
            "family_wise_alpha": spec.statistics.family_wise_alpha,
            "event_weighting": spec.statistics.event_weighting,
            "exclude_no_event_episodes": spec.statistics.exclude_no_event_episodes,
            "estimands": dict(spec.statistics.estimands),
        },
        "planned_confirmation_batches": list(planned_batches),
        "ab_relationship": spec.confirmation.ab_relationship,
        "notes": notes,
    }
    return FreezeManifest(payload)


def save_manifest(manifest: FreezeManifest, path: str | Path) -> None:
    json_dump(Path(path), manifest.payload)


def load_manifest(path: str | Path) -> FreezeManifest:
    source = Path(path)
    if not source.exists():
        raise FreezeMismatch(
            f"freeze manifest {source} does not exist; a confirmation attempt requires a committed frozen manifest"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != MANIFEST_SCHEMA:
        raise FreezeMismatch(f"unsupported freeze manifest schema {payload.get('schema_version')!r}")
    return FreezeManifest(payload)


def check_manifest(
    manifest: FreezeManifest,
    spec: BenchmarkSpec,
    *,
    project_root: Path,
    checkpoint_hashes: Sequence[str],
    batch_id: str,
    training_seeds: Mapping[int, Sequence[int]],
) -> None:
    """Reject any confirmation attempt that drifted from the frozen manifest."""

    problems: list[str] = []
    resolved = spec_hash(spec)
    if manifest.spec_hash != resolved:
        problems.append(
            f"specification hash: frozen {manifest.spec_hash[:12]}..., resolved {resolved[:12]}..."
        )
    if batch_id not in manifest.planned_batches:
        problems.append(
            f"confirmation batch {batch_id!r} was not in the frozen plan {manifest.planned_batches}"
        )
    frozen_hashes = manifest.checkpoint_hashes
    if list(checkpoint_hashes) != frozen_hashes:
        problems.append(
            "checkpoint hashes differ from the frozen selected models: "
            f"frozen {[h[:8] for h in frozen_hashes]}, resolved {[h[:8] for h in checkpoint_hashes]}"
        )
    frozen_seeds = manifest.payload["training"]["seeds"]
    resolved_seeds = {str(replica): list(seeds) for replica, seeds in sorted(training_seeds.items())}
    if frozen_seeds != resolved_seeds:
        problems.append("training stream identities differ from the frozen manifest")
    lock = dependency_lock(project_root)
    frozen_lock = manifest.payload["dependency_lock"]
    if lock["hash"] != frozen_lock["hash"]:
        problems.append(f"dependency lock hash: frozen {frozen_lock['hash']}, resolved {lock['hash']}")
    if problems:
        raise FreezeMismatch(
            "confirmation attempt does not match the frozen manifest:\n  - " + "\n  - ".join(problems)
        )
