"""Source identity manifest for the observation-noise phase."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..benchmark.evidence import git_metadata, sha256_file
from ..benchmark.spec import load_spec, spec_hash
from .spec import load_protocol


def build_freeze_manifest(
    project_root: Path,
    batches: list[str],
    notes: str,
    *,
    stage: str = "design_freeze",
    selected_candidate: str | None = None,
) -> dict[str, Any]:
    if stage not in {"design_freeze", "confirmation_freeze"}:
        raise ValueError("stage must be design_freeze or confirmation_freeze")
    if stage == "confirmation_freeze" and not selected_candidate:
        raise ValueError("confirmation_freeze requires a selected candidate")
    protocol_path = project_root / "aaa/noise/data/observation_noise_v1.json"
    reference_path = project_root / "aaa/benchmark/data/benchmark_v2_1.json"
    ledger_path = project_root / "benchmarks/observation_noise_candidate_ledger.json"
    protocol = load_protocol(protocol_path)
    protocol_hash = protocol.hash()
    git = git_metadata(project_root)
    return {
        "schema_version": "aaa.observation_noise_source_freeze.v1",
        "stage": stage,
        "protocol_version": "aaa.observation_noise.v1",
        "protocol_hash": protocol_hash,
        "protocol_file_sha256": sha256_file(protocol_path),
        "reference": {
            "path": "aaa/benchmark/data/benchmark_v2_1.json",
            "raw_sha256": sha256_file(reference_path),
            "resolved_sha256": spec_hash(load_spec(reference_path)),
        },
        "source": git,
        "dependency_lock": {
            "path": "requirements-lock.txt",
            "sha256": sha256_file(project_root / "requirements-lock.txt"),
        },
        "candidate_ledger_sha256": sha256_file(ledger_path),
        "planned_batches": list(batches),
        "selected_candidate": selected_candidate,
        "notes": notes,
        "manifest_identity": hashlib.sha256(
            json.dumps(
                {
                    "stage": stage,
                    "protocol_hash": protocol_hash,
                    "reference": sha256_file(reference_path),
                    "batches": batches,
                    "selected_candidate": selected_candidate,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def save_freeze_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
