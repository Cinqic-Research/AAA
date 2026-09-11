"""Independent recomputation of metrics and gates from retained raw evidence.

This module never trains a model and never runs the simulator. It reads the
compressed per-step logs an attempt retained, rebuilds every episode, replica
and family summary, recomputes the intervals and re-evaluates the gates. An
independent reviewer can therefore audit an adaptation claim without trusting
the original runtime aggregation code.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..experiment import StepRecord
from ..metrics import RecoveryConfig, normalized_errors
from .evidence import iter_raw_records, verify_checksums
from .families import (
    CHANGED_LAW_PREDICTOR_NAMES,
    FamilyCollector,
    MOTION_PREDICTOR_NAMES,
    ONLINE_PREDICTOR_NAMES,
)
from .gates import GateContext, evaluate_gates
from .spec import BenchmarkSpec, load_spec, spec_hash


def recovery_config(spec: BenchmarkSpec) -> RecoveryConfig:
    return RecoveryConfig(
        pre_event_reference_length=spec.recovery.pre_event_reference_length,
        post_event_horizon=spec.recovery.post_event_horizon,
        shock_window=spec.recovery.shock_window,
        shock_multiplier=spec.recovery.shock_multiplier,
        shock_floor=spec.recovery.shock_floor,
        tolerance_multiplier=spec.recovery.tolerance_multiplier,
        tolerance_floor=spec.recovery.tolerance_floor,
        rolling_window=spec.recovery.rolling_window,
        sustain_windows=spec.recovery.sustain_windows,
    )


def collector_key(family: str, branch: str) -> str:
    if family == "changed_law":
        return "changed_law:changed" if branch == "changed-law" else "changed_law:unchanged"
    return family


def predictor_names_for(spec: BenchmarkSpec, family: str, present: Sequence[str]) -> list[str]:
    if family == "changed_law":
        declared: Sequence[str] = CHANGED_LAW_PREDICTOR_NAMES
    elif "candidate_online" in present:
        declared = ONLINE_PREDICTOR_NAMES
    else:
        declared = MOTION_PREDICTOR_NAMES
    missing = [name for name in declared if name not in present]
    if missing:
        raise ValueError(f"family {family!r} evidence is missing predictors {missing}")
    return list(declared)


def rebuild_collectors(run_dir: Path, spec: BenchmarkSpec) -> dict[str, FamilyCollector]:
    """Rebuild every family collector purely from retained step records."""

    recovery = recovery_config(spec)
    episodes: dict[tuple[str, str], dict[tuple[int, int], list[StepRecord]]] = defaultdict(dict)
    prefixes: dict[tuple[int, int], list[StepRecord]] = {}
    for _, records in iter_raw_records(run_dir):
        if not records:
            continue
        identity = records[0].identity
        key = (identity.family, identity.branch)
        if identity.branch == "prefix":
            prefixes[(identity.replica_id, identity.episode)] = records
            continue
        episodes[key][(identity.replica_id, identity.episode)] = records

    collectors: dict[str, FamilyCollector] = {}
    for (family, branch), grouped in sorted(episodes.items()):
        any_records = next(iter(grouped.values()))
        names = predictor_names_for(spec, family, list(any_records[0].predictions))
        collector = FamilyCollector(family, names, recovery=recovery, branch=branch)
        for (replica, episode), records in sorted(grouped.items()):
            pre_event_errors = None
            if family == "changed_law":
                prefix = prefixes.get((replica, episode))
                if prefix is None:
                    raise ValueError(
                        f"changed-law episode r{replica} e{episode} has no retained prefix evidence"
                    )
                pre_event_errors = {
                    "persistence": normalized_errors(prefix, "persistence"),
                    "constant_motion": normalized_errors(prefix, "constant_motion"),
                    "constant_motion_reflected": normalized_errors(prefix, "constant_motion_reflected"),
                    "frozen": normalized_errors(prefix, "prefix_model"),
                    "online": normalized_errors(prefix, "prefix_model"),
                }
            collector.add(
                records,
                pre_event_errors=pre_event_errors,
                post_change_window=spec.recovery.post_event_horizon,
            )
        collectors[collector_key(family, branch)] = collector
    return collectors


def recompute_run(
    run_dir: str | Path, *, spec_path: str | Path | None = None, verify: bool = True
) -> dict[str, Any]:
    """Recompute summaries and gates from a completed attempt directory."""

    directory = Path(run_dir)
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{directory} does not contain summary.json")
    stored = json.loads(summary_path.read_text(encoding="utf-8"))

    checksum_report = verify_checksums(directory) if verify else {"ok": None, "skipped": True}
    if verify and not checksum_report.get("ok"):
        raise ValueError(f"checksum verification failed for {directory}: {checksum_report}")

    spec = load_spec(spec_path) if spec_path is not None else _spec_from_run(directory)
    resolved_hash = spec_hash(spec)
    stored_hash = stored.get("spec_hash")
    if stored_hash is not None and stored_hash != resolved_hash:
        raise ValueError(
            f"specification hash mismatch: stored {stored_hash}, resolved {resolved_hash}"
        )

    collectors = rebuild_collectors(directory, spec)
    results = {key: collector.finish() for key, collector in collectors.items()}
    context = GateContext(
        spec=spec,
        collectors=collectors,
        results=results,
        learning_curve=stored.get("learning_curve", {}),
        correctness=stored.get("correctness", {}),
        reproducibility=stored.get("reproducibility", {}),
        latency=stored.get("latency", {}),
        confirmation_attempts=int(stored.get("confirmation_attempts", 0)),
    )
    gates = evaluate_gates(context)
    return {
        "run_id": stored.get("run_id"),
        "spec_hash": resolved_hash,
        "checksums": checksum_report,
        "results": results,
        "gates": gates,
        "stored_gates": stored.get("gates"),
    }


def _spec_from_run(directory: Path) -> BenchmarkSpec:
    path = directory / "benchmark_spec.json"
    if not path.exists():
        raise FileNotFoundError(f"{directory} does not contain benchmark_spec.json")
    return load_spec(path)


def compare_results(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any], *, tolerance: float
) -> dict[str, Any]:
    """Deep numeric comparison of two result trees."""

    differences: list[dict[str, Any]] = []

    def walk(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append({"path": f"{path}.{key}", "reason": "key present on one side only"})
                    continue
                walk(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                differences.append({"path": path, "reason": f"length {len(left)} vs {len(right)}"})
                return
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{path}[{index}]")
        elif isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                differences.append({"path": path, "stored": left, "recomputed": right})
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if abs(float(left) - float(right)) > tolerance:
                differences.append({"path": path, "stored": left, "recomputed": right})
        elif left != right:
            differences.append({"path": path, "stored": left, "recomputed": right})

    walk(stored, recomputed, "results")
    return {"equivalent": not differences, "differences": differences[:50], "difference_count": len(differences)}
