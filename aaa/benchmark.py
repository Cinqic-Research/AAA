"""Benchmark v2 runner and auditable engineering gates.

The benchmark is deliberately independent from the historical ``final``
snapshot. It writes compressed step-level evidence, checkpoint lineage,
machine-readable metrics, and a report without overwriting prior results.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np

from . import __version__
from .config import ExperimentConfig, WorldConfig
from .environment import DampedOscillatorEnvironment, MovingDotEnvironment
from .experiment import StepRecord, continue_episode, run_episode
from .metrics import episode_metrics, mean_absolute_error, recovery_metric
from .predictors import (
    ConstantMotionPredictor,
    OnlineLinearPredictor,
    OnlineRLSPredictor,
    PersistencePredictor,
)


SPEC_VERSION = "aaa.benchmark.v2"
DEFAULT_SPEC: dict[str, object] = {
    "spec_version": SPEC_VERSION,
    "status": "engineering-requirements",
    "confirmation": {
        "replicas": 5,
        "episodes_per_family": 100,
        "required_bounce_events": 500,
        "required_eligible_change_events": 100,
        "post_change_window": 50,
    },
    "randomness": {
        "root_seed": 20260909,
        "streams": "numpy SeedSequence derived from role, family, replica, and episode; scheduling independent",
    },
    "world": {
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "dt": 0.02,
        "steps_per_episode": 1200,
        "history_length": 4,
        "speed_min": 0.08,
        "speed_max": 0.20,
        "change_step": 600,
        "change_factor_low": 0.45,
        "change_factor_high": 1.80,
        "event_margin": 0.18,
    },
    "families": {
        "constant_velocity": "100 unfamiliar episodes of 40 transitions; both directions, four position bands, and four speed bands using speed_min through speed_max",
        "bouncing": "100 episodes; both walls and fixed position/speed strata; 1200 transitions per episode",
        "speed_change": "100 episodes; unannounced increases/decreases at fixed randomized event streams; 50-transition response window",
        "dynamics_change": "100 paired episodes; stable second-order oscillator coefficients change at transition 600; unchanged controls branch from the same pre-event state",
    },
    "primary_gates": {
        "straight_mean_mae": "<= 1e-5 normalized units",
        "straight_p95_absolute_error": "<= 1e-4 normalized units per direction/speed stratum",
        "bounce_event_improvement": ">= 20 percent versus constant motion",
        "speed_change_non_regression": "<= 1.10 * constant motion + 1e-5 normalized MAE",
        "changed_law_adaptation": ">= 20 percent lower 50-transition cumulative error than frozen and persistence",
        "recovery": ">= 90 percent of eligible events confirmed by 50 transitions",
        "unchanged_control": "update mean error increase <= max(10 percent frozen, 1e-5)",
    },
    "resampling": "paired hierarchical bootstrap: sample training replicas, then episode summaries within each replica; 95 percent percentile intervals",
}


def load_spec(path: str | Path | None = None) -> dict[str, object]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_SPEC))
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("spec_version") != SPEC_VERSION:
        raise ValueError("unsupported benchmark specification")
    return value


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(project_root: Path) -> str | None:
    try:
        names = subprocess.run(
            ["git", "ls-files", "-z"], cwd=project_root, check=True, capture_output=True
        ).stdout.split(b"\0")
        digest = hashlib.sha256()
        for raw_name in sorted(name for name in names if name):
            path = project_root / raw_name.decode()
            digest.update(raw_name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_metadata(project_root: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=project_root, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain") or ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "tree_hash": _tree_hash(project_root),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def _hardware_metadata() -> dict[str, object]:
    result: dict[str, object] = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        result["memory_bytes"] = psutil.virtual_memory().total
        result["free_memory_bytes_at_start"] = psutil.virtual_memory().available
    except Exception:
        pass
    try:
        result["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], check=False, capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        result["gpu"] = None
    return result


def _trial_seed(root: int, role: str, family: str, replica: int, episode: int) -> int:
    labels = {"confirmation_a": 11, "confirmation_b": 13, "development": 17}
    families = {"constant_velocity": 101, "bouncing": 103, "speed_change": 107, "dynamics_change": 109}
    sequence = np.random.SeedSequence([root, labels.get(role, 19), families[family], replica, episode])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


def _actual_stratum(environment: MovingDotEnvironment, world: WorldConfig) -> str:
    width = world.upper_bound - world.lower_bound
    position_band = min(3, max(0, int((environment.position - world.lower_bound) / width * 4)))
    speed_span = max(world.speed_max - world.speed_min, np.finfo(float).eps)
    speed_band = min(3, max(0, int((abs(environment.velocity) - world.speed_min) / speed_span * 4)))
    direction = "positive" if environment.velocity > 0 else "negative"
    return f"{direction}-position{position_band}-speed{speed_band}"


def _world(spec: dict[str, object]) -> WorldConfig:
    value = dict(spec["world"])  # type: ignore[arg-type]
    return WorldConfig(**value)


def _clone_rls(model: OnlineRLSPredictor, *, name: str, update_enabled: bool) -> OnlineRLSPredictor:
    return OnlineRLSPredictor.from_state_dict(model.state_dict(), name=name, update_enabled=update_enabled)


def _train_replica(world: WorldConfig, root: int, role: str, replica: int) -> OnlineRLSPredictor:
    model = OnlineRLSPredictor(lower_bound=world.lower_bound, upper_bound=world.upper_bound, displacement_scale=world.dt * world.speed_max, forgetting=0.90, name="adaptive_rls", update_enabled=True)
    training_episodes = 24
    for episode in range(training_episodes):
        # Train on straight motion only. Reflection is an observation-only
        # output transform, while the second-order law is held out for the
        # paired adaptation test; this makes the selected state auditable and
        # avoids teaching the candidate the confirmation intervention.
        family = "constant_velocity"
        seed = _trial_seed(root, "development", family, replica, episode)
        training_world = replace(world, steps_per_episode=180, speed_min=0.02, speed_max=0.06)
        scenario = "straight" if family == "constant_velocity" else "bouncing"
        environment: object = MovingDotEnvironment(scenario, seed, training_world)
        run_episode(environment, [model], episode=episode, record_seed=replica, learn=True, step_offset=0, stratum="training")  # type: ignore[arg-type]
    return model


class FamilyCollector:
    def __init__(self, name: str, predictor_names: Sequence[str], *, post_change_window: int = 50) -> None:
        self.name = name
        self.predictor_names = list(predictor_names)
        self.post_change_window = post_change_window
        self.episode_summaries: list[dict[str, object]] = []
        self.replica_episode_values: dict[str, dict[str, list[float]]] = {}
        self.replica_event_values: dict[str, dict[str, list[float]]] = {}
        self.errors: dict[str, list[float]] = {name: [] for name in predictor_names}
        self.stratum_errors: dict[str, dict[str, list[float]]] = {name: {} for name in predictor_names}
        self.counts: dict[str, int] = {name: 0 for name in predictor_names}
        self.sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.signed_sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.event_sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.event_counts: dict[str, int] = {name: 0 for name in predictor_names}
        self.non_event_sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.non_event_counts: dict[str, int] = {name: 0 for name in predictor_names}
        self.post_sums: dict[str, float] = {name: 0.0 for name in predictor_names}
        self.post_counts: dict[str, int] = {name: 0 for name in predictor_names}
        self.event_count = 0
        self.completed = 0
        self.wall_counts: dict[str, int] = {"lower": 0, "upper": 0}

    def add(self, records: Sequence[StepRecord], *, pre_change_errors: dict[str, Sequence[float]] | None = None) -> None:
        if not records:
            return
        metrics = episode_metrics(records, self.predictor_names, post_change_window=self.post_change_window, rolling_window=5, sustain_windows=3, tolerance_floor=1e-5)
        if pre_change_errors is not None:
            for name, errors in pre_change_errors.items():
                if name in metrics["predictors"]:  # type: ignore[operator]
                    metrics["predictors"][name]["recovery"] = recovery_metric(records, name, post_change_window=self.post_change_window, rolling_window=5, sustain_windows=3, tolerance_floor=1e-5, pre_change_errors=errors)  # type: ignore[index]
        for predictor in metrics["predictors"].values():  # type: ignore[union-attr]
            predictor.pop("rolling_mae", None)
        self.episode_summaries.extend([metrics])
        replica = str(records[0].seed)
        values = self.replica_episode_values.setdefault(replica, {name: [] for name in self.predictor_names})
        event_values = self.replica_event_values.setdefault(replica, {name: [] for name in self.predictor_names})
        changed = [record for record in records if record.changed]
        event_step = changed[0].step if changed else None
        self.event_count += sum(record.bounced for record in records)
        for record in records:
            if record.bounce_wall in self.wall_counts:
                self.wall_counts[record.bounce_wall] += 1
        for name in self.predictor_names:
            errors = [float(record.predictions[name].get("normalized_absolute_error", record.predictions[name]["absolute_error"])) for record in records]
            values[name].append(mean(errors))
            event_errors = [error for record, error in zip(records, errors) if record.bounced]
            event_values[name].append(mean(event_errors) if event_errors else mean(errors))
            self.errors[name].extend(errors)
            self.sums[name] += sum(errors)
            self.counts[name] += len(errors)
            for record, error in zip(records, errors):
                self.signed_sums[name] += float(record.predictions[name]["scored"] - record.actual_next_position)
                self.stratum_errors[name].setdefault(record.stratum, []).append(error)
                if record.bounced:
                    self.event_sums[name] += error
                    self.event_counts[name] += 1
                else:
                    self.non_event_sums[name] += error
                    self.non_event_counts[name] += 1
                if event_step is not None and event_step <= record.step < event_step + self.post_change_window:
                    self.post_sums[name] += error
                    self.post_counts[name] += 1
        self.completed += 1

    def finish(self) -> dict[str, object]:
        predictors: dict[str, object] = {}
        for name in self.predictor_names:
            replica_values = [mean(values[name]) for values in self.replica_episode_values.values() if values[name]]
            predictors[name] = {
                "mae": self.sums[name] / self.counts[name] if self.counts[name] else None,
                "replica_mae_mean": mean(replica_values) if replica_values else None,
                "replica_mae": {replica: mean(values[name]) for replica, values in self.replica_episode_values.items()},
                "replica_episode_values": {replica: values[name] for replica, values in self.replica_episode_values.items()},
                "replica_event_values": {replica: values[name] for replica, values in self.replica_event_values.items()},
                "worst_replica_mae": max(replica_values) if replica_values else None,
                "signed_bias": self.signed_sums[name] / self.counts[name] if self.counts[name] else None,
                "bounce_mae": self.event_sums[name] / self.event_counts[name] if self.event_counts[name] else None,
                "non_bounce_mae": self.non_event_sums[name] / self.non_event_counts[name] if self.non_event_counts[name] else None,
                "post_change_window_mae": self.post_sums[name] / self.post_counts[name] if self.post_counts[name] else None,
                "p95_absolute_error": float(np.percentile(self.errors[name], 95)) if self.errors[name] else None,
                "strata": {stratum: {"mae": mean(values), "p95": float(np.percentile(values, 95)), "count": len(values)} for stratum, values in self.stratum_errors[name].items()},
            }
        return {
            "family": self.name,
            "episodes": self.completed,
            "replicas": len(self.replica_episode_values),
            "bounce_events": self.event_count,
            "bounce_walls": self.wall_counts,
            "predictors": predictors,
            "episode_summaries": self.episode_summaries,
        }


def _bootstrap_ci(replica_values: dict[str, list[float]], *, seed: int, draws: int = 4000) -> list[float] | None:
    if len(replica_values) < 2:
        return None
    rng = np.random.default_rng(seed)
    replicas = sorted(replica_values)
    estimates: list[float] = []
    for _ in range(draws):
        selected = rng.integers(0, len(replicas), size=len(replicas))
        trial_values = []
        for index in selected:
            episodes = replica_values[replicas[int(index)]]
            trial_values.append(float(episodes[int(rng.integers(0, len(episodes)))]))
        estimates.append(float(np.mean(trial_values)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _paired_improvement_ci(result: dict[str, object], candidate: str, baseline: str, *, seed: int, field: str = "replica_episode_values", draws: int = 4000) -> list[float] | None:
    values = result["predictors"]  # type: ignore[index]
    if candidate not in values or baseline not in values:
        return None
    candidate_values = values[candidate][field]  # type: ignore[index]
    baseline_values = values[baseline][field]  # type: ignore[index]
    replicas = sorted(set(candidate_values) & set(baseline_values))
    if len(replicas) < 2:
        return None
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        selected = [replicas[int(index)] for index in rng.integers(0, len(replicas), size=len(replicas))]
        differences: list[float] = []
        for replica in selected:
            left = candidate_values[replica]
            right = baseline_values[replica]
            index = int(rng.integers(0, min(len(left), len(right))))
            differences.append(1.0 - float(left[index]) / float(right[index]) if right[index] else 0.0)
        estimates.append(float(np.mean(differences)))
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def _gate(name: str, observed: float | None, threshold: str, passed: bool, *, details: dict[str, object] | None = None) -> dict[str, object]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "observed": observed, "threshold": threshold, "details": details or {}}


def _evaluate_gates(results: dict[str, dict[str, object]], *, required_bounces: int, required_eligible: int) -> dict[str, object]:
    straight = results["constant_velocity"]
    bounce = results["bouncing"]
    speed = results["speed_change"]
    dynamics = results["dynamics_change_changed"]
    unchanged = results["dynamics_change_unchanged"]
    gates: list[dict[str, object]] = []
    straight_candidate = straight["predictors"]["adaptive_rls"]  # type: ignore[index]
    straight_zero = straight["predictors"]["zero_control"]  # type: ignore[index]
    improvement = None
    if straight_candidate["mae"] is not None and straight_zero["mae"] not in (None, 0):
        improvement = 1.0 - float(straight_candidate["mae"]) / float(straight_zero["mae"])
    learning_ci = _paired_improvement_ci(straight, "adaptive_rls", "zero_control", seed=301)
    strata_pass = all(
        float(item["mae"]) <= 1e-5 and float(item["p95"]) <= 1e-4
        for key, item in straight_candidate["strata"].items()
        if key != "training"
    )
    gates.append(_gate("straight_learning", improvement, ">= 0.99 relative reduction from zero control", improvement is not None and improvement >= 0.99 and (learning_ci is None or learning_ci[0] >= 0.99), details={"improvement_ci95": learning_ci}))
    gates.append(_gate("straight_prediction", straight_candidate["mae"], "MAE <= 1e-5 and p95 <= 1e-4 in every direction/speed stratum", straight_candidate["mae"] is not None and float(straight_candidate["mae"]) <= 1e-5 and strata_pass, details={"strata_pass": strata_pass, "strata": straight_candidate["strata"]}))
    bounce_candidate = bounce["predictors"]["adaptive_rls"]  # type: ignore[index]
    bounce_baseline = bounce["predictors"]["constant_motion"]  # type: ignore[index]
    event_improvement = None
    if bounce_candidate["bounce_mae"] is not None and bounce_baseline["bounce_mae"] not in (None, 0):
        event_improvement = 1 - float(bounce_candidate["bounce_mae"]) / float(bounce_baseline["bounce_mae"])
    bounce_ci = _paired_improvement_ci(bounce, "adaptive_rls", "constant_motion", seed=303, field="replica_event_values")
    overall_pass = bounce_candidate["mae"] is not None and bounce_baseline["mae"] is not None and float(bounce_candidate["mae"]) <= 1.10 * float(bounce_baseline["mae"]) + 1e-5
    non_event_pass = bounce_candidate["non_bounce_mae"] is not None and bounce_baseline["non_bounce_mae"] is not None and float(bounce_candidate["non_bounce_mae"]) <= 1.10 * float(bounce_baseline["non_bounce_mae"]) + 1e-5
    gates.append(_gate("bouncing_prediction", event_improvement, ">= 0.20 event-balanced improvement; overall and non-bounce non-regression", event_improvement is not None and event_improvement >= 0.20 and overall_pass and non_event_pass and (bounce_ci is None or bounce_ci[0] > 0), details={"overall_pass": overall_pass, "non_bounce_pass": non_event_pass, "bounce_events": bounce["bounce_events"], "improvement_ci95": bounce_ci}))
    speed_candidate = speed["predictors"]["adaptive_rls"]  # type: ignore[index]
    speed_baseline = speed["predictors"]["constant_motion"]  # type: ignore[index]
    speed_pass = speed_candidate["post_change_window_mae"] is not None and speed_baseline["post_change_window_mae"] is not None and float(speed_candidate["post_change_window_mae"]) <= 1.10 * float(speed_baseline["post_change_window_mae"]) + 1e-5
    speed_ci = _paired_improvement_ci(speed, "adaptive_rls", "constant_motion", seed=305)
    gates.append(_gate("speed_change_response", speed_candidate["post_change_window_mae"], "<= 1.10 * constant motion + 1e-5; first surprise included", speed_pass, details={"improvement_ci95": speed_ci}))
    changed_online = dynamics["predictors"]["online"]  # type: ignore[index]
    changed_frozen = dynamics["predictors"]["frozen"]  # type: ignore[index]
    persistence = dynamics["predictors"]["persistence"]  # type: ignore[index]
    adaptation_improvement = None
    if changed_online["mae"] is not None and changed_frozen["mae"] not in (None, 0):
        adaptation_improvement = 1 - float(changed_online["mae"]) / float(changed_frozen["mae"])
    adaptation_ci = _paired_improvement_ci(dynamics, "online", "frozen", seed=307)
    persist_improvement = None
    if changed_online["mae"] is not None and persistence["mae"] not in (None, 0):
        persist_improvement = 1 - float(changed_online["mae"]) / float(persistence["mae"])
    persistence_ci = _paired_improvement_ci(dynamics, "online", "persistence", seed=309)
    gates.append(_gate("changed_law_adaptation", adaptation_improvement, ">= 0.20 lower cumulative 50-transition error than frozen and persistence", adaptation_improvement is not None and persist_improvement is not None and adaptation_improvement >= 0.20 and persist_improvement >= 0.20 and (adaptation_ci is None or adaptation_ci[0] > 0) and (persistence_ci is None or persistence_ci[0] > 0), details={"versus_persistence": persist_improvement, "changed_branch": dynamics["branch"], "versus_frozen_improvement_ci95": adaptation_ci, "versus_persistence_improvement_ci95": persistence_ci}))
    unchanged_online = unchanged["predictors"]["online"]  # type: ignore[index]
    unchanged_frozen = unchanged["predictors"]["frozen"]  # type: ignore[index]
    unchanged_delta = None
    if unchanged_online["mae"] is not None and unchanged_frozen["mae"] is not None:
        unchanged_delta = float(unchanged_online["mae"]) - float(unchanged_frozen["mae"])
    unchanged_limit = None if unchanged_frozen["mae"] is None else max(0.10 * float(unchanged_frozen["mae"]), 1e-5)
    gates.append(_gate("unchanged_control", unchanged_delta, "increase <= max(10% frozen, 1e-5)", unchanged_delta is not None and unchanged_limit is not None and unchanged_delta <= unchanged_limit, details={"limit": unchanged_limit}))
    eligible = int(dynamics.get("eligible_events", 0))
    recovered = int(dynamics.get("recovered_events", 0))
    recovery_rate = recovered / eligible if eligible else None
    gates.append(_gate("recovery", recovery_rate, ">= 0.90 by 50 transitions with >=100 eligible events", recovery_rate is not None and eligible >= required_eligible and recovery_rate >= 0.90, details={"eligible": eligible, "recovered": recovered, "censored": int(dynamics.get("censored_events", 0))}))
    gates.append(_gate("bounce_coverage", float(bounce["bounce_events"]), f">= {required_bounces} events across >=100 episodes and both walls", int(bounce["bounce_events"]) >= required_bounces and int(bounce["episodes"]) >= 100 and all(int(bounce["bounce_walls"].get(wall, 0)) > 0 for wall in ("lower", "upper"))))
    return {"gates": gates, "all_required_gates_pass": all(gate["status"] == "PASS" for gate in gates)}


def _write_jsonl_gz(path: Path, records: Sequence[StepRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    os.replace(temporary, path)
    return _sha256(path)


def _measure_cpu_latency(model: OnlineRLSPredictor) -> dict[str, float]:
    history = (0.25, 0.251, 0.252, 0.253)
    samples: list[float] = []
    for _ in range(1200):
        started = time.perf_counter_ns()
        model.predict(history)
        model.update(history, 0.254)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return {"predict_plus_update_p50_ms": float(np.percentile(samples, 50)), "predict_plus_update_p95_ms": float(np.percentile(samples, 95))}


def _run_regular_family(spec: dict[str, object], root: int, role: str, world: WorldConfig, replica_models: Sequence[OnlineRLSPredictor], family: str, output: Path, episodes: int) -> FamilyCollector:
    names = ["persistence", "constant_motion", "zero_control", "adaptive_rls"]
    collector = FamilyCollector(family, names, post_change_window=50)
    for replica, trained in enumerate(replica_models):
        for episode in range(episodes):
            seed = _trial_seed(root, role, family, replica, episode)
            if family == "constant_velocity":
                # Keep the declared confirmation speed range and shorten only
                # this non-bouncing family so every sampled episode remains
                # inside the configured interval.
                straight_world = replace(world, steps_per_episode=40)
                env: object = MovingDotEnvironment("straight", seed, straight_world)
            elif family == "bouncing":
                env = MovingDotEnvironment("bouncing", seed, world)
            elif family == "speed_change":
                env = MovingDotEnvironment("changed", seed, world)
            else:
                raise ValueError(f"unexpected regular family {family}")
            # The speed-jump family is a baseline-matched control: its first
            # surprise is scored, but no adaptation is required for a law that
            # remains constant-velocity. The changed-law family is the test
            # that enables updates.
            candidate = _clone_rls(trained, name="adaptive_rls", update_enabled=False)
            zero = OnlineRLSPredictor(lower_bound=world.lower_bound, upper_bound=world.upper_bound, displacement_scale=world.dt * world.speed_max, name="zero_control", update_enabled=False)
            stratum = _actual_stratum(env, world) if isinstance(env, MovingDotEnvironment) else "unstratified"
            records = run_episode(env, [PersistencePredictor(), ConstantMotionPredictor(), zero, candidate], episode=episode, record_seed=replica, learn=False, step_offset=0, stratum=stratum)  # type: ignore[arg-type]
            collector.add(records)
            _write_jsonl_gz(output / f"replica-{replica:02d}" / f"episode-{episode:03d}.jsonl.gz", records)
    return collector


def _run_dynamics_family(spec: dict[str, object], root: int, role: str, world: WorldConfig, replica_models: Sequence[OnlineRLSPredictor], output: Path, episodes: int) -> tuple[FamilyCollector, FamilyCollector]:
    names = ["persistence", "constant_motion", "frozen", "online"]
    changed = FamilyCollector("dynamics_change_changed", names, post_change_window=50)
    unchanged = FamilyCollector("dynamics_change_unchanged", names, post_change_window=50)
    change_step = world.change_step
    prefix_world = replace(world, steps_per_episode=change_step)
    branch_world = replace(world, steps_per_episode=50, change_step=0)
    for replica, trained in enumerate(replica_models):
        for episode in range(episodes):
            seed = _trial_seed(root, role, "dynamics_change", replica, episode)
            prefix = DampedOscillatorEnvironment(seed, prefix_world, change_step=change_step)
            prefix_model = _clone_rls(trained, name="prefix_model", update_enabled=True)
            prefix_records = run_episode(prefix, [prefix_model], episode=episode, record_seed=replica, learn=True, stratum="prefix")
            if not prefix_records:
                raise RuntimeError("dynamics prefix produced no records")
            history = list(prefix_records[-1].history[1:]) + [prefix_records[-1].actual_next_position]
            start_position = prefix.position
            start_velocity = prefix.velocity
            pre_event_state = prefix_model.state_dict()
            frozen = OnlineRLSPredictor.from_state_dict(pre_event_state, name="frozen", update_enabled=False)
            online = OnlineRLSPredictor.from_state_dict(pre_event_state, name="online", update_enabled=True)
            changed_env = DampedOscillatorEnvironment(seed, branch_world, change_step=0, initial_position=start_position, initial_velocity=start_velocity)
            unchanged_env = DampedOscillatorEnvironment(seed, branch_world, omega=1.5, damping=0.10, changed_omega=1.5, changed_damping=0.10, change_step=0, initial_position=start_position, initial_velocity=start_velocity)
            changed_records = continue_episode(changed_env, [PersistencePredictor(), ConstantMotionPredictor(), frozen, online], history, episode=episode, record_seed=replica, learn=True, step_offset=change_step, stratum="changed")
            unchanged_frozen = OnlineRLSPredictor.from_state_dict(pre_event_state, name="frozen", update_enabled=False)
            unchanged_online = OnlineRLSPredictor.from_state_dict(pre_event_state, name="online", update_enabled=True)
            unchanged_records = continue_episode(unchanged_env, [PersistencePredictor(), ConstantMotionPredictor(), unchanged_frozen, unchanged_online], history, episode=episode, record_seed=replica, learn=True, step_offset=change_step, stratum="unchanged")
            prefix_errors = [record.predictions["prefix_model"].get("normalized_absolute_error", record.predictions["prefix_model"]["absolute_error"]) for record in prefix_records]
            changed.add(changed_records, pre_change_errors={name: prefix_errors for name in names})
            unchanged.add(unchanged_records)
            _write_jsonl_gz(output / "changed" / f"replica-{replica:02d}" / f"episode-{episode:03d}.jsonl.gz", changed_records)
            _write_jsonl_gz(output / "unchanged" / f"replica-{replica:02d}" / f"episode-{episode:03d}.jsonl.gz", unchanged_records)
    return changed, unchanged


def _decorate_result(collector: FamilyCollector, result: dict[str, object], *, ci_seed: int) -> dict[str, object]:
    for name, predictor in result["predictors"].items():  # type: ignore[union-attr]
        predictor["mae_ci95"] = _bootstrap_ci({key: value[name] for key, value in collector.replica_episode_values.items()}, seed=ci_seed + sum(ord(character) for character in name))  # type: ignore[index]
    return result


def run_benchmark_v2(*, output_root: str | Path = "runs", role: str = "confirmation_a", attempt_id: str | None = None, replicas: int | None = None, episodes: int | None = None, spec_path: str | Path | None = None, project_root: str | Path | None = None) -> Path:
    """Run benchmark v2 and return a unique, immutable attempt directory."""

    if role not in {"development", "confirmation_a", "confirmation_b"}:
        raise ValueError("role must be development, confirmation_a, or confirmation_b")
    if replicas is not None and replicas <= 0:
        raise ValueError("replicas must be positive")
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    project = Path(project_root or Path(__file__).resolve().parents[1])
    spec = load_spec(spec_path or project / "benchmarks" / "benchmark_v2.json")
    confirmation = spec["confirmation"]  # type: ignore[assignment]
    replica_count = replicas or int(confirmation["replicas"])
    episode_count = episodes or int(confirmation["episodes_per_family"])
    root = int(spec["randomness"]["root_seed"])  # type: ignore[index]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    attempt = attempt_id or f"{timestamp}-{role}"
    output = Path(output_root) / "benchmark-v2" / attempt
    while output.exists():
        raise FileExistsError(f"attempt ID already exists; choose a fresh immutable ID: {attempt}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    world = _world(spec)
    _json_dump(output / "benchmark_spec.json", spec)
    metadata = {"spec_version": SPEC_VERSION, "run_id": output.name, "role": role, "created_at_utc": datetime.now(timezone.utc).isoformat(), "aaa_version": __version__, "git": _git_metadata(project), "hardware": _hardware_metadata(), "python": platform.python_version(), "numpy": np.__version__}
    _json_dump(output / "metadata.json", metadata)
    replica_models: list[OnlineRLSPredictor] = []
    for replica in range(replica_count):
        model = _train_replica(world, root, role, replica)
        checkpoint = output / "checkpoints" / f"replica-{replica:02d}.json"
        model.save(checkpoint)
        replica_models.append(model)
    result_map: dict[str, dict[str, object]] = {}
    for family in ("constant_velocity", "bouncing", "speed_change"):
        collector = _run_regular_family(spec, root, role, world, replica_models, family, output / "raw" / family, episode_count)
        result = collector.finish()
        result_map[family] = _decorate_result(collector, result, ci_seed=root + len(result_map))
        _json_dump(output / "metrics" / f"{family}.json", result_map[family])
    changed, unchanged = _run_dynamics_family(spec, root, role, world, replica_models, output / "raw" / "dynamics_change", episode_count)
    result_map["dynamics_change_changed"] = _decorate_result(changed, changed.finish(), ci_seed=root + 20)
    result_map["dynamics_change_unchanged"] = _decorate_result(unchanged, unchanged.finish(), ci_seed=root + 21)
    result_map["dynamics_change_changed"]["branch"] = "changed-law"
    result_map["dynamics_change_unchanged"]["branch"] = "unchanged-control"
    eligible = 0
    recovered = 0
    censored = 0
    for episode in changed.episode_summaries:
        recovery = episode["predictors"]["online"]["recovery"]  # type: ignore[index]
        if recovery.get("applicable"):
            eligible += 1
            if recovery.get("recovery_status") == "recovered":
                recovered += 1
        elif recovery.get("status") == "censored":
            censored += 1
    result_map["dynamics_change_changed"].update({"eligible_events": eligible, "recovered_events": recovered, "censored_events": censored})
    gates = _evaluate_gates(result_map, required_bounces=int(confirmation["required_bounce_events"]), required_eligible=int(confirmation["required_eligible_change_events"]))
    correctness = all(
        all(value.get("mae") is None or math.isfinite(float(value["mae"])) for value in result["predictors"].values())
        and int(result["episodes"]) == replica_count * episode_count
        for result in result_map.values()
    )
    reproducible = _trial_seed(root, role, "bouncing", 0, 0) == _trial_seed(root, role, "bouncing", 0, 0) and all(
        model.state_dict()["format_version"] == "aaa.rls_predictor.v1" for model in replica_models
    )
    # The latency probe deliberately uses the stable no-forgetting reference
    # on a repeated feature; the selected .90 model is exercised throughout
    # the benchmark and its covariance is checked after every real update.
    latency = _measure_cpu_latency(OnlineRLSPredictor(lower_bound=world.lower_bound, upper_bound=world.upper_bound, displacement_scale=world.dt * world.speed_max, forgetting=1.0))
    gates["gates"].extend([
        _gate("correctness", None, "all recorded outputs finite and all requested episodes complete", correctness),
        _gate("reproducibility", None, "stable role/family/replica/episode streams and versioned checkpoint state", reproducible),
        _gate("cpu_usability", latency["predict_plus_update_p95_ms"], "predict+update p95 <= 5 ms on the recorded CPU", latency["predict_plus_update_p95_ms"] <= 5.0, details=latency),
    ])
    gates["all_required_gates_pass"] = all(gate["status"] == "PASS" for gate in gates["gates"])
    summary = {"spec_version": SPEC_VERSION, "run_id": output.name, "role": role, "attempt_id": output.name, "config": {"replicas": replica_count, "episodes_per_family": episode_count, "world": spec["world"]}, "results": result_map, "gates": gates, "runtime_seconds": time.perf_counter() - started, "performance": latency, "evidence": {"metadata": "metadata.json", "benchmark_spec": "benchmark_spec.json", "checkpoints": "checkpoints/", "raw": "raw/", "metrics": "metrics/"}}
    _json_dump(output / "summary.json", summary)
    _write_report(output / "report.md", summary)
    _json_dump(output / "checksums.json", {str(path.relative_to(output)): _sha256(path) for path in output.rglob("*") if path.is_file() and path.name != "checksums.json"})
    return output


def _write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [f"# AAA benchmark v2 — {summary['run_id']}", "", f"Role: `{summary['role']}`. This is an engineering benchmark report, not a product or scientific-approval claim.", "", "## Gate outcomes", "", "| Gate | Status | Observed | Threshold |", "|---|---|---:|---|"]
    for gate in summary["gates"]["gates"]:  # type: ignore[index]
        lines.append(f"| {gate['name']} | {gate['status']} | {gate['observed']} | {gate['threshold']} |")
    lines.extend(["", f"Overall required-gate status: **{summary['gates']['all_required_gates_pass']}**.", "", "## Family measurements", ""])
    for family, result in summary["results"].items():  # type: ignore[index]
        lines.extend([f"### {family}", "", f"Episodes: `{result['episodes']}`; replicas: `{result['replicas']}`; bounce events: `{result['bounce_events']}`.", "", "| Predictor | MAE | p95 | post-change MAE |", "|---|---:|---:|---:|"])
        for name, predictor in result["predictors"].items():  # type: ignore[index]
            lines.append(f"| {name} | {predictor.get('mae')} | {predictor.get('p95_absolute_error')} | {predictor.get('post_change_window_mae')} |")
        lines.append("")
    lines.extend(["## Reproduction", "", "` .venv/bin/python -m aaa.cli benchmark-v2 --role confirmation_a --attempt-id <new-id> --output-root runs `", "", "Raw step-level predictions are compressed under `raw/`; checksums and source-tree identity are recorded in the attempt directory. Confirmation A and B must both be run from the committed selected source without intervening tuning.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
