"""Historical v1 evaluation track: train, generalize, adapt.

This module is retained so the original v1 result stays reproducible as a
regression track. It is *not* the acceptance protocol; that is
:mod:`aaa.benchmark`. Its defects have been repaired, but its scientific design
(one learner, one speed-change scenario, no fair reflected baseline) is
deliberately unchanged so the historical comparison remains meaningful.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import ExperimentConfig
from .environment import MovingDotEnvironment, Scenario, as_scenario
from .experiment import StepRecord, TrialIdentity, run_episode, write_step_records
from .metrics import RecoveryConfig, aggregate_metrics, mean_absolute_error, normalized_errors
from .predictors import ConstantMotionPredictor, OnlineLinearPredictor, PersistencePredictor, Predictor
from .reporting import write_experiment_report
from .visualization import write_all_plots

PHASE_OFFSETS = {"dev": 1_000_000, "train": 2_000_000, "final": 3_000_000}
SCENARIO_OFFSETS = {"straight": 17, "bouncing": 31, "changed": 47, "dynamics_change": 59}


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _episode_seed(seed: int, phase: str, episode: int, scenario: str) -> int:
    if phase not in PHASE_OFFSETS:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(PHASE_OFFSETS)}")
    if scenario not in SCENARIO_OFFSETS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {sorted(SCENARIO_OFFSETS)}")
    return int(seed + PHASE_OFFSETS[phase] + episode * 1009 + SCENARIO_OFFSETS[scenario])


def _new_environment(
    config: ExperimentConfig, seed: int, phase: str, episode: int, scenario: Scenario
) -> MovingDotEnvironment:
    return MovingDotEnvironment(
        scenario=scenario, seed=_episode_seed(seed, phase, episode, scenario), config=config.world
    )


def _identity(
    *, seed: int, phase: str, episode: int, scenario: str, environment_seed: int, update_mode: str
) -> TrialIdentity:
    return TrialIdentity(
        trial_id=f"v1:{phase}:s{seed}:e{episode:04d}:{scenario}",
        role="legacy_v1",
        family=phase,
        scenario=scenario,
        environment_seed=environment_seed,
        replica_id=seed,
        episode=episode,
        branch=phase,
        stratum="unstratified",
        update_mode=update_mode,
    )


def _training_scenario(config: ExperimentConfig, episode: int) -> Scenario:
    return as_scenario(config.training_scenarios[episode % len(config.training_scenarios)])


def _recovery(config: ExperimentConfig) -> RecoveryConfig:
    """Historical v1 recovery rule expressed in the repaired vocabulary."""

    return RecoveryConfig(
        pre_event_reference_length=min(config.world.change_step or 20, 20),
        post_event_horizon=config.post_change_window,
        shock_window=min(5, config.post_change_window),
        shock_multiplier=1.0 + config.meaningful_change_fraction,
        shock_floor=config.recovery_floor,
        tolerance_multiplier=config.recovery_multiplier,
        tolerance_floor=config.recovery_floor,
        rolling_window=config.rolling_window,
        sustain_windows=config.recovery_sustain_windows,
    )


def _run_training_seed(
    config: ExperimentConfig,
    seed: int,
    learning_rate: float,
    episodes: int,
    *,
    phase: str,
    include_baselines: bool,
) -> tuple[OnlineLinearPredictor, list[StepRecord]]:
    model = OnlineLinearPredictor(learning_rate=learning_rate, name="linear_online", update_enabled=True)
    records: list[StepRecord] = []
    for episode in range(episodes):
        scenario = _training_scenario(config, episode)
        predictors: list[Predictor] = [model]
        if include_baselines:
            predictors = [PersistencePredictor(), ConstantMotionPredictor(), model]
        environment = _new_environment(config, seed, phase, episode, scenario)
        records.extend(
            run_episode(
                environment,
                predictors,
                _identity(
                    seed=seed,
                    phase=phase,
                    episode=episode,
                    scenario=scenario,
                    environment_seed=environment.seed,
                    update_mode="online",
                ),
                learn=True,
                clip_predictions=config.clip_predictions,
            )
        )
    return model, records


def select_learning_rate(config: ExperimentConfig) -> dict[str, Any]:
    """Choose a rate using only the development split, before final runs."""

    candidates: list[dict[str, Any]] = []
    for learning_rate in config.learning_rate_candidates:
        seed_scores: dict[str, float] = {}
        for seed in config.dev_seeds:
            model, _ = _run_training_seed(
                config,
                seed,
                learning_rate,
                config.dev_training_episodes,
                phase="dev",
                include_baselines=False,
            )
            validation_records: list[StepRecord] = []
            for episode in range(config.dev_validation_episodes):
                scenario = _training_scenario(config, episode + config.dev_training_episodes)
                environment = _new_environment(config, seed, "dev", episode + 100, scenario)
                validation_records.extend(
                    run_episode(
                        environment,
                        [model],
                        _identity(
                            seed=seed,
                            phase="dev",
                            episode=episode + 100,
                            scenario=scenario,
                            environment_seed=environment.seed,
                            update_mode="frozen",
                        ),
                        learn=False,
                        clip_predictions=config.clip_predictions,
                    )
                )
            score = mean_absolute_error(normalized_errors(validation_records, "linear_online"))
            if score is None:
                raise RuntimeError("development validation produced no scored steps")
            seed_scores[str(seed)] = score
        candidates.append(
            {
                "learning_rate": learning_rate,
                "validation_mae_mean": sum(seed_scores.values()) / len(seed_scores),
                "validation_mae_by_seed": seed_scores,
            }
        )
    selected = min(candidates, key=lambda candidate: float(candidate["validation_mae_mean"]))
    return {
        "selection_rule": "minimum mean normalized validation MAE on development seeds; final seeds were not inspected",
        "candidates": candidates,
        "selected_learning_rate": selected["learning_rate"],
    }


def _train_canonical_checkpoint(config: ExperimentConfig, destination: Path) -> dict[str, Any]:
    """Train one checkpoint sequentially on the independent training split."""

    model = OnlineLinearPredictor(
        learning_rate=config.learning_rate, name="linear_online", update_enabled=True
    )
    episode_count = 0
    for seed in config.training_seeds:
        for local_episode in range(config.training_episodes):
            scenario = _training_scenario(config, local_episode)
            environment = _new_environment(config, seed, "train", local_episode, scenario)
            run_episode(
                environment,
                [model],
                _identity(
                    seed=seed,
                    phase="train",
                    episode=episode_count,
                    scenario=scenario,
                    environment_seed=environment.seed,
                    update_mode="online",
                ),
                learn=True,
                clip_predictions=config.clip_predictions,
            )
            episode_count += 1
    model.save(destination)
    return {
        "path": str(destination),
        "training_seeds": list(config.training_seeds),
        "episodes_per_seed": config.training_episodes,
        "total_updates": model.update_count,
        "state": model.state_dict(),
    }


def _run_learning_from_scratch(config: ExperimentConfig) -> list[StepRecord]:
    records: list[StepRecord] = []
    for seed in config.training_seeds:
        _, seed_records = _run_training_seed(
            config,
            seed,
            config.learning_rate,
            config.training_episodes,
            phase="train",
            include_baselines=True,
        )
        records.extend(seed_records)
    return records


def _run_frozen_generalization(
    config: ExperimentConfig, checkpoint: Path
) -> tuple[list[StepRecord], dict[str, Any]]:
    """Evaluate a frozen checkpoint and *measure* whether it stayed frozen."""

    records: list[StepRecord] = []
    reference = OnlineLinearPredictor.load(checkpoint, name="linear_frozen", update_enabled=False)
    weights_before = reference.weights.tolist()
    update_count_before = reference.update_count
    observed_after: list[list[float]] = []
    observed_update_counts: list[int] = []
    for seed in config.final_seeds:
        for scenario_index, scenario_name in enumerate(config.final_generalization_scenarios):
            scenario: Scenario = as_scenario(scenario_name)
            for episode in range(config.generalization_episodes_per_scenario):
                model = OnlineLinearPredictor.load(checkpoint, name="linear_frozen", update_enabled=False)
                global_episode = scenario_index * config.generalization_episodes_per_scenario + episode
                environment = _new_environment(config, seed, "final", global_episode, scenario)
                records.extend(
                    run_episode(
                        environment,
                        [PersistencePredictor(), ConstantMotionPredictor(), model],
                        _identity(
                            seed=seed,
                            phase="final",
                            episode=global_episode,
                            scenario=scenario,
                            environment_seed=environment.seed,
                            update_mode="frozen",
                        ),
                        learn=False,
                        clip_predictions=config.clip_predictions,
                    )
                )
                observed_after.append(model.weights.tolist())
                observed_update_counts.append(model.update_count)
    unchanged = all(after == weights_before for after in observed_after) and all(
        count == update_count_before for count in observed_update_counts
    )
    return records, {
        # Measured, not asserted: the value below is derived from the actual
        # post-evaluation state of every frozen copy.
        "checkpoint_weights_unchanged": unchanged,
        "evaluated_copies": len(observed_after),
        "weights_before": weights_before,
        "weights_after_distinct": [list(item) for item in {tuple(item) for item in observed_after}],
        "update_count_before": update_count_before,
        "update_counts_after_distinct": sorted(set(observed_update_counts)),
    }


def _run_online_adaptation(config: ExperimentConfig, checkpoint: Path) -> list[StepRecord]:
    records: list[StepRecord] = []
    checkpoint_state = json.loads(checkpoint.read_text(encoding="utf-8"))
    for seed in config.final_seeds:
        frozen = OnlineLinearPredictor.from_state_dict(
            checkpoint_state, name="linear_frozen", update_enabled=False
        )
        online = OnlineLinearPredictor.from_state_dict(
            checkpoint_state, name="linear_online", update_enabled=True
        )
        environment = _new_environment(config, seed, "final", 10_000, "changed")
        records.extend(
            run_episode(
                environment,
                [PersistencePredictor(), ConstantMotionPredictor(), frozen, online],
                _identity(
                    seed=seed,
                    phase="final",
                    episode=0,
                    scenario="changed",
                    environment_seed=environment.seed,
                    update_mode="mixed",
                ),
                learn=True,
                clip_predictions=config.clip_predictions,
            )
        )
    return records


def _make_run_directory(output_root: Path, label: str) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{label}"
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}-{label}-{suffix}"
        run_dir = output_root / run_id
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def run_full_evaluation(
    config: ExperimentConfig,
    *,
    output_root: str | Path = "runs",
    label: str = "full",
    project_root: str | Path | None = None,
) -> Path:
    """Run every requested stage and return the unique run directory.

    Every artifact is written under ``output_root``. Nothing is written into
    the repository source tree; a caller that selects an output root gets all
    of its outputs there and only there.
    """

    output_root_path = Path(output_root)
    run_id, run_dir = _make_run_directory(output_root_path, label)
    project_path = Path(project_root or Path(__file__).resolve().parents[1])
    from .benchmark.evidence import git_metadata

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aaa_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "matplotlib": _package_version("matplotlib"),
        "git": git_metadata(project_path),
    }
    _json_dump(run_dir / "metadata.json", metadata)

    selection = select_learning_rate(config)
    resolved_config = replace(config, learning_rate=float(selection["selected_learning_rate"]))
    recovery = _recovery(resolved_config)
    _json_dump(run_dir / "config.json", resolved_config.as_dict())
    _json_dump(run_dir / "dev_selection.json", selection)

    learning_records = _run_learning_from_scratch(resolved_config)
    learning_metrics = aggregate_metrics(
        learning_records,
        ["persistence", "constant_motion", "linear_online"],
        recovery=recovery,
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
    )
    _write_experiment_files(run_dir / "learning_from_scratch", learning_records, learning_metrics)

    checkpoint_info = _train_canonical_checkpoint(
        resolved_config, run_dir / "checkpoints" / "trained_linear.json"
    )
    _json_dump(run_dir / "checkpoints" / "checkpoint_metadata.json", checkpoint_info)

    frozen_records, frozen_integrity = _run_frozen_generalization(
        resolved_config, Path(checkpoint_info["path"])
    )
    frozen_metrics = aggregate_metrics(
        frozen_records,
        ["persistence", "constant_motion", "linear_frozen"],
        recovery=recovery,
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
    )
    _write_experiment_files(run_dir / "frozen_generalization", frozen_records, frozen_metrics)
    _json_dump(run_dir / "frozen_generalization" / "integrity.json", frozen_integrity)

    adaptation_records = _run_online_adaptation(resolved_config, Path(checkpoint_info["path"]))
    adaptation_metrics = aggregate_metrics(
        adaptation_records,
        ["persistence", "constant_motion", "linear_frozen", "linear_online"],
        recovery=recovery,
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
    )
    _write_experiment_files(run_dir / "online_adaptation", adaptation_records, adaptation_metrics)

    plots = write_all_plots(
        run_dir / "plots",
        learning_records=learning_records,
        frozen_records=frozen_records,
        adaptation_records=adaptation_records,
        config=resolved_config,
    )
    summary = {
        "run_id": run_id,
        "track": "historical-v1",
        "config": resolved_config.as_dict(),
        "metadata": metadata,
        "dev_selection": selection,
        "checkpoint": checkpoint_info,
        "frozen_integrity": frozen_integrity,
        "recovery_rule": recovery.to_dict(),
        "experiments": {
            "learning_from_scratch": learning_metrics,
            "frozen_generalization": frozen_metrics,
            "online_adaptation": adaptation_metrics,
        },
        "plots": plots,
        "report": str(run_dir / "experiment_report.md"),
    }
    _json_dump(run_dir / "summary.json", summary)
    write_experiment_report(run_dir / "experiment_report.md", summary)
    return run_dir


def _write_experiment_files(directory: Path, records: Sequence[StepRecord], metrics: dict[str, Any]) -> None:
    write_step_records(records, directory / "steps.jsonl", directory / "steps.csv")
    _json_dump(directory / "metrics.json", metrics)


def _package_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # pragma: no cover - environment-specific metadata
        return None
