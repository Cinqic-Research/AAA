"""Reproducible train, generalization, and adaptation experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import ExperimentConfig
from .environment import MovingDotEnvironment, Scenario
from .experiment import StepRecord, run_episode, write_step_records
from .metrics import aggregate_metrics, mean_absolute_error
from .predictors import ConstantMotionPredictor, OnlineLinearPredictor, PersistencePredictor
from .reporting import write_experiment_report
from .visualization import write_all_plots


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_metadata(project_root: Path) -> dict[str, object]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "working_tree_status": status.splitlines() if status else [],
    }


def _episode_seed(seed: int, phase: str, episode: int, scenario: str) -> int:
    phase_offsets = {"dev": 1_000_000, "train": 2_000_000, "final": 3_000_000}
    scenario_offsets = {"straight": 17, "bouncing": 31, "changed": 47}
    return int(seed + phase_offsets[phase] + episode * 1009 + scenario_offsets[scenario])


def _new_environment(config: ExperimentConfig, seed: int, phase: str, episode: int, scenario: Scenario) -> MovingDotEnvironment:
    return MovingDotEnvironment(
        scenario=scenario,
        seed=_episode_seed(seed, phase, episode, scenario),
        config=config.world,
    )


def _training_scenario(config: ExperimentConfig, episode: int) -> Scenario:
    return config.training_scenarios[episode % len(config.training_scenarios)]  # type: ignore[return-value]


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
        predictors = [model]
        if include_baselines:
            predictors = [PersistencePredictor(), ConstantMotionPredictor(), model]
        environment = _new_environment(config, seed, phase, episode, scenario)
        records.extend(
            run_episode(
                environment,
                predictors,
                episode=episode,
                record_seed=seed,
                learn=True,
                clip_predictions=config.clip_predictions,
            )
        )
    return model, records


def select_learning_rate(config: ExperimentConfig) -> dict[str, object]:
    """Choose a rate using only the development split, before final runs."""

    candidates: list[dict[str, object]] = []
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
                        episode=episode,
                        record_seed=seed,
                        learn=False,
                        clip_predictions=config.clip_predictions,
                    )
                )
            score = mean_absolute_error(
                record.predictions["linear_online"]["absolute_error"] for record in validation_records
            )
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
        "selection_rule": "minimum mean validation MAE on development seeds; final seeds were not inspected",
        "candidates": candidates,
        "selected_learning_rate": selected["learning_rate"],
    }


def _train_canonical_checkpoint(config: ExperimentConfig, destination: Path) -> dict[str, object]:
    """Train one checkpoint sequentially on the independent training split."""

    model = OnlineLinearPredictor(learning_rate=config.learning_rate, name="linear_online", update_enabled=True)
    episode_count = 0
    for seed in config.training_seeds:
        for local_episode in range(config.training_episodes):
            scenario = _training_scenario(config, local_episode)
            environment = _new_environment(config, seed, "train", local_episode, scenario)
            run_episode(
                environment,
                [model],
                episode=episode_count,
                record_seed=seed,
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


def _run_frozen_generalization(config: ExperimentConfig, checkpoint: Path) -> tuple[list[StepRecord], dict[str, object]]:
    records: list[StepRecord] = []
    weights_before = OnlineLinearPredictor.load(checkpoint, name="linear_frozen", update_enabled=False).weights.tolist()
    for seed in config.final_seeds:
        for scenario_index, scenario_name in enumerate(config.final_generalization_scenarios):
            scenario = scenario_name  # type: ignore[assignment]
            for episode in range(config.generalization_episodes_per_scenario):
                model = OnlineLinearPredictor.load(checkpoint, name="linear_frozen", update_enabled=False)
                global_episode = scenario_index * config.generalization_episodes_per_scenario + episode
                environment = _new_environment(config, seed, "final", global_episode, scenario)
                records.extend(
                    run_episode(
                        environment,
                        [PersistencePredictor(), ConstantMotionPredictor(), model],
                        episode=global_episode,
                        record_seed=seed,
                        learn=False,
                        clip_predictions=config.clip_predictions,
                    )
                )
                if model.weights.tolist() != weights_before:
                    raise AssertionError("frozen evaluation changed model parameters")
    return records, {
        "checkpoint_weights_unchanged": True,
        "weights_before": weights_before,
        "weights_after": weights_before,
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
                episode=0,
                record_seed=seed,
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
    """Run every requested stage and return the unique run directory."""

    output_root_path = Path(output_root)
    run_id, run_dir = _make_run_directory(output_root_path, label)
    project_path = Path(project_root or Path(__file__).resolve().parents[1])
    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aaa_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": _package_version("numpy"),
        "matplotlib": _package_version("matplotlib"),
        **_git_metadata(project_path),
    }
    _json_dump(run_dir / "metadata.json", metadata)

    selection = select_learning_rate(config)
    resolved_config = replace(config, learning_rate=float(selection["selected_learning_rate"]))
    _json_dump(run_dir / "config.json", resolved_config.as_dict())
    _json_dump(run_dir / "dev_selection.json", selection)

    learning_records = _run_learning_from_scratch(resolved_config)
    learning_metrics = aggregate_metrics(
        learning_records,
        ["persistence", "constant_motion", "linear_online"],
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
        sustain_windows=resolved_config.recovery_sustain_windows,
        tolerance_multiplier=resolved_config.recovery_multiplier,
        tolerance_floor=resolved_config.recovery_floor,
        meaningful_change_fraction=resolved_config.meaningful_change_fraction,
    )
    _write_experiment_files(run_dir / "learning_from_scratch", learning_records, learning_metrics)

    checkpoint_info = _train_canonical_checkpoint(resolved_config, run_dir / "checkpoints" / "trained_linear.json")
    _json_dump(run_dir / "checkpoints" / "checkpoint_metadata.json", checkpoint_info)

    frozen_records, frozen_integrity = _run_frozen_generalization(
        resolved_config, Path(checkpoint_info["path"])
    )
    frozen_metrics = aggregate_metrics(
        frozen_records,
        ["persistence", "constant_motion", "linear_frozen"],
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
        sustain_windows=resolved_config.recovery_sustain_windows,
        tolerance_multiplier=resolved_config.recovery_multiplier,
        tolerance_floor=resolved_config.recovery_floor,
        meaningful_change_fraction=resolved_config.meaningful_change_fraction,
    )
    _write_experiment_files(run_dir / "frozen_generalization", frozen_records, frozen_metrics)
    _json_dump(run_dir / "frozen_generalization" / "integrity.json", frozen_integrity)

    adaptation_records = _run_online_adaptation(resolved_config, Path(checkpoint_info["path"]))
    adaptation_metrics = aggregate_metrics(
        adaptation_records,
        ["persistence", "constant_motion", "linear_frozen", "linear_online"],
        post_change_window=resolved_config.post_change_window,
        rolling_window=resolved_config.rolling_window,
        sustain_windows=resolved_config.recovery_sustain_windows,
        tolerance_multiplier=resolved_config.recovery_multiplier,
        tolerance_floor=resolved_config.recovery_floor,
        meaningful_change_fraction=resolved_config.meaningful_change_fraction,
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
        "config": resolved_config.as_dict(),
        "metadata": metadata,
        "dev_selection": selection,
        "checkpoint": checkpoint_info,
        "frozen_integrity": frozen_integrity,
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
    report_copy = project_path / "reports" / f"experiment_report_{run_id}.md"
    write_experiment_report(report_copy, summary)
    return run_dir


def _write_experiment_files(directory: Path, records: Sequence[StepRecord], metrics: dict[str, object]) -> None:
    write_step_records(records, directory / "steps.jsonl", directory / "steps.csv")
    _json_dump(directory / "metrics.json", metrics)


def _package_version(package: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # pragma: no cover - environment-specific metadata
        return None
