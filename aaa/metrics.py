"""Metric definitions for position prediction and adaptation."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, stdev
from typing import Iterable, Sequence

from .experiment import StepRecord


def mean_absolute_error(errors: Iterable[float]) -> float | None:
    values = [float(value) for value in errors]
    if any(not __import__("math").isfinite(value) for value in values):
        raise ValueError("metrics cannot aggregate non-finite errors")
    return mean(values) if values else None


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    if window <= 0:
        raise ValueError("rolling window must be positive")
    if not values:
        return []
    return [
        mean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def complete_rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Trailing means only after a complete window is available."""

    if window <= 0:
        raise ValueError("rolling window must be positive")
    return [mean(values[index - window + 1 : index + 1]) for index in range(window - 1, len(values))]


def _error_values(records: Sequence[StepRecord], predictor: str) -> list[float]:
    return [record.predictions[predictor].get("normalized_absolute_error", record.predictions[predictor]["absolute_error"]) for record in records]


def _event_slice(records: Sequence[StepRecord], event: str) -> list[StepRecord]:
    if event == "bounce":
        return [record for record in records if record.bounced]
    if event == "change":
        return [record for record in records if record.changed]
    raise ValueError(f"unknown event: {event}")


def recovery_metric(
    records: Sequence[StepRecord],
    predictor: str,
    *,
    post_change_window: int = 20,
    rolling_window: int = 5,
    sustain_windows: int = 3,
    tolerance_multiplier: float = 1.5,
    tolerance_floor: float = 0.01,
    meaningful_change_fraction: float = 0.25,
    pre_change_errors: Sequence[float] | None = None,
) -> dict[str, object]:
    """Measure sustained post-change recovery using a pre-declared rule."""

    changed = _event_slice(records, "change")
    if not changed:
        return {"applicable": False, "reason": "no movement change in episode"}
    event_step = changed[0].step
    pre = [record for record in records if record.step < event_step]
    if pre_change_errors is not None:
        pre_mae = mean_absolute_error(pre_change_errors)
    else:
        pre_mae = mean_absolute_error(_error_values(pre, predictor))
    post = [record for record in records if record.step >= event_step]
    if pre_mae is None:
        return {"applicable": False, "reason": "no pre-change scored transitions"}
    if len(post) < post_change_window:
        return {
            "applicable": False,
            "status": "censored",
            "reason": "insufficient post-change data for complete response window",
            "event_step": event_step,
            "post_observations": len(post),
            "required_post_observations": post_change_window,
        }
    post_window = post[:post_change_window]
    post_window_mae = mean_absolute_error(_error_values(post_window, predictor))
    tolerance = max(tolerance_multiplier * pre_mae, tolerance_floor)
    meaningful_threshold = pre_mae + max(tolerance_floor, meaningful_change_fraction * pre_mae)
    if post_window_mae is None or post_window_mae <= meaningful_threshold:
        return {
            "applicable": False,
            "reason": "no meaningful post-change error increase",
            "event_step": event_step,
            "pre_change_mae": pre_mae,
            "post_change_window_mae": post_window_mae,
            "tolerance": tolerance,
        }

    post_errors = _error_values(post, predictor)
    rolling = complete_rolling_mean(post_errors, rolling_window)
    recovery_steps: int | None = None
    recovery_onset: int | None = None
    recovery_confirmation: int | None = None
    if len(rolling) >= sustain_windows:
        for index in range(len(rolling) - sustain_windows + 1):
            if all(value <= tolerance for value in rolling[index : index + sustain_windows]):
                recovery_onset = index + 1
                recovery_confirmation = index + rolling_window + sustain_windows - 1
                recovery_steps = recovery_confirmation
                break
    return {
        "applicable": True,
        "event_step": event_step,
        "pre_change_mae": pre_mae,
        "post_change_window_mae": post_window_mae,
        "tolerance": tolerance,
        "rolling_window": rolling_window,
        "sustain_windows": sustain_windows,
        "recovery_onset_transition": recovery_onset,
        "recovery_confirmation_transition": recovery_confirmation,
        "recovery_time_steps": recovery_steps,
        "recovery_status": (
            "recovered" if recovery_steps is not None else "not recovered within evaluation horizon"
        ),
    }


def episode_metrics(
    records: Sequence[StepRecord],
    predictor_names: Sequence[str],
    **recovery_kwargs: object,
) -> dict[str, object]:
    if not records:
        raise ValueError("cannot compute episode metrics with no records")
    result: dict[str, object] = {
        "seed": records[0].seed,
        "episode": records[0].episode,
        "scenario": records[0].scenario,
        "scored_steps": len(records),
        "predictors": {},
    }
    bounce_records = _event_slice(records, "bounce")
    non_bounce_records = [record for record in records if not record.bounced]
    change_records = _event_slice(records, "change")
    post_records: list[StepRecord] = []
    if change_records:
        event_step = change_records[0].step
        post_records = [
            record
            for record in records
            if event_step <= record.step < event_step + int(recovery_kwargs.get("post_change_window", 20))
        ]
    for predictor in predictor_names:
        errors = _error_values(records, predictor)
        predictor_result: dict[str, object] = {
            "mae": mean_absolute_error(errors),
            "bounce_mae": mean_absolute_error(_error_values(bounce_records, predictor)),
            "non_bounce_mae": mean_absolute_error(_error_values(non_bounce_records, predictor)),
            "change_mae": mean_absolute_error(_error_values(change_records, predictor)),
            "post_change_window_mae": mean_absolute_error(_error_values(post_records, predictor)),
            "bounce_transitions": len(bounce_records),
            "change_transitions": len(change_records),
            "rolling_mae": rolling_mean(errors, int(recovery_kwargs.get("rolling_window", 5))),
            "recovery": recovery_metric(records, predictor, **recovery_kwargs),
        }
        result["predictors"][predictor] = predictor_result
    return result


def aggregate_metrics(
    records: Sequence[StepRecord],
    predictor_names: Sequence[str],
    **recovery_kwargs: object,
) -> dict[str, object]:
    """Return overall, per-episode, per-seed, and adaptation metrics."""

    if not records:
        return {"scored_steps": 0, "predictors": {}, "episodes": [], "seeds": {}}
    episode_groups: dict[tuple[int, int, str], list[StepRecord]] = defaultdict(list)
    for record in records:
        episode_groups[(record.seed, record.episode, record.scenario)].append(record)
    episodes = [
        episode_metrics(group, predictor_names, **recovery_kwargs)
        for _, group in sorted(episode_groups.items())
    ]
    seed_groups: dict[int, list[StepRecord]] = defaultdict(list)
    for record in records:
        seed_groups[record.seed].append(record)
    by_seed: dict[str, object] = {}
    for seed, group in sorted(seed_groups.items()):
        seed_episode_groups: dict[tuple[int, str], list[StepRecord]] = defaultdict(list)
        for record in group:
            seed_episode_groups[(record.episode, record.scenario)].append(record)
        by_seed[str(seed)] = {
            "scored_steps": len(group),
            "predictors": {
                predictor: {
                    "mae": mean_absolute_error(_error_values(group, predictor)),
                    "episodes": len(seed_episode_groups),
                }
                for predictor in predictor_names
            },
        }

    aggregate_predictors: dict[str, object] = {}
    for predictor in predictor_names:
        errors = _error_values(records, predictor)
        episode_values = [
            episode["predictors"][predictor]["mae"]
            for episode in episodes
            if episode["predictors"][predictor]["mae"] is not None
        ]
        recovery_values = [
            episode["predictors"][predictor]["recovery"].get("recovery_time_steps")
            for episode in episodes
            if episode["predictors"][predictor]["recovery"].get("recovery_time_steps") is not None
        ]
        recovery_applicable = sum(
            bool(episode["predictors"][predictor]["recovery"].get("applicable")) for episode in episodes
        )
        aggregate_predictors[predictor] = {
            "mae": mean_absolute_error(errors),
            "episode_mae_mean": mean(episode_values) if episode_values else None,
            "episode_mae_sd": stdev(episode_values) if len(episode_values) > 1 else 0.0,
            "bounce_mae": mean_absolute_error(
                [
                    record.predictions[predictor]["absolute_error"]
                    for record in records
                    if record.bounced
                ]
            ),
            "non_bounce_mae": mean_absolute_error(
                [
                    record.predictions[predictor]["absolute_error"]
                    for record in records
                    if not record.bounced
                ]
            ),
            "change_mae": mean_absolute_error(
                [record.predictions[predictor]["absolute_error"] for record in records if record.changed]
            ),
            "post_change_window_mae_mean": mean(
                [
                    episode["predictors"][predictor]["post_change_window_mae"]
                    for episode in episodes
                    if episode["predictors"][predictor]["post_change_window_mae"] is not None
                ]
            )
            if any(episode["predictors"][predictor]["post_change_window_mae"] is not None for episode in episodes)
            else None,
            "recovery_applicable_episodes": recovery_applicable,
            "recovery_recovered_episodes": len(recovery_values),
            "recovery_time_steps_mean": mean(recovery_values) if recovery_values else None,
        }
    early_late_by_seed: dict[str, object] = {}
    for seed, group in sorted(seed_groups.items()):
        ordered = sorted(group, key=lambda record: (record.episode, record.step))
        split = max(1, len(ordered) // 3)
        early = ordered[:split]
        late = ordered[-split:]
        early_late_by_seed[str(seed)] = {
            predictor: {
                "early_mae": mean_absolute_error(_error_values(early, predictor)),
                "late_mae": mean_absolute_error(_error_values(late, predictor)),
                "late_minus_early": (
                    mean_absolute_error(_error_values(late, predictor))
                    - mean_absolute_error(_error_values(early, predictor))
                ),
            }
            for predictor in predictor_names
        }
    early_late: dict[str, object] = {}
    for predictor in predictor_names:
        seed_values = [early_late_by_seed[str(seed)][predictor] for seed in sorted(seed_groups)]
        early_values = [item["early_mae"] for item in seed_values]
        late_values = [item["late_mae"] for item in seed_values]
        delta_values = [item["late_minus_early"] for item in seed_values]
        early_late[predictor] = {
            "by_seed": {str(seed): early_late_by_seed[str(seed)][predictor] for seed in sorted(seed_groups)},
            "early_mae_mean": mean(early_values),
            "early_mae_sd": stdev(early_values) if len(early_values) > 1 else 0.0,
            "late_mae_mean": mean(late_values),
            "late_mae_sd": stdev(late_values) if len(late_values) > 1 else 0.0,
            "late_minus_early_mean": mean(delta_values),
        }
    return {
        "scored_steps": len(records),
        "episode_count": len(episodes),
        "seed_count": len(by_seed),
        "predictors": aggregate_predictors,
        "early_late": early_late,
        "episodes": episodes,
        "seeds": by_seed,
    }
