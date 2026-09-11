"""Metric definitions for position prediction and adaptation.

Unit contract
-------------
Every benchmark-facing aggregate is a **normalized** absolute error:

``normalized_absolute_error = |scored - actual| / (upper_bound - lower_bound)``

All normalized aggregates go through :func:`normalized_errors`, the single
canonical accessor, so a unit-width interval can never hide a mixed-unit
mistake. Raw-unit values are available through :func:`raw_errors` and are
always labelled ``*_raw`` where they appear in reports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Any

from .experiment import StepRecord

NORMALIZED_FIELD = "normalized_absolute_error"
RAW_FIELD = "absolute_error"


def mean_absolute_error(errors: Iterable[float]) -> float | None:
    values = [float(value) for value in errors]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("metrics cannot aggregate non-finite errors")
    return mean(values) if values else None


def normalized_errors(records: Sequence[StepRecord], predictor: str) -> list[float]:
    """Canonical normalized-error accessor used by every benchmark aggregate."""

    values: list[float] = []
    for record in records:
        prediction = record.predictions.get(predictor)
        if prediction is None:
            raise KeyError(f"record has no prediction for predictor {predictor!r}")
        if NORMALIZED_FIELD not in prediction:
            raise KeyError(
                f"record for predictor {predictor!r} lacks {NORMALIZED_FIELD!r}; "
                "normalized aggregates must never fall back to raw units"
            )
        values.append(float(prediction[NORMALIZED_FIELD]))
    return values


def raw_errors(records: Sequence[StepRecord], predictor: str) -> list[float]:
    """Raw-unit absolute errors. Never mixed into a normalized aggregate."""

    return [float(record.predictions[predictor][RAW_FIELD]) for record in records]


def signed_normalized_errors(records: Sequence[StepRecord], predictor: str) -> list[float]:
    values: list[float] = []
    for record in records:
        prediction = record.predictions[predictor]
        if "signed_error" in prediction:
            values.append(float(prediction["signed_error"]) / float(record.interval_width))
        else:
            values.append(
                (float(prediction["scored"]) - float(record.actual_next_position))
                / float(record.interval_width)
            )
    return values


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Expanding-then-trailing mean; useful for within-episode display only."""

    if window <= 0:
        raise ValueError("rolling window must be positive")
    if not values:
        return []
    return [mean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def complete_rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Trailing means only after a complete window is available."""

    if window <= 0:
        raise ValueError("rolling window must be positive")
    return [mean(values[index - window + 1 : index + 1]) for index in range(window - 1, len(values))]


def _event_slice(records: Sequence[StepRecord], event: str) -> list[StepRecord]:
    if event == "bounce":
        return [record for record in records if record.bounced]
    if event == "change":
        return [record for record in records if record.changed]
    raise ValueError(f"unknown event: {event}")


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

RECOVERY_STATUSES = (
    "no_intervention",
    "insufficient_pre_event_evidence",
    "insufficient_post_event_evidence",
    "no_measured_shock",
    "recovered",
    "unrecovered",
)

ELIGIBLE_STATUSES = ("recovered", "unrecovered")


@dataclass(frozen=True)
class RecoveryConfig:
    """Pre-declared recovery rule.

    Eligibility is defined from the *peak* post-event shock, not from a long
    post-event average. A large one-step shock followed by fast recovery is an
    eligible, recovered event; averaging it over a 50-transition window would
    otherwise erase the shock and quietly drop the episode from the
    denominator.
    """

    pre_event_reference_length: int = 50
    post_event_horizon: int = 50
    shock_window: int = 5
    shock_multiplier: float = 3.0
    shock_floor: float = 1e-4
    tolerance_multiplier: float = 1.5
    tolerance_floor: float = 1e-5
    rolling_window: int = 5
    sustain_windows: int = 3

    def __post_init__(self) -> None:
        integers = {
            "pre_event_reference_length": self.pre_event_reference_length,
            "post_event_horizon": self.post_event_horizon,
            "shock_window": self.shock_window,
            "rolling_window": self.rolling_window,
            "sustain_windows": self.sustain_windows,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.shock_window > self.post_event_horizon:
            raise ValueError("shock_window must fit inside post_event_horizon")
        if self.rolling_window + self.sustain_windows - 1 > self.post_event_horizon:
            raise ValueError("rolling and sustain windows must fit inside post_event_horizon")
        for name, number in (
            ("shock_multiplier", self.shock_multiplier),
            ("tolerance_multiplier", self.tolerance_multiplier),
        ):
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, number in (("shock_floor", self.shock_floor), ("tolerance_floor", self.tolerance_floor)):
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_event_reference_length": self.pre_event_reference_length,
            "post_event_horizon": self.post_event_horizon,
            "shock_window": self.shock_window,
            "shock_multiplier": self.shock_multiplier,
            "shock_floor": self.shock_floor,
            "tolerance_multiplier": self.tolerance_multiplier,
            "tolerance_floor": self.tolerance_floor,
            "rolling_window": self.rolling_window,
            "sustain_windows": self.sustain_windows,
        }


def recovery_metric(
    records: Sequence[StepRecord],
    predictor: str,
    config: RecoveryConfig | None = None,
    *,
    pre_event_errors: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Classify one episode into exactly one recovery status.

    ``pre_event_errors`` must be **this predictor's own** normalized errors
    from the pre-intervention prefix. Sharing one learner's error sequence
    across persistence, constant motion and frozen copies produces recovery
    statistics that describe the wrong model.
    """

    config = config or RecoveryConfig()
    changed = _event_slice(records, "change")
    if not changed:
        return {
            "status": "no_intervention",
            "eligible": False,
            "recovered": False,
            "reason": "episode contains no intervention transition",
        }
    event_step = changed[0].step

    if pre_event_errors is None:
        prefix = [record for record in records if record.step < event_step]
        reference_values = normalized_errors(prefix, predictor)
    else:
        reference_values = [float(value) for value in pre_event_errors]
    if len(reference_values) < config.pre_event_reference_length:
        return {
            "status": "insufficient_pre_event_evidence",
            "eligible": False,
            "recovered": False,
            "event_step": event_step,
            "pre_event_observations": len(reference_values),
            "required_pre_event_observations": config.pre_event_reference_length,
        }
    reference_window = reference_values[-config.pre_event_reference_length :]
    reference_mae = mean(reference_window)

    post = [record for record in records if record.step >= event_step]
    if len(post) < config.post_event_horizon:
        return {
            "status": "insufficient_post_event_evidence",
            "eligible": False,
            "recovered": False,
            "event_step": event_step,
            "pre_event_reference_mae": reference_mae,
            "post_event_observations": len(post),
            "required_post_event_observations": config.post_event_horizon,
        }
    post_errors = normalized_errors(post[: config.post_event_horizon], predictor)
    shock = max(post_errors[: config.shock_window])
    shock_threshold = max(config.shock_multiplier * reference_mae, config.shock_floor)
    tolerance = max(config.tolerance_multiplier * reference_mae, config.tolerance_floor)
    common = {
        "event_step": event_step,
        "pre_event_reference_mae": reference_mae,
        "pre_event_reference_length": config.pre_event_reference_length,
        "peak_shock": shock,
        "shock_threshold": shock_threshold,
        "tolerance": tolerance,
        "post_event_window_mae": mean(post_errors),
        "post_event_cumulative_error": float(sum(post_errors)),
        "rolling_window": config.rolling_window,
        "sustain_windows": config.sustain_windows,
    }
    if shock < shock_threshold:
        return {"status": "no_measured_shock", "eligible": False, "recovered": False, **common}

    rolling = complete_rolling_mean(post_errors, config.rolling_window)
    onset: int | None = None
    confirmation: int | None = None
    for index in range(len(rolling) - config.sustain_windows + 1):
        if all(value <= tolerance for value in rolling[index : index + config.sustain_windows]):
            onset = index + 1
            confirmation = index + config.rolling_window + config.sustain_windows - 1
            break
    if confirmation is None:
        return {"status": "unrecovered", "eligible": True, "recovered": False, **common}
    return {
        "status": "recovered",
        "eligible": True,
        "recovered": True,
        "recovery_onset_transition": onset,
        "recovery_confirmation_transition": confirmation,
        "recovery_time_steps": confirmation,
        **common,
    }


def episode_metrics(
    records: Sequence[StepRecord],
    predictor_names: Sequence[str],
    *,
    recovery: RecoveryConfig | None = None,
    pre_event_errors: Mapping[str, Sequence[float]] | None = None,
    post_change_window: int | None = None,
    include_rolling: bool = False,
    rolling_window: int = 5,
) -> dict[str, Any]:
    """Per-episode normalized summaries and recovery classification."""

    if not records:
        raise ValueError("cannot compute episode metrics with no records")
    recovery = recovery or RecoveryConfig()
    window = int(post_change_window if post_change_window is not None else recovery.post_event_horizon)
    identity = records[0].identity
    result: dict[str, Any] = {
        "trial_id": identity.trial_id,
        "replica_id": identity.replica_id,
        "episode": identity.episode,
        "family": identity.family,
        "branch": identity.branch,
        "scenario": identity.scenario,
        "stratum": identity.stratum,
        "environment_seed": identity.environment_seed,
        "scored_steps": len(records),
        "bounce_events": sum(record.bounce_count for record in records),
        "bounce_transitions": sum(1 for record in records if record.bounced),
        "predictors": {},
    }
    bounce_records = _event_slice(records, "bounce")
    non_bounce_records = [record for record in records if not record.bounced]
    change_records = _event_slice(records, "change")
    post_records: list[StepRecord] = []
    if change_records:
        event_step = change_records[0].step
        post_records = [record for record in records if event_step <= record.step < event_step + window]
    for predictor in predictor_names:
        errors = normalized_errors(records, predictor)
        predictor_result: dict[str, Any] = {
            "mae": mean_absolute_error(errors),
            "bounce_mae": mean_absolute_error(normalized_errors(bounce_records, predictor)),
            "non_bounce_mae": mean_absolute_error(normalized_errors(non_bounce_records, predictor)),
            "change_mae": mean_absolute_error(normalized_errors(change_records, predictor)),
            "post_change_window_mae": mean_absolute_error(normalized_errors(post_records, predictor)),
            "post_change_window_cumulative": (
                float(sum(normalized_errors(post_records, predictor))) if post_records else None
            ),
            "post_change_window_transitions": len(post_records),
            "signed_bias": mean(signed_normalized_errors(records, predictor)),
            "bounce_transitions": len(bounce_records),
            "change_transitions": len(change_records),
            "recovery": recovery_metric(
                records,
                predictor,
                recovery,
                pre_event_errors=None if pre_event_errors is None else pre_event_errors.get(predictor),
            ),
        }
        if include_rolling:
            predictor_result["rolling_mae"] = rolling_mean(errors, rolling_window)
        result["predictors"][predictor] = predictor_result
    return result


def aggregate_metrics(
    records: Sequence[StepRecord],
    predictor_names: Sequence[str],
    *,
    recovery: RecoveryConfig | None = None,
    post_change_window: int | None = None,
    rolling_window: int = 5,
) -> dict[str, Any]:
    """Return overall, per-episode, per-replica, and adaptation metrics.

    Every aggregate below is in normalized units. Raw-unit values are exposed
    separately under ``*_raw`` names.
    """

    if not records:
        return {"scored_steps": 0, "predictors": {}, "episodes": [], "replicas": {}}
    recovery = recovery or RecoveryConfig()
    episode_groups: dict[tuple[int, int, str], list[StepRecord]] = defaultdict(list)
    for record in records:
        episode_groups[(record.replica_id, record.episode, record.scenario)].append(record)
    episodes = [
        episode_metrics(
            group,
            predictor_names,
            recovery=recovery,
            post_change_window=post_change_window,
            include_rolling=True,
            rolling_window=rolling_window,
        )
        for _, group in sorted(episode_groups.items())
    ]
    replica_groups: dict[int, list[StepRecord]] = defaultdict(list)
    for record in records:
        replica_groups[record.replica_id].append(record)
    by_replica: dict[str, Any] = {}
    for replica, group in sorted(replica_groups.items()):
        nested: dict[tuple[int, str], list[StepRecord]] = defaultdict(list)
        for record in group:
            nested[(record.episode, record.scenario)].append(record)
        by_replica[str(replica)] = {
            "scored_steps": len(group),
            "predictors": {
                predictor: {
                    "mae": mean_absolute_error(normalized_errors(group, predictor)),
                    "episodes": len(nested),
                }
                for predictor in predictor_names
            },
        }

    aggregate_predictors: dict[str, Any] = {}
    for predictor in predictor_names:
        errors = normalized_errors(records, predictor)
        episode_values = [
            episode["predictors"][predictor]["mae"]
            for episode in episodes
            if episode["predictors"][predictor]["mae"] is not None
        ]
        statuses = [episode["predictors"][predictor]["recovery"]["status"] for episode in episodes]
        recovery_values = [
            episode["predictors"][predictor]["recovery"].get("recovery_time_steps")
            for episode in episodes
            if episode["predictors"][predictor]["recovery"].get("recovery_time_steps") is not None
        ]
        post_values = [
            episode["predictors"][predictor]["post_change_window_mae"]
            for episode in episodes
            if episode["predictors"][predictor]["post_change_window_mae"] is not None
        ]
        aggregate_predictors[predictor] = {
            "mae": mean_absolute_error(errors),
            "mae_raw": mean_absolute_error(raw_errors(records, predictor)),
            "episode_mae_mean": mean(episode_values) if episode_values else None,
            "episode_mae_sd": stdev(episode_values) if len(episode_values) > 1 else 0.0,
            "bounce_mae": mean_absolute_error(normalized_errors(_event_slice(records, "bounce"), predictor)),
            "non_bounce_mae": mean_absolute_error(
                normalized_errors([record for record in records if not record.bounced], predictor)
            ),
            "change_mae": mean_absolute_error(normalized_errors(_event_slice(records, "change"), predictor)),
            "post_change_window_mae_mean": mean(post_values) if post_values else None,
            "recovery_status_counts": {status: statuses.count(status) for status in RECOVERY_STATUSES},
            "recovery_eligible_episodes": sum(1 for status in statuses if status in ELIGIBLE_STATUSES),
            "recovery_recovered_episodes": statuses.count("recovered"),
            "recovery_unrecovered_episodes": statuses.count("unrecovered"),
            "recovery_time_steps_mean": mean(recovery_values) if recovery_values else None,
        }
    early_late = _early_late(replica_groups, predictor_names)
    return {
        "scored_steps": len(records),
        "episode_count": len(episodes),
        "replica_count": len(by_replica),
        "predictors": aggregate_predictors,
        "early_late": early_late,
        "episodes": episodes,
        "replicas": by_replica,
    }


def _early_late_row(early: float | None, late: float | None) -> dict[str, float | None]:
    """A missing side stays missing rather than becoming a fabricated zero."""

    return {
        "early_mae": early,
        "late_mae": late,
        "late_minus_early": None if early is None or late is None else late - early,
    }


def _early_late(
    replica_groups: dict[int, list[StepRecord]],
    predictor_names: Sequence[str],
) -> dict[str, Any]:
    """Within-run early/late comparison. Descriptive only, not a learning gate."""

    per_replica: dict[str, Any] = {}
    for replica, group in sorted(replica_groups.items()):
        ordered = sorted(group, key=lambda record: (record.episode, record.step))
        split = max(1, len(ordered) // 3)
        early, late = ordered[:split], ordered[-split:]
        per_replica[str(replica)] = {
            predictor: _early_late_row(
                mean_absolute_error(normalized_errors(early, predictor)),
                mean_absolute_error(normalized_errors(late, predictor)),
            )
            for predictor in predictor_names
        }
    result: dict[str, Any] = {}
    for predictor in predictor_names:
        rows = [per_replica[str(replica)][predictor] for replica in sorted(replica_groups)]
        early_values = [row["early_mae"] for row in rows]
        late_values = [row["late_mae"] for row in rows]
        delta_values = [row["late_minus_early"] for row in rows]
        result[predictor] = {
            "by_replica": {
                str(replica): per_replica[str(replica)][predictor] for replica in sorted(replica_groups)
            },
            "early_mae_mean": mean(early_values),
            "early_mae_sd": stdev(early_values) if len(early_values) > 1 else 0.0,
            "late_mae_mean": mean(late_values),
            "late_mae_sd": stdev(late_values) if len(late_values) > 1 else 0.0,
            "late_minus_early_mean": mean(delta_values),
        }
    return result
