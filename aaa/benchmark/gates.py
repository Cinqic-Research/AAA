"""Gate definitions, evaluation and status semantics.

Statuses
--------
``PASS``
    The declared criterion was evaluated and met.
``FAIL``
    The declared criterion was evaluated and not met.
``NOT_VERIFIED``
    The criterion could not be evaluated at all (a required artifact, check or
    interval is missing). Absence of evidence is never converted into ``PASS``.
``INSUFFICIENT_EVIDENCE``
    The evidence exists but does not reach the declared minimum coverage or
    replication for the claim to be made.

Only ``PASS`` satisfies a required gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..metrics import ELIGIBLE_STATUSES
from .families import FamilyCollector
from .seeds import stable_label
from .spec import BenchmarkSpec, GateSpec
from .stats import (
    Interval,
    MultiplicityResult,
    PairedSamples,
    hierarchical_bootstrap,
    holm_bonferroni,
    relative_improvement_statistic,
    mean_statistic,
)

PASS = "PASS"
FAIL = "FAIL"
NOT_VERIFIED = "NOT_VERIFIED"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

SATISFACTORY = (PASS,)

PRIMARY_COMPARISONS = (
    "learning_progress.reduction",
    "changed_law.online_vs_frozen",
    "changed_law.online_vs_persistence",
)


@dataclass
class GateResult:
    name: str
    status: str
    required: bool
    observed: float | None
    threshold: dict[str, Any]
    description: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "observed": self.observed,
            "threshold": self.threshold,
            "description": self.description,
            "details": self.details,
        }


@dataclass
class GateContext:
    spec: BenchmarkSpec
    collectors: Mapping[str, FamilyCollector]
    results: Mapping[str, dict[str, Any]]
    learning_curve: dict[str, Any]
    correctness: dict[str, Any]
    reproducibility: dict[str, Any]
    latency: dict[str, Any]
    intervals: dict[str, Interval] = field(default_factory=dict)
    multiplicity: MultiplicityResult | None = None
    confirmation_attempts: int = 0


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def margin_statistic(candidate: np.ndarray, baseline: np.ndarray | None, *, max_ratio: float, floor: float) -> float:
    """``mean(candidate) - max_ratio * mean(baseline) - floor``.

    A single statistic for every non-regression claim. ``<= 0`` means the
    criterion holds, so the gate simply asks whether the upper end of the
    interval stays at or below zero.
    """

    if baseline is None:
        raise ValueError("non-regression requires a baseline")
    if candidate.size == 0 or baseline.size == 0:
        raise ValueError("no observations")
    return float(np.mean(candidate)) - max_ratio * float(np.mean(baseline)) - floor


def _paired(collector: FamilyCollector, candidate: str, baseline: str, field_name: str) -> PairedSamples:
    container = getattr(collector, field_name)
    return PairedSamples(
        candidate={key: list(value.get(candidate, [])) for key, value in container.items()},
        baseline={key: list(value.get(baseline, [])) for key, value in container.items()},
    )


def _single(collector: FamilyCollector, predictor: str, field_name: str) -> PairedSamples:
    container = getattr(collector, field_name)
    return PairedSamples(candidate={key: list(value.get(predictor, [])) for key, value in container.items()})


def collect_intervals(context: GateContext) -> None:
    """Compute every interval the gates need, before any decision is made."""

    spec = context.spec
    statistics = spec.statistics
    seed = spec.randomness.bootstrap_seed
    draws = statistics.draws
    level = statistics.interval

    def bootstrap(key: str, samples: PairedSamples, statistic, estimand: str, *, offset: int, null_value=None) -> None:
        context.intervals[key] = hierarchical_bootstrap(
            samples,
            statistic,
            estimand=estimand,
            seed=seed + offset,
            draws=draws,
            level=level,
            null_value=null_value,
        )

    # learning progress -------------------------------------------------
    curve = context.learning_curve.get("curve", {})
    budgets = context.learning_curve.get("budgets", [])
    if curve and budgets:
        first, last = str(budgets[0]), str(budgets[-1])
        samples = PairedSamples(
            candidate={k: list(v) for k, v in curve[last]["replica_episode_values"].items()},
            baseline={k: list(v) for k, v in curve[first]["replica_episode_values"].items()},
        )
        gate = spec.gate("learning_progress")
        bootstrap(
            "learning_progress.reduction",
            samples,
            relative_improvement_statistic,
            statistics.estimands["relative_improvement"],
            offset=11,
            null_value=float(gate.threshold["min_reduction"]),
        )

    # changed law --------------------------------------------------------
    changed = context.collectors.get("changed_law:changed")
    if changed is not None:
        gate = spec.gate("changed_law_adaptation")
        for offset, (key, baseline) in enumerate(
            (("changed_law.online_vs_frozen", "frozen"), ("changed_law.online_vs_persistence", "persistence"))
        ):
            threshold_key = (
                "min_improvement_vs_frozen" if baseline == "frozen" else "min_improvement_vs_persistence"
            )
            bootstrap(
                key,
                _paired(changed, "online", baseline, "replica_post_change_cumulative"),
                relative_improvement_statistic,
                statistics.estimands["relative_improvement"],
                offset=21 + offset,
                null_value=float(gate.threshold[threshold_key]),
            )
        bootstrap(
            "changed_law.online_vs_constant_motion_margin",
            _paired(changed, "online", "constant_motion", "replica_post_change_cumulative"),
            partial(
                margin_statistic,
                max_ratio=float(gate.threshold["constant_motion_max_ratio"]),
                floor=float(gate.threshold["constant_motion_absolute_floor"]),
            ),
            statistics.estimands["absolute_difference"],
            offset=31,
        )
    unchanged = context.collectors.get("changed_law:unchanged")
    if unchanged is not None:
        gate = spec.gate("unchanged_control")
        bootstrap(
            "unchanged_control.margin",
            _paired(unchanged, "online", "frozen", "replica_episode_mae"),
            partial(
                margin_statistic,
                max_ratio=float(gate.threshold["max_ratio"]),
                floor=float(gate.threshold["absolute_floor"]),
            ),
            statistics.estimands["absolute_difference"],
            offset=41,
        )

    # bounce parity ------------------------------------------------------
    bouncing = context.collectors.get("bouncing")
    if bouncing is not None:
        gate = spec.gate("bounce_event_accuracy")
        bootstrap(
            "bounce.parity_margin",
            _paired(bouncing, str(gate.threshold["candidate"]), str(gate.threshold["baseline"]), "replica_event_mae"),
            partial(
                margin_statistic,
                max_ratio=float(gate.threshold["max_ratio"]),
                floor=float(gate.threshold["absolute_floor"]),
            ),
            statistics.estimands["absolute_difference"],
            offset=51,
        )
        bootstrap(
            "bounce.candidate_event_mae",
            _single(bouncing, str(gate.threshold["candidate"]), "replica_event_mae"),
            mean_statistic,
            statistics.estimands["mean"],
            offset=52,
        )
        # decomposition of any apparent bounce advantage
        for offset, (key, baseline) in enumerate(
            (
                ("bounce.decomposition_vs_raw_constant_motion", "constant_motion"),
                ("bounce.decomposition_vs_reflected_constant_motion", "constant_motion_reflected"),
                ("bounce.decomposition_vs_persistence", "persistence"),
            )
        ):
            bootstrap(
                key,
                _paired(bouncing, "candidate_frozen", baseline, "replica_event_mae"),
                relative_improvement_statistic,
                statistics.estimands["relative_improvement"],
                offset=61 + offset,
                null_value=0.0,
            )
        bootstrap(
            "bounce.decomposition_no_reflect_vs_raw_constant_motion",
            _paired(bouncing, "candidate_no_reflect", "constant_motion", "replica_event_mae"),
            relative_improvement_statistic,
            statistics.estimands["relative_improvement"],
            offset=71,
            null_value=0.0,
        )

    # generic non-regression gates --------------------------------------
    for gate in spec.gates:
        if gate.evaluator != "non_regression":
            continue
        collector = context.collectors.get(str(gate.threshold["family"]))
        if collector is None:
            continue
        field_name = {
            "mae": "replica_episode_mae",
            "post_change_window_mae": "replica_post_change_mae",
        }[str(gate.threshold["metric"])]
        bootstrap(
            f"{gate.name}.margin",
            _paired(collector, str(gate.threshold["candidate"]), str(gate.threshold["baseline"]), field_name),
            partial(
                margin_statistic,
                max_ratio=float(gate.threshold["max_ratio"]),
                floor=float(gate.threshold["absolute_floor"]),
            ),
            statistics.estimands["absolute_difference"],
            offset=81 + (stable_label(gate.name) % 100),
        )

    # absolute accuracy gates -------------------------------------------
    for gate in spec.gates:
        if gate.evaluator not in ("absolute_accuracy", "absolute_accuracy_by_stratum"):
            continue
        family = str(gate.threshold["family"])
        collector = context.collectors.get(family)
        if collector is None:
            continue
        predictor = str(gate.threshold["predictor"])
        bootstrap(
            f"{gate.name}.mean",
            _single(collector, predictor, "replica_episode_mae"),
            mean_statistic,
            statistics.estimands["mean"],
            offset=181 + (stable_label(gate.name) % 100),
        )


def apply_multiplicity(context: GateContext) -> None:
    """Holm-Bonferroni across primary comparisons and repeated attempts."""

    statistics = context.spec.statistics
    p_values = {
        key: context.intervals[key].p_value
        for key in PRIMARY_COMPARISONS
        if key in context.intervals
    }
    context.multiplicity = holm_bonferroni(
        p_values,
        alpha=statistics.family_wise_alpha,
        extra_family_size=max(0, context.confirmation_attempts - 1),
        method=statistics.multiplicity,
    )


# ---------------------------------------------------------------------------
# evaluators
# ---------------------------------------------------------------------------


def _interval_dict(context: GateContext, key: str) -> dict[str, Any] | None:
    interval = context.intervals.get(key)
    return None if interval is None else interval.to_dict()


def evaluate_coverage(context: GateContext, gate: GateSpec) -> GateResult:
    spec = context.spec
    required = set(spec.required_strata())
    missing: dict[str, list[str]] = {}
    thin: dict[str, list[str]] = {}
    thin_replicas: dict[str, list[str]] = {}
    for family_name, family in spec.motion_families.items():
        if not family.stratified:
            continue
        collector = context.collectors.get(family_name)
        if collector is None:
            return GateResult(
                gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                {"reason": f"family {family_name} was not executed"},
            )
        present = collector.stratum_episodes
        absent = sorted(required - set(present))
        if absent:
            missing[family_name] = absent
        thin[family_name] = sorted(
            name for name in required & set(present)
            if present[name] < spec.stratification.minimum_episodes_per_stratum
        )
        thin_replicas[family_name] = sorted(
            name for name in required & set(present)
            if len(collector.stratum_replicas.get(name, set())) < spec.stratification.minimum_replicas_per_stratum
        )
    bouncing = context.collectors.get("bouncing")
    wall_ok = False
    bounce_events = 0
    if bouncing is not None:
        bounce_events = bouncing.bounce_events
        required_walls = spec.motion_families["bouncing"].required_walls
        wall_ok = all(bouncing.wall_counts.get(wall, 0) > 0 for wall in required_walls)
    minimum_bounces = spec.motion_families["bouncing"].minimum_bounce_events or 0
    changed = context.collectors.get("changed_law:changed")
    eligible = 0
    if changed is not None:
        counts = changed.recovery_status_counts.get("online", {})
        eligible = sum(counts.get(status, 0) for status in ELIGIBLE_STATUSES)
    details = {
        "required_strata": len(required),
        "missing_strata": missing,
        "under_populated_strata": {k: v for k, v in thin.items() if v},
        "under_replicated_strata": {k: v for k, v in thin_replicas.items() if v},
        "bounce_events": bounce_events,
        "minimum_bounce_events": minimum_bounces,
        "both_walls_observed": wall_ok,
        "eligible_change_events": eligible,
        "minimum_eligible_change_events": spec.confirmation.minimum_eligible_change_events,
    }
    ok = (
        not missing
        and not any(thin.values())
        and not any(thin_replicas.values())
        and bounce_events >= minimum_bounces
        and wall_ok
        and eligible >= spec.confirmation.minimum_eligible_change_events
    )
    status = PASS if ok else INSUFFICIENT_EVIDENCE
    return GateResult(gate.name, status, gate.required, float(bounce_events), gate.threshold, gate.description, details)


def evaluate_absolute_accuracy(context: GateContext, gate: GateSpec) -> GateResult:
    family = str(gate.threshold["family"])
    predictor = str(gate.threshold["predictor"])
    metric = str(gate.threshold.get("metric", "episode_balanced_mae"))
    result = context.results.get(family)
    if result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": f"family {family} was not executed"})
    key = {"mae": "episode_balanced_mae"}.get(metric, metric)
    observed = result["predictors"][predictor].get(key)
    interval = _interval_dict(context, f"{gate.name}.mean")
    if observed is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": f"metric {key} unavailable", "interval": interval})
    limit = float(gate.threshold["max_value"])
    status = PASS if float(observed) <= limit else FAIL
    return GateResult(gate.name, status, gate.required, float(observed), gate.threshold, gate.description,
                      {"interval": interval, "metric": key})


def evaluate_absolute_accuracy_by_stratum(context: GateContext, gate: GateSpec) -> GateResult:
    family = str(gate.threshold["family"])
    predictor = str(gate.threshold["predictor"])
    result = context.results.get(family)
    if result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": f"family {family} was not executed"})
    predictor_result = result["predictors"][predictor]
    observed = predictor_result.get("episode_balanced_mae")
    strata = predictor_result.get("strata", {})
    required = set(context.spec.required_strata())
    present = required & set(strata)
    if not present or present != required:
        return GateResult(
            gate.name, INSUFFICIENT_EVIDENCE, gate.required, observed, gate.threshold, gate.description,
            {
                "reason": "required strata are missing; an empty or partial stratum set cannot pass",
                "required_strata": len(required),
                "present_strata": len(present),
                "missing_strata": sorted(required - present),
            },
        )
    max_stratum_mae = float(gate.threshold["max_stratum_mae"])
    max_stratum_p95 = float(gate.threshold["max_stratum_p95"])
    failures = [
        {"stratum": name, "mae": strata[name]["mae"], "p95": strata[name]["p95"]}
        for name in sorted(required)
        if strata[name]["mae"] > max_stratum_mae or strata[name]["p95"] > max_stratum_p95
    ]
    overall_ok = observed is not None and float(observed) <= float(gate.threshold["max_mae"])
    status = PASS if overall_ok and not failures else FAIL
    return GateResult(
        gate.name, status, gate.required, observed, gate.threshold, gate.description,
        {
            "interval": _interval_dict(context, f"{gate.name}.mean"),
            "strata_evaluated": len(required),
            "failing_strata": failures,
            "overall_within_limit": overall_ok,
        },
    )


def evaluate_learning_progress(context: GateContext, gate: GateSpec) -> GateResult:
    curve = context.learning_curve.get("curve", {})
    budgets = context.learning_curve.get("budgets", [])
    if not curve or len(budgets) < 2:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "learning curve was not measured"})
    means = [float(curve[str(budget)]["mean"]) for budget in budgets]
    first, last = means[0], means[-1]
    reduction = None if first == 0 else 1.0 - last / first
    interval = context.intervals.get("learning_progress.reduction")
    monotone = all(later <= earlier * (1 + 1e-9) + 1e-15 for earlier, later in zip(means, means[1:]))
    adjusted = None if context.multiplicity is None else context.multiplicity.adjusted("learning_progress.reduction")
    details = {
        "budget_means": {str(budget): value for budget, value in zip(budgets, means)},
        "monotone_non_increasing": monotone,
        "interval": None if interval is None else interval.to_dict(),
        "multiplicity": adjusted,
        "probe_episodes": context.learning_curve.get("probe_episodes"),
    }
    if interval is None or not interval.available:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, reduction, gate.threshold, gate.description,
                          {**details, "reason": "required uncertainty interval could not be computed"})
    if reduction is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {**details, "reason": "zero-budget probe error is zero; reduction is undefined"})
    ok = (
        reduction >= float(gate.threshold["min_reduction"])
        and interval.lower is not None
        and interval.lower >= float(gate.threshold["min_ci_lower"])
    )
    if bool(gate.threshold.get("require_monotone_non_increasing")):
        ok = ok and monotone
    if adjusted is not None and adjusted.get("reject") is False:
        ok = False
        details["multiplicity_rejected"] = False
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, reduction, gate.threshold, gate.description, details)


def evaluate_event_accuracy_and_parity(context: GateContext, gate: GateSpec) -> GateResult:
    family = str(gate.threshold["family"])
    result = context.results.get(family)
    collector = context.collectors.get(family)
    if result is None or collector is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": f"family {family} was not executed"})
    candidate = str(gate.threshold["candidate"])
    baseline = str(gate.threshold["baseline"])
    observed = result["predictors"][candidate].get("event_episode_balanced_mae")
    baseline_value = result["predictors"][baseline].get("event_episode_balanced_mae")
    margin = context.intervals.get("bounce.parity_margin")
    decomposition = {
        "vs_raw_constant_motion": _interval_dict(context, "bounce.decomposition_vs_raw_constant_motion"),
        "vs_reflected_constant_motion": _interval_dict(context, "bounce.decomposition_vs_reflected_constant_motion"),
        "vs_persistence": _interval_dict(context, "bounce.decomposition_vs_persistence"),
        "no_reflect_vs_raw_constant_motion": _interval_dict(
            context, "bounce.decomposition_no_reflect_vs_raw_constant_motion"
        ),
    }
    details: dict[str, Any] = {
        "candidate_event_mae": observed,
        "baseline_event_mae": baseline_value,
        "event_episodes": collector.event_episode_count,
        "no_event_episodes_excluded": collector.no_event_episode_count,
        "candidate_event_p95": result["predictors"][candidate].get("event_p95"),
        "candidate_event_p99": result["predictors"][candidate].get("event_p99"),
        "parity_margin_interval": None if margin is None else margin.to_dict(),
        "advantage_decomposition": decomposition,
    }
    if observed is None or baseline_value is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, observed, gate.threshold, gate.description,
                          {**details, "reason": "no event episodes were observed"})
    if margin is None or not margin.available:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, observed, gate.threshold, gate.description,
                          {**details, "reason": "required parity interval could not be computed"})
    accuracy_ok = float(observed) <= float(gate.threshold["max_event_mae"])
    parity_ok = margin.estimate is not None and margin.estimate <= 0.0 and margin.upper is not None and margin.upper <= 0.0
    details["absolute_accuracy_within_limit"] = accuracy_ok
    details["parity_within_limit"] = parity_ok
    return GateResult(
        gate.name, PASS if accuracy_ok and parity_ok else FAIL, gate.required, observed,
        gate.threshold, gate.description, details,
    )


def evaluate_non_regression(context: GateContext, gate: GateSpec) -> GateResult:
    family = str(gate.threshold["family"])
    result = context.results.get(family)
    if result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": f"family {family} was not executed"})
    candidate = str(gate.threshold["candidate"])
    baseline = str(gate.threshold["baseline"])
    metric = str(gate.threshold["metric"])
    key = {"mae": "episode_balanced_mae", "post_change_window_mae": "post_change_window_mae"}[metric]
    observed = result["predictors"][candidate].get(key)
    baseline_value = result["predictors"][baseline].get(key)
    interval = context.intervals.get(f"{gate.name}.margin")
    details = {
        "candidate": observed,
        "baseline": baseline_value,
        "metric": key,
        "margin_interval": None if interval is None else interval.to_dict(),
    }
    if observed is None or baseline_value is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, observed, gate.threshold, gate.description,
                          {**details, "reason": f"metric {key} unavailable for the comparison"})
    if interval is None or not interval.available:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, observed, gate.threshold, gate.description,
                          {**details, "reason": "required non-regression interval could not be computed"})
    ok = interval.estimate is not None and interval.estimate <= 0.0 and interval.upper is not None and interval.upper <= 0.0
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, observed, gate.threshold, gate.description, details)


def evaluate_changed_law_adaptation(context: GateContext, gate: GateSpec) -> GateResult:
    result = context.results.get("changed_law:changed")
    if result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "changed-law family was not executed"})
    online = result["predictors"]["online"]
    frozen = result["predictors"]["frozen"]
    persistence = result["predictors"]["persistence"]
    constant_motion = result["predictors"]["constant_motion"]
    vs_frozen = context.intervals.get("changed_law.online_vs_frozen")
    vs_persistence = context.intervals.get("changed_law.online_vs_persistence")
    vs_constant = context.intervals.get("changed_law.online_vs_constant_motion_margin")
    details: dict[str, Any] = {
        "online_post_change_cumulative": online.get("post_change_cumulative_mean"),
        "frozen_post_change_cumulative": frozen.get("post_change_cumulative_mean"),
        "persistence_post_change_cumulative": persistence.get("post_change_cumulative_mean"),
        "constant_motion_post_change_cumulative": constant_motion.get("post_change_cumulative_mean"),
        "online_post_change_mae": online.get("post_change_window_mae"),
        "frozen_post_change_mae": frozen.get("post_change_window_mae"),
        "versus_frozen": None if vs_frozen is None else vs_frozen.to_dict(),
        "versus_persistence": None if vs_persistence is None else vs_persistence.to_dict(),
        "versus_constant_motion_margin": None if vs_constant is None else vs_constant.to_dict(),
        "multiplicity": {
            key: None if context.multiplicity is None else context.multiplicity.adjusted(key)
            for key in ("changed_law.online_vs_frozen", "changed_law.online_vs_persistence")
        },
        "first_surprise_included": True,
    }
    missing = [
        name
        for name, interval in (
            ("online_vs_frozen", vs_frozen),
            ("online_vs_persistence", vs_persistence),
            ("online_vs_constant_motion", vs_constant),
        )
        if interval is None or not interval.available
    ]
    if missing:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {**details, "reason": f"required intervals unavailable: {missing}"})
    assert vs_frozen is not None and vs_persistence is not None and vs_constant is not None
    floor = float(gate.threshold["require_ci_lower_above"])
    frozen_ok = (
        vs_frozen.estimate is not None
        and vs_frozen.estimate >= float(gate.threshold["min_improvement_vs_frozen"])
        and vs_frozen.lower is not None
        and vs_frozen.lower > floor
    )
    persistence_ok = (
        vs_persistence.estimate is not None
        and vs_persistence.estimate >= float(gate.threshold["min_improvement_vs_persistence"])
        and vs_persistence.lower is not None
        and vs_persistence.lower > floor
    )
    constant_ok = (
        vs_constant.estimate is not None and vs_constant.estimate <= 0.0
        and vs_constant.upper is not None and vs_constant.upper <= 0.0
    )
    multiplicity_ok = True
    if context.multiplicity is not None:
        for key in ("changed_law.online_vs_frozen", "changed_law.online_vs_persistence"):
            decision = context.multiplicity.rejects(key)
            if decision is False:
                multiplicity_ok = False
    details.update(
        {
            "versus_frozen_ok": frozen_ok,
            "versus_persistence_ok": persistence_ok,
            "constant_motion_non_regression_ok": constant_ok,
            "multiplicity_ok": multiplicity_ok,
        }
    )
    ok = frozen_ok and persistence_ok and constant_ok and multiplicity_ok
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, vs_frozen.estimate, gate.threshold, gate.description, details)


def evaluate_unchanged_control(context: GateContext, gate: GateSpec) -> GateResult:
    result = context.results.get("changed_law:unchanged")
    if result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "unchanged control branch was not executed"})
    interval = context.intervals.get("unchanged_control.margin")
    online = result["predictors"]["online"].get("episode_balanced_mae")
    frozen = result["predictors"]["frozen"].get("episode_balanced_mae")
    details = {
        "online_mae": online,
        "frozen_mae": frozen,
        "margin_interval": None if interval is None else interval.to_dict(),
    }
    if interval is None or not interval.available:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {**details, "reason": "required interval could not be computed"})
    ok = interval.estimate is not None and interval.estimate <= 0.0 and interval.upper is not None and interval.upper <= 0.0
    observed = None if online is None or frozen is None else float(online) - float(frozen)
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, observed, gate.threshold, gate.description, details)


def evaluate_recovery(context: GateContext, gate: GateSpec) -> GateResult:
    collector = context.collectors.get("changed_law:changed")
    result = context.results.get("changed_law:changed")
    if collector is None or result is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "changed-law family was not executed"})
    counts = collector.recovery_status_counts.get("online", {})
    eligible = sum(counts.get(status, 0) for status in ELIGIBLE_STATUSES)
    recovered = counts.get("recovered", 0)
    unrecovered = counts.get("unrecovered", 0)
    per_replica: dict[str, dict[str, int]] = {}
    for summary in collector.episode_summaries:
        replica = str(summary["replica_id"])
        status = str(summary["predictors"]["online"]["recovery"]["status"])  # type: ignore[index]
        bucket = per_replica.setdefault(replica, {})
        bucket[status] = bucket.get(status, 0) + 1
    rate = recovered / eligible if eligible else None
    details = {
        "status_counts": dict(sorted(counts.items())),
        "eligible": eligible,
        "recovered": recovered,
        "unrecovered": unrecovered,
        "per_replica_status_counts": per_replica,
        "per_replica_recovery_rate": {
            replica: (
                bucket.get("recovered", 0)
                / sum(bucket.get(status, 0) for status in ELIGIBLE_STATUSES)
                if sum(bucket.get(status, 0) for status in ELIGIBLE_STATUSES)
                else None
            )
            for replica, bucket in sorted(per_replica.items())
        },
        "recovery_time_steps": result["predictors"]["online"].get("recovery_time_steps"),
        "recovery_by_predictor": {
            name: dict(sorted(collector.recovery_status_counts.get(name, {}).items()))
            for name in collector.predictor_names
        },
    }
    minimum = int(gate.threshold["min_eligible_events"])
    if eligible < minimum:
        return GateResult(gate.name, INSUFFICIENT_EVIDENCE, gate.required, rate, gate.threshold, gate.description,
                          {**details, "reason": f"only {eligible} eligible events; {minimum} required"})
    ok = rate is not None and rate >= float(gate.threshold["min_recovery_rate"])
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, rate, gate.threshold, gate.description, details)


def evaluate_correctness(context: GateContext, gate: GateSpec) -> GateResult:
    report = context.correctness
    if not report:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "correctness checks were not executed"})
    failures = [name for name, value in report.get("checks", {}).items() if value is not True]
    status = PASS if not failures and report.get("executed") else FAIL
    if not report.get("executed"):
        status = NOT_VERIFIED
    return GateResult(gate.name, status, gate.required, float(len(failures)), gate.threshold, gate.description,
                      {**report, "failed_checks": failures})


def evaluate_reproducibility(context: GateContext, gate: GateSpec) -> GateResult:
    report = context.reproducibility
    if not report or not report.get("executed"):
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {**(report or {}), "reason": "no rerun evidence was produced"})
    failures = [name for name, value in report.get("checks", {}).items() if value is not True]
    return GateResult(gate.name, PASS if not failures else FAIL, gate.required, float(len(failures)),
                      gate.threshold, gate.description, {**report, "failed_checks": failures})


def evaluate_latency(context: GateContext, gate: GateSpec) -> GateResult:
    report = context.latency
    if not report:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description,
                          {"reason": "latency was not measured"})
    observed = report.get("predict_plus_update_p95_ms")
    limit = context.spec.latency.p95_limit_ms
    if observed is None:
        return GateResult(gate.name, NOT_VERIFIED, gate.required, None, gate.threshold, gate.description, report)
    ok = float(observed) <= limit and int(report.get("failures", 0)) == 0
    return GateResult(gate.name, PASS if ok else FAIL, gate.required, float(observed), gate.threshold,
                      gate.description, {**report, "p95_limit_ms": limit})


EVALUATORS: dict[str, Callable[[GateContext, GateSpec], GateResult]] = {
    "coverage": evaluate_coverage,
    "absolute_accuracy": evaluate_absolute_accuracy,
    "absolute_accuracy_by_stratum": evaluate_absolute_accuracy_by_stratum,
    "learning_progress": evaluate_learning_progress,
    "event_accuracy_and_parity": evaluate_event_accuracy_and_parity,
    "non_regression": evaluate_non_regression,
    "changed_law_adaptation": evaluate_changed_law_adaptation,
    "unchanged_control": evaluate_unchanged_control,
    "recovery": evaluate_recovery,
    "correctness": evaluate_correctness,
    "reproducibility": evaluate_reproducibility,
    "latency": evaluate_latency,
}


def evaluate_gates(context: GateContext) -> dict[str, Any]:
    """Evaluate every declared gate and summarize required-gate status."""

    collect_intervals(context)
    apply_multiplicity(context)
    results: list[GateResult] = []
    for gate in context.spec.gates:
        evaluator = EVALUATORS.get(gate.evaluator)
        if evaluator is None:
            raise ValueError(f"gate {gate.name!r} declares unknown evaluator {gate.evaluator!r}")
        results.append(evaluator(context, gate))
    required = [item for item in results if item.required]
    unmet = [item.name for item in required if item.status not in SATISFACTORY]
    return {
        "gates": [item.to_dict() for item in results],
        "required_gate_names": [item.name for item in required],
        "unmet_required_gates": unmet,
        "all_required_gates_pass": not unmet,
        "multiplicity": None if context.multiplicity is None else context.multiplicity.to_dict(),
        "intervals": {key: value.to_dict() for key, value in sorted(context.intervals.items())},
    }
