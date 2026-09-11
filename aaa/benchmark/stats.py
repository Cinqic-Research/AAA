"""Paired hierarchical bootstrap with estimands matched to the gates.

Design
------
Every interval resamples the same two levels the design actually has:

1. training replicas, with replacement;
2. within each selected replica, that replica's **own** episode summaries with
   replacement, preserving how many observations that replica contributed.

Adjacent transitions are never treated as independent units, and a draw never
collapses a replica to a single episode. Comparisons preserve episode pairing:
the candidate and the baseline are read from the same resampled episode, so a
paired difference is a paired difference.

Every draw recomputes *exactly* the statistic the gate uses. When the gate is
``1 - mean(candidate) / mean(baseline)``, the interval is a ratio-of-means
interval, not a mean-of-ratios interval.

References
----------
- B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, ch. 6, 8, 13.
- A. J. Field and A. P. Welsh, "Bootstrapping clustered data",
  *J. R. Statist. Soc. B* 69(3), 2007 — cluster/hierarchical resampling.
- S. Holm, "A simple sequentially rejective multiple test procedure",
  *Scand. J. Statist.* 6(2), 1979 — the multiplicity correction used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class PairedSamples:
    """Per-replica aligned episode values for one candidate/baseline pair."""

    candidate: dict[str, list[float]]
    baseline: dict[str, list[float]] | None = None

    def replicas(self) -> list[str]:
        if self.baseline is None:
            return sorted(key for key, values in self.candidate.items() if values)
        return sorted(
            key
            for key in set(self.candidate) & set(self.baseline)
            if self.candidate[key] and len(self.candidate[key]) == len(self.baseline[key])
        )

    def validate(self) -> None:
        if self.baseline is None:
            return
        for key in set(self.candidate) & set(self.baseline):
            if len(self.candidate[key]) != len(self.baseline[key]):
                raise ValueError(
                    f"paired samples for replica {key!r} are misaligned: "
                    f"{len(self.candidate[key])} candidate vs {len(self.baseline[key])} baseline episodes"
                )


@dataclass(frozen=True)
class Interval:
    """A bootstrap result, or an explicit statement that it could not be computed."""

    estimate: float | None
    lower: float | None
    upper: float | None
    draws: int
    level: float
    estimand: str
    status: str = "OK"
    reason: str | None = None
    p_value: float | None = None
    replicas: int = 0
    episodes: int = 0

    @property
    def available(self) -> bool:
        return self.status == "OK" and self.lower is not None and self.upper is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "draws": self.draws,
            "level": self.level,
            "estimand": self.estimand,
            "status": self.status,
            "reason": self.reason,
            "p_value": self.p_value,
            "replicas": self.replicas,
            "episodes": self.episodes,
        }

    @classmethod
    def unavailable(cls, estimand: str, reason: str, *, level: float, draws: int) -> "Interval":
        return cls(
            estimate=None,
            lower=None,
            upper=None,
            draws=draws,
            level=level,
            estimand=estimand,
            status=INSUFFICIENT,
            reason=reason,
        )


def _resample_indices(rng: np.random.Generator, replicas: Sequence[str], counts: Mapping[str, int]):
    """One hierarchical draw: replicas with replacement, then their episodes."""

    chosen = rng.integers(0, len(replicas), size=len(replicas))
    draw: list[tuple[str, np.ndarray]] = []
    for index in chosen:
        replica = replicas[int(index)]
        size = counts[replica]
        draw.append((replica, rng.integers(0, size, size=size)))
    return draw


def _flatten(values: Mapping[str, Sequence[float]], draw) -> np.ndarray:
    parts = [np.asarray(values[replica], dtype=float)[indices] for replica, indices in draw]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def hierarchical_bootstrap(
    samples: PairedSamples,
    statistic: Callable[[np.ndarray, np.ndarray | None], float],
    *,
    estimand: str,
    seed: int,
    draws: int = 4000,
    level: float = 0.95,
    null_value: float | None = None,
    minimum_replicas: int = 2,
) -> Interval:
    """Percentile interval for ``statistic`` under paired hierarchical resampling.

    ``statistic`` receives the flattened resampled candidate values and, when a
    baseline is present, the paired baseline values. It must compute exactly
    the quantity the gate compares against its threshold.
    """

    samples.validate()
    replicas = samples.replicas()
    if len(replicas) < minimum_replicas:
        return Interval.unavailable(
            estimand,
            f"only {len(replicas)} usable replica(s); at least {minimum_replicas} are required",
            level=level,
            draws=draws,
        )
    counts = {replica: len(samples.candidate[replica]) for replica in replicas}
    total_episodes = sum(counts.values())

    observed_candidate = np.concatenate([np.asarray(samples.candidate[r], dtype=float) for r in replicas])
    observed_baseline = (
        np.concatenate([np.asarray(samples.baseline[r], dtype=float) for r in replicas])
        if samples.baseline is not None
        else None
    )
    try:
        point = float(statistic(observed_candidate, observed_baseline))
    except (ZeroDivisionError, FloatingPointError, ValueError) as error:
        return Interval.unavailable(estimand, f"point estimate could not be computed: {error}", level=level, draws=draws)
    if not math.isfinite(point):
        return Interval.unavailable(estimand, "point estimate is not finite", level=level, draws=draws)

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(draws):
        draw = _resample_indices(rng, replicas, counts)
        candidate_values = _flatten(samples.candidate, draw)
        baseline_values = _flatten(samples.baseline, draw) if samples.baseline is not None else None
        try:
            value = float(statistic(candidate_values, baseline_values))
        except (ZeroDivisionError, FloatingPointError, ValueError):
            continue
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < max(200, draws // 10):
        return Interval.unavailable(
            estimand,
            f"only {len(estimates)} of {draws} bootstrap draws produced a finite statistic",
            level=level,
            draws=draws,
        )
    array = np.asarray(estimates, dtype=float)
    alpha = (1.0 - level) / 2.0
    lower = float(np.percentile(array, 100 * alpha))
    upper = float(np.percentile(array, 100 * (1 - alpha)))
    p_value = None
    if null_value is not None:
        # One-sided bootstrap p-value for H0: statistic <= null_value.
        p_value = float((np.sum(array <= null_value) + 1) / (len(array) + 1))
    return Interval(
        estimate=point,
        lower=lower,
        upper=upper,
        draws=draws,
        level=level,
        estimand=estimand,
        p_value=p_value,
        replicas=len(replicas),
        episodes=total_episodes,
    )


# ---------------------------------------------------------------------------
# statistics used by the gates
# ---------------------------------------------------------------------------


def mean_statistic(candidate: np.ndarray, baseline: np.ndarray | None) -> float:
    if candidate.size == 0:
        raise ValueError("no observations")
    return float(np.mean(candidate))


def relative_improvement_statistic(candidate: np.ndarray, baseline: np.ndarray | None) -> float:
    """``1 - mean(candidate) / mean(baseline)`` — a ratio of means."""

    if baseline is None:
        raise ValueError("relative improvement requires a baseline")
    if candidate.size == 0 or baseline.size == 0:
        raise ValueError("no observations")
    denominator = float(np.mean(baseline))
    if denominator == 0.0:
        raise ZeroDivisionError("baseline mean is zero; relative improvement is undefined")
    return 1.0 - float(np.mean(candidate)) / denominator


def absolute_difference_statistic(candidate: np.ndarray, baseline: np.ndarray | None) -> float:
    if baseline is None:
        raise ValueError("absolute difference requires a baseline")
    if candidate.size == 0 or baseline.size == 0:
        raise ValueError("no observations")
    return float(np.mean(candidate)) - float(np.mean(baseline))


def ratio_statistic(candidate: np.ndarray, baseline: np.ndarray | None) -> float:
    if baseline is None:
        raise ValueError("ratio requires a baseline")
    denominator = float(np.mean(baseline))
    if denominator == 0.0:
        raise ZeroDivisionError("baseline mean is zero; ratio is undefined")
    return float(np.mean(candidate)) / denominator


def point_relative_improvement(candidate: Mapping[str, Sequence[float]], baseline: Mapping[str, Sequence[float]]) -> float | None:
    """Point estimate computed with exactly the bootstrap statistic."""

    replicas = sorted(set(candidate) & set(baseline))
    values = np.concatenate([np.asarray(candidate[r], dtype=float) for r in replicas]) if replicas else np.asarray([])
    others = np.concatenate([np.asarray(baseline[r], dtype=float) for r in replicas]) if replicas else np.asarray([])
    if values.size == 0 or others.size == 0 or float(np.mean(others)) == 0.0:
        return None
    return 1.0 - float(np.mean(values)) / float(np.mean(others))


def episode_balanced_mean(values: Mapping[str, Sequence[float]]) -> float | None:
    flat = [float(value) for series in values.values() for value in series]
    return float(np.mean(flat)) if flat else None


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------


@dataclass
class MultiplicityResult:
    method: str
    family_size: int
    alpha: float
    entries: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "family_size": self.family_size,
            "alpha": self.alpha,
            "entries": self.entries,
        }

    def adjusted(self, name: str) -> dict[str, object] | None:
        for entry in self.entries:
            if entry["name"] == name:
                return entry
        return None

    def rejects(self, name: str) -> bool | None:
        entry = self.adjusted(name)
        if entry is None or entry.get("adjusted_p_value") is None:
            return None
        return bool(entry["reject"])


def holm_bonferroni(
    p_values: Mapping[str, float | None],
    *,
    alpha: float,
    extra_family_size: int = 0,
    method: str = "holm_bonferroni",
) -> MultiplicityResult:
    """Holm-Bonferroni step-down correction over the primary comparisons.

    ``extra_family_size`` accounts for repeated confirmation attempts against
    the same frozen protocol: every previously spent confirmation batch adds to
    the family so that re-testing is paid for rather than ignored.
    """

    named = [(name, value) for name, value in p_values.items()]
    family_size = len(named) + max(0, int(extra_family_size))
    if method == "none":
        entries = [
            {
                "name": name,
                "p_value": value,
                "adjusted_p_value": value,
                "reject": None if value is None else bool(value <= alpha),
            }
            for name, value in sorted(named)
        ]
        return MultiplicityResult("none", len(named), alpha, entries)

    usable = sorted([(value, name) for name, value in named if value is not None])
    entries: list[dict[str, object]] = []
    running = 0.0
    rejected: dict[str, bool] = {}
    adjusted: dict[str, float] = {}
    for index, (value, name) in enumerate(usable):
        multiplier = family_size - index
        candidate_p = min(1.0, value * multiplier)
        running = max(running, candidate_p)
        adjusted[name] = running
        rejected[name] = running <= alpha
    # Holm is step-down: once a hypothesis fails to be rejected, none after it are.
    stop = False
    for _, name in usable:
        if stop or not rejected[name]:
            stop = True
            rejected[name] = False
    for name, value in sorted(named):
        entries.append(
            {
                "name": name,
                "p_value": value,
                "adjusted_p_value": adjusted.get(name),
                "reject": None if value is None else rejected.get(name, False),
            }
        )
    return MultiplicityResult(method, family_size, alpha, entries)
