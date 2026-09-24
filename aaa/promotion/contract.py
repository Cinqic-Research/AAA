"""The frozen statistical contract ``aaa.promotion.crossed.v1``.

This is the prospective successor to the ``aaa.1k.v2`` K5 decision path
(``AAA-180``). It is a *specification*, shared by two structurally independent
implementations (:mod:`aaa.promotion.primary` and
:mod:`aaa.promotion.independent`); it contains no resampling code of its own.

Estimand
--------
For each declared group ``g`` the ratio of means

    R_g = mean(challenger_g) / mean(reference_g)

is taken over the complete declared ``initialization x series`` grid of that
group, and the groups are combined by a geometric mean

    G = exp(mean_g log R_g).

Lower is better (the metric is an error). The groups are the declared suite and
are never resampled.

Uncertainty and the two group designs
-------------------------------------
Initializations are shared by every group, so a bootstrap draw resamples them
once. What happens to the series dimension is *declared per group before any
observation*:

``crossed``
    The group has at least two observed series, treated as a sample from a
    population of series. Each draw also resamples that group's series. The
    interval describes the population of series.
``conditional_on_single_series``
    The group has exactly one observed series (Monash ``saugeen`` is the
    motivating case). There is no series dimension to resample and none is
    invented: the sole series is held fixed and only the shared
    initializations are resampled. That group's contribution to the interval
    is *conditional on the observed series*. It says nothing about other
    series from the same source.

A result whose groups include a conditional one carries the scope
``conditional_on_observed_series`` and names those groups; a population claim
is made only when every group is ``crossed``. The rule was chosen because it
is the only one of the options identified in ``AAA-180`` that (a) keeps the
declared estimand unchanged, (b) uses every declared group, and (c) invents no
variance that was not observed. Excluding the group from interval estimation
would change the estimand; treating one series as a population would fabricate
a dimension. It was fixed from those properties, not from any verdict.

Fewer than two declared initializations leave nothing to resample at all:
every criterion is then ``INSUFFICIENT_EVIDENCE``, never ``PASS``.

Criteria and verdicts
---------------------
Each criterion compares an interval of ``G`` with a threshold on the ratio
scale:

``noninferior``  PASS if upper <= threshold; FAIL if lower > threshold; else INCONCLUSIVE
``superior``     PASS if upper <  threshold; FAIL if lower >= threshold; else INCONCLUSIVE

A criterion without a measured interval is ``INSUFFICIENT_EVIDENCE``. The
verdict is intersection-union: any ``FAIL`` -> ``REJECT``; every criterion
``PASS`` -> ``PROMOTE``; any ``INSUFFICIENT_EVIDENCE`` -> ``INSUFFICIENT_EVIDENCE``;
otherwise ``INCONCLUSIVE``. Adjudication adds two fail-closed verdicts:
``INVALID_EVIDENCE`` (malformed, missing, extra, non-finite, mismatched or
duplicated primitives) and ``DISAGREEMENT`` (the two implementations differ).
Neither ever promotes.

Agreement tolerance
-------------------
Point values must agree to a relative ``1e-12``: they are deterministic
functions of the primitives. The two implementations draw different bootstrap
samples, so interval bounds agree only up to Monte Carlo error. For a
percentile bound at level ``p`` from ``B`` draws of an approximately normal
statistic with standard deviation ``s``, the standard error of the bound is
about ``sqrt(p (1 - p) / B) / phi(z_p) * s``; at ``p = 0.025`` that is
``2.67 s / sqrt(B)``, and the difference of two independent estimates has
standard error ``3.78 s / sqrt(B)``. The 95% interval is about ``3.92 s`` wide,
so with the default ``B = 20000`` one standard error of the difference is
``0.0068`` interval widths. The declared tolerance, ``0.05`` interval widths,
is more than seven standard errors: a genuine implementation difference larger
than that is detected, while Monte Carlo noise essentially never trips it.
Statuses must agree exactly. A bound that lies within Monte Carlo error of a
threshold can therefore produce ``DISAGREEMENT``; that is resolved by *not*
promoting, never by choosing either side.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

CONTRACT_ID = "aaa.promotion.crossed.v1"
RECORD_KEYS = ("arm", "group", "init", "series", "value")
DESIGNS = ("crossed", "conditional_on_single_series")
RULES = ("noninferior", "superior")

# Contract identities that must never adjudicate a promotion again, with the reason.
FORBIDDEN_CONTRACTS: Mapping[str, str] = {
    "aaa.1k.v2.k1-k5": (
        "AAA-180: frozen aaa.1k.v2 run_confirmation omitted the Monash groups from K5, and its primary "
        "and independent K5 implementations disagree on the single-series saugeen group. The frozen "
        "source and evidence are retained unchanged as history; the path may not promote any candidate."
    ),
}

CriterionStatus = Literal["PASS", "FAIL", "INCONCLUSIVE", "INSUFFICIENT_EVIDENCE"]
Verdict = Literal[
    "PROMOTE", "REJECT", "INCONCLUSIVE", "INSUFFICIENT_EVIDENCE", "INVALID_EVIDENCE", "DISAGREEMENT"
]


class ContractError(ValueError):
    """The contract itself is malformed or forbidden."""


class PrimitiveError(ValueError):
    """The primitives do not match the declared design; adjudication must fail closed."""


def require_admissible(contract_id: str) -> None:
    """Raise if ``contract_id`` is a retired decision path that may not promote."""

    if contract_id in FORBIDDEN_CONTRACTS:
        raise ContractError(f"{contract_id} is forbidden for promotion: {FORBIDDEN_CONTRACTS[contract_id]}")
    if contract_id != CONTRACT_ID:
        raise ContractError(f"unknown promotion contract {contract_id!r}; expected {CONTRACT_ID}")


def _identity(value: Any, what: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise ContractError(f"{what} identities must be strings or integers, got {value!r}")
    if isinstance(value, str) and not value:
        raise ContractError(f"{what} identities must be non-empty")
    return value


@dataclass(frozen=True)
class GroupDeclaration:
    name: str
    design: str
    series: tuple[str | int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ContractError("a group needs a non-empty name")
        if self.design not in DESIGNS:
            raise ContractError(f"group {self.name}: unknown design {self.design!r}")
        series = tuple(_identity(s, f"group {self.name} series") for s in self.series)
        if len(set(series)) != len(series):
            raise ContractError(f"group {self.name}: duplicate series identities")
        if self.design == "crossed" and len(series) < 2:
            raise ContractError(
                f"group {self.name}: a crossed group needs at least two series; declare "
                "conditional_on_single_series instead of inventing a series dimension"
            )
        if self.design == "conditional_on_single_series" and len(series) != 1:
            raise ContractError(
                f"group {self.name}: conditional_on_single_series requires exactly one series"
            )


@dataclass(frozen=True)
class Criterion:
    name: str
    rule: str
    threshold: float

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ContractError(f"criterion {self.name}: unknown rule {self.rule!r}")
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ContractError(f"criterion {self.name}: threshold must be a finite positive ratio")


@dataclass(frozen=True)
class Contract:
    """Everything the decision depends on, fixed before any candidate evidence is observed."""

    reference: str
    challenger: str
    initializations: tuple[str | int, ...]
    groups: tuple[GroupDeclaration, ...]
    criteria: tuple[Criterion, ...]
    seed: int
    draws: int = 20000
    confidence: float = 0.95
    bound_tolerance_widths: float = 0.05
    point_rel_tolerance: float = 1e-12
    contract_id: str = CONTRACT_ID
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_admissible(self.contract_id)
        if not self.reference or not self.challenger or self.reference == self.challenger:
            raise ContractError("reference and challenger must be distinct non-empty arm names")
        inits = tuple(_identity(i, "initialization") for i in self.initializations)
        if not inits or len(set(inits)) != len(inits):
            raise ContractError("initializations must be non-empty and unique")
        names = [g.name for g in self.groups]
        if not names or len(set(names)) != len(names):
            raise ContractError("groups must be non-empty with unique names")
        if not self.criteria or len({c.name for c in self.criteria}) != len(self.criteria):
            raise ContractError("criteria must be non-empty with unique names")
        if isinstance(self.draws, bool) or not isinstance(self.draws, int) or self.draws < 1000:
            raise ContractError("draws must be an integer of at least 1000")
        if not 0.5 < self.confidence < 1.0:
            raise ContractError("confidence must lie in (0.5, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ContractError("seed must be a non-negative 63-bit integer")
        if not 0 < self.bound_tolerance_widths < 0.5 or not 0 < self.point_rel_tolerance < 1e-6:
            raise ContractError("agreement tolerances are out of their declared range")

    @property
    def scope(self) -> str:
        return (
            "population"
            if all(g.design == "crossed" for g in self.groups)
            else "conditional_on_observed_series"
        )

    @property
    def conditional_groups(self) -> list[str]:
        return [g.name for g in self.groups if g.design == "conditional_on_single_series"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "reference": self.reference,
            "challenger": self.challenger,
            "initializations": list(self.initializations),
            "groups": [{"name": g.name, "design": g.design, "series": list(g.series)} for g in self.groups],
            "criteria": [{"name": c.name, "rule": c.rule, "threshold": c.threshold} for c in self.criteria],
            "seed": self.seed,
            "draws": self.draws,
            "confidence": self.confidence,
            "bound_tolerance_widths": self.bound_tolerance_widths,
            "point_rel_tolerance": self.point_rel_tolerance,
            "scope": self.scope,
            "conditional_groups": self.conditional_groups,
        }


def criterion_status(
    rule: str, lower: float | None, upper: float | None, threshold: float
) -> CriterionStatus:
    """The declared rule; ``None`` bounds (no measured interval) are never a pass."""

    if lower is None or upper is None:
        return "INSUFFICIENT_EVIDENCE"
    if rule == "noninferior":
        return "PASS" if upper <= threshold else ("FAIL" if lower > threshold else "INCONCLUSIVE")
    if rule == "superior":
        return "PASS" if upper < threshold else ("FAIL" if lower >= threshold else "INCONCLUSIVE")
    raise ContractError(f"unknown rule {rule!r}")


def combine(statuses: Sequence[str]) -> Verdict:
    """Intersection-union over criterion statuses."""

    if not statuses:
        return "INSUFFICIENT_EVIDENCE"
    if any(s == "FAIL" for s in statuses):
        return "REJECT"
    if all(s == "PASS" for s in statuses):
        return "PROMOTE"
    if any(s == "INSUFFICIENT_EVIDENCE" for s in statuses):
        return "INSUFFICIENT_EVIDENCE"
    return "INCONCLUSIVE"
