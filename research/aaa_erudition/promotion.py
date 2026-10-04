"""Promotion contract ``aaa.promotion.paired.v1``.

Why a new contract. ``aaa.promotion.crossed.v1`` estimates a ratio of positive
means over a crossed initialization x series grid. Juniper 1 adaptation
evidence is different: each endpoint is a per-stream *rate* that can
legitimately be zero (a stream with no failures, no false adaptations), the
compared conditions run on identical streams (common random numbers), and
the claim is about one frozen Erudition artifact rather than a population of
initializations. Forcing that into ``crossed.v1`` would either refuse valid
zero rates or invent an initialization dimension. ``crossed.v1`` is unchanged
and remains the contract for evidence it fits.

Estimand. For each criterion, the mean over declared streams of the paired
difference ``metric(challenger, s) - metric(reference, s)``. Lower is better
for every metric. The interval is a percentile bootstrap over streams.

Criteria. ``superior``: PASS if upper < threshold, FAIL if lower >= threshold.
``noninferior``: PASS if upper <= threshold, FAIL if lower > threshold.
Otherwise INCONCLUSIVE. Fewer than ``MIN_STREAMS`` streams is
INSUFFICIENT_EVIDENCE. The verdict is intersection-union: any FAIL gives
REJECT, all PASS gives PROMOTE, any INSUFFICIENT_EVIDENCE (with no FAIL)
gives INSUFFICIENT_EVIDENCE, otherwise INCONCLUSIVE. A promotion therefore
needs both its target improvements and its bounded-regression criteria.

Two implementations. The primary resamples stream indices from a PCG64
stream. The independent one validates by identity sets, sorts, and
resamples with multinomial count weights from a separately salted Philox
stream; it re-implements the status and verdict rules and imports only the
declared constants. ``adjudicate`` returns a verdict only when both agree on
the point value (relative 1e-12), interval status, bounds (within 0.05
interval widths), every criterion and the verdict; otherwise DISAGREEMENT.
Missing, duplicated, extra, non-finite or out-of-range values are
INVALID_EVIDENCE. Neither failure verdict promotes.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

CONTRACT = "aaa.promotion.paired.v1"
MIN_STREAMS = 6
DRAWS = 20_000
LEVEL = 0.95
BOUND_TOLERANCE = 0.05
POINT_TOLERANCE = 1e-12
INDEPENDENT_SALT = 0x5EED


class EvidenceError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Criterion:
    name: str
    metric: str
    challenger: str
    reference: str
    kind: str
    threshold: float

    def __post_init__(self) -> None:
        if self.kind not in ("superior", "noninferior"):
            raise ValueError(f"unknown criterion kind {self.kind!r}")


@dataclasses.dataclass(frozen=True)
class Contract:
    name: str
    streams: tuple[str, ...]
    criteria: tuple[Criterion, ...]
    seed: int
    draws: int = DRAWS
    contract: str = CONTRACT


def _status(kind: str, lower: float, upper: float, threshold: float) -> str:
    if kind == "superior":
        if upper < threshold:
            return "PASS"
        if lower >= threshold:
            return "FAIL"
        return "INCONCLUSIVE"
    if upper <= threshold:
        return "PASS"
    if lower > threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def _verdict(statuses: list[str]) -> str:
    if "FAIL" in statuses:
        return "REJECT"
    if all(s == "PASS" for s in statuses):
        return "PROMOTE"
    if "INSUFFICIENT_EVIDENCE" in statuses:
        return "INSUFFICIENT_EVIDENCE"
    return "INCONCLUSIVE"


def _grid(
    contract: Contract, primitives: list[dict[str, Any]], criterion: Criterion
) -> tuple[np.ndarray, np.ndarray]:
    wanted = {(arm, s) for arm in (criterion.challenger, criterion.reference) for s in contract.streams}
    cells: dict[tuple[str, str], float] = {}
    for p in primitives:
        if p.get("metric") != criterion.metric or p.get("arm") not in (
            criterion.challenger,
            criterion.reference,
        ):
            continue
        key = (p["arm"], p["stream"])
        if key not in wanted:
            raise EvidenceError(f"undeclared cell {key}")
        if key in cells:
            raise EvidenceError(f"duplicate cell {key}")
        value: Any = p.get("value")
        # Exact JSON number types only, as the independent path requires.
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise EvidenceError(f"cell {key} has invalid value {value!r}")
        cells[key] = float(value)
    if set(cells) != wanted:
        raise EvidenceError(f"missing cells {sorted(wanted - set(cells))[:4]}")
    challenger = np.array([cells[(criterion.challenger, s)] for s in contract.streams])
    reference = np.array([cells[(criterion.reference, s)] for s in contract.streams])
    return challenger, reference


def primary(contract: Contract, primitives: list[dict[str, Any]]) -> dict[str, Any]:
    if not contract.criteria:
        raise EvidenceError("a contract with no criteria decides nothing")
    results: list[dict[str, Any]] = []
    generator = np.random.Generator(np.random.PCG64(contract.seed))
    n = len(contract.streams)
    for criterion in contract.criteria:
        challenger, reference = _grid(contract, primitives, criterion)
        diff = challenger - reference
        point = float(diff.mean())
        if n < MIN_STREAMS:
            results.append(
                {
                    "criterion": criterion.name,
                    "point": point,
                    "interval": "INSUFFICIENT_EVIDENCE",
                    "lower": None,
                    "upper": None,
                    "status": "INSUFFICIENT_EVIDENCE",
                }
            )
            continue
        draws = []
        for start in range(0, contract.draws, 4096):
            size = min(4096, contract.draws - start)
            index = generator.integers(0, n, size=(size, n))
            draws.append(diff[index].mean(axis=1))
        sample = np.concatenate(draws)
        alpha = (1 - LEVEL) / 2
        lower, upper = (float(v) for v in np.quantile(sample, [alpha, 1 - alpha]))
        results.append(
            {
                "criterion": criterion.name,
                "point": point,
                "interval": "MEASURED",
                "lower": lower,
                "upper": upper,
                "status": _status(criterion.kind, lower, upper, criterion.threshold),
            }
        )
    return {
        "implementation": "primary",
        "criteria": results,
        "verdict": _verdict([r["status"] for r in results]),
    }


def independent(contract: Contract, primitives: list[dict[str, Any]]) -> dict[str, Any]:
    """Recomputation by a different route: identity sets, sorting, count weights, another bit generator."""

    streams = sorted(contract.streams)
    if len(contract.criteria) == 0:
        raise EvidenceError("no criteria declared")
    if len(set(streams)) != len(streams):
        raise EvidenceError("declared streams repeat")
    rng = np.random.Generator(np.random.Philox(contract.seed ^ INDEPENDENT_SALT))
    out = []
    for criterion in contract.criteria:
        rows = sorted(
            (
                p
                for p in primitives
                if p.get("metric") == criterion.metric
                and p.get("arm") in (criterion.challenger, criterion.reference)
            ),
            key=lambda p: (str(p.get("arm")), str(p.get("stream"))),
        )
        seen: list[tuple[Any, Any]] = [(p.get("arm"), p.get("stream")) for p in rows]
        expected: list[tuple[str, str]] = sorted(
            (arm, s) for arm in {criterion.challenger, criterion.reference} for s in streams
        )
        if sorted(seen, key=str) != sorted(expected, key=str) or len(set(seen)) != len(seen):
            raise EvidenceError(f"{criterion.name}: cells do not match the declared streams")
        lookup = {}
        for p in rows:
            v: Any = p.get("value")
            if type(v) not in (int, float) or v != v or v in (float("inf"), float("-inf")) or v < 0 or v > 1:
                raise EvidenceError(f"{criterion.name}: invalid value {v!r}")
            lookup[(p["arm"], p["stream"])] = float(v)
        diff = np.array(
            [lookup[(criterion.challenger, s)] - lookup[(criterion.reference, s)] for s in streams]
        )
        point = float(np.sum(diff) / len(diff))
        if len(streams) < MIN_STREAMS:
            out.append(
                {
                    "criterion": criterion.name,
                    "point": point,
                    "interval": "INSUFFICIENT_EVIDENCE",
                    "lower": None,
                    "upper": None,
                    "status": "INSUFFICIENT_EVIDENCE",
                }
            )
            continue
        counts = rng.multinomial(len(streams), np.full(len(streams), 1 / len(streams)), size=contract.draws)
        means = counts @ diff / len(streams)
        means.sort()
        lo_rank = (1 - LEVEL) / 2 * (contract.draws - 1)
        hi_rank = (1 + LEVEL) / 2 * (contract.draws - 1)
        lower = float(np.interp(lo_rank, np.arange(contract.draws), means))
        upper = float(np.interp(hi_rank, np.arange(contract.draws), means))
        if criterion.kind == "superior":
            status = (
                "PASS"
                if upper < criterion.threshold
                else ("FAIL" if not lower < criterion.threshold else "INCONCLUSIVE")
            )
        else:
            status = (
                "PASS"
                if not upper > criterion.threshold
                else ("FAIL" if lower > criterion.threshold else "INCONCLUSIVE")
            )
        out.append(
            {
                "criterion": criterion.name,
                "point": point,
                "interval": "MEASURED",
                "lower": lower,
                "upper": upper,
                "status": status,
            }
        )
    statuses = [r["status"] for r in out]
    if any(s == "FAIL" for s in statuses):
        verdict = "REJECT"
    elif statuses and all(s == "PASS" for s in statuses):
        verdict = "PROMOTE"
    elif any(s == "INSUFFICIENT_EVIDENCE" for s in statuses):
        verdict = "INSUFFICIENT_EVIDENCE"
    else:
        verdict = "INCONCLUSIVE"
    return {"implementation": "independent", "criteria": out, "verdict": verdict}


def adjudicate(contract: Contract, primitives: list[dict[str, Any]]) -> dict[str, Any]:
    if contract.contract != CONTRACT:
        return {
            "contract": CONTRACT,
            "verdict": "INVALID_EVIDENCE",
            "reason": f"contract {contract.contract!r} is not {CONTRACT}",
        }
    try:
        a = primary(contract, primitives)
        b = independent(contract, primitives)
    except (EvidenceError, KeyError, TypeError) as error:
        return {"contract": CONTRACT, "verdict": "INVALID_EVIDENCE", "reason": str(error)}
    problems = []
    for ra, rb in zip(a["criteria"], b["criteria"], strict=True):
        if abs(ra["point"] - rb["point"]) > POINT_TOLERANCE * max(1.0, abs(ra["point"])):
            problems.append(f"{ra['criterion']}: point")
        if ra["interval"] != rb["interval"] or ra["status"] != rb["status"]:
            problems.append(f"{ra['criterion']}: status")
        if ra["interval"] == "MEASURED":
            width = max(ra["upper"] - ra["lower"], 1e-12)
            if (
                abs(ra["lower"] - rb["lower"]) > BOUND_TOLERANCE * width
                or abs(ra["upper"] - rb["upper"]) > BOUND_TOLERANCE * width
            ):
                problems.append(f"{ra['criterion']}: bounds")
    if a["verdict"] != b["verdict"]:
        problems.append("verdict")
    if problems:
        return {
            "contract": CONTRACT,
            "verdict": "DISAGREEMENT",
            "problems": problems,
            "primary": a,
            "independent": b,
        }
    return {"contract": CONTRACT, "verdict": a["verdict"], "primary": a, "independent": b}
