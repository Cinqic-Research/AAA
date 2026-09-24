"""Independent recomputation for ``aaa.promotion.crossed.v1``: count-weighted resampling.

Shares only the declared :class:`~aaa.promotion.contract.Contract` values with
the primary evaluator. Everything else is written separately:

* validation compares *sets* of observed identities with the declared sets,
  instead of indexing a pre-shaped grid;
* grids are assembled by sorting records, not by dictionary lookup;
* a bootstrap draw assigns multinomial counts to initializations and to each
  crossed group's series and takes weighted means (equal in distribution to
  index resampling, not equal draw by draw). A group declared
  ``conditional_on_single_series`` gets the explicit weight vector ``[1]``:
  its sole series is fixed by declaration, not by an accident of
  ``multinomial(1, [1])`` as in the frozen v2 recomputation;
* criterion statuses and the verdict are re-implemented from the rule text.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contract import Contract, PrimitiveError

FIELDS = frozenset({"arm", "group", "init", "series", "value"})


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise PrimitiveError(f"non-numeric primitive {value!r}")
    number = float(value)
    if not (number > 0 and math.isfinite(number)):
        raise PrimitiveError(f"primitive outside (0, inf): {value!r}")
    return number


def assemble(contract: Contract, records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    arms = {contract.reference, contract.challenger}
    by_group: dict[tuple[str, str], list[tuple[Any, Any, float]]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != FIELDS:
            raise PrimitiveError("a primitive record has missing or extra fields")
        if not all(type(record[k]) in (str, int) for k in ("arm", "group", "init", "series")):
            raise PrimitiveError("identities must be plain strings or integers")
        if record["arm"] not in arms:
            raise PrimitiveError(f"undeclared arm {record['arm']!r}")
        by_group.setdefault((record["arm"], record["group"]), []).append(
            (record["init"], record["series"], _number(record["value"]))
        )
    declared_groups = {g.name for g in contract.groups}
    observed_groups = {group for _, group in by_group}
    if observed_groups - declared_groups:
        raise PrimitiveError(
            f"undeclared groups present: {sorted(map(str, observed_groups - declared_groups))}"
        )
    out: dict[tuple[str, str], np.ndarray] = {}
    init_order = list(contract.initializations)
    for group in contract.groups:
        for arm in (contract.reference, contract.challenger):
            cells = by_group.get((arm, group.name))
            if not cells:
                raise PrimitiveError(f"declared group {group.name} has no primitives for {arm}")
            pairs = [(i, s) for i, s, _ in cells]
            if len(set(pairs)) != len(pairs):
                raise PrimitiveError(f"{group.name}/{arm}: a cell appears more than once")
            if any(isinstance(i, bool) for i, _ in pairs) or {i for i, _ in pairs} != set(init_order):
                raise PrimitiveError(f"{group.name}/{arm}: initializations differ from the declared set")
            if any(isinstance(s, bool) for _, s in pairs) or {s for _, s in pairs} != set(group.series):
                raise PrimitiveError(f"{group.name}/{arm}: series differ from the declared set")
            if len(pairs) != len(init_order) * len(group.series):
                raise PrimitiveError(f"{group.name}/{arm}: the grid is incomplete")
            rank_i = {v: k for k, v in enumerate(init_order)}
            rank_s = {v: k for k, v in enumerate(group.series)}
            ordered = sorted(cells, key=lambda c: (rank_i[c[0]], rank_s[c[1]]))
            out[(arm, group.name)] = np.array([c[2] for c in ordered]).reshape(
                len(init_order), len(group.series)
            )
    return out


def _status(rule: str, lower: float | None, upper: float | None, threshold: float) -> str:
    if lower is None or upper is None:
        return "INSUFFICIENT_EVIDENCE"
    if rule == "noninferior":
        if upper <= threshold:
            return "PASS"
        return "FAIL" if lower > threshold else "INCONCLUSIVE"
    if upper < threshold:
        return "PASS"
    return "FAIL" if lower >= threshold else "INCONCLUSIVE"


def _verdict(statuses: Sequence[str]) -> str:
    if "FAIL" in statuses:
        return "REJECT"
    if statuses and set(statuses) == {"PASS"}:
        return "PROMOTE"
    return "INSUFFICIENT_EVIDENCE" if "INSUFFICIENT_EVIDENCE" in statuses or not statuses else "INCONCLUSIVE"


def recompute(contract: Contract, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grids = assemble(contract, records)
    ref_name, cha_name = contract.reference, contract.challenger
    point_logs = [
        math.log(float(np.mean(grids[(cha_name, g.name)])) / float(np.mean(grids[(ref_name, g.name)])))
        for g in contract.groups
    ]
    geometric = math.exp(math.fsum(point_logs) / len(point_logs))
    n_init = len(contract.initializations)
    lower: float | None = None
    upper: float | None = None
    if n_init >= 2:
        # A different stream from the primary's: the seed is combined with a fixed salt.
        rng = np.random.default_rng([contract.seed, 0x1DE9E7])
        draws = contract.draws
        wi = rng.multinomial(n_init, np.full(n_init, 1.0 / n_init), size=draws).astype(float)
        logs = np.zeros(draws)
        for group in contract.groups:
            ref = grids[(ref_name, group.name)]
            cha = grids[(cha_name, group.name)]
            n_series = ref.shape[1]
            if group.design == "crossed":
                ws = rng.multinomial(n_series, np.full(n_series, 1.0 / n_series), size=draws).astype(float)
            else:
                ws = np.ones((draws, 1))
            num = np.einsum("di,is,ds->d", wi, cha, ws)
            den = np.einsum("di,is,ds->d", wi, ref, ws)
            logs += np.log(num / den)  # weights sum identically in num and den, so they cancel
        values = np.exp(logs / len(contract.groups))
        tail = (1.0 - contract.confidence) / 2.0
        lower = float(np.quantile(values, tail))
        upper = float(np.quantile(values, 1.0 - tail))
    statuses = {c.name: _status(c.rule, lower, upper, c.threshold) for c in contract.criteria}
    return {
        "implementation": "independent:count-weighted",
        "geometric_ratio": geometric,
        "interval_status": "MEASURED" if lower is not None else "INSUFFICIENT_EVIDENCE",
        "lower": lower,
        "upper": upper,
        "criteria": statuses,
        "verdict": _verdict(list(statuses.values())),
    }
