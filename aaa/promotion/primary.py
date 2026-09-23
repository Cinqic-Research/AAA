"""Primary evaluator for ``aaa.promotion.crossed.v1``: direct index-based resampling.

Implements the estimand in :mod:`aaa.promotion.contract` by *selecting*
resampled rows and columns of each ``initialization x series`` grid. The
independent recomputation (:mod:`aaa.promotion.independent`) implements the
same estimand with multinomial count weights and its own validation; the two
share the contract and nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .contract import RECORD_KEYS, Contract, PrimitiveError, combine, criterion_status

CHUNK = 256


def build_grids(contract: Contract, records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    """``{group: {"reference": grid, "challenger": grid}}``, refusing anything off-design."""

    arms = (contract.reference, contract.challenger)
    declared = {g.name: g for g in contract.groups}
    init_index = {init: i for i, init in enumerate(contract.initializations)}
    series_index = {g.name: {s: j for j, s in enumerate(g.series)} for g in contract.groups}
    grids = {
        g.name: {arm: np.full((len(init_index), len(g.series)), np.nan) for arm in arms}
        for g in contract.groups
    }
    seen: set[tuple[Any, ...]] = set()
    for position, record in enumerate(records):
        if not isinstance(record, Mapping) or tuple(sorted(record)) != tuple(sorted(RECORD_KEYS)):
            raise PrimitiveError(f"record {position}: fields must be exactly {RECORD_KEYS}")
        arm, group, init, series, value = (record[k] for k in RECORD_KEYS)
        if arm not in arms:
            raise PrimitiveError(f"record {position}: arm {arm!r} is not declared")
        if group not in declared:
            raise PrimitiveError(f"record {position}: group {group!r} is not declared")
        if isinstance(init, bool) or init not in init_index:
            raise PrimitiveError(f"record {position}: initialization {init!r} is not declared")
        if isinstance(series, bool) or series not in series_index[group]:
            raise PrimitiveError(f"record {position}: series {series!r} is not declared for {group}")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PrimitiveError(f"record {position}: value must be a number, got {type(value).__name__}")
        if not math.isfinite(value) or value <= 0:
            raise PrimitiveError(f"record {position}: value must be finite and positive, got {value!r}")
        key = (arm, group, init, series)
        if key in seen:
            raise PrimitiveError(f"record {position}: duplicate cell {key}")
        seen.add(key)
        grids[group][arm][init_index[init], series_index[group][series]] = float(value)
    for group, pair in grids.items():
        for arm, grid in pair.items():
            if np.isnan(grid).any():
                missing = int(np.isnan(grid).sum())
                raise PrimitiveError(f"group {group}, arm {arm}: {missing} declared cells are missing")
    return grids


def evaluate(contract: Contract, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grids = build_grids(contract, records)
    names = [g.name for g in contract.groups]
    ratios = {
        name: float(grids[name][contract.challenger].mean() / grids[name][contract.reference].mean())
        for name in names
    }
    point = float(math.exp(sum(math.log(r) for r in ratios.values()) / len(ratios)))
    n_init = len(contract.initializations)
    result: dict[str, Any] = {
        "implementation": "primary:index-resampling",
        "geometric_ratio": point,
        "group_ratios": ratios,
        "scope": contract.scope,
        "conditional_groups": contract.conditional_groups,
    }
    if n_init < 2:
        result.update(interval_status="INSUFFICIENT_EVIDENCE", lower=None, upper=None)
    else:
        rng = np.random.default_rng(contract.seed)
        logs = np.empty(contract.draws)
        designs = {g.name: g.design for g in contract.groups}
        for start in range(0, contract.draws, CHUNK):
            size = min(CHUNK, contract.draws - start)
            rows = rng.integers(0, n_init, size=(size, n_init))
            total = np.zeros(size)
            for name in names:
                ref = grids[name][contract.reference]
                cha = grids[name][contract.challenger]
                if designs[name] == "crossed":
                    cols = rng.integers(0, ref.shape[1], size=(size, ref.shape[1]))
                    num = cha[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
                    den = ref[rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
                else:  # the sole observed series is held fixed
                    num = cha[rows, 0].mean(axis=1)
                    den = ref[rows, 0].mean(axis=1)
                total += np.log(num / den)
            logs[start : start + size] = total / len(names)
        values = np.exp(logs)
        alpha = (1.0 - contract.confidence) / 2.0
        result.update(
            interval_status="MEASURED",
            lower=float(np.quantile(values, alpha)),
            upper=float(np.quantile(values, 1.0 - alpha)),
        )
    statuses = {
        c.name: criterion_status(c.rule, result["lower"], result["upper"], c.threshold)
        for c in contract.criteria
    }
    result["criteria"] = statuses
    result["verdict"] = combine(list(statuses.values()))
    return result
