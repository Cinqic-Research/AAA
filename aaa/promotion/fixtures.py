"""Deterministic synthetic primitives for testing ``aaa.promotion.crossed.v1``.

These are fixtures, not evidence: seeded, positive, lognormal-like errors with
an initialization effect, a series effect and a declared multiplicative
challenger effect, so a test knows which verdict each case should reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .contract import Contract, Criterion, GroupDeclaration


def contract(
    group_series: Sequence[int],
    *,
    initializations: int = 8,
    threshold: float = 1.02,
    rule: str = "noninferior",
    seed: int = 20260923,
    draws: int = 20000,
) -> Contract:
    """One group per entry of ``group_series``; a single-series group is declared conditional."""

    groups = tuple(
        GroupDeclaration(
            name=f"g{k}",
            design="crossed" if count >= 2 else "conditional_on_single_series",
            series=tuple(f"g{k}:s{j}" for j in range(count)),
        )
        for k, count in enumerate(group_series)
    )
    return Contract(
        reference="reference",
        challenger="challenger",
        initializations=tuple(range(1000, 1000 + initializations)),
        groups=groups,
        criteria=(Criterion("external_noninferiority", rule, threshold),),
        seed=seed,
        draws=draws,
    )


def primitives(
    declared: Contract, *, effect: float = 1.0, noise: float = 0.15, seed: int = 7
) -> list[dict[str, Any]]:
    """Reference and challenger cells for every declared group, with ``challenger = effect x reference x noise``."""

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    for group in declared.groups:
        init_effect = rng.normal(0.0, 0.2, size=len(declared.initializations))
        series_effect = rng.normal(0.0, 0.3, size=len(group.series))
        for i, init in enumerate(declared.initializations):
            for j, series in enumerate(group.series):
                base = float(np.exp(init_effect[i] + series_effect[j] + rng.normal(0.0, noise)))
                paired = float(base * effect * np.exp(rng.normal(0.0, noise)))
                records.append(
                    {
                        "arm": declared.reference,
                        "group": group.name,
                        "init": init,
                        "series": series,
                        "value": base,
                    }
                )
                records.append(
                    {
                        "arm": declared.challenger,
                        "group": group.name,
                        "init": init,
                        "series": series,
                        "value": paired,
                    }
                )
    return records
