"""Post-confirmation capacity diagnostic (exploratory; cannot change the confirmation decision).

Question: is the failure that remains after the final 1K pass plausibly caused
by insufficient capacity, or by architecture, objective, data, optimization or
benchmark problems?

Design (fixed before it runs):

* the final 1K system's architecture family at four widths chosen to sit
  nearest 0.5K, 1K, 2K and 4K trainable parameters, everything else (feature
  set, target rule, learning rule, horizon, learning rate, clip) unchanged;
* fresh ``capacity`` identities: 5 initializations x 8 streams of every dot
  family (held-out families included: confirmation has already observed them)
  plus NARMA-10, NARMA-20 and Mackey-Glass capacity realizations;
* per family, the crossed-bootstrap relative MAE of each width against the
  ~1K width.

Verdict rule over the 26 dot families at the ~4K width:

``CAPACITY_LIMITED``      at least two thirds improve by more than 5% with the upper bound below 0
                          and improve monotonically from ~1K to ~2K to ~4K in point estimate;
``NOT_CAPACITY_LIMITED``  at most one third improve by more than 5% with the upper bound below 0;
``MIXED``                 otherwise, with the improving and non-improving families listed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from . import plan
from .arms import ArmSpec, core_spec
from .attack import evaluate_arm
from .identities import seeds_of
from .stats import relative_crossed
from .tournament import build_cells, crossed_grid

TARGETS = (500, 1000, 2000, 4000)


def width_for(arm: ArmSpec, target: int) -> int:
    """The hidden width whose parameter count is nearest ``target`` (ties to the smaller width)."""

    core = dict(arm.core)
    best, best_gap = 1, None
    for hidden in range(1, 200):
        candidate = replace(
            arm,
            core=core_spec(
                core["kind"],
                core["inputs"],
                hidden,
                core["outputs"],
                **({"readout_hidden": core["readout_hidden"]} if "readout_hidden" in core else {}),
            ),
        )
        gap = abs(candidate.build_core().parameter_count() - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = hidden, gap
    return best


def ladder(arm: ArmSpec) -> dict[int, ArmSpec]:
    core = dict(arm.core)
    extra = {"readout_hidden": core["readout_hidden"]} if "readout_hidden" in core else {}
    out = {}
    for target in TARGETS:
        hidden = width_for(arm, target)
        out[target] = replace(
            arm,
            name=f"{arm.name}@{target}",
            kind="probe",
            core=core_spec(core["kind"], core["inputs"], hidden, core["outputs"], **extra),
        )
    return out


def run_capacity(
    registry: Mapping[str, Any], arm: ArmSpec, *, workers: Any, log: Callable[[str], None] = print
) -> dict[str, Any]:
    families = list(plan.STAGE_FAMILIES["confirmation"])
    external = ("narma10", "narma20", "mackey_glass17")
    cells = build_cells(registry, "capacity", families, external)
    arms = ladder(arm)
    primitives = {}
    for target, member in arms.items():
        primitives[target] = evaluate_arm(member, cells, device="cpu", workers=workers, external=external)
        log(
            f"  {member.name} ({member.build_core().parameter_count()} parameters): {len(primitives[target])} cells"
        )
    bootstrap = seeds_of(dict(registry), "v2-capacity-bootstrap")
    comparisons: dict[str, dict[str, Any]] = {}
    for index, family in enumerate([*families, *external]):
        metric = (
            "mae"
            if family in families
            else ("prequential_nmse" if family.startswith("narma") else "prequential_nrmse")
        )
        base, _, _ = crossed_grid(primitives[1000], family, metric)
        row: dict[str, Any] = {}
        for target in (500, 2000, 4000):
            other, _, _ = crossed_grid(primitives[target], family, metric)
            if np.all(np.isfinite(base)) and np.all(np.isfinite(other)):
                row[str(target)] = relative_crossed(
                    base, other, seed=bootstrap[0] % 2**32 + 10 * index + target, draws=plan.BOOTSTRAP_DRAWS
                )
            else:
                row[str(target)] = {"interval_status": "INCOMPLETE"}
        comparisons[family] = row

    def improves(entry: Mapping[str, Any]) -> bool:
        return (
            entry.get("interval_status") == "MEASURED" and entry["relative"] < -0.05 and entry["upper"] < 0.0
        )

    improving = [f for f in families if improves(comparisons[f]["4000"])]
    monotone = [
        f
        for f in improving
        if comparisons[f]["2000"].get("interval_status") == "MEASURED"
        and comparisons[f]["4000"]["relative"] < comparisons[f]["2000"]["relative"] < 0.0
    ]
    share = len(improving) / len(families)
    if share >= 2 / 3 and len(monotone) >= 2 / 3 * len(families):
        verdict = "CAPACITY_LIMITED"
    elif share <= 1 / 3:
        verdict = "NOT_CAPACITY_LIMITED"
    else:
        verdict = "MIXED"
    return {
        "schema": "aaa.1k.v2.capacity.v1",
        "stage": "capacity",
        "base_arm": arm.to_dict(),
        "ladder": {
            str(t): {"arm": a.to_dict(), "parameters": a.build_core().parameter_count()}
            for t, a in arms.items()
        },
        "comparisons": comparisons,
        "improving_at_4k": improving,
        "monotone_improving": monotone,
        "verdict": verdict,
        "primitives": {str(t): p for t, p in primitives.items()},
    }
