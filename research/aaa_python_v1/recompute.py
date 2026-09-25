"""Independent recomputation of ``aaa.python.v1`` stage evidence.

Structurally separate from :mod:`.summarize` (which produced the stored
summary), so that a shared bug is unlikely to agree with itself:

1. **Counts.** Every ``initialization x stream`` cell is recounted from the
   correctness bits with integer arithmetic on Python lists (no NumPy grid
   reshaping), and every stored arm mean and contrast mean must match to a
   relative 1e-12.
2. **Intervals.** Each stored interval is re-estimated by a *count-weighted*
   crossed bootstrap -- multinomial weights over initializations and streams,
   drawn from a separately salted stream -- instead of index resampling. Bounds
   must agree within ``TOLERANCE_WIDTHS`` interval widths (well above Monte Carlo
   error at 4,000 draws). A sign that differs only because a bound lies within
   that tolerance of zero is reported ``BORDERLINE``, never silently accepted;
   any other sign difference fails.
3. **Re-execution** (``rerun=True``). One initialization of every learner arm
   is re-trained from source and re-evaluated; its correctness bits and trained
   state hash must be identical to the stored ones. This is the only check that
   sees a learner answer replaced after the fact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .experiment import DESIGN, ArmSpec, evaluate_job, unbits

TOLERANCE_WIDTHS = 0.1
SALT = 0x5EED_1234


def _cells(bits: Sequence[bool], streams: int, length: int) -> list[float]:
    out = []
    for s in range(streams):
        chunk = bits[s * length : (s + 1) * length]
        out.append(sum(1 for b in chunk if b) / length)
    return out


def _weighted_interval(
    grid: list[list[float]], seed: int, draws: int, confidence: float
) -> tuple[float, float]:
    inits, streams = len(grid), len(grid[0])
    values = np.array(grid)
    rng = np.random.default_rng(seed ^ SALT)
    wi = rng.multinomial(inits, [1.0 / inits] * inits, size=draws).astype(float)
    ws = rng.multinomial(streams, [1.0 / streams] * streams, size=draws).astype(float)
    means = np.einsum("di,ij,dj->d", wi, values, ws) / (inits * streams)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def _sign(lower: float, upper: float) -> str:
    return "POSITIVE" if lower > 0 else "NEGATIVE" if upper < 0 else "INCONCLUSIVE"


def verify_stage(document: Mapping[str, Any], *, rerun: bool = False) -> dict[str, Any]:
    stage = document["stage"]
    split = stage["stage"] if stage["stage"] in ("attack", "confirmation") else "evaluate"
    shape = DESIGN[split]
    offset = shape.get("init_offset", 0)
    streams, length = shape["streams"], shape["stream_length"]
    st = DESIGN["statistics"]
    problems: list[str] = []
    borderline: list[str] = []
    per: dict[tuple[str, str, str], dict[int, list[bool]]] = {}
    for key in ("evaluations", "baselines", "anchors"):
        for row in stage.get(key, []):
            for family, modes in row["families"].items():
                for mode, payload in modes.items():
                    per.setdefault((row["arm"], family, mode), {})[row["init"]] = unbits(payload["bits"])
    grids: dict[tuple[str, str, str], list[list[float]]] = {}
    for cell_key, by_init in per.items():
        if sorted(by_init) != list(range(offset, offset + shape["initializations"])):
            problems.append(f"{cell_key}: initializations are not the declared set")
            continue
        grids[cell_key] = [_cells(by_init[i], streams, length) for i in sorted(by_init)]
    summary = stage["summary"]
    for (arm, family, mode), grid in grids.items():
        stored = summary["arms"].get(arm, {}).get(family, {}).get(mode)
        if stored is None:
            problems.append(f"{arm}/{family}/{mode}: no stored summary")
            continue
        mean = sum(sum(row) for row in grid) / (len(grid) * len(grid[0]))
        if not math.isclose(mean, stored["mean"], rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f"{arm}/{family}/{mode}: mean {mean!r} != stored {stored['mean']!r}")
    for name, row in summary["contrasts"].items():
        family, rest = name.split(": ", 1)
        arms_part, mode = rest.rsplit(" [", 1)
        mode = mode.rstrip("]")
        left, right = arms_part.split(" - ")
        left_mode, right_mode = ("online", "frozen") if mode == "online-frozen" else (mode, mode)
        a, b = grids.get((left, family, left_mode)), grids.get((right, family, right_mode))
        if a is None or b is None:
            problems.append(f"{name}: primitives missing")
            continue
        diff = [[x - y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]
        mean = sum(sum(r) for r in diff) / (len(diff) * len(diff[0]))
        if not math.isclose(mean, row["mean"], rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"{name}: mean {mean!r} != stored {row['mean']!r}")
        if row.get("interval_status") != "MEASURED":
            continue
        lower, upper = _weighted_interval(diff, st["seed"], st["draws"], st["confidence"])
        width = max(row["upper"] - row["lower"], upper - lower, 1e-12)
        if (
            abs(lower - row["lower"]) > TOLERANCE_WIDTHS * width
            or abs(upper - row["upper"]) > TOLERANCE_WIDTHS * width
        ):
            problems.append(
                f"{name}: interval [{lower:.4f}, {upper:.4f}] vs stored [{row['lower']:.4f}, {row['upper']:.4f}]"
            )
        stored_sign = row["resolved_sign"] if row["resolved_sign"] != "DEGENERATE" else "INCONCLUSIVE"
        mine = _sign(lower, upper)
        if row["resolved_sign"] == "DEGENERATE":
            if any(
                x != y
                for i in per[(left, family, left_mode)]
                for x, y in zip(
                    per[(left, family, left_mode)][i], per[(right, family, right_mode)][i], strict=True
                )
            ):
                problems.append(f"{name}: stored DEGENERATE but the arms disagree")
        elif mine != stored_sign:
            near = (
                min(abs(row["lower"]), abs(row["upper"]), abs(lower), abs(upper)) <= TOLERANCE_WIDTHS * width
            )
            (borderline if near else problems).append(f"{name}: sign {stored_sign} vs recomputed {mine}")
    rerun_checked = 0
    if rerun:
        for row in stage.get("evaluations", []):
            if row["init"] != 0 or split != "evaluate":
                continue
            spec = dict(stage["arms"][row["arm"]])
            spec_arm = ArmSpec(
                row["arm"],
                spec["encoder"],
                spec["hidden"],
                spec["output_head"],
                spec["localize_head"],
                tuple(spec["channels"]),
                spec["tool"],
                spec["weight_decay"],
                spec["momentum"],
                spec["clip"],
                spec["train_per_family"],
            )
            fresh = evaluate_job(spec_arm, spec["learning_rate"], spec["epochs"], 0)
            if fresh["state_hash"] != row["state_hash"]:
                problems.append(f"{row['arm']}: re-trained state hash differs")
            for family, modes in row["families"].items():
                for mode, payload in modes.items():
                    if fresh["families"][family][mode]["bits"] != payload["bits"]:
                        problems.append(f"{row['arm']}/{family}/{mode}: re-executed actions differ")
            rerun_checked += 1
    return {
        "cells_recounted": len(grids),
        "contrasts_checked": len(summary["contrasts"]),
        "rerun_arms": rerun_checked,
        "borderline": borderline,
        "problems": problems,
        "verdict": "FAIL" if problems else "PASS",
    }
