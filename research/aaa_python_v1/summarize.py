"""Every reported ``aaa.python.v1`` development number, derived from retained primitives only.

The primitives are per-task correctness bits (and log-losses for learners), in
evaluation order, per ``(arm, initialization, family, mode)``. A stream is a
fixed slice of that order, so the ``initialization x stream`` cells, per-slice
accuracies, disagreement counts and every interval are recomputed from them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import generator
from .experiment import DESIGN, FAMILIES, layout, unbits
from .stats import crossed, holm, resolved_sign, variance_components

Contrast = tuple[
    str, str, str
]  # (left arm, right arm, mode); mode "online-frozen" = left online - right frozen


def _modes(mode: str) -> tuple[str, str]:
    if mode == "online-frozen":
        return "online", "frozen"
    if mode in ("frozen", "online"):
        return mode, mode
    raise ValueError(f"unknown contrast mode {mode!r}")


def _slices() -> list[str]:
    start, _ = layout("evaluate")
    count = DESIGN["evaluate"]["streams"] * DESIGN["evaluate"]["stream_length"]
    return [generator.slice_of("development", index) for index in range(start, start + count)]


def primitives(stage: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[int, list[bool]]]:
    """``{(arm, family, mode): {init: bits}}`` for learners, baselines and anchors."""

    out: dict[tuple[str, str, str], dict[int, list[bool]]] = {}
    for key in ("evaluations", "baselines", "anchors"):
        for row in stage.get(key, []):
            for family, modes in row["families"].items():
                for mode, payload in modes.items():
                    cell = out.setdefault((row["arm"], family, mode), {})
                    if row["init"] in cell:
                        raise ValueError(
                            f"duplicate primitives for {row['arm']}/{family}/{mode}/{row['init']}"
                        )
                    cell[row["init"]] = unbits(payload["bits"])
    return out


def nll(stage: Mapping[str, Any]) -> dict[tuple[str, str, str], float]:
    sums: dict[tuple[str, str, str], list[float]] = {}
    for row in stage.get("evaluations", []):
        for family, modes in row["families"].items():
            for mode, payload in modes.items():
                if "nll" in payload:
                    sums.setdefault((row["arm"], family, mode), []).extend(payload["nll"])
    return {key: float(np.mean(v)) for key, v in sums.items()}


def grid_of(per_init: Mapping[int, Sequence[bool]]) -> np.ndarray:
    streams, length = DESIGN["evaluate"]["streams"], DESIGN["evaluate"]["stream_length"]
    inits = sorted(per_init)
    grid = np.empty((len(inits), streams))
    for r, init in enumerate(inits):
        values = per_init[init]
        if len(values) != streams * length:
            raise ValueError(f"initialization {init}: {len(values)} tasks, expected {streams * length}")
        grid[r] = np.asarray(values, dtype=float).reshape(streams, length).mean(axis=1)
    return grid


def summarize(
    stage: Mapping[str, Any], primary: Sequence[Contrast], secondary: Sequence[Contrast] = ()
) -> dict[str, Any]:
    st = DESIGN["statistics"]
    prims = primitives(stage)
    losses = nll(stage)
    slices = _slices()
    expected_inits = set(range(DESIGN["evaluate"]["initializations"]))
    arms: dict[str, dict[str, Any]] = {}
    for (arm, family, mode), per_init in sorted(prims.items()):
        if set(per_init) != expected_inits:
            raise ValueError(
                f"{arm}/{family}/{mode}: initializations {sorted(per_init)} are not the declared set"
            )
        grid = grid_of(per_init)
        by_slice: dict[str, list[int]] = {}
        for values in per_init.values():
            for s, v in zip(slices, values, strict=True):
                counts = by_slice.setdefault(s, [0, 0])
                counts[0] += 1
                counts[1] += int(v)
        entry = crossed(grid, seed=st["seed"], draws=st["draws"], confidence=st["confidence"])
        entry["variance_components"] = variance_components(grid)
        entry["by_slice"] = {s: c[1] / c[0] for s, c in sorted(by_slice.items())}
        if (arm, family, mode) in losses:
            entry["mean_nll"] = losses[(arm, family, mode)]
        arms.setdefault(arm, {}).setdefault(family, {})[mode] = entry
    contrasts: dict[str, dict[str, Any]] = {}
    groups: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for left, right, mode in [*primary, *secondary]:
        for family in FAMILIES:
            left_mode, right_mode = _modes(mode)
            a, b = prims.get((left, family, left_mode)), prims.get((right, family, right_mode))
            if a is None or b is None:
                raise ValueError(f"declared contrast {left} - {right} ({mode}) is missing {family}")
            name = f"{family}: {left} - {right} [{mode}]"
            diff = grid_of(a) - grid_of(b)
            disagreements = sum(int(x != y) for i in a for x, y in zip(a[i], b[i], strict=True))
            result = crossed(diff, seed=st["seed"], draws=st["draws"], confidence=st["confidence"])
            result["disagreements"] = disagreements
            result["resolved_sign"] = resolved_sign(result, disagreements=disagreements)
            result["variance_components"] = variance_components(diff)
            result["primary"] = (left, right, mode) in primary
            contrasts[name] = result
            if result["primary"]:
                groups.setdefault((left, right, mode), {})[name] = diff
    # Holm across the five families of each primary contrast (the declared capacity rule).
    for grids in groups.values():
        adjusted = holm(contrasts, grids, seed=st["seed"], draws=st["draws"], alpha=st["holm_alpha"])
        for name, row in adjusted.items():
            contrasts[name]["holm"] = row
    return {"arms": arms, "contrasts": contrasts}


def summarize_adapt(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per arm and family: online advantage on changed and control branches, their difference, forgetting."""

    st = DESIGN["statistics"]
    streams = DESIGN["adapt"]["streams"]
    out: dict[str, Any] = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["init"])
        if [r["init"] for r in mine] != list(range(DESIGN["adapt"]["initializations"])):
            raise ValueError(f"{arm}: adaptation initializations are not the declared set")
        entry: dict[str, Any] = {}
        for family in FAMILIES:

            def grid(key: str, family: str = family, mine: Sequence[Mapping[str, Any]] = mine) -> np.ndarray:
                values = np.empty((len(mine), streams))
                for r, row in enumerate(mine):
                    cells = row["families"][family]
                    if [c["stream"] for c in cells] != list(range(streams)):
                        raise ValueError("adaptation streams are not the declared set")
                    for s, cell in enumerate(cells):
                        bits = unbits(cell[key])
                        values[r, s] = 1.0 - sum(bits) / len(bits)
                return values

            changed = grid("changed_frozen") - grid("changed_online")
            control = grid("control_frozen") - grid("control_online")
            measures = {
                "online_advantage_changed": changed,
                "online_advantage_control": control,
                "difference_of_differences": changed - control,
                "forgetting_after_changed": grid("probe_after_changed") - grid("probe_before"),
                "forgetting_after_control": grid("probe_after_control") - grid("probe_before"),
            }
            fam: dict[str, Any] = {}
            for name, values in measures.items():
                result = crossed(values, seed=st["seed"], draws=st["draws"], confidence=st["confidence"])
                result["resolved_sign"] = resolved_sign(result)
                result["variance_components"] = variance_components(values)
                fam[name] = result
            entry[family] = fam
        out[arm] = entry
    return out
