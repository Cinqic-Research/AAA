"""Independent recomputation of the v2 confirmation decision from retained primitives.

Shares no decision code with :mod:`research.aaa_1k_v2.confirmation` or
:mod:`research.aaa_1k_v2.stats`:

* thresholds and criterion rules are read from the committed freeze, never
  from ``plan``;
* the bootstrap is count-weighted: each draw assigns multinomial counts to
  initializations and to streams and takes weighted means, instead of indexing
  resampled rows and columns (the two are equal in distribution, not in draws);
* the initializations and stream identifiers found in the primitives are
  checked against the seeds the frozen confirmation blocks name.

Point estimates must agree with the stored decision exactly. Interval-based
statuses can in principle differ from the stored ones when a bound lies within
Monte Carlo error of a threshold; any such disagreement is reported, never
resolved in either direction.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DRAWS_FALLBACK = 4000


def _grid(
    cells: Sequence[Mapping[str, Any]], group: str, key: str, keep: set[str] | None = None
) -> tuple[np.ndarray, list[Any], list[str]]:
    rows = [
        c
        for c in cells
        if c["group"] == group and (keep is None or c.get("series_id", c.get("stream_id")) in keep)
    ]
    inits = sorted({c["init"] for c in rows})
    streams = sorted({c.get("stream_id", c.get("series_id")) for c in rows})
    lookup = {(c["init"], c.get("stream_id", c.get("series_id"))): c for c in rows}
    grid = np.full((len(inits), len(streams)), np.nan)
    for i, init in enumerate(inits):
        for j, stream in enumerate(streams):
            cell = lookup.get((init, stream))
            if cell is not None and not cell.get("failed") and cell.get(key) is not None:
                grid[i, j] = float(cell[key])
    return grid, inits, streams


def _weighted_ratio(a: np.ndarray, b: np.ndarray, wi: np.ndarray, ws: np.ndarray) -> np.ndarray:
    weights = wi[:, :, None] * ws[:, None, :]
    return (weights * b).sum(axis=(1, 2)) / (weights * a).sum(axis=(1, 2))


def relative(a: np.ndarray, b: np.ndarray, seed: int, draws: int, confidence: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    wi = rng.multinomial(a.shape[0], [1.0 / a.shape[0]] * a.shape[0], size=draws).astype(float)
    ws = rng.multinomial(a.shape[1], [1.0 / a.shape[1]] * a.shape[1], size=draws).astype(float)
    ratios = _weighted_ratio(a, b, wi, ws) - 1.0
    alpha = (1.0 - confidence) / 2.0
    return {
        "relative": float(b.mean() / a.mean() - 1.0),
        "lower": float(np.quantile(ratios, alpha)),
        "upper": float(np.quantile(ratios, 1.0 - alpha)),
    }


def geometric(
    pairs: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int, draws: int, confidence: float
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_init = next(iter(pairs.values()))[0].shape[0]
    wi = rng.multinomial(n_init, [1.0 / n_init] * n_init, size=draws).astype(float)
    logs = np.zeros(draws)
    point_logs = []
    for name in sorted(pairs):
        a, b = pairs[name]
        ws = rng.multinomial(a.shape[1], [1.0 / a.shape[1]] * a.shape[1], size=draws).astype(float)
        logs += np.log(_weighted_ratio(a, b, wi, ws))
        point_logs.append(math.log(b.mean() / a.mean()))
    values = np.exp(logs / len(pairs))
    alpha = (1.0 - confidence) / 2.0
    return {
        "geometric_ratio": float(math.exp(np.mean(point_logs))),
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
    }


def _status_noninferior(upper: float, lower: float, margin: float) -> str:
    return "PASS" if upper <= margin else ("FAIL" if lower > margin else "INCONCLUSIVE")


def _combine(statuses: Sequence[str]) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    return "PASS" if statuses and all(s == "PASS" for s in statuses) else "INCONCLUSIVE"


def recompute(
    confirmation: Mapping[str, Any], manifest: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    from .identities import seeds_of

    frozen = manifest["frozen"]
    thresholds = frozen["thresholds"]
    stats = frozen["statistics"]
    draws = int(stats.get("bootstrap_draws", DRAWS_FALLBACK))
    confidence = float(stats["confidence"])
    challenger = frozen["challenger"]
    if challenger is None:
        return {"outcome": "NO_CHALLENGER", "agrees": confirmation["decision"]["outcome"] == "NO_CHALLENGER"}
    name = challenger["name"]
    ref = confirmation["primitives"]["c1_champion1"]
    cha = confirmation["primitives"][name]
    problems: list[str] = []
    inits = seeds_of(dict(registry), "v2-confirmation-init")
    if sorted({c["init"] for c in ref}) != sorted(inits) or sorted({c["init"] for c in cha}) != sorted(inits):
        problems.append("primitives use initializations other than the frozen block's")
    for family in frozen["families"]:
        expected = {f"{family}:{seed}" for seed in seeds_of(dict(registry), f"v2-confirmation-env.{family}")}
        seen = {c["stream_id"] for c in cha if c["group"] == family}
        if seen != expected:
            problems.append(f"{family}: streams differ from the frozen block's")
    seeds = seeds_of(dict(registry), "v2-confirmation-bootstrap")
    persistence = {
        r["stream_id"]: r["persistence"]
        for records in confirmation["baselines"]["dot"].values()
        for r in records
    }
    external_base = {
        sid: m for task in confirmation["baselines"]["external"].values() for sid, m in task.items()
    }
    factor = thresholds["divergence_factor"]

    def unstable(cell: Mapping[str, Any]) -> tuple[bool, bool]:
        if cell.get("failed"):
            return True, True
        if "stream_id" in cell:
            return False, cell["mae"] > factor * persistence[cell["stream_id"]]
        metric = "prequential_nmse" if cell["group"].startswith("narma") else "prequential_nrmse"
        value = cell.get(metric)
        base = external_base[cell["series_id"]]["persistence"][metric]
        return False, value is None or value > factor * base or cell.get("forecast_finite") is False

    ref_flags = [unstable(c) for c in ref]
    cha_flags = [unstable(c) for c in cha]
    k1 = (
        "PASS"
        if sum(f for f, _ in cha_flags) == 0 and sum(d for _, d in cha_flags) <= sum(d for _, d in ref_flags)
        else "FAIL"
    )
    v1 = [f for f in frozen["families"] if f.startswith("v1_")]
    stress = [f for f in frozen["families"] if not f.startswith("v1_")]
    k2_parts = {}
    for offset, family in enumerate(v1):
        a, _, _ = _grid(ref, family, "mae")
        b, _, _ = _grid(cha, family, "mae")
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            k2_parts[family] = {"status": "FAIL"}
            continue
        result = relative(a, b, seeds[0] % 2**32 + offset, draws, confidence)
        k2_parts[family] = {
            **result,
            "status": _status_noninferior(
                result["upper"], result["lower"], thresholds["noninferiority_margin"]
            ),
        }
    k2 = _combine([p["status"] for p in k2_parts.values()])
    pairs = {}
    for family in stress:
        a, _, _ = _grid(ref, family, "mae")
        b, _, _ = _grid(cha, family, "mae")
        pairs[family] = (a, b)
    complete = all(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) for a, b in pairs.values())
    if complete:
        k3r = geometric(pairs, seeds[1] % 2**32, draws, confidence)
        improvement = thresholds["stress_improvement"]
        k3 = (
            "PASS"
            if k3r["geometric_ratio"] <= improvement and k3r["upper"] < 1.0
            else ("FAIL" if k3r["lower"] > improvement else "INCONCLUSIVE")
        )
    else:
        k3r, k3 = {}, "FAIL"
    k4_parts = {}
    for offset, family in enumerate(stress):
        a, b = pairs[family]
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            k4_parts[family] = {"status": "FAIL"}
            continue
        result = relative(a, b, seeds[2] % 2**32 + offset, draws, confidence)
        k4_parts[family] = {
            **result,
            "status": "FAIL" if result["lower"] > thresholds["regression_guard"] else "PASS",
        }
    k4 = _combine([p["status"] for p in k4_parts.values()])
    external = list(frozen["external_tasks"])
    k5_pairs = {}
    monash = confirmation.get("monash_primitives", {})
    for task in external:
        metric = "prequential_nmse" if task.startswith("narma") else "prequential_nrmse"
        k5_pairs[task] = (_grid(ref, task, metric)[0], _grid(cha, task, metric)[0])
    if name in monash:
        for task in frozen["monash_tasks"]:
            keep = {
                sid
                for sid, record in confirmation["baselines"]["monash"][task].items()
                if record["persistence"].get("forecast_mase") is not None
                and math.isfinite(record["persistence"]["forecast_mase"])
            }
            k5_pairs[task] = (
                _grid(monash["c1_champion1"], task, "forecast_mase", keep)[0],
                _grid(monash[name], task, "forecast_mase", keep)[0],
            )
    if all(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) for a, b in k5_pairs.values()):
        k5r = geometric(k5_pairs, seeds[3] % 2**32, draws, confidence)
        k5 = _status_noninferior(k5r["upper"] - 1.0, k5r["lower"] - 1.0, thresholds["external_margin"])
    else:
        k5r, k5 = {}, "FAIL"
    statuses = {
        "K1_stability": k1,
        "K2_v1_noninferiority": k2,
        "K3_stress_improvement": k3,
        "K4_regression_guard": k4,
        "K5_external_noninferiority": k5,
    }
    outcome = {"PASS": "PROMOTE", "FAIL": "REJECT"}.get(_combine(list(statuses.values())), "INCONCLUSIVE")
    stored = confirmation["decision"]
    point_checks = {}
    if complete and "geometric_ratio" in stored["criteria"].get("K3_stress_improvement", {}):
        point_checks["K3_geometric_ratio"] = math.isclose(
            k3r["geometric_ratio"],
            stored["criteria"]["K3_stress_improvement"]["geometric_ratio"],
            rel_tol=1e-12,
        )
    if "geometric_ratio" in k5r and "geometric_ratio" in stored["criteria"].get(
        "K5_external_noninferiority", {}
    ):
        point_checks["K5_geometric_ratio"] = math.isclose(
            k5r["geometric_ratio"],
            stored["criteria"]["K5_external_noninferiority"]["geometric_ratio"],
            rel_tol=1e-12,
        )
    return {
        "schema": "aaa.1k.v2.recomputation.v1",
        "method": "count-weighted multinomial bootstrap; thresholds from the freeze; seeds checked against the registry",
        "problems": problems,
        "statuses": statuses,
        "outcome": outcome,
        "stored_outcome": stored["outcome"],
        "stored_statuses": stored["statuses"],
        "agrees": outcome == stored["outcome"] and statuses == stored["statuses"] and not problems,
        "point_estimates_agree": point_checks,
        "details": {"K2": k2_parts, "K3": k3r, "K4": k4_parts, "K5": k5r},
    }
