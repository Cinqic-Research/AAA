"""Iteration 0004, diagnosis round 1: which gradient does online TBPTT take, and does it matter for M2?

Audit finding R-02 (``AAA_Research_Audit_2026-09-21``): the champion's
``backward()`` walks cached activations produced under older parameters
through the *current* recurrent matrices. :mod:`research.aaa_1k_loop.tbptt`
defines the alternatives exactly and proves each against a finite-difference
reference. This round measures, on realistic streams, how far the live rule
is from them and whether switching rules changes the long-horizon runaway
(M2).

Everything below -- conditions, arms, divergence definition, hypotheses and
thresholds -- was committed before the diagnostic block was run. The scratch
block exists only to shake the instrument down (does it run, how long does it
take) and is never cited as evidence.

Arms
----
``gru_live``
    :class:`~research.aaa_1k_loop.tbptt.ComparingGRU`: bitwise the champion
    (tested), additionally reporting the snapshot, replay and T=1 update on its
    exact state at every learning step.
``gru_snapshot``, ``gru_replay``, ``gru_t1``
    the champion's construction and initialization learning with that rule.
``persistence``
    the divergence reference.

A cell *diverges* for an arm when its mean error exceeds twice persistence on
the same stream, exactly as in diagnosis round 3.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from research.aaa_1k.agents import PersistenceAgent
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.stats import crossed_paired_difference
from research.aaa_1k.streams import coarse_speed_stream, motion_compat_stream, occlusion_stream

from .arms import CHAMPION_CONFIGURATION
from .develop import plan_streams
from .harness import Cell, InstrumentedAgent, arm_errors, run_cells
from .identities import block_seeds, find_block, require_usable
from .tbptt import ComparingGRU, build_rule_model

ITERATION_ID = "aaa1k-loop-0004"
DIAGNOSIS4_SCHEMA = "aaa.loop.diagnosis.v4"
DIAGNOSTIC_BLOCK = "aaa1k-loop-0004/diagnostic/tbptt"
SCRATCH_BLOCK = "aaa1k-loop-0004/diagnostic/scratch"

LONG_STREAMS = 16
PLAN_STREAMS_PER_ENTRY = 8
PLAN_ENTRIES = 6
BLOCK_SIZE = LONG_STREAMS + PLAN_STREAMS_PER_ENTRY * PLAN_ENTRIES
INITIALIZATIONS = 5
LONG_STEPS = 1120
DIVERGENCE_FACTOR = 2.0
MINIMUM_LIVE_DIVERGENCES = 5
SEGMENTS = 8

LONG_CONDITIONS = ("long_coarse_no_switch", "long_coarse_switching", "long_bouncing", "long_occlusion")
COARSE_CONDITIONS = ("long_coarse_no_switch", "long_coarse_switching")
RULE_ARMS = ("gru_live", "gru_snapshot", "gru_replay", "gru_t1")
ALTERNATIVES = ("snapshot", "replay", "t1")

NEGLIGIBLE_MEDIAN = 1e-3
NEGLIGIBLE_Q99 = 1e-2
MATERIAL_MEDIAN = 1e-2
EQUIVALENCE_BAND = 0.01
"""Plan-entry MAE: a rule is EQUIVALENT to live when the interval of the relative difference lies within ±1%."""

HYPOTHESES: tuple[dict[str, str], ...] = (
    {
        "id": "H20",
        "name": "live is numerically the realized-trajectory gradient where the champion is stable",
        "statement": (
            "on cells where the live champion does not diverge, the stale recurrent matrices change the "
            "update negligibly: the live update is the snapshot update to within numerical noise"
        ),
        "prediction": (
            f"pooled over non-diverged cells, the median per-step relative difference between the live and "
            f"snapshot update vectors is <= {NEGLIGIBLE_MEDIAN:g} and its 99th percentile <= {NEGLIGIBLE_Q99:g}"
        ),
        "contradicted_by": f"a pooled median > {MATERIAL_MEDIAN:g}",
    },
    {
        "id": "H21",
        "name": "the stale-matrix mismatch drives M2",
        "statement": (
            "backpropagating old activations through current matrices produces a biased update that "
            "feeds the long-horizon runaway"
        ),
        "prediction": "the snapshot rule diverges in <= 20% as many long coarse cells as the live rule",
        "contradicted_by": "the snapshot rule diverging in >= 50% as many long coarse cells",
    },
    {
        "id": "H22",
        "name": "M2 does not need multi-step credit assignment",
        "statement": "the runaway arises in the one-step update; truncated backpropagation beyond it is not required",
        "prediction": "the T=1 rule diverges in >= 50% as many long coarse cells as the live rule",
        "contradicted_by": "the T=1 rule diverging in <= 20% as many long coarse cells",
    },
    {
        "id": "H23",
        "name": "chunk-consistent credit prevents M2",
        "statement": (
            "gradients recomputed under one parameter value across the window (constant-parameter chunked "
            "TBPTT) remove the runaway"
        ),
        "prediction": "the replay rule diverges in <= 20% as many long coarse cells as the live rule",
        "contradicted_by": "the replay rule diverging in >= 50% as many long coarse cells",
    },
)


# ----------------------------------------------------------------------
# arms
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RuleArms:
    """Picklable factory for the four rule arms plus persistence."""

    def __call__(self, seed: int) -> Sequence[Any]:
        live = ComparingGRU(seed=seed, **CHAMPION_CONFIGURATION)
        agents: list[Any] = [InstrumentedAgent(live, name="gru_live")]
        for rule in ALTERNATIVES:
            agents.append(
                InstrumentedAgent(build_rule_model(rule, seed, CHAMPION_CONFIGURATION), name=f"gru_{rule}")
            )
        agents.append(PersistenceAgent())
        return agents


# ----------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------
def long_stream(condition: str, seed: int, *, steps: int = LONG_STEPS) -> Any:
    if condition == "long_coarse_no_switch":
        return coarse_speed_stream(seed, steps=steps, regime_length=10_000)
    if condition == "long_coarse_switching":
        return coarse_speed_stream(seed, steps=steps, regime_length=60)
    if condition == "long_bouncing":
        return motion_compat_stream("bouncing", seed, steps=steps, change_step=None)
    if condition == "long_occlusion":
        return occlusion_stream(seed, steps=steps)
    raise ValueError(condition)


def diagnostic_cells(
    ledger: Mapping[str, Any],
    *,
    block_id: str = DIAGNOSTIC_BLOCK,
    long_streams: int = LONG_STREAMS,
    per_entry: int = PLAN_STREAMS_PER_ENTRY,
    initializations: int = INITIALIZATIONS,
    steps: int = LONG_STEPS,
) -> list[Cell]:
    block = require_usable(ledger, block_id, purpose="selection")
    if block["role"] != "diagnostic":
        raise ValueError("diagnosis round 4 must run on a diagnostic block")
    seeds = block_seeds(find_block(ledger, block_id))
    if len(seeds) < long_streams + per_entry * PLAN_ENTRIES:
        raise ValueError("the block holds too few seeds for this design")
    long_seeds, plan_seeds = (
        seeds[:long_streams],
        seeds[long_streams : long_streams + per_entry * PLAN_ENTRIES],
    )
    streams: list[tuple[str, Any]] = [
        (condition, long_stream(condition, seed, steps=steps))
        for condition in LONG_CONDITIONS
        for seed in long_seeds
    ]
    streams.extend((f"plan:{entry}", stream) for entry, stream in plan_streams(plan_seeds, per_entry))
    return [
        Cell(condition, init, derive_seed("model_init", init), stream)
        for condition, stream in streams
        for init in range(initializations)
    ]


# ----------------------------------------------------------------------
# reduction
# ----------------------------------------------------------------------
def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": None, "q90": None, "q99": None, "max": None, "count": 0}
    return {
        "median": float(np.median(finite)),
        "q90": float(np.quantile(finite, 0.9)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "count": int(finite.size),
    }


HISTOGRAM_LOW = -16.0
HISTOGRAM_WIDTH = 0.05
HISTOGRAM_BINS = 360
"""log10 bins of width 0.05 over [1e-16, 1e2); exact zeros and overflow are counted separately."""


def log_histogram(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    zeros = int(np.sum(finite == 0.0))
    positive = finite[finite > 0.0]
    index = np.floor((np.log10(positive) - HISTOGRAM_LOW) / HISTOGRAM_WIDTH).astype(int)
    below = int(np.sum(index < 0))
    above = int(np.sum(index >= HISTOGRAM_BINS))
    inside = index[(index >= 0) & (index < HISTOGRAM_BINS)]
    bins, counts = np.unique(inside, return_counts=True)
    return {
        "zeros": zeros,
        "below": below,
        "above": above,
        "nonfinite": int(values.size - finite.size),
        "bins": {str(int(b)): int(c) for b, c in zip(bins, counts, strict=True)},
    }


def histogram_quantile(histograms: Sequence[Mapping[str, Any]], q: float) -> float | None:
    """Pooled quantile, reported at the bin's *upper* edge (conservative for an upper threshold)."""

    counts = np.zeros(HISTOGRAM_BINS, dtype=np.int64)
    zeros = below = above = 0
    for h in histograms:
        zeros += h["zeros"]
        below += h["below"]
        above += h["above"]
        for b, c in h["bins"].items():
            counts[int(b)] += c
    total = zeros + below + above + int(counts.sum())
    if total == 0:
        return None
    rank = q * (total - 1)
    cumulative = zeros + below
    if rank < zeros:
        return 0.0
    if rank < cumulative:
        return 10.0**HISTOGRAM_LOW
    for index in range(HISTOGRAM_BINS):
        cumulative += int(counts[index])
        if rank < cumulative:
            return 10.0 ** (HISTOGRAM_LOW + (index + 1) * HISTOGRAM_WIDTH)
    return float("inf")


def pooled(histograms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "median": histogram_quantile(histograms, 0.5),
        "q90": histogram_quantile(histograms, 0.9),
        "q99": histogram_quantile(histograms, 0.99),
        "cells": len(histograms),
        "steps": sum(h["zeros"] + h["below"] + h["above"] + sum(h["bins"].values()) for h in histograms),
        "nonfinite": sum(h["nonfinite"] for h in histograms),
    }


COMPARISON_FIELDS = (
    "window_parameter_drift",
    "live_gradient_norm",
    "live_update_norm",
    *(
        f"{rule}_{kind}_{metric}"
        for rule in ALTERNATIVES
        for kind in ("gradient", "update")
        for metric in ("relative_difference", "cosine")
    ),
)


def reduce_cell(_cell: Cell, result: RunResult, agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    live_model = next(agent.model for agent in agents if agent.name == "gru_live")
    comparisons = live_model.comparisons
    columns = {
        name: np.asarray([record[name] for record in comparisons], dtype=float) for name in COMPARISON_FIELDS
    }
    # Time-resolved medians: equal learning-step segments, so a late runaway is not averaged away.
    bounds = np.linspace(0, len(comparisons), SEGMENTS + 1).astype(int)
    segments = [
        {
            name: (float(np.median(columns[name][lo:hi])) if hi > lo else None)
            for name in (
                "window_parameter_drift",
                "live_gradient_norm",
                "snapshot_update_relative_difference",
                "replay_update_relative_difference",
                "t1_update_relative_difference",
            )
        }
        for lo, hi in itertools.pairwise(bounds)
    ]
    quarter = max(1, len(result.steps) // 4)
    by_name = {agent.name: agent for agent in agents}
    return {
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "last_quarter_mae": {name: float(np.mean(values[-quarter:])) for name, values in errors.items()},
        "clip_fraction": {
            name: (
                by_name[name].model.clip_events / by_name[name].model.update_count
                if by_name[name].model.update_count
                else 0.0
            )
            for name in RULE_ARMS
            if name in by_name
        },
        "comparison": {name: _quantiles(values) for name, values in columns.items()},
        "comparison_segments": segments,
        # Retained for the pooled H20 statistic without storing every step.
        "snapshot_update_histogram": log_histogram(columns["snapshot_update_relative_difference"]),
    }


def run_diagnosis4(
    ledger: Mapping[str, Any], *, workers: int | None = None, **design: Any
) -> list[dict[str, Any]]:
    return run_cells(
        diagnostic_cells(ledger, **design), RuleArms(), reduce_cell, isolate_failures=True, workers=workers
    )


# ----------------------------------------------------------------------
# adjudication
# ----------------------------------------------------------------------
def diverged(record: Mapping[str, Any], arm: str) -> bool:
    if arm in record.get("failures", {}):
        return True
    return record["mae"][arm] > DIVERGENCE_FACTOR * record["mae"]["persistence"]


def _ratio_verdict(count: int, reference: int, *, supported_below: bool) -> tuple[str, float | None]:
    if reference < MINIMUM_LIVE_DIVERGENCES:
        return "INSUFFICIENT_EVIDENCE", None
    ratio = count / reference
    if supported_below:
        return ("SUPPORTED" if ratio <= 0.2 else "CONTRADICTED" if ratio >= 0.5 else "INCONCLUSIVE"), ratio
    return ("SUPPORTED" if ratio >= 0.5 else "CONTRADICTED" if ratio <= 0.2 else "INCONCLUSIVE"), ratio


def _crossed(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    inits = sorted({r["init_index"] for r in records})
    streams = sorted({r["stream_id"] for r in records})
    index = {(r["init_index"], r["stream_id"]): r for r in records}
    first = np.array([[index[(i, s)]["mae"]["gru_live"] for s in streams] for i in inits])
    second = np.array([[index[(i, s)]["mae"][arm] for s in streams] for i in inits])
    summary = crossed_paired_difference(first, second)
    live_mean = float(np.mean(first))
    low, high = summary["ci_low"] / live_mean, summary["ci_high"] / live_mean
    if summary["interval_status"] != "MEASURED":
        verdict = "INSUFFICIENT_EVIDENCE"
    elif low >= -EQUIVALENCE_BAND and high <= EQUIVALENCE_BAND:
        verdict = "EQUIVALENT"
    elif low > 0:
        verdict = "WORSE_THAN_LIVE"
    elif high < 0:
        verdict = "BETTER_THAN_LIVE"
    else:
        verdict = "INCONCLUSIVE"
    return {
        "live_mean": live_mean,
        "rule_mean": float(np.mean(second)),
        "relative_difference": summary["mean_difference"] / live_mean,
        "relative_ci_low": low,
        "relative_ci_high": high,
        "per_initialization_difference": summary["per_initialization_difference"],
        "verdict": verdict,
    }


def adjudicate4(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    conditions = sorted({r["condition"] for r in records})
    divergence = {
        condition: {
            arm: sum(diverged(r, arm) for r in records if r["condition"] == condition) for arm in RULE_ARMS
        }
        | {"cells": sum(1 for r in records if r["condition"] == condition)}
        for condition in conditions
    }
    coarse = [r for r in records if r["condition"] in COARSE_CONDITIONS]
    coarse_counts = {arm: sum(diverged(r, arm) for r in coarse) for arm in RULE_ARMS}
    live = coarse_counts["gru_live"]

    stable = [r for r in records if not diverged(r, "gru_live")]
    pooled_q = pooled([r["snapshot_update_histogram"] for r in stable])
    diverged_cells = [r for r in records if diverged(r, "gru_live") and r["condition"] in COARSE_CONDITIONS]
    pooled_diverged = pooled([r["snapshot_update_histogram"] for r in diverged_cells])
    median, q99 = pooled_q["median"], pooled_q["q99"]
    if median is None or q99 is None:
        h20 = "INSUFFICIENT_EVIDENCE"
    elif median <= NEGLIGIBLE_MEDIAN and q99 <= NEGLIGIBLE_Q99:
        h20 = "SUPPORTED"
    elif median > MATERIAL_MEDIAN:
        h20 = "CONTRADICTED"
    else:
        h20 = "INCONCLUSIVE"

    h21, r21 = _ratio_verdict(coarse_counts["gru_snapshot"], live, supported_below=True)
    h22, r22 = _ratio_verdict(coarse_counts["gru_t1"], live, supported_below=False)
    h23, r23 = _ratio_verdict(coarse_counts["gru_replay"], live, supported_below=True)

    plan_conditions = [c for c in conditions if c.startswith("plan:")]
    plan = {
        condition: {
            arm: _crossed([r for r in records if r["condition"] == condition], arm)
            for arm in ("gru_snapshot", "gru_replay", "gru_t1")
        }
        for condition in plan_conditions
    }
    return {
        "divergence_by_condition": divergence,
        "long_coarse_divergences": coarse_counts,
        "update_difference_live_vs_snapshot": {
            "non_diverged_cells": pooled_q,
            "diverged_coarse_cells": pooled_diverged,
        },
        "plan_mae_versus_live": plan,
        "verdicts": {
            "H20": {"verdict": h20, "pooled_median": median, "pooled_q99": q99},
            "H21": {"verdict": h21, "ratio": r21},
            "H22": {"verdict": h22, "ratio": r22},
            "H23": {"verdict": h23, "ratio": r23},
        },
    }
