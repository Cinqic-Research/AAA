"""Diagnostics of the known AAA-1K weaknesses, with verdict rules fixed in advance.

Committed before the diagnostics run. Every design uses **diagnostic**
identities and **development** families only, so attack-only and held-out
families stay unseen until their stages. Hypotheses carry numeric verdict
rules (``SUPPORTED`` / ``PARTIAL`` / ``CONTRADICTED`` / ``NOT_RESOLVED``)
computed by the code below, not by reading the numbers.

M1 / Q4 -- the gated core's coarse-family deficit
    Arms: Champion 1 (GRU-16, v1 features, reach-gated); the historical
    ungated control rebuilt in v2 (Elman-28, 954 parameters, its own v1
    configuration lr 0.01 / no clip / T 4, same target rule); Elman-16 (354,
    fixed width); Champion 1 with keep-gate bias -2. Families
    ``v1_coarse_speed``, ``quantized``, ``long_coarse``, ``v1_occlusion``,
    ``long_gap_recall``; a slow-speed sweep and a quantum sweep of the coarse
    construction; gate statistics of the GRU arms.

AAA-162 -- what does ``coarse_speed_v1`` reward?
    Model-independent: windowed least-squares extrapolation (the historical
    ``WindowedLinearFitAgent``) at windows 2-32 on paired streams of the
    coarse construction with switching speed (regime 60) and with the speed
    held fixed. No learned model enters the verdict.

AAA-172 -- does the v2.1 RLS learner frame-lock or stall?
    The unmodified ``RLSAgent``, instrumented by a subclass that only reads
    state around each ``update`` call, on long quantized and smooth streams.

Optimizer (brief, section Optimizer)
    SGD versus momentum versus Adam on the development-selected challenger
    (or, without one, the best-scoring candidate), under full state accounting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from aaa.predictors import unfold_observation
from research.aaa_1k.agents import RLSAgent, WindowedLinearFitAgent
from research.aaa_1k.streams import coarse_speed_stream

from . import plan, stress
from .arms import CHAMPION_1, ArmSpec, core_spec
from .engine import CellConfig
from .identities import seeds_of
from .jobs import DotJob, build, run_jobs
from .runner import StreamBatch
from .stats import relative_crossed

M1_FAMILIES = ("v1_coarse_speed", "quantized", "long_coarse", "v1_occlusion", "long_gap_recall")
SLOW_SPEEDS = (0.06, 0.08, 0.10, 0.12, 0.15)
QUANTA = (0.0025, 0.004, 0.005, 0.006, 0.008)
WINDOWS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
RLS_FAMILIES = ("long_coarse", "quantized", "long_bouncing", "v1_coarse_speed")
LOCK_RUN = 10
STALL_RUN = 50
MOMENTUM_RATES = (0.001, 0.003, 0.01)
ADAM_RATES = (0.0001, 0.0003, 0.001, 0.003)

DIAGNOSTIC_BLOCKS: tuple[dict[str, Any], ...] = (
    {
        "block_id": "v2-diagnostic-diag.coarse-speed-sweep",
        "role": "diagnostic",
        "namespace": "diag.coarse-speed-sweep",
        "start": 0,
        "count": 8,
        "purpose": "M1 slow-speed sweep of the coarse construction (seeds shared across speeds: paired)",
    },
    {
        "block_id": "v2-diagnostic-diag.coarse-quantum-sweep",
        "role": "diagnostic",
        "namespace": "diag.coarse-quantum-sweep",
        "start": 0,
        "count": 8,
        "purpose": "M1 quantum sweep of the coarse construction (paired)",
    },
    {
        "block_id": "v2-diagnostic-diag.coarse-fixed-vs-switching",
        "role": "diagnostic",
        "namespace": "diag.coarse-fixed-vs-switching",
        "start": 0,
        "count": 32,
        "purpose": "AAA-162 model-independent decomposition (paired fixed/switching)",
    },
)

UNGATED_V1 = CellConfig(learning_rate=0.01, gradient_clip=None, error_loss_weight=0.25)
M1_ARMS: dict[str, ArmSpec] = {
    "c1_champion1": CHAMPION_1,
    "elman28_v1": replace(
        CHAMPION_1,
        name="elman28_v1",
        kind="control",
        core=core_spec("elman", 3, 28),
        config=UNGATED_V1,
        description="the round-3 ungated control (954) with Champion 1's target rule",
    ),
    "elman16_v1": replace(
        CHAMPION_1,
        name="elman16_v1",
        kind="probe",
        core=core_spec("elman", 3, 16),
        config=UNGATED_V1,
        description="ungated at Champion 1's width (354)",
    ),
    "gru_keep-2_v1": replace(
        CHAMPION_1,
        name="gru_keep-2_v1",
        kind="probe",
        init=(("keep_bias", -2.0),),
        description="Champion 1 with keep-gate bias -2 at initialization",
    ),
}

HYPOTHESES: dict[str, str] = {
    "H-M1a": "on fresh v1_coarse_speed streams the ungated Elman-28 control beats Champion 1 (relative MAE upper bound < 0 -> SUPPORTED; lower > 0 -> CONTRADICTED)",
    "H-M1b": "operating point, not parameters: keep-bias -2 closes at least half of the Elman-28 minus Champion 1 gap on v1_coarse_speed (fraction >= 0.5 SUPPORTED, <= 0.2 CONTRADICTED, else PARTIAL; NOT_RESOLVED if the gap itself is not resolved)",
    "H-M1c": "width is not the explanation: Elman-16 (354 parameters) also beats Champion 1 on v1_coarse_speed (upper < 0 SUPPORTED; lower > 0 CONTRADICTED)",
    "H-M1d": "Champion 1's gates sit near their zero-bias operating point on v1_coarse_speed: mean |z - 0.5| < 0.1 over cells and steps (SUPPORTED if so, else CONTRADICTED)",
    "H-M1e": "the deficit depends on the speed-to-quantum ratio: Champion 1's relative excess MAE over Elman-28 at slow speed 0.06-0.08 exceeds that at 0.15 by more than 5 points (SUPPORTED), is smaller at slow speed (CONTRADICTED), else NOT_RESOLVED",
    "H-M1f": "keep-bias -2 costs occlusion accuracy: relative MAE versus Champion 1 on v1_occlusion lower bound > +2% (SUPPORTED); upper < +2% (CONTRADICTED)",
    "H-162a": "the construction rewards integrating history at fixed speed: the best window's MAE at fixed speed is at most 0.8 x the window-2 MAE (SUPPORTED); at least 0.95 x (CONTRADICTED); else PARTIAL",
    "H-162b": "switching adds a regime-inference cost: the best MAE under switching exceeds the best at fixed speed by at least 10% (SUPPORTED); by less than 5% (CONTRADICTED); else PARTIAL",
    "H-162c": "switching shortens the optimal window: the best window under switching is strictly shorter than at fixed speed (SUPPORTED), equal (NOT_RESOLVED), longer (CONTRADICTED)",
    "H-172-lock": f"the v2.1 RLS learner frame-locks: some cell has >= {LOCK_RUN} consecutive applied updates on a mirrored (unfolded) target (SUPPORTED per family if any cell does)",
    "H-172-stall": f"the v2.1 RLS learner stalls: some cell has >= {STALL_RUN} consecutive update attempts skipped by skip_after_reflected_prediction (SUPPORTED per family if any cell does)",
    "H-OPT": "a stateful optimizer earns its state: its best eligible configuration improves the development-style score over SGD by at least 5% with no extra failed or diverged cell (SUPPORTED), else NOT_SUPPORTED",
}


def register(registry: Mapping[str, Any]) -> dict[str, Any]:
    from .identities import reserve

    return reserve(registry, DIAGNOSTIC_BLOCKS)


def _verdict_upper_lower(result: Mapping[str, Any], *, supported_if_upper_below: float) -> str:
    if result.get("interval_status") != "MEASURED":
        return "NOT_RESOLVED"
    if result["upper"] < supported_if_upper_below:
        return "SUPPORTED"
    if result["lower"] > supported_if_upper_below:
        return "CONTRADICTED"
    return "NOT_RESOLVED"


def crossed(
    records: Sequence[Mapping[str, Any]], inits: Sequence[int], streams: Sequence[str], key: str = "mae"
) -> np.ndarray:
    table = {(r["seed"], r["stream_id"]): r for r in records}
    out = np.full((len(inits), len(streams)), np.nan)
    for i, init in enumerate(inits):
        for j, stream in enumerate(streams):
            record = table.get((init, stream))
            if record is not None and not record.get("failed") and record.get(key) is not None:
                out[i, j] = float(record[key])
    return out


def _crossed_job(arm: ArmSpec, inits: Sequence[int], batch: StreamBatch, tag: str) -> DotJob:
    seeds, rows = [], []
    for init in inits:
        for row in range(batch.size):
            seeds.append(init)
            rows.append(row)
    return DotJob(arm=arm, seeds=tuple(seeds), batch=batch.take(rows), tag=tag)


def gate_statistics(arm: ArmSpec, inits: Sequence[int], batch: StreamBatch) -> dict[str, float]:
    """Mean keep gate, mean reset gate and saturation over every cell and step (GRU arms)."""

    seeds = [init for init in inits for _ in range(batch.size)]
    rows = [row for _ in inits for row in range(batch.size)]
    sub = batch.take(rows)
    backend, learner, adapter = build(arm, seeds, [arm.config] * len(seeds), "cpu", stress_scales())
    xp = backend.xp
    truth = backend.asarray(sub.truth)
    observed = backend.asarray(sub.observed, dtype=bool)
    learner.reset_state()
    adapter.begin()
    adapter.accept(None, truth[:, 0], xp.ones(len(seeds), dtype=bool), xp.ones(len(seeds), dtype=bool))
    z_dev, r_mean, saturated, count = 0.0, 0.0, 0.0, 0
    T = sub.truth.shape[1]
    with np.errstate(all="ignore"):
        for t in range(T - 1):
            adapter.predict(learner)
            cache = learner._last
            assert cache is not None
            valid = backend.to_host(xp.asarray(t + 1 < backend.asarray(sub.length)))
            z = backend.to_host(cache["z"])[valid]
            r = backend.to_host(cache["r"])[valid]
            z_dev += float(np.abs(z - 0.5).sum())
            r_mean += float(r.sum())
            saturated += float(((z > 0.95) | (z < 0.05)).sum())
            count += z.size
            adapter.accept(learner, truth[:, t + 1], observed[:, t + 1], xp.ones(len(seeds), dtype=bool))
            learner.advance_clock()
    return {
        "mean_abs_keep_minus_half": z_dev / count,
        "mean_reset": r_mean / count,
        "keep_saturated_fraction": saturated / count,
    }


def stress_scales() -> Any:
    from .adapters import Scales

    return Scales()


def _coarse_batch(
    seeds: Sequence[int],
    *,
    slow: float = 0.12,
    quantum: float = 0.005,
    regime: int = 60,
    steps: int = 240,
    label: str,
) -> StreamBatch:
    rows = []
    for seed in seeds:
        stream = coarse_speed_stream(
            seed, steps=steps, quantum=quantum, slow_speed=slow, regime_length=regime
        )
        rows.append(stream)
    T = max(len(s.steps) for s in rows)
    truth = np.zeros((len(rows), T))
    obs = np.zeros((len(rows), T), dtype=bool)
    length = np.zeros(len(rows), dtype=int)
    regimes = np.zeros((len(rows), T), dtype=np.int16)
    for i, stream in enumerate(rows):
        n = len(stream.steps)
        truth[i, :n] = [s.true_position for s in stream.steps]
        obs[i, :n] = True
        length[i] = n
        regimes[i, :n] = [0 if s.regime == "slow" else 1 for s in stream.steps]
    return StreamBatch(
        truth,
        obs,
        length,
        [f"{label}:{s.seed}" for s in rows],
        labels={"condition": regimes},
        vocab={"condition": ["slow", "fast"]},
    )


def m1(registry: Mapping[str, Any], *, workers: Any, log: Callable[[str], None] = print) -> dict[str, Any]:
    inits = seeds_of(dict(registry), "v2-diagnostic-init")
    batches = {
        family: stress.batch(family, seeds_of(dict(registry), f"v2-diagnostic-env.{family}"))
        for family in M1_FAMILIES
    }
    speed_seeds = seeds_of(dict(registry), "v2-diagnostic-diag.coarse-speed-sweep")
    quantum_seeds = seeds_of(dict(registry), "v2-diagnostic-diag.coarse-quantum-sweep")
    for slow in SLOW_SPEEDS:
        batches[f"speed_{slow:g}"] = _coarse_batch(speed_seeds, slow=slow, label=f"speed_{slow:g}")
    for quantum in QUANTA:
        batches[f"quantum_{quantum:g}"] = _coarse_batch(
            quantum_seeds, quantum=quantum, label=f"quantum_{quantum:g}"
        )
    jobs, keys = [], []
    for arm_name, arm in M1_ARMS.items():
        for group, batch in batches.items():
            jobs.append(_crossed_job(arm, inits, batch, f"{arm_name}:{group}"))
            keys.append((arm_name, group))
    results = run_jobs(jobs, workers=workers)
    cells = {key: result["cells"] for key, result in zip(keys, results, strict=True)}
    log(f"M1: {sum(len(r['cells']) for r in results)} cells")

    def rel(first: str, second: str, group: str, seed: int) -> dict[str, Any]:
        streams = batches[group].stream_ids
        a = crossed(cells[(first, group)], inits, streams)
        b = crossed(cells[(second, group)], inits, streams)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return {
                "interval_status": "INCOMPLETE",
                "failed_first": int(np.sum(~np.isfinite(a))),
                "failed_second": int(np.sum(~np.isfinite(b))),
            }
        return relative_crossed(a, b, seed=seed, draws=plan.BOOTSTRAP_DRAWS)

    bootstrap = seeds_of(dict(registry), "v2-diagnostic-bootstrap")
    comparisons: dict[str, Any] = {}
    for index, (first, second) in enumerate(
        (("c1_champion1", "elman28_v1"), ("c1_champion1", "elman16_v1"), ("c1_champion1", "gru_keep-2_v1"))
    ):
        for offset, group in enumerate(batches):
            comparisons[f"{second}_vs_{first}:{group}"] = rel(
                first, second, group, bootstrap[0] % 2**32 + 100 * index + offset
            )
    gap = comparisons["elman28_v1_vs_c1_champion1:v1_coarse_speed"]
    keep = comparisons["gru_keep-2_v1_vs_c1_champion1:v1_coarse_speed"]
    verdicts: dict[str, Any] = {}
    verdicts["H-M1a"] = _verdict_upper_lower(gap, supported_if_upper_below=0.0)
    verdicts["H-M1c"] = _verdict_upper_lower(
        comparisons["elman16_v1_vs_c1_champion1:v1_coarse_speed"], supported_if_upper_below=0.0
    )
    if verdicts["H-M1a"] == "SUPPORTED" and keep.get("interval_status") == "MEASURED":
        fraction = keep["relative"] / gap["relative"]
        verdicts["H-M1b"] = (
            "SUPPORTED" if fraction >= 0.5 else ("CONTRADICTED" if fraction <= 0.2 else "PARTIAL")
        )
        verdicts["H-M1b_fraction"] = fraction
    else:
        verdicts["H-M1b"] = "NOT_RESOLVED"
    gates = {
        name: gate_statistics(M1_ARMS[name], inits[:3], batches["v1_coarse_speed"])
        for name in ("c1_champion1", "gru_keep-2_v1")
    }
    gates["c1_champion1:v1_occlusion"] = gate_statistics(CHAMPION_1, inits[:3], batches["v1_occlusion"])
    verdicts["H-M1d"] = (
        "SUPPORTED" if gates["c1_champion1"]["mean_abs_keep_minus_half"] < 0.1 else "CONTRADICTED"
    )

    def excess(group: str) -> float | None:
        entry = comparisons[f"elman28_v1_vs_c1_champion1:{group}"]
        return (
            None
            if entry.get("interval_status") != "MEASURED"
            else -entry["relative"] / (1.0 + entry["relative"])
        )

    slow_excess = [excess(f"speed_{s:g}") for s in (0.06, 0.08)]
    fast_excess = excess("speed_0.15")
    if all(value is not None for value in slow_excess) and fast_excess is not None:
        slow_mean = float(np.mean(slow_excess))  # type: ignore[arg-type]
        difference = slow_mean - fast_excess
        verdicts["H-M1e"] = (
            "SUPPORTED" if difference > 0.05 else ("CONTRADICTED" if difference < 0 else "NOT_RESOLVED")
        )
        verdicts["H-M1e_difference"] = difference
    else:
        verdicts["H-M1e"] = "NOT_RESOLVED"
    verdicts["H-M1f"] = _verdict_upper_lower_margin(
        comparisons["gru_keep-2_v1_vs_c1_champion1:v1_occlusion"], margin=0.02
    )
    return {
        "arms": {name: arm.to_dict() for name, arm in M1_ARMS.items()},
        "inits": inits,
        "groups": {group: batch.stream_ids for group, batch in batches.items()},
        "comparisons": comparisons,
        "gate_statistics": gates,
        "verdicts": verdicts,
        "cells": {f"{a}:{g}": c for (a, g), c in cells.items()},
    }


def _verdict_upper_lower_margin(result: Mapping[str, Any], *, margin: float) -> str:
    if result.get("interval_status") != "MEASURED":
        return "NOT_RESOLVED"
    if result["lower"] > margin:
        return "SUPPORTED"
    if result["upper"] < margin:
        return "CONTRADICTED"
    return "NOT_RESOLVED"


# ----------------------------------------------------------------------
# AAA-162: model-independent decomposition
# ----------------------------------------------------------------------
def aaa162(registry: Mapping[str, Any]) -> dict[str, Any]:
    seeds = seeds_of(dict(registry), "v2-diagnostic-diag.coarse-fixed-vs-switching")
    conditions = {"switching": 60, "fixed": 10**9}
    curves: dict[str, dict[str, Any]] = {}
    for condition, regime in conditions.items():
        per_window: dict[int, list[float]] = {w: [] for w in WINDOWS}
        per_window_regime: dict[str, dict[int, list[float]]] = {
            "slow": {w: [] for w in WINDOWS},
            "fast": {w: [] for w in WINDOWS},
        }
        for seed in seeds:
            stream = coarse_speed_stream(seed, steps=240, regime_length=regime)
            truth = [s.true_position for s in stream.steps]
            agents = {w: WindowedLinearFitAgent(window=w) for w in WINDOWS}
            errors: dict[int, list[float]] = {w: [] for w in WINDOWS}
            regimes = []
            for agent in agents.values():
                agent.begin_episode()
                agent.accept_observation(truth[0])
            for t in range(len(truth) - 1):
                for w, agent in agents.items():
                    errors[w].append(abs(agent.predict() - truth[t + 1]))
                regimes.append(stream.steps[t + 1].regime)
                for agent in agents.values():
                    agent.accept_observation(truth[t + 1])
            for w in WINDOWS:
                per_window[w].append(float(np.mean(errors[w])))
                for regime_name in ("slow", "fast"):
                    selected = [e for e, r in zip(errors[w], regimes, strict=True) if r == regime_name]
                    if selected:
                        per_window_regime[regime_name][w].append(float(np.mean(selected)))
        means = {w: float(np.mean(v)) for w, v in per_window.items()}
        best = min(WINDOWS, key=lambda w: (means[w], w))
        curves[condition] = {
            "mae_by_window": {str(w): means[w] for w in WINDOWS},
            "per_stream": {str(w): per_window[w] for w in WINDOWS},
            "by_regime": {
                name: {str(w): float(np.mean(v)) if v else None for w, v in table.items()}
                for name, table in per_window_regime.items()
            },
            "best_window": best,
            "best_mae": means[best],
        }
    fixed, switching = curves["fixed"], curves["switching"]
    ratio_a = fixed["best_mae"] / fixed["mae_by_window"]["2"]
    ratio_b = switching["best_mae"] / fixed["best_mae"]
    verdicts = {
        "H-162a": "SUPPORTED" if ratio_a <= 0.8 else ("CONTRADICTED" if ratio_a >= 0.95 else "PARTIAL"),
        "H-162a_ratio": ratio_a,
        "H-162b": "SUPPORTED" if ratio_b >= 1.10 else ("CONTRADICTED" if ratio_b < 1.05 else "PARTIAL"),
        "H-162b_ratio": ratio_b,
        "H-162c": "SUPPORTED"
        if switching["best_window"] < fixed["best_window"]
        else ("NOT_RESOLVED" if switching["best_window"] == fixed["best_window"] else "CONTRADICTED"),
    }
    return {
        "seeds": seeds,
        "windows": list(WINDOWS),
        "curves": curves,
        "verdicts": verdicts,
        "estimator": "research.aaa_1k.agents.WindowedLinearFitAgent (unmodified), public reflection",
    }


# ----------------------------------------------------------------------
# AAA-172: RLS instrumentation
# ----------------------------------------------------------------------
class InstrumentedRLSAgent(RLSAgent):
    """``RLSAgent`` unchanged; records, around every ``update`` call, whether the target was mirrored and whether it was skipped."""

    def begin_episode(self) -> None:
        super().begin_episode()
        self.trace: list[tuple[bool, bool]] = []

    def accept_observation(self, observation: float | None) -> None:
        will_update = (
            observation is not None
            and len(self.history) >= self.HISTORY
            and self.update_enabled
            and bool(self.observed)
            and self.observed[-1]
        )
        if will_update:
            assert observation is not None
            history = tuple(self.history[-self.HISTORY :])
            raw = self.predictor.raw_predict(history)
            mirrored = unfold_observation(
                float(observation), raw, self.predictor.lower_bound, self.predictor.upper_bound
            ) != float(observation)
            skips_before = self.predictor.reflection_skips
            super().accept_observation(observation)
            skipped = self.predictor.reflection_skips > skips_before
            self.trace.append((mirrored and not skipped, skipped))
            return
        super().accept_observation(observation)


def _longest(flags: Sequence[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def aaa172(registry: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"families": {}, "verdicts": {}}
    for family in RLS_FAMILIES:
        seeds = seeds_of(dict(registry), f"v2-diagnostic-env.{family}")
        batch = stress.batch(family, seeds)
        cells: list[dict[str, Any]] = []
        for b in range(batch.size):
            n = int(batch.length[b])
            agent = InstrumentedRLSAgent()
            agent.begin_episode()
            agent.accept_observation(float(batch.truth[b, 0]))
            errors, persistence = [], []
            last = float(batch.truth[b, 0])
            for t in range(n - 1):
                prediction = agent.predict()
                target = float(batch.scoring[b, t + 1])
                errors.append(abs(prediction - target))
                persistence.append(abs(last - target))
                value = float(batch.truth[b, t + 1]) if batch.observed[b, t + 1] else None
                agent.accept_observation(value)
                if value is not None:
                    last = value
            mirrored = [m for m, _ in agent.trace]
            skipped = [s for _, s in agent.trace]
            cells.append(
                {
                    "stream_id": batch.stream_ids[b],
                    "mae": float(np.mean(errors)),
                    "persistence_mae": float(np.mean(persistence)),
                    "updates_attempted": len(agent.trace),
                    "mirrored_applied": int(sum(mirrored)),
                    "skipped": int(sum(skipped)),
                    "longest_mirrored_run": _longest(mirrored),
                    "longest_skip_run": _longest(skipped),
                    "diverged": bool(np.mean(errors) > plan.DIVERGENCE_FACTOR * np.mean(persistence)),
                }
            )
        lock = any(int(c["longest_mirrored_run"]) >= LOCK_RUN for c in cells)
        stall = any(int(c["longest_skip_run"]) >= STALL_RUN for c in cells)
        out["families"][family] = cells
        out["verdicts"][family] = {
            "H-172-lock": "SUPPORTED" if lock else "CONTRADICTED",
            "H-172-stall": "SUPPORTED" if stall else "CONTRADICTED",
            "diverged_cells": sum(bool(c["diverged"]) for c in cells),
            "classification": "FRAME_LOCK" if lock else ("STALL" if stall else "NO_MATERIAL_DEFECT"),
        }
    return out


# ----------------------------------------------------------------------
# optimizer diagnostic
# ----------------------------------------------------------------------
def optimizer_probe(
    registry: Mapping[str, Any], arm: ArmSpec, *, workers: Any, log: Callable[[str], None] = print
) -> dict[str, Any]:
    """SGD against momentum and Adam on the development families' diagnostic streams."""

    from .tournament import group_means

    inits = seeds_of(dict(registry), "v2-diagnostic-init")
    families = plan.STAGE_FAMILIES["development"]
    batches = {f: stress.batch(f, seeds_of(dict(registry), f"v2-diagnostic-env.{f}")) for f in families}
    variants: dict[str, CellConfig] = {"sgd": arm.config}
    for rate in MOMENTUM_RATES:
        variants[f"momentum:lr={rate:g}"] = replace(arm.config, learning_rate=rate, optimizer="momentum")
    for rate in ADAM_RATES:
        variants[f"adam:lr={rate:g}"] = replace(arm.config, learning_rate=rate, optimizer="adam")
    results: dict[str, list[dict[str, Any]]] = {}
    champion: list[dict[str, Any]] = []
    jobs, keys = [], []
    for family, batch in batches.items():
        jobs.append(_crossed_job(CHAMPION_1, inits, batch, f"champion:{family}"))
        keys.append(("champion", family))
        for name, config in variants.items():
            jobs.append(_crossed_job(replace(arm, config=config), inits, batch, f"{name}:{family}"))
            keys.append((name, family))
    outputs = run_jobs(jobs, workers=workers)
    log(f"optimizer probe: {sum(len(o['cells']) for o in outputs)} cells")
    for (name, family), output in zip(keys, outputs, strict=True):
        cells = [{**c, "group": family, "init": c["seed"]} for c in output["cells"]]
        (champion if name == "champion" else results.setdefault(name, [])).extend(cells)
    persistence: dict[str, float] = {}
    from .baselines import baselines_parallel

    for batch in batches.values():
        for record in baselines_parallel(batch, workers=workers):
            persistence[record["stream_id"]] = record["persistence"]
    reference = group_means(champion)

    def summarize(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        unstable = sum(
            1
            for c in cells
            if c.get("failed")
            or c["mae"] is None
            or c["mae"] > plan.DIVERGENCE_FACTOR * persistence[c["stream_id"]]
        )
        means = group_means(cells)
        score = (
            float(math.exp(np.mean([math.log(means[f] / reference[f]) for f in families])))
            if all(f in means for f in families)
            else None
        )
        return {"unstable_cells": unstable, "score_vs_champion": score, "group_means": means}

    summaries = {name: summarize(cells) for name, cells in results.items()}
    sgd = summaries["sgd"]
    stateful = {
        n: s
        for n, s in summaries.items()
        if n != "sgd" and s["unstable_cells"] <= sgd["unstable_cells"] and s["score_vs_champion"] is not None
    }
    best = min(stateful, key=lambda n: stateful[n]["score_vs_champion"]) if stateful else None
    improvement = (
        None
        if best is None or sgd["score_vs_champion"] is None
        else 1.0 - stateful[best]["score_vs_champion"] / sgd["score_vs_champion"]
    )
    trainable = arm.build_core().parameter_count()
    return {
        "arm": arm.to_dict(),
        "variants": {name: config.to_dict() for name, config in variants.items()},
        "optimizer_state_scalars": {"sgd": 0, "momentum": trainable, "adam": 2 * trainable},
        "summaries": summaries,
        "best_stateful": best,
        "improvement_over_sgd": improvement,
        "verdicts": {
            "H-OPT": "SUPPORTED" if improvement is not None and improvement >= 0.05 else "NOT_SUPPORTED"
        },
    }
