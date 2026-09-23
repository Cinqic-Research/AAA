"""Attack stage: try to break the development-designated challenger (criteria A1-A7).

Every criterion comes from :data:`research.aaa_1k_v2.plan.ATTACK`, committed
before development. Attack identities are fresh (5 initializations, 8 streams
for every development and attack-only family, fresh external realizations).

A1  challenger: zero failed cells and diverged cells <= Champion 1's
A2  every v1 family non-inferior at +2%
A3  stress families (development and attack-only): geometric-mean upper bound < 1.0
A4  no stress family with relative-MAE lower bound above +5%
A5  NARMA-10, NARMA-20, Mackey-Glass and twelve dysts systems: geometric-mean upper bound <= 1.02
A6  the verdicts computed from a CUDA re-run equal the CPU verdicts. The frozen
    text names the v1 families; the re-run covers every attack dot family, and
    both the literal (A1 and A2 on the v1 families) and the full (A1-A3)
    comparisons are reported. A6 passes only if the literal comparison agrees.
A7  the challenger with input/recurrent initial weights x1.5 passes A1
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from . import plan
from .arms import CHAMPION_1, ArmSpec
from .baselines import baselines_parallel
from .identities import seeds_of
from .jobs import DotJob, SeriesJob, run_jobs
from .stats import combine, geometric_relative, guard, noninferior, relative_crossed, superior
from .tournament import build_cells, crossed_grid, divergence_metric, external_baselines, external_metric

ATTACK_EXTERNAL = tuple(plan.external_tasks())


def arm_from_record(record: Mapping[str, Any], template: ArmSpec) -> ArmSpec:
    """Rebuild a development-selected arm from its artifact record (hyperparameters from ``selected``)."""

    from .tournament import GridPoint, arm_at

    selected = record["decision"]["selected"]
    point = GridPoint(
        selected["learning_rate"], selected["tbptt_steps"], selected["rule"], selected["gradient_clip"]
    )
    return arm_at(template, point)


def _dot_jobs(arm: ArmSpec, cells: Any, device: str) -> tuple[list[DotJob], list[str]]:
    jobs, groups = [], []
    for family, batch in cells.families.items():
        seeds = [init for init in cells.inits for _ in range(batch.size)]
        rows = [row for _ in cells.inits for row in range(batch.size)]
        jobs.append(
            DotJob(
                arm=arm, seeds=tuple(seeds), batch=batch.take(rows), device=device, tag=f"{arm.name}:{family}"
            )
        )
        groups.append(family)
    return jobs, groups


def _series_jobs(
    arm: ArmSpec, cells: Any, tasks: Sequence[str], device: str
) -> tuple[list[SeriesJob], list[str]]:
    jobs, groups = [], []
    for task in tasks:
        problems = cells.external[task]
        seeds = [init for init in cells.inits for _ in problems]
        probs = [p for _ in cells.inits for p in problems]
        jobs.append(
            SeriesJob(
                arm=arm, seeds=tuple(seeds), problems=tuple(probs), device=device, tag=f"{arm.name}:{task}"
            )
        )
        groups.append(task)
    return jobs, groups


def evaluate_arm(
    arm: ArmSpec, cells: Any, *, device: str, workers: Any, external: Sequence[str]
) -> list[dict[str, Any]]:
    djobs, dgroups = _dot_jobs(arm, cells, device)
    sjobs, sgroups = _series_jobs(arm, cells, external, device)
    results = run_jobs([*djobs, *sjobs], workers=workers)
    out: list[dict[str, Any]] = []
    for result, group in zip(results, [*dgroups, *sgroups], strict=True):
        for cell in result["cells"]:
            out.append({**cell, "group": group, "init": cell["seed"]})
    return out


def unstable(
    cell: Mapping[str, Any], persistence: Mapping[str, float], external_base: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, bool]:
    """``(failed, diverged)`` under the plan's rules."""

    if cell.get("failed"):
        return True, True
    if "stream_id" in cell:
        return False, bool(cell["mae"] > plan.DIVERGENCE_FACTOR * persistence[cell["stream_id"]])
    metric = divergence_metric(cell["group"])
    base = external_base[cell["series_id"]]["persistence"][metric]
    value = cell.get(metric)
    return False, bool(
        value is None or value > plan.DIVERGENCE_FACTOR * base or cell.get("forecast_finite") is False
    )


def stability(
    reference: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    persistence: Mapping[str, float],
    external_base: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ref = [unstable(c, persistence, external_base) for c in reference]
    cha = [unstable(c, persistence, external_base) for c in challenger]
    failed = sum(f for f, _ in cha)
    diverged_ref = sum(d for _, d in ref)
    diverged_cha = sum(d for _, d in cha)
    return {
        "challenger_failed": failed,
        "challenger_diverged": diverged_cha,
        "champion_failed": sum(f for f, _ in ref),
        "champion_diverged": diverged_ref,
        "status": "PASS" if failed == 0 and diverged_cha <= diverged_ref else "FAIL",
    }


def v1_statuses(
    reference: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for offset, family in enumerate(families):
        a, _, _ = crossed_grid(reference, family)
        b, _, _ = crossed_grid(challenger, family)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            out[family] = {"status": "FAIL", "reason": "failed or missing cells"}
            continue
        result = relative_crossed(a, b, seed=seed + offset, draws=plan.BOOTSTRAP_DRAWS)
        out[family] = {**result, "status": noninferior(result, plan.NONINFERIORITY_MARGIN)}
    return {"families": out, "status": combine([entry["status"] for entry in out.values()])}


def geometric_status(
    reference: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
    seed: int,
    *,
    external: bool,
    keep_series: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Geometric-mean status over ``groups``; ``keep_series`` restricts a group to data-defined series."""

    pairs = {}
    for group in groups:
        value = external_metric(group) if external else "mae"
        ref_cells, cha_cells = reference, challenger
        if keep_series is not None and group in keep_series:
            allowed = keep_series[group]
            ref_cells = [
                c
                for c in reference
                if c.get("series_id", c.get("stream_id")) in allowed or c["group"] != group
            ]
            cha_cells = [
                c
                for c in challenger
                if c.get("series_id", c.get("stream_id")) in allowed or c["group"] != group
            ]
        a, _, _ = crossed_grid(ref_cells, group, value)
        b, _, _ = crossed_grid(cha_cells, group, value)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return {"status": "FAIL", "reason": f"failed or missing cells in {group}"}
        pairs[group] = (a, b)
    result = geometric_relative(pairs, seed=seed, draws=plan.BOOTSTRAP_DRAWS)
    if external:
        shifted = (
            {**result, "upper": result["upper"] - 1.0, "lower": result["lower"] - 1.0}
            if result.get("interval_status") == "MEASURED"
            else result
        )
        return {**result, "status": noninferior(shifted, plan.EXTERNAL_MARGIN)}
    return {**result, "status": superior(result, 1.0)}


def regression_guard(
    reference: Sequence[Mapping[str, Any]],
    challenger: Sequence[Mapping[str, Any]],
    groups: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for offset, group in enumerate(groups):
        a, _, _ = crossed_grid(reference, group)
        b, _, _ = crossed_grid(challenger, group)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            out[group] = {"status": "FAIL", "reason": "failed or missing cells"}
            continue
        result = relative_crossed(a, b, seed=seed + offset, draws=plan.BOOTSTRAP_DRAWS)
        out[group] = {**result, "status": guard(result, plan.REGRESSION_GUARD)}
    return {"families": out, "status": combine([entry["status"] for entry in out.values()])}


def run_attack(
    registry: Mapping[str, Any],
    challenger: ArmSpec,
    *,
    workers: Any,
    cuda_device: str | None = "cuda:0",
    log: Any = print,
) -> dict[str, Any]:
    started = time.time()
    families = plan.STAGE_FAMILIES["attack"]
    cells = build_cells(registry, "attack", families, ATTACK_EXTERNAL)
    log(
        f"attack cells: {len(cells.inits)} inits x {sum(b.size for b in cells.families.values())} dot streams + {sum(len(v) for v in cells.external.values())} external"
    )
    baseline_records = {
        family: baselines_parallel(batch, workers=workers) for family, batch in cells.families.items()
    }
    persistence = {r["stream_id"]: r["persistence"] for records in baseline_records.values() for r in records}
    external_base = external_baselines(cells, ATTACK_EXTERNAL)
    external_flat = {sid: metrics for task in external_base.values() for sid, metrics in task.items()}
    perturbed = replace(
        challenger,
        name=f"{challenger.name}:init_x1.5",
        kind="probe",
        init=(*challenger.init, ("weight_scale", 1.5)),
    )
    arms = {"c1_champion1": CHAMPION_1, challenger.name: challenger, perturbed.name: perturbed}
    primitives = {}
    for name, arm in arms.items():
        t0 = time.time()
        primitives[name] = evaluate_arm(
            arm,
            cells,
            device="cpu",
            workers=workers,
            external=ATTACK_EXTERNAL if name != perturbed.name else (),
        )
        log(f"  {name}: {len(primitives[name])} cells, {time.time() - t0:.0f}s")
    bootstrap = seeds_of(dict(registry), "v2-attack-bootstrap")
    v1 = [f for f in families if f.startswith("v1_")]
    stress_groups = [f for f in families if not f.startswith("v1_")]
    ref, cha = primitives["c1_champion1"], primitives[challenger.name]
    criteria: dict[str, Any] = {}
    criteria["A1_stability"] = stability(ref, cha, persistence, external_flat)
    criteria["A2_v1_noninferiority"] = v1_statuses(ref, cha, v1, bootstrap[0] % 2**32)
    criteria["A3_stress_superiority"] = geometric_status(
        ref, cha, stress_groups, bootstrap[1] % 2**32, external=False
    )
    criteria["A4_regression_guard"] = regression_guard(ref, cha, stress_groups, bootstrap[2] % 2**32)
    criteria["A5_external_noninferiority"] = geometric_status(
        ref, cha, list(ATTACK_EXTERNAL), bootstrap[3] % 2**32, external=True
    )
    dot_only_ref = [c for c in ref if "stream_id" in c]
    criteria["A7_perturbed_initialization"] = stability(
        dot_only_ref, primitives[perturbed.name], persistence, external_flat
    )
    if cuda_device is not None:
        cuda = {}
        for name in ("c1_champion1", challenger.name):
            t0 = time.time()
            cuda[name] = evaluate_arm(arms[name], cells, device=cuda_device, workers=1, external=())
            log(f"  {name} on {cuda_device}: {len(cuda[name])} cells, {time.time() - t0:.0f}s")
        cpu_dot = {name: [c for c in primitives[name] if "stream_id" in c] for name in cuda}
        literal_cpu = {
            "A1": stability(
                [c for c in cpu_dot["c1_champion1"] if c["group"] in v1],
                [c for c in cpu_dot[challenger.name] if c["group"] in v1],
                persistence,
                external_flat,
            )["status"],
            "A2": v1_statuses(cpu_dot["c1_champion1"], cpu_dot[challenger.name], v1, bootstrap[0] % 2**32)[
                "status"
            ],
        }
        literal_cuda = {
            "A1": stability(
                [c for c in cuda["c1_champion1"] if c["group"] in v1],
                [c for c in cuda[challenger.name] if c["group"] in v1],
                persistence,
                external_flat,
            )["status"],
            "A2": v1_statuses(cuda["c1_champion1"], cuda[challenger.name], v1, bootstrap[0] % 2**32)[
                "status"
            ],
        }
        full_cuda = {
            "A1": stability(cuda["c1_champion1"], cuda[challenger.name], persistence, external_flat)[
                "status"
            ],
            "A2": v1_statuses(cuda["c1_champion1"], cuda[challenger.name], v1, bootstrap[0] % 2**32)[
                "status"
            ],
            "A3": geometric_status(
                cuda["c1_champion1"],
                cuda[challenger.name],
                stress_groups,
                bootstrap[1] % 2**32,
                external=False,
            )["status"],
        }
        full_cpu = {
            "A1": stability(cpu_dot["c1_champion1"], cpu_dot[challenger.name], persistence, external_flat)[
                "status"
            ],
            "A2": literal_cpu["A2"],
            "A3": criteria["A3_stress_superiority"]["status"],
        }
        drift = []
        for name in cuda:
            table = {(c["seed"], c["stream_id"]): c for c in cpu_dot[name]}
            for cell in cuda[name]:
                other = table[(cell["seed"], cell["stream_id"])]
                if cell.get("mae") is not None and other.get("mae") is not None and other["mae"] > 0:
                    drift.append(abs(cell["mae"] - other["mae"]) / other["mae"])
        criteria["A6_backend_verdicts"] = {
            "device": cuda_device,
            "literal_cpu": literal_cpu,
            "literal_cuda": literal_cuda,
            "full_cpu": full_cpu,
            "full_cuda": full_cuda,
            "per_cell_relative_mae_drift": {
                "max": float(np.max(drift)) if drift else None,
                "median": float(np.median(drift)) if drift else None,
                "cells": len(drift),
            },
            "status": "PASS" if literal_cpu == literal_cuda else "FAIL",
            "full_agreement": full_cpu == full_cuda,
        }
    else:
        criteria["A6_backend_verdicts"] = {"status": "INCONCLUSIVE", "reason": "no CUDA device"}
    statuses = {key: value["status"] for key, value in criteria.items()}
    return {
        "schema": "aaa.1k.v2.attack.v1",
        "stage": "attack",
        "challenger": challenger.to_dict(),
        "perturbed": perturbed.to_dict(),
        "families": list(families),
        "external": list(ATTACK_EXTERNAL),
        "inits": cells.inits,
        "criteria": criteria,
        "statuses": statuses,
        "advance": all(status == "PASS" for status in statuses.values()),
        "primitives": primitives,
        "baselines": {"dot": baseline_records, "external": external_base},
        "compute_seconds": time.time() - started,
    }
