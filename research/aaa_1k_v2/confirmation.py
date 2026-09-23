"""Confirmation: observe fresh identities once, under a committed freeze, and decide (K1-K5).

Admission (``confirm``) refuses to run unless the freeze manifest is tracked,
identical to ``HEAD``, and agrees with the live repository (source
fingerprint, locks, registry seeds, arms, thresholds). It then marks every
confirmation block spent -- before the first cell runs -- and only then builds
cells, whose seeds may be read solely by the run that spent them.

Observed: every dot family (8 initializations x 16 streams, held-out families
included), the external suite (NARMA-10/20, Mackey-Glass, twelve dysts
systems, four Monash datasets at their test horizons), the capability designs
(online/frozen twins, paired-change adaptation, retention probe bank), all
arms listed in the freeze, and the analytic baselines.

The decision uses only the challenger and Champion 1 (K1-K5). Everything else
is exploratory and labelled so.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import plan
from .arms import ArmSpec
from .attack import evaluate_arm, geometric_status, regression_guard, stability, v1_statuses
from .baselines import baselines_parallel
from .capability import adaptation, retention
from .external import tasks as ext
from .identities import seeds_of
from .jobs import DotJob, SeriesJob, run_jobs
from .stats import combine
from .tournament import Cells, external_baselines, external_metric

EXTERNAL = tuple(plan.external_tasks())
MONASH = tuple(plan.monash_tasks())


def confirmation_cells(registry: Mapping[str, Any], observer: str) -> Cells:
    from .tournament import build_cells

    return build_cells(
        registry, "confirmation", plan.STAGE_FAMILIES["confirmation"], EXTERNAL, observer=observer
    )


def monash_problems(root: Any) -> dict[str, list[ext.SeriesProblem]]:
    return {task: ext.monash_problems(task.split(":", 1)[1], "confirmation", root=root) for task in MONASH}


def online_frozen(arm: ArmSpec, cells: Cells, *, workers: Any) -> list[dict[str, Any]]:
    jobs, groups = [], []
    for family, batch in cells.families.items():
        seeds = [init for init in cells.inits for _ in range(batch.size)]
        rows = [row for _ in cells.inits for row in range(batch.size)]
        branch_at = int(batch.length.min()) // 4
        jobs.append(
            DotJob(
                arm=arm,
                seeds=tuple(seeds),
                batch=batch.take(rows),
                branch_at=branch_at,
                tag=f"{arm.name}:q1:{family}",
            )
        )
        groups.append(family)
    out = []
    for result, group in zip(run_jobs(jobs, workers=workers), groups, strict=True):
        for twin in result["cells"]:
            online, frozen = twin["online"], twin["frozen"]
            record = {
                "group": group,
                "init": online["seed"],
                "stream_id": online["stream_id"],
                "failed": online["failed"] or frozen["failed"],
            }
            if not record["failed"]:
                record.update(
                    online_mae=online["mae"],
                    frozen_mae=frozen["mae"],
                    online_advantage=frozen["mae"] - online["mae"],
                )
            out.append(record)
    return out


def monash_arm(
    arm: ArmSpec, problems: Mapping[str, Sequence[ext.SeriesProblem]], inits: Sequence[int], *, workers: Any
) -> list[dict[str, Any]]:
    jobs, groups = [], []
    for task, items in problems.items():
        # series within one dataset share a horizon; lengths may differ, so one job per series length
        by_length: dict[int, list[ext.SeriesProblem]] = {}
        for problem in items:
            by_length.setdefault(problem.length, []).append(problem)
        for length_problems in by_length.values():
            seeds = [init for init in inits for _ in length_problems]
            probs = [p for _ in inits for p in length_problems]
            jobs.append(
                SeriesJob(arm=arm, seeds=tuple(seeds), problems=tuple(probs), tag=f"{arm.name}:{task}")
            )
            groups.append(task)
    out = []
    for result, group in zip(run_jobs(jobs, workers=workers), groups, strict=True):
        for cell in result["cells"]:
            out.append({**cell, "group": group, "init": cell["seed"]})
    return out


def monash_baselines(
    problems: Mapping[str, Sequence[ext.SeriesProblem]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task, items in problems.items():
        out[task] = {}
        for problem in items:
            record = {
                name: ext.problem_metrics(problem, *function(problem))
                for name, function in ext.BASELINES.items()
            }
            record["seasonal_naive"] = ext.problem_metrics(problem, *ext.seasonal_naive(problem))
            out[task][problem.series_id] = record
    return out


K3_RULE = (
    "PASS if the geometric-mean ratio is <= 0.95 and its upper bound < 1.0; FAIL if its lower bound > 0.95 "
    "(an improvement of at least 5% is excluded); otherwise INCONCLUSIVE"
)


def k3_status(result: Mapping[str, Any]) -> str:
    if result.get("interval_status") != "MEASURED":
        return "FAIL" if result.get("status") == "FAIL" else "INCONCLUSIVE"
    if result["geometric_ratio"] <= plan.STRESS_IMPROVEMENT and result["upper"] < 1.0:
        return "PASS"
    if result["lower"] > plan.STRESS_IMPROVEMENT:
        return "FAIL"
    return "INCONCLUSIVE"


def decide(
    primitives: Mapping[str, Sequence[Mapping[str, Any]]],
    challenger: str,
    *,
    persistence: Mapping[str, float],
    external_base: Mapping[str, Mapping[str, Any]],
    bootstrap: Sequence[int],
    families: Sequence[str],
    external: Sequence[str],
    monash_primitives: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    monash_keep: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """K1-K5 for ``challenger`` against Champion 1, from per-cell primitives.

    K5 covers NARMA-10/20, Mackey-Glass, the dysts systems and, when given, the
    Monash test horizons (forecast MASE). A Monash series whose MASE scale is
    undefined in the data (the archive drops those too) is excluded for every
    arm alike (``monash_keep``).
    """

    ref = primitives["c1_champion1"]
    cha = primitives[challenger]
    v1 = [f for f in families if f.startswith("v1_")]
    stress_groups = [f for f in families if not f.startswith("v1_")]
    dot_ref = [c for c in ref if "stream_id" in c]
    dot_cha = [c for c in cha if "stream_id" in c]
    criteria: dict[str, Any] = {}
    criteria["K1_stability"] = stability(ref, cha, persistence, external_base)
    criteria["K2_v1_noninferiority"] = v1_statuses(dot_ref, dot_cha, v1, bootstrap[0] % 2**32)
    k3 = geometric_status(dot_ref, dot_cha, stress_groups, bootstrap[1] % 2**32, external=False)
    k3["status"] = k3_status(k3)
    criteria["K3_stress_improvement"] = k3
    criteria["K4_regression_guard"] = regression_guard(dot_ref, dot_cha, stress_groups, bootstrap[2] % 2**32)
    k5_ref, k5_cha, k5_groups = list(ref), list(cha), list(external)
    if monash_primitives is not None:
        k5_ref += list(monash_primitives["c1_champion1"])
        k5_cha += list(monash_primitives[challenger])
        k5_groups += sorted({c["group"] for c in monash_primitives["c1_champion1"]})
    criteria["K5_external_noninferiority"] = geometric_status(
        k5_ref, k5_cha, k5_groups, bootstrap[3] % 2**32, external=True, keep_series=monash_keep
    )
    statuses = {key: value["status"] for key, value in criteria.items()}
    outcome = combine(list(statuses.values()))
    return {
        "criteria": criteria,
        "statuses": statuses,
        "outcome": {"PASS": "PROMOTE", "FAIL": "REJECT"}.get(outcome, "INCONCLUSIVE"),
    }


def run_confirmation(
    registry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    arms: Mapping[str, ArmSpec],
    *,
    observer: str,
    workers: Any,
    data_root: Any,
    log: Any = print,
) -> dict[str, Any]:
    started = time.time()
    cells = confirmation_cells(registry, observer)
    families = list(cells.families)
    log(
        f"confirmation cells: {len(cells.inits)} inits x {sum(b.size for b in cells.families.values())} dot streams, {sum(len(v) for v in cells.external.values())} external realizations"
    )
    baseline_records = {
        family: baselines_parallel(batch, workers=workers) for family, batch in cells.families.items()
    }
    persistence = {r["stream_id"]: r["persistence"] for records in baseline_records.values() for r in records}
    external_base = external_baselines(cells, EXTERNAL)
    external_flat = {sid: metrics for task in external_base.values() for sid, metrics in task.items()}
    problems = monash_problems(data_root)
    monash_base = monash_baselines(problems)
    primitives: dict[str, list[dict[str, Any]]] = {}
    for name, arm in arms.items():
        t0 = time.time()
        external = EXTERNAL if manifest["arms"][name]["external"] else ()
        primitives[name] = evaluate_arm(
            arm, cells, device=plan.CONFIRMATION_BACKEND, workers=workers, external=external
        )
        log(f"  {name}: {len(primitives[name])} cells, {time.time() - t0:.0f}s")
    monash: dict[str, list[dict[str, Any]]] = {}
    for name in manifest["monash_arms"]:
        t0 = time.time()
        monash[name] = monash_arm(arms[name], problems, cells.inits, workers=workers)
        log(f"  monash {name}: {len(monash[name])} cells, {time.time() - t0:.0f}s")
    capability: dict[str, Any] = {}
    paired = seeds_of(dict(registry), "v2-confirmation-env.paired_change")
    retention_seeds = seeds_of(dict(registry), "v2-confirmation-env.retention")
    probe_seeds = seeds_of(dict(registry), "v2-confirmation-probe.v1_aba")
    for name in manifest["capability_arms"]:
        t0 = time.time()
        capability[name] = {
            "online_frozen": online_frozen(arms[name], cells, workers=workers),
            "adaptation": adaptation(arms[name], cells.inits, paired),
            "retention": retention(arms[name], cells.inits, retention_seeds, probe_seeds),
        }
        log(f"  capability {name}: {time.time() - t0:.0f}s")
    bootstrap = seeds_of(dict(registry), "v2-confirmation-bootstrap")
    challenger = manifest.get("challenger")
    decision = (
        decide(
            primitives,
            challenger["name"],
            persistence=persistence,
            external_base=external_flat,
            bootstrap=bootstrap,
            families=families,
            external=EXTERNAL,
        )
        if challenger
        else {"outcome": "NO_CHALLENGER", "statuses": {}, "criteria": {}}
    )
    return {
        "schema": "aaa.1k.v2.confirmation.v1",
        "stage": "confirmation",
        "observer": observer,
        "freeze_sha256": manifest["_sha256"],
        "families": families,
        "external": list(EXTERNAL),
        "monash": list(MONASH),
        "inits": cells.inits,
        "primitives": primitives,
        "monash_primitives": monash,
        "capability": capability,
        "baselines": {"dot": baseline_records, "external": external_base, "monash": monash_base},
        "decision": decision,
        "compute_seconds": time.time() - started,
    }


def monash_summary(
    monash: Mapping[str, Sequence[Mapping[str, Any]]],
    baselines: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Mean forecast MASE per dataset and arm (per-series MASE averaged over inits first), with baselines."""

    from .external.monash import PUBLISHED_MASE

    out: dict[str, Any] = {}
    for task in MONASH:
        dataset = task.split(":", 1)[1]
        entry: dict[str, Any] = {"published": PUBLISHED_MASE[dataset], "arms": {}, "baselines": {}}
        for arm, cells in monash.items():
            per_series: dict[str, list[float]] = {}
            for cell in cells:
                if (
                    cell["group"] == task
                    and not cell.get("failed")
                    and cell.get("forecast_mase") is not None
                    and np.isfinite(cell["forecast_mase"])
                ):
                    per_series.setdefault(cell["series_id"], []).append(float(cell["forecast_mase"]))
            values = [float(np.mean(v)) for v in per_series.values()]
            entry["arms"][arm] = {
                "mean_mase": float(np.mean(values)) if values else None,
                "series": len(values),
            }
        for name in ("persistence", "seasonal_naive", "ar_rls"):
            values = [record[name].get("forecast_mase") for record in baselines[task].values()]
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            entry["baselines"][name] = {
                "mean_mase": float(np.mean(finite)) if finite else None,
                "series": len(finite),
            }
        out[task] = entry
    return out


def external_metric_of(group: str) -> str:
    return external_metric(group)


def _finite(value: Any) -> bool:
    return value is not None and bool(np.isfinite(value))
