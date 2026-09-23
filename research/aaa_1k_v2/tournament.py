"""Development stage: per-candidate hyperparameter selection, then the screen.

Everything this module decides follows :mod:`research.aaa_1k_v2.plan`, which
was committed before any development identity was observed.

1. **Grid.** Every candidate runs every configuration in
   ``LEARNING_RATES x TBPTT_OPTIONS x GRADIENT_CLIPS`` (``RTRL`` candidates:
   ``LEARNING_RATES x GRADIENT_CLIPS``) on every development cell: 4
   initializations x 12 streams x 17 development families, plus NARMA-10,
   NARMA-20 and Mackey-Glass development realizations. Champion 1 runs its
   frozen configuration on the same cells.
2. **Eligibility.** A configuration is eliminated by any failed or diverged
   development cell (dot: MAE above twice persistence's; external: a failure,
   a non-finite forecast, or prequential error above twice persistence's) and
   by the stability margin: its learning rate must sit at least two grid
   positions below the lowest learning rate at which the *unclipped* reference
   configuration (T=4 live, or RTRL) diverged anywhere.
3. **Selection score.** Geometric mean, over the 17 dot families and the three
   external tasks, of the configuration's mean error divided by Champion 1's
   on the same cells. Lower is better. Ties within 2% follow ``plan.TIE_MARGIN``.
4. **Screen.** Each candidate's selected configuration against Champion 1 on
   the same development cells: S2 v1 non-inferiority, S3 stress superiority,
   S4 external non-inferiority, each with the crossed bootstrap. The passing
   candidate with the lowest S3 point estimate becomes the challenger.

The committed artifact keeps, for every configuration, its eligibility, its
failure and divergence counts, its per-group means and its score; and for
every candidate's selected configuration and for Champion 1, the complete
per-cell primitives, so the screen can be recomputed from them.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import PHASE_VERSION, plan, stress
from .arms import CANDIDATE_TEMPLATES, CHAMPION_1, ArmSpec, audit
from .baselines import baselines_parallel
from .engine import CellConfig
from .external import tasks as ext
from .jobs import DotJob, SeriesJob, run_jobs
from .runner import StreamBatch
from .stats import combine, geometric_relative, noninferior, relative_crossed, superior

EXTERNAL_SELECTION_TASKS = ("narma10", "narma20", "mackey_glass17")


def candidate_template(name: str) -> ArmSpec:
    if name == "gru_v1_retuned":
        return replace(
            CHAMPION_1,
            name="gru_v1_retuned",
            kind="candidate",
            description=(
                "Champion 1's architecture and target rule with hyperparameters re-selected under the v2 rule, "
                "separating architecture from hyperparameter effects"
            ),
        )
    return CANDIDATE_TEMPLATES[name]


@dataclass(frozen=True)
class GridPoint:
    learning_rate: float
    tbptt_steps: int
    rule: str
    gradient_clip: float | None

    def key(self) -> str:
        clip = "none" if self.gradient_clip is None else f"{self.gradient_clip:g}"
        return f"lr={self.learning_rate:g};T={self.tbptt_steps};rule={self.rule};clip={clip}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "tbptt_steps": self.tbptt_steps,
            "rule": self.rule,
            "gradient_clip": self.gradient_clip,
        }


def grid(template: ArmSpec) -> list[GridPoint]:
    if template.rule == "rtrl":
        return [
            GridPoint(lr, 0, "rtrl", clip)
            for lr, clip in itertools.product(plan.LEARNING_RATES, plan.GRADIENT_CLIPS)
        ]
    return [
        GridPoint(lr, steps, rule, clip)
        for lr, (steps, rule), clip in itertools.product(
            plan.LEARNING_RATES, plan.TBPTT_OPTIONS, plan.GRADIENT_CLIPS
        )
    ]


def arm_at(template: ArmSpec, point: GridPoint) -> ArmSpec:
    return replace(
        template,
        rule=point.rule,
        tbptt_steps=point.tbptt_steps if point.rule != "rtrl" else 0,
        config=CellConfig(
            learning_rate=point.learning_rate,
            gradient_clip=point.gradient_clip,
            error_loss_weight=plan.ERROR_LOSS_WEIGHT,
        ),
    )


# ----------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------
@dataclass
class Cells:
    """The development (or attack) cell set: dot families and external problems."""

    role: str
    inits: list[int]
    families: dict[str, StreamBatch]
    external: dict[str, list[ext.SeriesProblem]]


def build_cells(
    registry: Mapping[str, Any],
    role: str,
    families: Sequence[str],
    external: Sequence[str],
    observer: str | None = None,
) -> Cells:
    inits = plan.block_seeds_for(dict(registry), role, "init", observer=observer)
    batches = {}
    for family in families:
        seeds = plan.block_seeds_for(dict(registry), role, f"env.{family}", observer=observer)
        batches[family] = stress.batch(family, seeds)
    problems: dict[str, list[ext.SeriesProblem]] = {}
    for task in external:
        if task.startswith("dysts:"):
            system = task.split(":", 1)[1]
            seeds = plan.block_seeds_for(dict(registry), role, f"ext.dysts-{system}", observer=observer)
            problems[task] = ext.dysts_problems(system, seeds)
        elif task == "mackey_glass17":
            problems[task] = ext.mackey_glass_problems(
                plan.block_seeds_for(dict(registry), role, f"ext.{task}", observer=observer)
            )
        else:
            problems[task] = ext.narma_problems(
                task, plan.block_seeds_for(dict(registry), role, f"ext.{task}", observer=observer)
            )
    return Cells(role, list(inits), batches, problems)


def dot_jobs(
    arm: ArmSpec, points: Sequence[GridPoint], cells: Cells, device: str, tag: str
) -> tuple[list[DotJob], list[list[tuple[str, int, str]]]]:
    """One job per family; each job crosses points x inits x streams. Returns jobs and their cell index."""

    jobs, index = [], []
    for family, batch in cells.families.items():
        seeds, configs, rows, labels = [], [], [], []
        for point in points:
            config = arm_at(arm, point).config
            for init in cells.inits:
                for row in range(batch.size):
                    seeds.append(init)
                    configs.append(config)
                    rows.append(row)
                    labels.append((point.key(), init, batch.stream_ids[row]))
        base = arm_at(arm, points[0])
        jobs.append(
            DotJob(
                arm=base,
                seeds=tuple(seeds),
                batch=batch.take(rows),
                configs=tuple(configs),
                device=device,
                tag=f"{tag}:{family}",
            )
        )
        index.append(labels)
    return jobs, index


def series_jobs(
    arm: ArmSpec, points: Sequence[GridPoint], cells: Cells, tasks: Sequence[str], device: str, tag: str
) -> tuple[list[SeriesJob], list[list[tuple[str, int, str]]]]:
    jobs, index = [], []
    for task in tasks:
        problems = cells.external[task]
        seeds, configs, probs, labels = [], [], [], []
        for point in points:
            config = arm_at(arm, point).config
            for init in cells.inits:
                for problem in problems:
                    seeds.append(init)
                    configs.append(config)
                    probs.append(problem)
                    labels.append((point.key(), init, problem.series_id))
        jobs.append(
            SeriesJob(
                arm=arm_at(arm, points[0]),
                seeds=tuple(seeds),
                problems=tuple(probs),
                configs=tuple(configs),
                device=device,
                tag=f"{tag}:{task}",
            )
        )
        index.append(labels)
    return jobs, index


def external_metric(task: str) -> str:
    """The scoring metric of an external task: NMSE for NARMA, forecast MASE for Monash, else NRMSE."""

    if task.startswith("monash:"):
        return "forecast_mase"
    return "prequential_nmse" if task.startswith("narma") else "prequential_nrmse"


def divergence_metric(task: str) -> str:
    """Divergence is judged on the prequential (online) error for every external task."""

    return "prequential_nmse" if task.startswith("narma") else "prequential_nrmse"


def external_baselines(cells: Cells, tasks: Sequence[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for task in tasks:
        out[task] = {}
        for problem in cells.external[task]:
            out[task][problem.series_id] = {
                name: ext.problem_metrics(problem, *function(problem))
                for name, function in ext.BASELINES.items()
            }
    return out


# ----------------------------------------------------------------------
# collection
# ----------------------------------------------------------------------
def collect(
    results: Sequence[dict[str, Any]], index: Sequence[Sequence[tuple[str, int, str]]]
) -> dict[str, list[dict[str, Any]]]:
    """Per grid-point key, the flat list of cell records (with their init index and group)."""

    by_point: dict[str, list[dict[str, Any]]] = {}
    for result, labels in zip(results, index, strict=True):
        group = result["tag"].split(":", 1)[1]
        # run_jobs concatenates chunk cells in submission order, which is the label order
        if len(result["cells"]) != len(labels):
            raise RuntimeError(f"{result['tag']}: {len(result['cells'])} cells for {len(labels)} labels")
        for cell, (key, init, stream) in zip(result["cells"], labels, strict=True):
            if cell.get("stream_id", cell.get("series_id")) != stream or cell["seed"] != init:
                raise RuntimeError(f"{result['tag']}: cell order does not match its labels")
            by_point.setdefault(key, []).append({**cell, "group": group, "init": init})
    return by_point


def diverged(
    cell: Mapping[str, Any],
    persistence: Mapping[str, float],
    external_persistence: Mapping[str, Mapping[str, Any]],
) -> bool:
    if cell.get("failed"):
        return True
    if "stream_id" in cell:
        return cell["mae"] is None or cell["mae"] > plan.DIVERGENCE_FACTOR * persistence[cell["stream_id"]]
    metric = divergence_metric(cell["group"])
    base = external_persistence[cell["series_id"]]["persistence"][metric]
    value = cell.get(metric)
    return (
        value is None
        or not math.isfinite(value)
        or value > plan.DIVERGENCE_FACTOR * base
        or cell.get("forecast_finite") is False
    )


def group_means(cells: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for cell in cells:
        if cell.get("failed"):
            continue
        value = cell["mae"] if "stream_id" in cell else cell.get(external_metric(cell["group"]))
        if value is not None and math.isfinite(value):
            sums.setdefault(cell["group"], []).append(float(value))
    return {group: float(np.mean(values)) for group, values in sums.items()}


def score(means: Mapping[str, float], reference: Mapping[str, float], groups: Sequence[str]) -> float | None:
    if any(group not in means or group not in reference for group in groups):
        return None
    return float(math.exp(np.mean([math.log(means[g] / reference[g]) for g in groups])))


def select(points: Sequence[GridPoint], summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply eligibility, the stability margin and the tie rule. Returns the decision record."""

    rates = list(plan.LEARNING_RATES)
    reference = [
        p
        for p in points
        if p.gradient_clip is None and (p.rule == "rtrl" or (p.tbptt_steps == 4 and p.rule == "live"))
    ]
    diverging = sorted(p.learning_rate for p in reference if summaries[p.key()]["unstable_cells"] > 0)
    if diverging:
        boundary = diverging[0]
        limit_index = rates.index(boundary) - plan.MARGIN_STEPS
        eligible_rates = set(rates[: max(limit_index + 1, 0)])
        margin_note = (
            f"unclipped reference diverged at lr={boundary:g}; eligible lr <= {rates[limit_index]:g}"
            if limit_index >= 0
            else f"unclipped reference diverged at lr={boundary:g}; no rate is eligible"
        )
    else:
        boundary = None
        eligible_rates = set(rates)
        margin_note = "the unclipped reference never diverged on this grid: the margin was not demonstrated"
    eligible = [
        p
        for p in points
        if p.learning_rate in eligible_rates
        and summaries[p.key()]["unstable_cells"] == 0
        and summaries[p.key()]["score"] is not None
    ]
    if not eligible:
        return {"selected": None, "boundary_learning_rate": boundary, "margin": margin_note, "eligible": 0}
    best = min(summaries[p.key()]["score"] for p in eligible)
    tied = [p for p in eligible if summaries[p.key()]["score"] <= best * (1.0 + plan.TIE_MARGIN)]

    def preference(p: GridPoint) -> tuple[Any, ...]:
        clip_rank = math.inf if p.gradient_clip is None else p.gradient_clip
        return (
            p.learning_rate,
            0 if p.rule in ("live", "rtrl") else 1,
            p.tbptt_steps,
            -clip_rank,
            summaries[p.key()]["score"],
        )

    chosen = min(tied, key=preference)
    return {
        "selected": chosen.to_dict(),
        "selected_key": chosen.key(),
        "selected_score": summaries[chosen.key()]["score"],
        "best_score": best,
        "tied_within_margin": [p.key() for p in tied],
        "boundary_learning_rate": boundary,
        "margin": margin_note,
        "eligible": len(eligible),
    }


# ----------------------------------------------------------------------
# the screen (also used, with other thresholds, by attack and confirmation)
# ----------------------------------------------------------------------
def crossed_grid(
    cells: Sequence[Mapping[str, Any]], group: str, value: str = "mae"
) -> tuple[np.ndarray, list[int], list[str]]:
    rows = [c for c in cells if c["group"] == group]
    inits = sorted({c["init"] for c in rows})
    streams = sorted({c.get("stream_id", c.get("series_id")) for c in rows})
    table = {(c["init"], c.get("stream_id", c.get("series_id"))): c for c in rows}
    grid_values = np.full((len(inits), len(streams)), np.nan)
    for i, init in enumerate(inits):
        for j, stream in enumerate(streams):
            cell = table.get((init, stream))
            if cell is not None and not cell.get("failed") and cell.get(value) is not None:
                grid_values[i, j] = float(cell[value])
    return grid_values, inits, streams


def compare(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    v1: Sequence[str],
    stress_groups: Sequence[str],
    external: Sequence[str],
    seed_base: int,
    draws: int,
) -> dict[str, Any]:
    """S2/S3/S4-style statistics of ``candidate`` against ``reference`` on shared cells."""

    out: dict[str, Any] = {"v1": {}, "stress": None, "external": None, "incomplete": []}
    for offset, family in enumerate(v1):
        a, _, _ = crossed_grid(reference, family)
        b, _, _ = crossed_grid(candidate, family)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            out["incomplete"].append(family)
            out["v1"][family] = {"status": "FAIL", "reason": "failed or missing cells"}
            continue
        result = relative_crossed(a, b, seed=seed_base + offset, draws=draws)
        out["v1"][family] = {**result, "status": noninferior(result, plan.NONINFERIORITY_MARGIN)}

    def geometric(groups: Sequence[str], value_of: Any, seed: int) -> dict[str, Any] | None:
        pairs = {}
        for group in groups:
            a, _, _ = crossed_grid(reference, group, value_of(group))
            b, _, _ = crossed_grid(candidate, group, value_of(group))
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                out["incomplete"].append(group)
                return None
            pairs[group] = (a, b)
        return geometric_relative(pairs, seed=seed, draws=draws)

    stress_result = geometric(stress_groups, lambda _g: "mae", seed_base + 100)
    out["stress"] = stress_result
    if external:
        out["external"] = geometric(external, external_metric, seed_base + 200)
    return out


def screen_statuses(comparison: Mapping[str, Any]) -> dict[str, str]:
    s2 = combine([entry["status"] for entry in comparison["v1"].values()])
    stress_result = comparison["stress"]
    s3 = "FAIL" if stress_result is None else superior(stress_result, 1.0)
    external = comparison["external"]
    s4 = (
        "FAIL"
        if external is None
        else noninferior(
            {**external, "upper": external["upper"] - 1.0, "lower": external["lower"] - 1.0}
            if external.get("interval_status") == "MEASURED"
            else external,
            plan.EXTERNAL_MARGIN,
        )
    )
    return {"S2_v1_noninferiority": s2, "S3_stress_superiority": s3, "S4_external_noninferiority": s4}


# ----------------------------------------------------------------------
# the stage
# ----------------------------------------------------------------------
def run_development(
    registry: Mapping[str, Any],
    *,
    workers: int | str | None,
    device: str = plan.DEVELOPMENT_BACKEND,
    candidates: Sequence[str] = plan.CANDIDATES,
    role: str = "development",
    raw_dir: Any = None,
    log: Any = print,
) -> dict[str, Any]:
    """The development stage (``role="scratch"`` runs the identical pipeline on scratch identities).

    ``raw_dir``, if given, receives one gzip strict-JSON archive per arm with
    every grid point's per-cell primitives; the artifact records its SHA-256.
    """

    if role not in ("development", "scratch"):
        raise ValueError("the development stage runs on development identities (or scratch for shakedowns)")
    started = time.time()
    families = plan.STAGE_FAMILIES["development"]
    cells = build_cells(registry, role, families, EXTERNAL_SELECTION_TASKS)
    log(
        f"development cells: {len(cells.inits)} inits x {sum(b.size for b in cells.families.values())} dot streams + external {sum(len(v) for v in cells.external.values())}"
    )
    baseline_records = {
        family: baselines_parallel(batch, workers=workers) for family, batch in cells.families.items()
    }
    persistence = {r["stream_id"]: r["persistence"] for records in baseline_records.values() for r in records}
    external_base = external_baselines(cells, EXTERNAL_SELECTION_TASKS)
    external_persistence = {sid: metrics for task in external_base.values() for sid, metrics in task.items()}
    groups = [*families, *EXTERNAL_SELECTION_TASKS]

    def evaluate(arm: ArmSpec, points: Sequence[GridPoint], tag: str) -> dict[str, list[dict[str, Any]]]:
        results_by_point: dict[str, list[dict[str, Any]]] = {}

        # one job set per (rule, horizon): a lockstep batch needs one window length
        def keyfn(point: GridPoint) -> tuple[str, int]:
            return (point.rule, point.tbptt_steps)

        for _key, group in itertools.groupby(sorted(points, key=keyfn), key=keyfn):
            members = list(group)
            template = arm_at(arm, members[0])
            djobs, dindex = dot_jobs(template, members, cells, device, tag)
            sjobs, sindex = series_jobs(template, members, cells, EXTERNAL_SELECTION_TASKS, device, tag)
            t0 = time.time()
            results = run_jobs([*djobs, *sjobs], workers=workers)
            log(
                f"  {tag} {template.rule} T={template.tbptt_steps}: {len(members)} configs, {sum(len(r['cells']) for r in results)} cells, {time.time() - t0:.0f}s"
            )
            for key, value in collect(results, [*dindex, *sindex]).items():
                results_by_point.setdefault(key, []).extend(value)
        return results_by_point

    champion_point = GridPoint(
        CHAMPION_1.config.learning_rate,
        CHAMPION_1.tbptt_steps,
        CHAMPION_1.rule,
        CHAMPION_1.config.gradient_clip,
    )
    champion_cells = evaluate(CHAMPION_1, [champion_point], "c1_champion1")[champion_point.key()]
    champion_means = group_means(champion_cells)
    champion_unstable = sum(diverged(c, persistence, external_persistence) for c in champion_cells)

    report: dict[str, Any] = {}
    selected_cells: dict[str, list[dict[str, Any]]] = {"c1_champion1": champion_cells}
    for name in candidates:
        template = candidate_template(name)
        points = grid(template)
        by_point = evaluate(template, points, name)
        summaries: dict[str, dict[str, Any]] = {}
        for point in points:
            point_cells = by_point[point.key()]
            unstable = [c for c in point_cells if diverged(c, persistence, external_persistence)]
            means = group_means(point_cells)
            summaries[point.key()] = {
                **point.to_dict(),
                "cells": len(point_cells),
                "failed_cells": sum(1 for c in point_cells if c.get("failed")),
                "unstable_cells": len(unstable),
                "unstable_groups": sorted({c["group"] for c in unstable}),
                "group_means": means,
                "score": score(means, champion_means, groups) if not unstable else None,
                "clip_rate_mean": float(
                    np.mean([c.get("clip_rate", 0.0) for c in point_cells if "clip_rate" in c])
                )
                if any("clip_rate" in c for c in point_cells)
                else None,
            }
        decision = select(points, summaries)
        report[name] = {
            "arm": template.to_dict(),
            "audit": audit(template),
            "grid": summaries,
            "decision": decision,
        }
        if raw_dir is not None:
            from pathlib import Path

            from .evidence import write

            archive = Path(raw_dir) / f"{role}_{name}.json.gz"
            report[name]["raw_archive"] = {
                "file": archive.name,
                "sha256": write(archive, by_point, overwrite=True, indent=None),
            }
        if decision["selected"] is not None:
            selected_cells[name] = by_point[decision["selected_key"]]
        log(f"{name}: selected {decision.get('selected_key')} score {decision.get('selected_score')}")

    stress_groups = [f for f in families if not f.startswith("v1_")]
    v1_groups = [f for f in families if f.startswith("v1_")]
    from .identities import seeds_of

    bootstrap = seeds_of(dict(registry), f"v2-{role}-bootstrap")
    screens: dict[str, Any] = {}
    for offset, name in enumerate(candidates):
        if name not in selected_cells:
            screens[name] = {
                "statuses": {"S1_stability": "FAIL"},
                "passed": False,
                "reason": "no eligible configuration",
            }
            continue
        comparison = compare(
            champion_cells,
            selected_cells[name],
            v1=v1_groups,
            stress_groups=stress_groups,
            external=EXTERNAL_SELECTION_TASKS,
            seed_base=bootstrap[offset] % (2**32),
            draws=plan.BOOTSTRAP_DRAWS,
        )
        statuses = {"S1_stability": "PASS", **screen_statuses(comparison)}
        screens[name] = {
            "statuses": statuses,
            "passed": all(v == "PASS" for v in statuses.values()),
            "comparison": comparison,
        }
    passing = [n for n in candidates if screens[n]["passed"]]
    challenger = (
        min(passing, key=lambda n: screens[n]["comparison"]["stress"]["geometric_ratio"]) if passing else None
    )
    return {
        "schema": "aaa.1k.v2.development.v1",
        "phase": PHASE_VERSION,
        "protocol": plan.PROTOCOL_VERSION,
        "stage": role,
        "backend": device,
        "workers": str(workers),
        "families": list(families),
        "external_selection_tasks": list(EXTERNAL_SELECTION_TASKS),
        "inits": cells.inits,
        "champion": {
            "arm": CHAMPION_1.to_dict(),
            "group_means": champion_means,
            "unstable_cells": champion_unstable,
        },
        "candidates": report,
        "screens": screens,
        "challenger": challenger,
        "selected_cells": selected_cells,
        "baselines": {"dot": baseline_records, "external": external_base},
        "compute_seconds": time.time() - started,
    }
