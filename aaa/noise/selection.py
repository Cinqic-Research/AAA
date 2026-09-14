"""Bounded, development-only observation-noise candidate selection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..benchmark.evidence import sha256_file
from .candidates import CandidateDefinition, candidate_catalog
from .runner import AttemptResult, project_root, run_attempt
from .spec import canonical_protocol_hash, load_protocol
from .verifier import iter_records, verify_attempt


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _candidate_metrics(attempt: AttemptResult) -> dict[str, Any]:
    """Compute descriptive development metrics from primitive records."""

    records = list(iter_records(attempt.directory))
    by_cell: dict[tuple[str, str, str, float, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    adaptation: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        trial = record["trial"]
        if trial["channel"] not in {"gaussian", "uniform"} or float(trial["scale"]) not in {
            0.0005,
            0.002,
        }:
            continue
        if trial["branch"] == "stationary" and trial["family"] in {
            "constant_velocity",
            "bouncing",
            "changed_law",
            "speed_change",
        }:
            cell_key = (
                str(trial["condition"]),
                str(trial["family"]),
                str(trial["channel"]),
                float(trial["scale"]),
                str(trial["branch"]),
            )
            for predictor in (
                "selected_candidate_online",
                "constant_motion_reflected",
                "incumbent_square_root_rls",
            ):
                if predictor in record["predictions"]:
                    by_cell[cell_key][predictor].append(
                        float(record["predictions"][predictor]["latent_normalized_absolute_error"])
                    )
        if (
            trial["family"] == "changed_law"
            and trial["branch"] in {"frozen", "online"}
            and int(record["step"]) < 350
        ):
            adaptation_key = (str(trial["condition"]), str(trial["channel"]), float(trial["scale"]))
            predictor = (
                "selected_candidate_branch_online"
                if trial["branch"] == "online"
                else "selected_candidate_branch_frozen"
            )
            if predictor in record["predictions"]:
                adaptation[adaptation_key][predictor].append(
                    float(record["predictions"][predictor]["latent_normalized_absolute_error"])
                )
    cell_rows: list[dict[str, Any]] = []
    for cell_key, cell_values in sorted(by_cell.items(), key=str):
        candidate = _mean(cell_values.get("selected_candidate_online", []))
        baseline = _mean(cell_values.get("constant_motion_reflected", []))
        incumbent = _mean(cell_values.get("incumbent_square_root_rls", []))
        cell_rows.append(
            {
                "condition": cell_key[0],
                "family": cell_key[1],
                "channel": cell_key[2],
                "scale": cell_key[3],
                "branch": cell_key[4],
                "candidate_mae": candidate,
                "reflected_baseline_mae": baseline,
                "incumbent_mae": incumbent,
                "candidate_minus_1_10_baseline": (
                    None if candidate is None or baseline is None else candidate - 1.10 * baseline
                ),
                "incumbent_minus_candidate": (
                    None if candidate is None or incumbent is None else incumbent - candidate
                ),
            }
        )
    adaptation_rows: list[dict[str, Any]] = []
    for adaptation_key, adaptation_values in sorted(adaptation.items(), key=str):
        frozen = _mean(adaptation_values.get("selected_candidate_branch_frozen", []))
        online = _mean(adaptation_values.get("selected_candidate_branch_online", []))
        adaptation_rows.append(
            {
                "condition": adaptation_key[0],
                "channel": adaptation_key[1],
                "scale": adaptation_key[2],
                "frozen_first50_mae": frozen,
                "online_first50_mae": online,
                "point_reduction": (
                    None if frozen in {None, 0.0} or online is None else (frozen - online) / frozen
                ),
            }
        )
    candidates = [row["candidate_mae"] for row in cell_rows if row["candidate_mae"] is not None]
    baselines = [
        row["candidate_minus_1_10_baseline"]
        for row in cell_rows
        if row["candidate_minus_1_10_baseline"] is not None
    ]
    incumbent_gains = [
        row["incumbent_minus_candidate"] for row in cell_rows if row["incumbent_minus_candidate"] is not None
    ]
    reductions = [row["point_reduction"] for row in adaptation_rows if row["point_reduction"] is not None]
    return {
        "primary_candidate_mae_mean": _mean(candidates),
        "baseline_margin_mean": _mean(baselines),
        "incumbent_gain_mean": _mean(incumbent_gains),
        "adaptation_reduction_mean": _mean(reductions),
        "cells": cell_rows,
        "adaptation_cells": adaptation_rows,
        "primitive_verifier": verify_attempt(attempt.directory)["verdict"],
    }


def _ledger_entry(
    definition: CandidateDefinition,
    attempt: AttemptResult,
    metrics: dict[str, Any],
    selected_candidate: str,
) -> dict[str, Any]:
    outcome = "selected" if definition.candidate_id == selected_candidate else "rejected"
    reason = (
        "Selected by the preregistered conjunctive development ranking; confirmation remains unexecuted."
        if outcome == "selected"
        else "Not selected by the preregistered development ranking; retained as a rejected bounded attempt."
    )
    entry = definition.to_dict()
    entry.update(
        {
            "development_attempts": [attempt.directory.name],
            "outcome": outcome,
            "outcome_reason": reason,
            "development_evidence": {
                "attempt_id": attempt.directory.name,
                "archive_locator": "transient_local_selection_root_not_committed",
                "summary_sha256": sha256_file(attempt.directory / "summary.json"),
                "records_sha256": sha256_file(attempt.directory / "records.jsonl"),
                "metrics": metrics,
            },
        }
    )
    return entry


def run_development_selection(
    *,
    output_root: str | Path,
    evidence_path: str | Path,
    quick: bool,
    reuse_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the committed catalog without consuming confirmation data."""

    protocol = load_protocol()
    plan_path = project_root() / "benchmarks/observation_noise_development_selection.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "selection_plan_frozen"
        or plan.get("frozen_before_comparative_development") is not True
    ):
        raise ValueError("development selection plan is not frozen")
    catalog = candidate_catalog()
    if list(catalog) != plan.get("candidates"):
        raise ValueError("development plan candidate catalog differs from code")
    if len(catalog) > int(protocol.raw["search_budget"]["max_candidate_configurations"]):
        raise ValueError("candidate search exceeds the frozen configuration budget")
    attempts: dict[str, AttemptResult] = {}
    metrics_by_candidate: dict[str, dict[str, Any]] = {}
    for candidate_id in catalog:
        if reuse_root is None:
            attempt = run_attempt(
                role="development",
                output_root=output_root,
                attempt_label=f"selection-{candidate_id}",
                quick=quick,
                candidate_id=candidate_id,
            )
        else:
            directory = Path(reuse_root) / "observation-noise-v1" / f"selection-{candidate_id}"
            summary_path = directory / "summary.json"
            if not summary_path.is_file():
                raise ValueError(f"cannot reuse incomplete selection attempt: {directory}")
            attempt = AttemptResult(
                directory=directory, summary=json.loads(summary_path.read_text(encoding="utf-8"))
            )
        attempts[candidate_id] = attempt
        metrics_by_candidate[candidate_id] = _candidate_metrics(attempt)
    # The incumbent is the no-refinement control. A refinement is eligible only
    # if its development point estimates satisfy every declared practical
    # constraint; ties resolve by configuration hash then ID.
    eligible = [
        candidate_id
        for candidate_id, metrics in metrics_by_candidate.items()
        if candidate_id != "incumbent-no-refinement-v1"
        and (metrics["primary_candidate_mae_mean"] or float("inf")) <= 0.02
        and (metrics["baseline_margin_mean"] or float("inf")) <= 0.00001
        and (metrics["adaptation_reduction_mean"] or float("-inf")) >= 0.10
    ]
    selected = min(
        eligible or ["incumbent-no-refinement-v1"],
        key=lambda candidate_id: (catalog[candidate_id].configuration_hash, candidate_id),
    )
    selection = {
        "schema_version": "aaa.observation_noise_development_selection_evidence.v1",
        "protocol_version": protocol.protocol_version,
        "protocol_hash": canonical_protocol_hash(),
        "plan_path": str(plan_path),
        "quick": quick,
        "selected_candidate": selected,
        "candidate_order": list(catalog),
        "attempts": {candidate_id: attempts[candidate_id].directory.name for candidate_id in catalog},
        "archive_locator": "transient_local_selection_root_not_committed",
        "metrics": metrics_by_candidate,
        "ranking": {
            "eligible_refinements": eligible,
            "tie_break": "configuration_hash ascending then candidate_id ascending",
            "practical_margin": 0.00001,
            "decision": "No refinement is selected unless all preregistered development constraints are met.",
        },
    }
    evidence_destination = Path(evidence_path)
    evidence_destination.parent.mkdir(parents=True, exist_ok=True)
    evidence_destination.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_path = project_root() / protocol.raw["search_budget"]["candidate_ledger_file"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["status"] = "development_complete"
    ledger["selected_candidate"] = selected
    ledger["selection_evidence"] = str(evidence_destination)
    ledger["entries"] = []
    for candidate_id, definition in catalog.items():
        ledger["entries"].append(
            _ledger_entry(definition, attempts[candidate_id], metrics_by_candidate[candidate_id], selected)
        )
    ledger["notes"] = "All bounded development attempts are retained. Confirmation data was not inspected."
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return selection
