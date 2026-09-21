"""Champion 0: the reviewed AAA-1K GRU, frozen as the loop's starting point.

The champion record is *derived*, not typed. It is built from the live
repository (phase fingerprint, parameter arrays, state footprint) and from the
committed evidence files (selected hyperparameters, round-3 capability
vector), and :func:`verify_champion` rebuilds it and refuses any disagreement.
A stale fingerprint, a rewritten evidence file or a changed parameter count is
therefore caught mechanically rather than noticed by a reader.

Nothing here modifies AAA-1K. Round 3 stays the historical baseline.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.aaa_1k import PHASE_VERSION
from research.aaa_1k.controls import VanillaRNNControl
from research.aaa_1k.identity import phase_fingerprint
from research.aaa_1k.model import AAA1KGRU, ARCHITECTURE_ID, MODEL_FORMAT_VERSION, expected_parameter_count
from research.aaa_1k.streams import BENCHMARK_VERSION

from .evidence import read_strict_json

CHAMPION_SCHEMA = "aaa.loop.champion.v1"
CHAMPION_ID = "aaa1k-champion-0"

DEVELOPMENT_EVIDENCE = "docs/evidence/aaa_1k_development_selection.json"
ROUND3_EVIDENCE = "docs/evidence/aaa_1k_evaluation_round3.json"
CHARACTERIZATION_EVIDENCE = "docs/evidence/aaa_1k_characterization.json"

KNOWN_LIMITATIONS = (
    "Q4 gating vs ungated control is INCONCLUSIVE overall and NEGATIVE on the memory families",
    "Q4 mean and median disagree in sign; a minority of streams carries the aggregate",
    "Q5 forgetting is INCONCLUSIVE; no forgetting measured on the fixed probe bank only",
    "Q6 self-error head is weakly informative and over-predicts",
    "Q7 comparisons are exploratory; loses to analytic baselines on smooth fully observed motion",
    "occlusion missingness code (inputs 2 and 3 zero) is ambiguous with genuine stationarity",
    "TBPTT horizon barely matters on these benchmarks; no long-range credit assignment is exercised",
    "occlusion_v1 and coarse_speed_v1 were designed by the implementer whose model they evaluate",
    "five initializations is a small second bootstrap level",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_verdict(record: Mapping[str, Any]) -> str:
    """Recomputed from the interval, never read from a stored label."""

    if record.get("interval_status") != "MEASURED":
        return "INSUFFICIENT_EVIDENCE"
    low, high = float(record["ci_low"]), float(record["ci_high"])
    if not (math.isfinite(low) and math.isfinite(high)):
        return "INSUFFICIENT_EVIDENCE"
    if low > 0:
        return "POSITIVE"
    if high < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def _headline(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean_difference": float(record["mean_difference"]),
        "median_difference": float(record["median_difference"]),
        "ci_low": float(record["ci_low"]),
        "ci_high": float(record["ci_high"]),
        "favours_first": int(record["favours_first"]),
        "favours_second": int(record["favours_second"]),
        "streams": int(record["streams"]),
        "verdict": interval_verdict(record),
    }


def capability_summary(round3: Mapping[str, Any]) -> dict[str, Any]:
    """Q1-Q7 headline records, with verdicts recomputed from the intervals."""

    dims = round3["capability_vector"]["dimensions"]
    q6 = dims["q6_error_calibration"]
    q7 = dims["q7_baseline_competitiveness"]["by_family"]
    return {
        "Q1_online_learning": _headline(dims["q1_online_learning"]["overall"]),
        "Q2_adaptation": _headline(dims["q2_adaptation"]["adaptation_effect"]),
        "Q3_hidden_state_vs_stateless": _headline(dims["q3_hidden_state"]["versus_stateless_mlp"]),
        "Q3_hidden_state_vs_state_reset": _headline(dims["q3_hidden_state"]["versus_state_reset"]),
        "Q4_gating_all_families": _headline(dims["q4_gating"]["all_families"]),
        "Q4_gating_memory_families": _headline(dims["q4_gating"]["memory_families"]),
        "Q5_forgetting": _headline(dims["q5_retention"]["forgetting"]),
        "Q6_error_calibration": {
            "spearman_mean": float(q6["spearman"]["mean"]),
            "bias_mean": float(q6["bias"]["mean"]),
            "cells_with_monotone_bins": int(q6["cells_with_monotone_bins"]),
            "cells_measured": int(q6["cells_measured"]),
        },
        "Q7_baselines": {
            family: {baseline: interval_verdict(record) for baseline, record in sorted(rows.items())}
            for family, rows in sorted(q7.items())
        },
    }


def build_champion(root: Path, *, source_commit: str) -> dict[str, Any]:
    """Derive Champion 0 from the repository at ``root``."""

    fingerprint = phase_fingerprint(root)
    selection = read_strict_json(root / DEVELOPMENT_EVIDENCE)
    round3 = read_strict_json(root / ROUND3_EVIDENCE)
    if selection["scientific_fingerprint"] != fingerprint["sha256"]:
        raise RuntimeError("development selection evidence does not carry the current phase fingerprint")
    if round3["scientific_fingerprint"] != fingerprint["sha256"]:
        raise RuntimeError("round-3 evidence does not carry the current phase fingerprint")
    selected = dict(selection["architecture_configurations"]["AAA1KGRU"])
    model = AAA1KGRU(seed=0, **selected)
    ungated = VanillaRNNControl(seed=0, **selection["architecture_configurations"]["VanillaRNNControl"])
    return {
        "schema": CHAMPION_SCHEMA,
        "champion_id": CHAMPION_ID,
        "parent_model_id": None,
        "lineage_note": "the reviewed AAA-1K round-3 model; aaa.1k.v1 is not redefined by this loop",
        "source_commit": source_commit,
        "phase_version": PHASE_VERSION,
        "phase_fingerprint": fingerprint["sha256"],
        "phase_fingerprint_file_count": fingerprint["file_count"],
        "model_format": MODEL_FORMAT_VERSION,
        "architecture_id": ARCHITECTURE_ID,
        "parameter_count": model.parameter_count(),
        "parameter_count_formula": expected_parameter_count(),
        "state_footprint": model.state_footprint(),
        "selected_hyperparameters": selected,
        "control_hyperparameters": {
            name: dict(value)
            for name, value in selection["architecture_configurations"].items()
            if name != "AAA1KGRU"
        },
        "ungated_control": {
            "architecture_id": ungated.architecture_id,
            "parameter_count": ungated.parameter_count(),
            "hidden_units": ungated.hidden_size,
        },
        "benchmark_version": BENCHMARK_VERSION,
        "development_evidence": {
            "path": DEVELOPMENT_EVIDENCE,
            "sha256": sha256_file(root / DEVELOPMENT_EVIDENCE),
        },
        "round3_evidence": {
            "path": ROUND3_EVIDENCE,
            "sha256": sha256_file(root / ROUND3_EVIDENCE),
            "schema": round3["schema"],
            "cells": len(round3["cells"]),
            "seed_offsets": round3["design"]["evaluation_seed_offsets"],
            "initializations": round3["design"]["initializations"],
        },
        "characterization_evidence": {
            "path": CHARACTERIZATION_EVIDENCE,
            "sha256": sha256_file(root / CHARACTERIZATION_EVIDENCE),
        },
        "capability_vector": capability_summary(round3),
        "known_limitations": list(KNOWN_LIMITATIONS),
    }


_VERIFIED_FIELDS = (
    "phase_fingerprint",
    "phase_fingerprint_file_count",
    "model_format",
    "architecture_id",
    "parameter_count",
    "parameter_count_formula",
    "state_footprint",
    "selected_hyperparameters",
    "control_hyperparameters",
    "ungated_control",
    "benchmark_version",
    "development_evidence",
    "round3_evidence",
    "characterization_evidence",
    "capability_vector",
)


def verify_champion(record: Mapping[str, Any], root: Path) -> list[str]:
    """Rebuild the champion from the repository and list every disagreement."""

    if record.get("schema") != CHAMPION_SCHEMA:
        return ["unknown champion schema"]
    live = build_champion(root, source_commit=str(record.get("source_commit")))
    problems = [
        f"{field}: recorded {record.get(field)!r} but repository gives {live[field]!r}"
        for field in _VERIFIED_FIELDS
        if record.get(field) != live[field]
    ]
    if record.get("parameter_count") != record.get("parameter_count_formula"):
        problems.append("parameter accounting disagrees with the architecture formula")
    return problems
