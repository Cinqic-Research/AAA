"""Champion 1: Champion 0 with c10, promoted by iteration 0006.

    python -m research.aaa_1k_loop.champion1 round3          # writes round3_champion_1.json
    python -m research.aaa_1k_loop.champion1 record          # writes champion_1.json
    python -m research.aaa_1k_loop.champion1 verify

Champion 1 is the same network as Champion 0 -- same code, same 994
parameters, same 1414 adaptive state scalars, same hyperparameters, same
phase fingerprint -- trained through one different target-construction rule,
:class:`~research.aaa_1k_loop.unfolding.ReachGatedUnfoldAgent`. It lives in
the loop package; ``research/aaa_1k/`` and ``aaa.1k.v1`` are unchanged.

Champion 1's capability vector is *measured*, not inherited: :func:`run_round3_champion_1`
re-runs the whole of round 3 (720 cells, 120 adaptation and 60 retention
trials, on round 3's own identities and configuration) with the reach-gated
target rule substituted for every neural learner -- primary, ablations and
controls, so architecture comparisons keep one agent rule -- including the
arms :meth:`NeuralAgent.branch` clones. :func:`compare_round3` then requires
every cell and trial to equal the committed round 3 exactly, except those
where the gate changed something, and lists those. This replays already-
observed round-3 identities as a reproduction; it selects nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from research.aaa_1k.round3 import run_round3
from research.aaa_1k.selection import Configuration

from .arms import CHAMPION_CONFIGURATION, gru
from .champion import CHARACTERIZATION_EVIDENCE as CHARACTERIZATION
from .champion import DEVELOPMENT_EVIDENCE as DEVELOPMENT_SELECTION
from .champion import KNOWN_LIMITATIONS, ROUND3_EVIDENCE, capability_summary, sha256_file
from .evidence import read_strict_json, write_strict_json
from .unfolding import ReachGatedUnfoldAgent

E0 = "docs/evidence/aaa1k_loop_0001/champion_0.json"
E6 = Path("docs/evidence/aaa1k_loop_0006")
REPLAY = E6 / "round3_champion_1.json"
RECORD = E6 / "champion_1.json"
PROMOTION_EVIDENCE = (
    str(E6 / "development.json"),
    str(E6 / "attack.json"),
    str(E6 / "freeze_2.json"),
    str(E6 / "confirmation_2.json"),
    str(E6 / "recomputation_2.json"),
)
AGENT_SOURCE = "research/aaa_1k_loop/unfolding.py"


@contextmanager
def champion_1_world() -> Iterator[None]:
    """Every neural learner round 3 builds, including branch clones, uses the reach-gated rule."""

    import research.aaa_1k.agents as agents_module
    import research.aaa_1k.experiments as experiments_module
    import research.aaa_1k.measurements as measurements_module

    modules = (agents_module, experiments_module, measurements_module)
    saved = [getattr(module, "NeuralAgent") for module in modules]  # noqa: B009
    for module in modules:
        setattr(module, "NeuralAgent", ReachGatedUnfoldAgent)  # noqa: B010
    try:
        yield
    finally:
        for module, original in zip(modules, saved, strict=True):
            setattr(module, "NeuralAgent", original)  # noqa: B010


def run_round3_champion_1(root: Path) -> dict[str, Any]:
    from research.aaa_1k.identity import git_provenance

    selection = read_strict_json(root / DEVELOPMENT_SELECTION)
    characterization = read_strict_json(root / CHARACTERIZATION)
    chosen = selection["selected"]
    configuration = Configuration(
        float(chosen["learning_rate"]),
        int(chosen["tbptt_steps"]),
        float(chosen["error_loss_weight"]),
        chosen.get("gradient_clip", 1.0),
    )
    fairness = next(
        (p for p in characterization["probes"] if p["probe"] == "per_architecture_selection"), None
    )
    with champion_1_world():
        evidence = run_round3(
            configuration,
            architecture_configurations=selection.get("architecture_configurations"),
            fairness=fairness,
        )
    evidence["champion"] = "aaa1k-champion-1"
    evidence["agent_rule"] = "research.aaa_1k_loop.unfolding.ReachGatedUnfoldAgent for every neural learner"
    evidence["git"] = git_provenance(root)
    evidence["comparison_with_round3"] = compare_round3(read_strict_json(root / ROUND3_EVIDENCE), evidence)
    return evidence


def _differences(first: Any, second: Any, path: str = "") -> list[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        out = [f"{path}.{k}: key set differs" for k in sorted(set(first) ^ set(second))]
        for key in sorted(set(first) & set(second)):
            out.extend(_differences(first[key], second[key], f"{path}.{key}"))
        return out
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return [f"{path}: length {len(first)} != {len(second)}"]
        return [
            d
            for i, (a, b) in enumerate(zip(first, second, strict=True))
            for d in _differences(a, b, f"{path}[{i}]")
        ]
    return [] if first == second else [path]


def compare_round3(round3: Mapping[str, Any], champion_1: Mapping[str, Any]) -> dict[str, Any]:
    """Cell-by-cell and trial-by-trial equality with the committed round 3."""

    def keyed(cells: list[Mapping[str, Any]]) -> dict[tuple[int, str], Mapping[str, Any]]:
        return {(int(c["model_seed_index"]), str(c["stream_id"])): c for c in cells}

    old, new = keyed(round3["cells"]), keyed(champion_1["cells"])
    changed_cells = []
    for key in sorted(set(old) | set(new)):
        if key not in old or key not in new:
            changed_cells.append({"cell": list(key), "fields": ["missing"]})
            continue
        fields = sorted({d.split(".")[1].split("[")[0] for d in _differences(old[key], new[key])})
        if fields:
            changed_cells.append({"cell": list(key), "fields": fields})
    trials = {}
    for dimension in ("q2_adaptation", "q5_retention"):
        before = round3["capability_vector"]["dimensions"][dimension]["trials"]
        after = champion_1["capability_vector"]["dimensions"][dimension]["trials"]
        trials[dimension] = {
            "trials": len(after),
            "changed": [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
            if len(before) == len(after)
            else "length differs",
        }
    return {
        "cells": len(new),
        "cells_changed": changed_cells,
        "trials": trials,
        "same_streams": round3["streams"] == champion_1["streams"],
        "same_design": round3["design"] == champion_1["design"],
        "same_configuration": round3["configuration"] == champion_1["configuration"],
    }


def build_record(root: Path) -> dict[str, Any]:
    rerun = read_strict_json(root / REPLAY)
    comparison = rerun["comparison_with_round3"]
    parent = read_strict_json(root / E0)
    confirmation = read_strict_json(root / E6 / "confirmation_2.json")
    probe = gru(0)
    faithful = comparison["same_streams"] and comparison["same_design"] and comparison["same_configuration"]
    return {
        "schema": "aaa.loop.champion.v1",
        "champion_id": "aaa1k-champion-1",
        "parent_model_id": parent["champion_id"],
        "parent_record": {"path": E0, "sha256": sha256_file(root / E0)},
        "promoted_by": {
            "iteration_id": "aaa1k-loop-0006",
            "challenger": "c10",
            "outcome": confirmation["decision"]["outcome"],
            "evidence": [{"path": p, "sha256": sha256_file(root / p)} for p in PROMOTION_EVIDENCE],
        },
        "lineage_note": (
            "Champion 0's network, weights initialization and hyperparameters, trained through the reach-gated "
            "unfolding target rule; research/aaa_1k and aaa.1k.v1 are unchanged"
        ),
        "change": {
            "agent_class": "research.aaa_1k_loop.unfolding.ReachGatedUnfoldAgent",
            "agent_source": {"path": AGENT_SOURCE, "sha256": sha256_file(root / AGENT_SOURCE)},
            "rule": (
                "refuse a mirrored unfolding branch when the input position is farther from the crossed wall "
                "than |velocity estimate| + |observed displacement|"
            ),
        },
        "phase_fingerprint": parent["phase_fingerprint"],
        "architecture_id": parent["architecture_id"],
        "model_format": parent["model_format"],
        "parameter_count": probe.parameter_count(),
        "parameter_count_parent": parent["parameter_count"],
        "state_footprint": probe.state_footprint(),
        "selected_hyperparameters": dict(CHAMPION_CONFIGURATION),
        "round3_evidence": {
            "path": str(REPLAY),
            "sha256": sha256_file(root / REPLAY),
            "round3_identities_design_and_configuration_unchanged": faithful,
            "cells": comparison["cells"],
            "cells_changed": comparison["cells_changed"],
            "trials_changed": {k: v["changed"] for k, v in comparison["trials"].items()},
        },
        "capability_vector": capability_summary(rerun) if faithful else None,
        "capability_vector_parent": parent["capability_vector"],
        "capability_vector_basis": (
            "measured: round 3 re-run on its own identities with the reach-gated rule for every neural learner; "
            "every cell and trial not listed in round3_evidence equals the committed round 3 exactly"
        ),
        "repaired": [
            "M2 (long-horizon runaway on quantized streams): 0% divergence against Champion 0's 21.3% on fresh "
            "confirmation (1120-step coarse conditions)",
        ],
        "known_limitations": [
            limitation for limitation in KNOWN_LIMITATIONS if not limitation.startswith("TBPTT horizon")
        ]
        + [
            "TBPTT horizon barely matters on these benchmarks; the online rule is the live cached-activation "
            "rule, numerically within ~1e-4 of the realized-trajectory gradient (iteration 0004, H20)",
            "M2 repair is verified on the tested quantized and smooth families at 1120 steps; other boundary "
            "geometries, noise and missing-observation patterns near walls are untested",
            "the reach gate uses the tracker's one-step velocity estimate; streams where the previous "
            "displacement is unrepresentative of the next step (large accelerations at walls) are untested",
        ],
    }


def verify(root: Path) -> list[str]:
    record = read_strict_json(root / RECORD)
    live = build_record(root)
    return [f"{key}: recorded differs from repository" for key in live if record.get(key) != live[key]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_1k_loop.champion1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("round3")
    sub.add_parser("record")
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.command == "round3":
        result = run_round3_champion_1(root)
        write_strict_json(root / REPLAY, result)
        comparison = result["comparison_with_round3"]
        print(
            f"{comparison['cells']} cells, {len(comparison['cells_changed'])} changed: "
            f"{[c['cell'] for c in comparison['cells_changed']]}; trials changed: "
            f"{ {k: v['changed'] for k, v in comparison['trials'].items()} }; "
            f"identities/design/configuration unchanged: "
            f"{comparison['same_streams'] and comparison['same_design'] and comparison['same_configuration']}"
        )
        return 0
    if args.command == "record":
        write_strict_json(root / RECORD, build_record(root))
        print(f"wrote {RECORD}")
        return 0
    problems = verify(root)
    for problem in problems:
        print(f"STALE: {problem}", file=sys.stderr)
    print("champion 1: " + ("VALID" if not problems else "INVALID"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
