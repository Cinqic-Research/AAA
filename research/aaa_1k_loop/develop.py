"""Inner loop, development stage: evaluate the declared candidates.

Runs on the ``development/all`` block only: the six round-3 evaluation plan
entries, eight fresh development streams each, crossed with the champion's
five initializations. The candidates and the selection rule live in
:mod:`research.aaa_1k_loop.challengers` and were committed before this ran.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from research.aaa_1k.experiments import EVALUATION_PLAN
from research.aaa_1k.runner import RunResult
from research.aaa_1k.seeds import derive_seed
from research.aaa_1k.streams import build_stream

from .arms import ArmSet
from .challengers import CANDIDATES
from .harness import Cell, arm_errors, run_cells
from .identities import block_seeds, find_block, require_usable

DEVELOPMENT_SCHEMA = "aaa.loop.development.v1"
DEVELOPMENT_BLOCK = "aaa1k-loop-0001/development/all"
STREAMS_PER_ENTRY = 8
INITIALIZATIONS = 5

ARMS = ArmSet(("gru", *(candidate["arm"] for candidate in CANDIDATES), "rnn28", "persistence"))


def plan_entry_name(family: str, options: Mapping[str, Any]) -> str:
    return f"{family}:{options['scenario']}" if family == "motion_compat" else family


def plan_streams(seeds: Sequence[int], per_entry: int) -> list[tuple[str, Any]]:
    """``(plan_entry, stream)`` pairs, consuming ``seeds`` in plan order like round 3 does."""

    if len(seeds) < per_entry * len(EVALUATION_PLAN):
        raise ValueError("not enough seeds for the plan")
    out: list[tuple[str, Any]] = []
    cursor = 0
    for family, options in EVALUATION_PLAN:
        for _ in range(per_entry):
            out.append(
                (plan_entry_name(family, options), build_stream(family, seeds[cursor], **dict(options)))
            )
            cursor += 1
    return out


def development_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = require_usable(ledger, DEVELOPMENT_BLOCK, purpose="selection")
    if block["role"] != "development":
        raise ValueError("development must run on a development block")
    streams = plan_streams(block_seeds(find_block(ledger, DEVELOPMENT_BLOCK)), STREAMS_PER_ENTRY)
    return [
        Cell(entry, init, derive_seed("model_init", init), stream)
        for entry, stream in streams
        for init in range(INITIALIZATIONS)
    ]


def reduce_cell(cell: Cell, result: RunResult, _agents: Sequence[Any]) -> dict[str, Any]:
    errors = arm_errors(result)
    return {
        "plan_entry": cell.condition,
        "mae": {name: float(np.mean(values)) for name, values in errors.items()},
        "max_step_error": {name: float(np.max(values)) for name, values in errors.items()},
    }


def run_development(ledger: Mapping[str, Any], *, workers: int | None = None) -> list[dict[str, Any]]:
    return run_cells(development_cells(ledger), ARMS, reduce_cell, workers=workers, isolate_failures=True)
