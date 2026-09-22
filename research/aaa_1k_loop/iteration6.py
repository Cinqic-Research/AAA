"""Iteration 0006: the last M2 candidate of this effort, under the rules of 0004.

Iterations 0004 (c7, c8) and 0005 (c9) ended REJECTED at their precommitted
screens. Each removed M2 completely; each cost more than the 2% margin on
long smooth bouncing. A disclosed exploratory look at 0005's already-observed
development cells (long bouncing and long coarse, two initializations) found
the champion's mirrored-target runs on smooth motion are only ever one or two
steps long -- the crossing and the step after it, next to the wall -- while
frame locks run eleven steps or more. c9's implied-step bound refused some of
those useful short runs. That observation, from development identities that
may inform selection, designs the single candidate here:

c10 (:class:`~research.aaa_1k_loop.unfolding.ReachGatedUnfoldAgent`)
    the champion's unfolding, refused only when the input position is farther
    from the crossed wall than one observed step (``|v| + |y - p|``) -- a
    necessary condition for any one-step crossing.

This is declared in advance as the last M2 candidate of this effort. If it
fails, M2 stays open with its measured tradeoff. Across 0004-0006 four
candidates (c7-c10) will have been screened; that multiplicity is disclosed,
and a screen pass is only a licence to be attacked and then confirmed on
fresh identities, which is where a lucky screen would be caught.

Everything else -- budget of one, uncertainty-aware screen with the 2% margin,
one attack, the K1-K3 decision -- is 0004's, imported unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from research.aaa_1k.agents import PersistenceAgent

from . import iteration4 as it4
from .arms import gru
from .harness import Cell, run_cells
from .identities import block_seeds, find_block, require_usable
from .unfolding import ReachGatedUnfoldAgent, UnfoldTraceAgent

ITERATION_ID = "aaa1k-loop-0006"

DEVELOPMENT_BLOCK = "aaa1k-loop-0006/development/all"
ATTACK_ENV_BLOCK = "aaa1k-loop-0006/attack/env"
ATTACK_INIT_BLOCK = "aaa1k-loop-0006/attack/init"
CONFIRMATION_ENV_BLOCK = "aaa1k-loop-0006/confirmation/env"
CONFIRMATION_INIT_BLOCK = "aaa1k-loop-0006/confirmation/init"

DEVELOPMENT_BLOCK_SIZE = it4.DEVELOPMENT_BLOCK_SIZE
ATTACK_ENV_BLOCK_SIZE = it4.ATTACK_ENV_BLOCK_SIZE
CONFIRMATION_ENV_COUNT = it4.CONFIRMATION_ENV_COUNT
CONFIRMATION_INIT_COUNT = it4.CONFIRMATION_INIT_COUNT

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "id": "c10",
        "arm": "c10_reach_gated_unfold",
        "change": (
            "keep the champion's unfolding but refuse a mirrored branch when the input position is farther from "
            "the crossed wall than the previous displacement plus the observed displacement"
        ),
        "sentence": (
            "this change should remove M2 because a frame lock mirrors targets after the input has moved out of "
            "reach of the wall (H32-H34), while the champion's short post-bounce mirror runs, which 0005 showed "
            "are useful on smooth motion, start within reach and are kept"
        ),
    },
)
CANDIDATE_ARMS = tuple(candidate["arm"] for candidate in CANDIDATES)
LEARNERS = ("gru", *CANDIDATE_ARMS)

FROZEN_RULES = {**it4.FROZEN_RULES, "shared_with": "aaa1k-loop-0004 (identical functions, imported)"}

screen = it4.screen
adjudicate_attack = it4.adjudicate_attack
decide = it4.decide4
reduce_cell = it4.reduce_cell


def select_for_attack(screens: Mapping[str, Mapping[str, Any]]) -> str | None:
    for candidate in CANDIDATES:
        if screens[candidate["arm"]]["passed"]:
            return candidate["arm"]
    return None


@dataclass(frozen=True)
class Iteration6Arms:
    arms: tuple[str, ...] = LEARNERS

    def __call__(self, seed: int) -> Sequence[Any]:
        builders = {
            "gru": lambda: UnfoldTraceAgent(gru(seed), name="gru"),
            "c10_reach_gated_unfold": lambda: ReachGatedUnfoldAgent(gru(seed), name="c10_reach_gated_unfold"),
        }
        return [builders[name]() for name in self.arms] + [PersistenceAgent()]


def run(
    cells: Sequence[Cell], *, arms: tuple[str, ...] = LEARNERS, workers: int | None = None
) -> list[dict[str, Any]]:
    return run_cells(cells, Iteration6Arms(arms), reduce_cell, isolate_failures=True, workers=workers)


def development_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    block = require_usable(ledger, DEVELOPMENT_BLOCK, purpose="selection")
    if block["role"] != "development":
        raise ValueError("development must run on a development block")
    seeds = block_seeds(find_block(ledger, DEVELOPMENT_BLOCK))
    split = it4.DEVELOPMENT_PER_ENTRY * it4.PLAN_ENTRY_COUNT
    return it4._cells(
        seeds[:split],
        it4.DEVELOPMENT_PER_ENTRY,
        seeds[split:][: it4.DEVELOPMENT_LONG],
        it4.LONG_CONDITIONS,
        it4.init_seeds(None, it4.INITIALIZATIONS),
    )


def attack_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    env = require_usable(ledger, ATTACK_ENV_BLOCK, purpose="selection")
    init = require_usable(ledger, ATTACK_INIT_BLOCK, purpose="selection")
    if env["role"] != "attack" or init["role"] != "attack":
        raise ValueError("the attack must run on attack blocks")
    seeds = block_seeds(find_block(ledger, ATTACK_ENV_BLOCK))
    split = it4.ATTACK_PER_ENTRY * it4.PLAN_ENTRY_COUNT
    rest = seeds[split:]
    long, nearby = rest[: it4.ATTACK_LONG], rest[it4.ATTACK_LONG : it4.ATTACK_LONG + it4.ATTACK_NEARBY]
    inits = it4.init_seeds(block_seeds(find_block(ledger, ATTACK_INIT_BLOCK)), it4.INITIALIZATIONS)
    cells = it4._cells(seeds[:split], it4.ATTACK_PER_ENTRY, long, it4.LONG_CONDITIONS, inits)
    cells.extend(
        Cell(condition, index, seed, it4.long_stream(condition, stream_seed))
        for condition in it4.NEARBY_CONDITIONS
        for stream_seed in nearby
        for index, seed in inits
    )
    return cells


def confirmation_cells(ledger: Mapping[str, Any]) -> list[Cell]:
    require_usable(ledger, CONFIRMATION_ENV_BLOCK, purpose="confirmation")
    require_usable(ledger, CONFIRMATION_INIT_BLOCK, purpose="confirmation")
    seeds = block_seeds(find_block(ledger, CONFIRMATION_ENV_BLOCK))
    split = it4.CONFIRMATION_PER_ENTRY * it4.PLAN_ENTRY_COUNT
    inits = it4.init_seeds(block_seeds(find_block(ledger, CONFIRMATION_INIT_BLOCK)), it4.INITIALIZATIONS)
    return it4._cells(
        seeds[:split],
        it4.CONFIRMATION_PER_ENTRY,
        seeds[split:][: it4.CONFIRMATION_LONG],
        it4.LONG_CONDITIONS,
        inits,
    )
