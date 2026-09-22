"""Iteration 0005: one minimal candidate for M2, under the same prospective rules as 0004.

Iteration 0004 ended REJECTED at its precommitted screen. Its development
evidence (``docs/evidence/aaa1k_loop_0004/development.json``) showed that
replacing the unfolding reference *everywhere* (c7) removes M2 but costs
accuracy around genuine bounces on smooth motion, and that not unfolding at
all (c8) costs more. The lesson is that the champion's own-prediction
reference is right at genuine crossings and wrong only where it mirrors a
transition that crossed nothing. That lesson, not any tuned number, designs
the single candidate here:

c9 (:class:`~research.aaa_1k_loop.unfolding.GatedUnfoldAgent`)
    the champion's unfolding, unchanged, except that a mirrored branch is
    refused when the one-step crossing it implies is longer than the observed
    motion can account for (``|u - p| > |v| + |y - p|``). The bound is derived
    from reflection geometry; nothing in it was fitted to 0004's data.

Rules carried over from iteration 0004 verbatim (``iteration4.py`` docstring):
finite budget (here exactly one candidate), no screen-failed advancement,
uncertainty-aware screens with the 2% margin, one attack with prevalidated
instruments, scratch identities, long-horizon stability as a primary gate,
and the three-valued K1-K3 confirmation decision. The screen, attack
adjudication and decision are the *same functions* as 0004's, imported, so
nothing about the judgement was re-chosen after seeing 0004's result. Every
identity is fresh: new development, attack and confirmation blocks under this
iteration's salt.
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
from .unfolding import GatedUnfoldAgent, UnfoldTraceAgent

ITERATION_ID = "aaa1k-loop-0005"

DEVELOPMENT_BLOCK = "aaa1k-loop-0005/development/all"
ATTACK_ENV_BLOCK = "aaa1k-loop-0005/attack/env"
ATTACK_INIT_BLOCK = "aaa1k-loop-0005/attack/init"
CONFIRMATION_ENV_BLOCK = "aaa1k-loop-0005/confirmation/env"
CONFIRMATION_INIT_BLOCK = "aaa1k-loop-0005/confirmation/init"

DEVELOPMENT_BLOCK_SIZE = it4.DEVELOPMENT_BLOCK_SIZE
ATTACK_ENV_BLOCK_SIZE = it4.ATTACK_ENV_BLOCK_SIZE
CONFIRMATION_ENV_COUNT = it4.CONFIRMATION_ENV_COUNT
CONFIRMATION_INIT_COUNT = it4.CONFIRMATION_INIT_COUNT

CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "id": "c9",
        "arm": "c9_gated_unfold",
        "change": (
            "keep the champion's unfolding but refuse a mirrored branch whose implied one-step crossing is "
            "longer than the previous displacement plus the observed displacement"
        ),
        "sentence": (
            "this change should remove M2 because the frame lock consists of mirrored targets for transitions "
            "that crossed nothing (H32-H34), which the gate refuses, while genuine crossings -- where 0004 "
            "showed the champion's reference is right -- are unfolded exactly as before"
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
class Iteration5Arms:
    arms: tuple[str, ...] = LEARNERS

    def __call__(self, seed: int) -> Sequence[Any]:
        builders = {
            "gru": lambda: UnfoldTraceAgent(gru(seed), name="gru"),
            "c9_gated_unfold": lambda: GatedUnfoldAgent(gru(seed), name="c9_gated_unfold"),
        }
        return [builders[name]() for name in self.arms] + [PersistenceAgent()]


def run(
    cells: Sequence[Cell], *, arms: tuple[str, ...] = LEARNERS, workers: int | None = None
) -> list[dict[str, Any]]:
    return run_cells(cells, Iteration5Arms(arms), reduce_cell, isolate_failures=True, workers=workers)


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
