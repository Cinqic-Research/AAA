"""Iteration 0005 challenger stages: develop, attack, freeze, confirm.

A copy of :mod:`research.aaa_1k_loop.outer4` with iteration 0005's spec. The
duplication is deliberate: the protocol says to generalize the outer-loop
machinery only after a second real use shows what is reusable, and this is
that second use.

    python -m research.aaa_1k_loop.outer5 develop
    python -m research.aaa_1k_loop.outer5 attack
    python -m research.aaa_1k_loop.outer5 freeze      # then commit the freeze
    python -m research.aaa_1k_loop.outer5 confirm     # observes the confirmation once

This module is on the confirmation path, so everything it imports is in
:data:`CONFIRMATION_SOURCES_5` and a test holds it to that. It deliberately
does not import :mod:`research.aaa_1k_loop.cli` (whose imports reach every
earlier iteration) and repeats the three small helpers it needs.

Each stage refuses to run out of order: attack needs a development record
whose recomputed screen selects a candidate; freeze needs an attack whose
recomputed adjudication advances it; confirm needs a committed, verified
freeze and a durable remote claim on each confirmation block.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from research.aaa_1k.identity import git_provenance

from . import LOOP_PROTOCOL_VERSION
from .evidence import read_strict_json, write_strict_json
from .identities import LEDGER_PATH, load_ledger, mark, prove_fresh, require_usable, reserve, write_ledger
from .iteration5 import (
    ATTACK_ENV_BLOCK,
    ATTACK_INIT_BLOCK,
    CANDIDATES,
    CONFIRMATION_ENV_BLOCK,
    CONFIRMATION_ENV_COUNT,
    CONFIRMATION_INIT_BLOCK,
    CONFIRMATION_INIT_COUNT,
    DEVELOPMENT_BLOCK,
    FROZEN_RULES,
    ITERATION_ID,
    adjudicate_attack,
    attack_cells,
    confirmation_cells,
    decide,
    development_cells,
    run,
    screen,
    select_for_attack,
)

EVIDENCE_DIR = Path("docs/evidence/aaa1k_loop_0005")
DEVELOPMENT = EVIDENCE_DIR / "development.json"
ATTACK = EVIDENCE_DIR / "attack.json"
FREEZE = EVIDENCE_DIR / "freeze.json"
CONFIRMATION = EVIDENCE_DIR / "confirmation.json"
CHAMPION_RECORD = "docs/evidence/aaa1k_loop_0001/champion_0.json"

CONFIRMATION_SOURCES_5 = (
    "research/__init__.py",
    "research/aaa_1k_loop/__init__.py",
    "research/aaa_1k_loop/arms.py",
    "research/aaa_1k_loop/bounded.py",
    "research/aaa_1k_loop/challengers.py",
    "research/aaa_1k_loop/develop.py",
    "research/aaa_1k_loop/dynamics.py",
    "research/aaa_1k_loop/evidence.py",
    "research/aaa_1k_loop/freeze.py",
    "research/aaa_1k_loop/harness.py",
    "research/aaa_1k_loop/identities.py",
    "research/aaa_1k_loop/iteration4.py",
    "research/aaa_1k_loop/iteration5.py",
    "research/aaa_1k_loop/outer5.py",
    "research/aaa_1k_loop/unfolding.py",
    "aaa/noise/reservation.py",
)
"""Iteration 0005's confirmation path, in addition to every AAA-1K phase file."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ledger(root: Path) -> Path:
    return root / LEDGER_PATH


def _stamp(payload: dict[str, Any], root: Path, started: float) -> dict[str, Any]:
    return {
        **payload,
        "loop_protocol_version": LOOP_PROTOCOL_VERSION,
        "iteration_id": ITERATION_ID,
        "git": git_provenance(root),
        "compute_seconds": time.perf_counter() - started,
    }


def frozen_content(selected: str) -> dict[str, Any]:
    candidate = next(c for c in CANDIDATES if c["arm"] == selected)
    return {
        "iteration": ITERATION_ID,
        "champion": "aaa1k-champion-0",
        "challenger": dict(candidate),
        "claim": (
            f"Champion 0 with {candidate['id']} ({candidate['change']}) removes the long-horizon runaway M2 "
            "without regressing any round-3 standard family or long-horizon non-coarse condition by more "
            "than the margin"
        ),
        "rules": FROZEN_RULES,
        "consequence": (
            "PROMOTE makes the challenger Champion 1 (same network, same 994 parameters, same state; one "
            "target-construction change). REJECT keeps Champion 0 and M2 open. INCONCLUSIVE keeps Champion 0 "
            "and records that fresh evidence did not resolve the claim; the spent identities are not reused."
        ),
    }


def _selected(root: Path) -> str | None:
    development = read_strict_json(root / DEVELOPMENT)
    screens = {arm: screen(development["records"], arm) for arm in (c["arm"] for c in CANDIDATES)}
    return select_for_attack(screens)


def spend_and_build(ledger: Mapping[str, Any], observer: str) -> tuple[dict[str, Any], list[Any]]:
    """Mark both confirmation blocks spent by ``observer``, then build the cells they name.

    One function, so the exact admission-to-cells sequence is tested (the
    sequence whose defect aborted iteration 0006's first attempt).
    """

    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        ledger = mark(ledger, block_id, status="spent", observed_by=observer)
    return dict(ledger), confirmation_cells(ledger, observer=observer)


# ----------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------
def command_develop(args: argparse.Namespace) -> int:
    root = project_root()
    started = time.perf_counter()
    records = run(development_cells(load_ledger(_ledger(root))), workers=args.workers)
    screens = {c["arm"]: screen(records, c["arm"]) for c in CANDIDATES}
    selected = select_for_attack(screens)
    payload = {
        "schema": "aaa.loop.development.v5",
        "evidence_role": "development",
        "identity_block": DEVELOPMENT_BLOCK,
        "candidates": list(CANDIDATES),
        "screens": screens,
        "selected_for_attack": selected,
        "records": records,
    }
    write_strict_json(root / args.output, _stamp(payload, root, started))
    for arm, result in screens.items():
        print(f"{arm:14s} screen {'PASS' if result['passed'] else 'NOT PASSED'}")
    print(f"selected for attack: {selected}")
    return 0


def command_attack(args: argparse.Namespace) -> int:
    root = project_root()
    selected = _selected(root)
    if selected is None:
        print("refusing: no candidate passed the recomputed development screen", file=sys.stderr)
        return 1
    started = time.perf_counter()
    records = run(attack_cells(load_ledger(_ledger(root))), arms=("gru", selected), workers=args.workers)
    adjudication = adjudicate_attack(records, selected)
    payload = {
        "schema": "aaa.loop.attack.v5",
        "evidence_role": "attack",
        "identity_blocks": [ATTACK_ENV_BLOCK, ATTACK_INIT_BLOCK],
        "attacked": selected,
        "adjudication": adjudication,
        "records": records,
    }
    write_strict_json(root / args.output, _stamp(payload, root, started))
    for key, value in adjudication["outcome"].items():
        print(f"{key:40s} {value}")
    print(f"advance to freeze: {adjudication['advance_to_freeze']}")
    return 0


def command_freeze(_args: argparse.Namespace) -> int:
    from .freeze import build_freeze

    root = project_root()
    if (root / FREEZE).exists():
        print(f"refusing: {FREEZE} exists; a freeze is never rewritten", file=sys.stderr)
        return 2
    selected = _selected(root)
    attack = read_strict_json(root / ATTACK)
    if selected is None or attack["attacked"] != selected:
        print("refusing: the attack evidence is not for the recomputed selection", file=sys.stderr)
        return 1
    if not adjudicate_attack(attack["records"], selected)["advance_to_freeze"]:
        print("refusing: the recomputed attack does not advance this challenger", file=sys.stderr)
        return 1
    ledger = load_ledger(_ledger(root))
    ledger = reserve(
        ledger,
        block_id=CONFIRMATION_ENV_BLOCK,
        role="confirmation",
        namespace="confirmation_env",
        count=CONFIRMATION_ENV_COUNT,
        purpose="iteration 0005 fresh confirmation streams (M2 challenger)",
        iteration_id=ITERATION_ID,
    )
    ledger = reserve(
        ledger,
        block_id=CONFIRMATION_INIT_BLOCK,
        role="confirmation",
        namespace="confirmation_init",
        count=CONFIRMATION_INIT_COUNT,
        purpose="iteration 0005 fresh confirmation initializations (M2 challenger)",
        iteration_id=ITERATION_ID,
    )
    freshness = prove_fresh(ledger, root)
    manifest = build_freeze(
        root,
        ledger=ledger,
        freshness=freshness,
        champion_path=CHAMPION_RECORD,
        attack_path=str(ATTACK),
        blocks=(CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK),
        frozen_content=frozen_content(selected),
        sources=CONFIRMATION_SOURCES_5,
    )
    write_ledger(_ledger(root), ledger)
    write_strict_json(
        root / FREEZE,
        {**manifest, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        overwrite=False,
    )
    print(f"freeze written: {FREEZE}; commit it (and the ledger) before running confirm")
    return 0


def command_confirm(args: argparse.Namespace) -> int:
    from aaa.noise.reservation import ReservationError, reserve_confirmation_batch

    from .freeze import load_freeze, require_committed, require_committed_confirmation_source, verify_freeze

    root = project_root()
    output = root / args.output
    if output.exists():
        print(f"refusing: {args.output} exists; a confirmation is observed once", file=sys.stderr)
        return 2
    manifest = load_freeze(root / FREEZE)
    head = require_committed(root, str(FREEZE))
    require_committed(root, LEDGER_PATH)
    require_committed_confirmation_source(root, manifest)
    selected = manifest["frozen"]["challenger"]["arm"]
    ledger = load_ledger(_ledger(root))
    problems = verify_freeze(
        manifest, root, ledger=ledger, frozen_content=frozen_content(selected), sources=CONFIRMATION_SOURCES_5
    )
    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        try:
            require_usable(ledger, block_id, purpose="confirmation")
        except Exception as error:
            problems.append(str(error))
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}", file=sys.stderr)
        return 1
    prove_fresh(ledger, root)
    observer = f"{ITERATION_ID} confirmation at {head}"
    fingerprint = manifest["confirmation_source_fingerprint"]["sha256"]
    for block_id in (CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK):
        try:
            reserve_confirmation_batch(
                root,
                batch_id=block_id.replace("/", "-"),
                role="loop-confirmation",
                attempt_id=observer,
                scientific_fingerprint_sha256=fingerprint,
                resume=False,
            )
        except ReservationError as error:
            print(f"REFUSED: durable confirmation reservation failed: {error}", file=sys.stderr)
            return 1
    ledger, cells = spend_and_build(ledger, observer)
    # Spent before the first cell runs: a crash after this point still consumes the identities.
    write_ledger(_ledger(root), ledger)
    started = time.perf_counter()
    primitives = run(cells, arms=("gru", selected), workers=args.workers)
    payload: dict[str, Any] = {
        "schema": "aaa.loop.confirmation.v5",
        "evidence_role": "confirmation",
        "freeze": {
            "path": str(FREEZE),
            "sha256": hashlib.sha256((root / FREEZE).read_bytes()).hexdigest(),
            "commit": head,
        },
        "confirmation_source_fingerprint": fingerprint,
        "champion_phase_fingerprint": manifest["champion_phase_fingerprint"],
        "challenger": selected,
        "identity_blocks": [CONFIRMATION_ENV_BLOCK, CONFIRMATION_INIT_BLOCK],
        "primitives": primitives,
    }
    try:
        payload["decision"] = decide(primitives, selected)
    except Exception as error:  # the primitives are kept whatever happens
        payload["decision"] = {"outcome": None, "error": f"{type(error).__name__}: {error}"}
        write_strict_json(output, _stamp(payload, root, started), overwrite=False)
        print(f"decision failed after observation; primitives retained: {error}", file=sys.stderr)
        return 1
    write_strict_json(output, _stamp(payload, root, started), overwrite=False)
    for name, status in payload["decision"]["statuses"].items():
        print(f"{name:40s} {status}")
    print(f"OUTCOME: {payload['decision']['outcome']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_1k_loop.outer5")
    sub = parser.add_subparsers(dest="command", required=True)
    develop = sub.add_parser("develop", help="screen the precommitted candidate on the development block")
    develop.add_argument("--output", default=str(DEVELOPMENT))
    develop.add_argument("--workers", type=int)
    attack = sub.add_parser("attack", help="attack the one candidate the recomputed screen selects")
    attack.add_argument("--output", default=str(ATTACK))
    attack.add_argument("--workers", type=int)
    sub.add_parser("freeze", help="reserve confirmation identities and write the freeze")
    confirm = sub.add_parser("confirm", help="observe the frozen confirmation once")
    confirm.add_argument("--output", default=str(CONFIRMATION))
    confirm.add_argument("--workers", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "develop": command_develop,
        "attack": command_attack,
        "freeze": command_freeze,
        "confirm": command_confirm,
    }
    return int(handlers[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
