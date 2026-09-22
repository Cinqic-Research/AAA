"""Independent recomputation of iteration 0004's confirmation, from primitives only.

Does not import :mod:`research.aaa_1k_loop.iteration4`. Grids are rebuilt
here, the crossed bootstrap is the count-weighted ``R D C^T`` form in
:func:`research.aaa_1k_loop.recompute.crossed` rather than AAA-1K's explicit
submatrix indexing, and every threshold and bootstrap index is read from the
*freeze manifest*, so editing the live decision code after the freeze cannot
change what this check expects. It also checks that the primitives cover
exactly the frozen confirmation identities.

    python -m research.aaa_1k_loop.recompute4 [--output PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import read_strict_json, write_strict_json
from .freeze import seed_list_sha256
from .identities import LEDGER_PATH, block_seeds, find_block, load_ledger
from .recompute import crossed

DRAWS = 10_000
CONFIDENCE = 0.95
TOLERANCE = 1e-9


def _table(records: Sequence[Mapping[str, Any]], arm: str, field: str) -> np.ndarray:
    keys_i = sorted({int(r["init_index"]) for r in records})
    keys_s = sorted({f"{r['condition']}|{r['stream_id']}" for r in records})
    out = np.full((len(keys_i), len(keys_s)), np.nan)
    for r in records:
        if arm in r.get("failures", {}):
            raise ValueError(f"{arm} failed in a confirmation cell; the decision has no primitive for it")
        out[keys_i.index(int(r["init_index"])), keys_s.index(f"{r['condition']}|{r['stream_id']}")] = float(
            r["arms"][arm][field]
        )
    if np.isnan(out).any():
        raise ValueError("confirmation primitives do not form a complete crossing")
    return out


def _count(records: Sequence[Mapping[str, Any]], arm: str) -> int:
    return int(sum(1 for r in records if arm in r.get("failures", {}) or r["arms"][arm]["diverged"]))


def recompute(primitives: Sequence[Mapping[str, Any]], arm: str, rules: Mapping[str, Any]) -> dict[str, Any]:
    margin = float(rules["margin"])
    index = rules["bootstrap_indices"]
    by = lambda conditions: [r for r in primitives if r["condition"] in conditions]  # noqa: E731
    statuses: dict[str, str] = {}

    coarse = by(("long:coarse_no_switch", "long:coarse_switching"))
    champion, challenger = _table(coarse, "gru", "diverged"), _table(coarse, arm, "diverged")
    k1 = crossed(challenger, champion, index=int(index["k1"]), draws=DRAWS, confidence=CONFIDENCE)
    fraction = float(challenger.mean())
    if k1["low"] > 0 and fraction <= 0.05:
        statuses["K1_long_coarse_stability"] = "PASS"
    elif k1["high"] <= 0 or fraction > 0.20:
        statuses["K1_long_coarse_stability"] = "FAIL"
    else:
        statuses["K1_long_coarse_stability"] = "INCONCLUSIVE"

    def relative(records: Sequence[Mapping[str, Any]], bootstrap: int) -> tuple[float, float]:
        first, second = _table(records, "gru", "mae"), _table(records, arm, "mae")
        result = crossed(first, second, index=bootstrap, draws=DRAWS, confidence=CONFIDENCE)
        base = float(first.mean())
        return result["low"] / base, result["high"] / base

    for offset, condition in enumerate(("long:bouncing", "long:occlusion")):
        records = by((condition,))
        low, high = relative(records, int(index["k2"][offset]))
        c_gru, c_arm = _count(records, "gru"), _count(records, arm)
        if low > margin or c_arm > c_gru + 2:
            status = "FAIL"
        elif high <= margin and c_arm <= c_gru:
            status = "PASS"
        else:
            status = "INCONCLUSIVE"
        statuses[f"K2_{condition}"] = status

    plan = sorted({r["condition"] for r in primitives if r["condition"].startswith("plan:")})
    for offset, condition in enumerate(plan):
        low, high = relative(by((condition,)), int(index["k3_start"]) + offset)
        statuses[f"K3_{condition}"] = (
            "PASS" if high <= margin else ("FAIL" if low > margin else "INCONCLUSIVE")
        )

    values = list(statuses.values())
    outcome = (
        "REJECT" if "FAIL" in values else ("PROMOTE" if all(v == "PASS" for v in values) else "INCONCLUSIVE")
    )
    return {
        "statuses": statuses,
        "outcome": outcome,
        "k1_interval": [k1["low"], k1["high"]],
        "k1_fraction": fraction,
    }


def verify(confirmation_path: Path, freeze_path: Path, ledger: Mapping[str, Any]) -> dict[str, Any]:
    confirmation = read_strict_json(confirmation_path)
    manifest = read_strict_json(freeze_path)
    problems: list[str] = []
    if confirmation["freeze"]["sha256"] != hashlib.sha256(freeze_path.read_bytes()).hexdigest():
        problems.append("confirmation does not reference this freeze's bytes")
    if (
        confirmation["confirmation_source_fingerprint"]
        != manifest["confirmation_source_fingerprint"]["sha256"]
    ):
        problems.append("confirmation ran under a different source fingerprint than the freeze")
    arm = manifest["frozen"]["challenger"]["arm"]
    if confirmation["challenger"] != arm:
        problems.append("confirmation evaluated a different challenger than the frozen one")
    for block_id, frozen in manifest["confirmation_blocks"].items():
        if seed_list_sha256(ledger, block_id) != frozen["seed_list_sha256"]:
            problems.append(f"ledger seeds for {block_id} differ from the freeze")
        block = find_block(ledger, block_id)
        if block["status"] != "spent":
            problems.append(f"{block_id} is not marked spent")
    primitives = confirmation["primitives"]
    env = set(block_seeds(find_block(ledger, "aaa1k-loop-0004/confirmation/env")))
    init = set(block_seeds(find_block(ledger, "aaa1k-loop-0004/confirmation/init")))
    used_env = {int(r["stream_seed"]) for r in primitives}
    used_init = {int(r["init_seed"]) for r in primitives}
    if used_env != env:
        problems.append("primitives do not use exactly the frozen confirmation stream seeds")
    if used_init != init:
        problems.append("primitives do not use exactly the frozen confirmation initialization seeds")
    independent = recompute(primitives, arm, manifest["frozen"]["rules"])
    stored = confirmation.get("decision", {})
    if stored.get("outcome") != independent["outcome"]:
        problems.append(f"stored outcome {stored.get('outcome')} != independent {independent['outcome']}")
    if stored.get("statuses") != independent["statuses"]:
        problems.append("stored criterion statuses differ from the independent recomputation")
    return {"independent": independent, "problems": problems, "agrees": not problems}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_1k_loop.recompute4")
    parser.add_argument("--confirmation", default="docs/evidence/aaa1k_loop_0004/confirmation.json")
    parser.add_argument("--freeze", default="docs/evidence/aaa1k_loop_0004/freeze.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    result = verify(root / args.confirmation, root / args.freeze, load_ledger(root / LEDGER_PATH))
    if args.output:
        write_strict_json(root / args.output, result)
    for problem in result["problems"]:
        print(f"DISAGREES: {problem}", file=sys.stderr)
    print(f"independent outcome: {result['independent']['outcome']}; agrees with stored: {result['agrees']}")
    return 0 if result["agrees"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
