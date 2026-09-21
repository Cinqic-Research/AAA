"""Independent recomputation of a confirmation, from primitives only.

This module does not import :mod:`research.aaa_1k_loop.decision`'s evaluation
code. It rebuilds each statistic with a different algorithm -- the crossed
bootstrap as count-weighted means (``R D C^T``) instead of explicit submatrix
indexing, sign counts and means from plain NumPy -- re-derives every criterion
status from the frozen criterion *text's* numbers, and recomputes the outcome.
It then compares against what the confirmation artifact stored and against the
freeze. A stored ``"PROMOTE"`` is only ever compared, never trusted.

The frozen thresholds are read from the freeze manifest, not from the live
decision module, so changing a threshold after the freeze cannot change what
this check expects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from research.aaa_1k.seeds import derive_seed

from .evidence import read_strict_json
from .freeze import seed_list_sha256
from .identities import block_seeds, find_block

TOLERANCE = 1e-9


def _grid(records: Sequence[Mapping[str, Any]], arm: str) -> np.ndarray:
    inits = sorted({r["init_index"] for r in records})
    streams = sorted({r["stream_id"] for r in records})
    table = {(r["init_index"], r["stream_id"]): r["mae"][arm] for r in records}
    if len(table) != len(records) or len(table) != len(inits) * len(streams):
        raise ValueError("records do not form a complete crossing")
    return np.array([[table[(i, s)] for s in streams] for i in inits], dtype=float)


def crossed(
    first: np.ndarray, second: np.ndarray, *, index: int, draws: int, confidence: float
) -> dict[str, Any]:
    """``second - first`` with a count-weighted crossed bootstrap."""

    difference = second - first
    inits, streams = difference.shape
    rng = np.random.default_rng(derive_seed("bootstrap", index))
    rows = rng.integers(0, inits, size=(draws, inits))
    columns = rng.integers(0, streams, size=(draws, streams))
    row_counts = np.zeros((draws, inits))
    column_counts = np.zeros((draws, streams))
    np.add.at(row_counts, (np.arange(draws)[:, None], rows), 1.0)
    np.add.at(column_counts, (np.arange(draws)[:, None], columns), 1.0)
    means = np.einsum("di,is,ds->d", row_counts, difference, column_counts) / (inits * streams)
    alpha = (1.0 - confidence) / 2.0
    per_stream = difference.mean(axis=0)
    return {
        "mean": float(difference.mean()),
        "low": float(np.quantile(means, alpha)),
        "high": float(np.quantile(means, 1.0 - alpha)),
        "positive_streams": int(np.sum(per_stream > 0)),
        "streams": int(streams),
        "per_initialization": [float(v) for v in difference.mean(axis=1)],
    }


def recompute_decision(primitives: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    plan = primitives["plan"]
    fixed = primitives["fixed_speed"]
    long = primitives["long_fixed_speed"]
    draws, confidence = int(thresholds["bootstrap_draws"]), float(thresholds["confidence"])
    idx = thresholds["bootstrap_indices"]
    coarse = [r for r in plan if r["family"] == "coarse_speed_v1"]
    other = [r for r in plan if r["family"] != "coarse_speed_v1"]

    def c(records: Sequence[Mapping[str, Any]], a: str, b: str, key: str) -> dict[str, Any]:
        return crossed(
            _grid(records, a), _grid(records, b), index=int(idx[key]), draws=draws, confidence=confidence
        )

    stats: dict[str, Any] = {}
    status: dict[str, str] = {}

    d = c(coarse, "rnn28", "gru", "C1")
    share = d["positive_streams"] / d["streams"]
    stats["C1"] = {**d, "stream_share": share}
    if d["low"] > 0 and share >= thresholds["C1_min_stream_share"] and min(d["per_initialization"]) > 0:
        status["C1_systematic_family_deficit"] = "PASS"
    elif d["high"] < 0 or share < thresholds["C1_fail_below_stream_share"]:
        status["C1_systematic_family_deficit"] = "FAIL"
    else:
        status["C1_systematic_family_deficit"] = "INCONCLUSIVE"

    q4c = c(coarse, "gru", "rnn28", "C2_coarse")
    q4o = c(other, "gru", "rnn28", "C2_without")
    stats["C2"] = {"coarse": q4c, "without": q4o}
    if q4c["high"] < 0 and q4o["mean"] >= 0:
        status["C2_q4_sign_carried_by_coarse"] = "PASS"
    elif q4c["low"] > 0 or q4o["high"] < 0:
        status["C2_q4_sign_carried_by_coarse"] = "FAIL"
    else:
        status["C2_q4_sign_carried_by_coarse"] = "INCONCLUSIVE"

    sw = c(coarse, "rnn28", "mlp", "C3_switch")
    fx = c(fixed, "rnn28", "mlp", "C3_fixed")
    memory = fx["mean"] / sw["mean"] if sw["mean"] > 0 else None
    stats["C3"] = {"switching": sw, "fixed": fx, "share": memory}
    if memory is not None and fx["low"] > 0 and memory >= thresholds["C3_share_supported"]:
        status["C3_memory_rewarded_without_switches"] = "PASS"
    elif fx["high"] < 0 or (memory is not None and memory < thresholds["C3_share_contradicted"]):
        status["C3_memory_rewarded_without_switches"] = "FAIL"
    else:
        status["C3_memory_rewarded_without_switches"] = "INCONCLUSIVE"

    ch = c(fixed, "gru", "mlp", "C4")
    stats["C4"] = ch
    if ch["mean"] <= 0:
        status["C4_fixed_speed_observation_is_the_champions"] = "PASS"
    elif ch["low"] > 0:
        status["C4_fixed_speed_observation_is_the_champions"] = "FAIL"
    else:
        status["C4_fixed_speed_observation_is_the_champions"] = "INCONCLUSIVE"

    slow = np.mean(
        [
            r["regime_mae"]["slow"]["gru"] - r["regime_mae"]["slow"]["rnn28"]
            for r in coarse
            if "slow" in r["regime_mae"]
        ]
    )
    fast = np.mean(
        [
            r["regime_mae"]["fast"]["gru"] - r["regime_mae"]["fast"]["rnn28"]
            for r in coarse
            if "fast" in r["regime_mae"]
        ]
    )
    gru_mode = float(np.median([r["jacobian_negative_mode_median"]["gru"] for r in coarse]))
    rnn_mode = float(np.median([r["jacobian_negative_mode_median"]["rnn28"] for r in coarse]))
    stats["C5"] = {"slow": float(slow), "fast": float(fast), "gru_mode": gru_mode, "rnn28_mode": rnn_mode}
    if (
        slow > fast
        and gru_mode < thresholds["C5_negative_mode_absent"]
        and rnn_mode > thresholds["C5_negative_mode_present"]
    ):
        status["C5_mechanism_signature"] = "PASS"
    elif slow <= fast or gru_mode >= thresholds["C5_negative_mode_present"]:
        status["C5_mechanism_signature"] = "FAIL"
    else:
        status["C5_mechanism_signature"] = "INCONCLUSIVE"

    values = list(status.values())
    outcome = (
        "REJECT" if "FAIL" in values else ("PROMOTE" if all(v == "PASS" for v in values) else "INCONCLUSIVE")
    )
    divergent = int(
        sum(r["mae"]["gru"] > thresholds["divergence_factor"] * r["mae"]["persistence"] for r in long)
    )
    return {"statuses": status, "outcome": outcome, "statistics": stats, "long_divergent": divergent}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


def verify_confirmation(
    confirmation_path: Path, freeze_path: Path, ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute everything and list every disagreement; empty list means agreement."""

    confirmation = read_strict_json(confirmation_path)
    freeze = read_strict_json(freeze_path)
    problems: list[str] = []
    freeze_sha = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    if confirmation["freeze"]["sha256"] != freeze_sha:
        problems.append("confirmation was not produced under this freeze manifest")
    thresholds = freeze["frozen"]["thresholds"]
    if confirmation["thresholds"] != thresholds:
        problems.append("confirmation thresholds differ from the frozen thresholds")
    if confirmation["confirmation_source_fingerprint"] != freeze["confirmation_source_fingerprint"]["sha256"]:
        problems.append("confirmation ran under a different source fingerprint than the freeze")

    # identities: every cell uses frozen, spent confirmation seeds, each cell exactly once
    blocks = sorted(freeze["confirmation_blocks"])
    env_block = next(block for block in blocks if block.endswith("/env"))
    init_block = next(block for block in blocks if block.endswith("/init"))
    for block_id in (env_block, init_block):
        block = find_block(ledger, block_id)
        if block["status"] != "spent":
            problems.append(f"confirmation block {block_id} is not marked spent")
        if seed_list_sha256(ledger, block_id) != freeze["confirmation_blocks"][block_id]["seed_list_sha256"]:
            problems.append(f"confirmation block {block_id} seeds differ from the freeze")
    env_seeds = set(block_seeds(find_block(ledger, env_block)))
    init_seeds = set(block_seeds(find_block(ledger, init_block)))
    keys = set()
    for section, records in confirmation["primitives"].items():
        for record in records:
            if record["stream_seed"] not in env_seeds or record["init_seed"] not in init_seeds:
                problems.append(f"{section} cell {record['stream_id']} uses a non-confirmation identity")
            key = (section, record["condition"], record["init_seed"], record["stream_id"])
            if key in keys:
                problems.append(f"duplicate confirmation cell {key}")
            keys.add(key)

    independent = recompute_decision(confirmation["primitives"], thresholds)
    stored = confirmation["decision"]
    if independent["outcome"] != stored["outcome"]:
        problems.append(f"outcome: stored {stored['outcome']} recomputed {independent['outcome']}")
    for name, value in independent["statuses"].items():
        if stored["statuses"].get(name) != value:
            problems.append(f"{name}: stored {stored['statuses'].get(name)} recomputed {value}")
    criteria = stored["evaluation"]["criteria"]
    pairs = [
        (
            "C1 mean",
            criteria["C1_systematic_family_deficit"]["deficit"]["mean_difference"],
            independent["statistics"]["C1"]["mean"],
        ),
        (
            "C1 low",
            criteria["C1_systematic_family_deficit"]["deficit"]["ci_low"],
            independent["statistics"]["C1"]["low"],
        ),
        (
            "C1 high",
            criteria["C1_systematic_family_deficit"]["deficit"]["ci_high"],
            independent["statistics"]["C1"]["high"],
        ),
        (
            "C2 coarse high",
            criteria["C2_q4_sign_carried_by_coarse"]["coarse_family_q4"]["ci_high"],
            independent["statistics"]["C2"]["coarse"]["high"],
        ),
        (
            "C2 without mean",
            criteria["C2_q4_sign_carried_by_coarse"]["q4_without_coarse"]["mean_difference"],
            independent["statistics"]["C2"]["without"]["mean"],
        ),
        (
            "C3 fixed low",
            criteria["C3_memory_rewarded_without_switches"]["ungated_memory_advantage_fixed_speed"]["ci_low"],
            independent["statistics"]["C3"]["fixed"]["low"],
        ),
        (
            "C4 mean",
            criteria["C4_fixed_speed_observation_is_the_champions"]["champion_memory_advantage_fixed_speed"][
                "mean_difference"
            ],
            independent["statistics"]["C4"]["mean"],
        ),
        (
            "C5 slow",
            criteria["C5_mechanism_signature"]["regime_deficit"]["slow"],
            independent["statistics"]["C5"]["slow"],
        ),
    ]
    for label, a, b in pairs:
        if not _close(float(a), float(b)):
            problems.append(f"{label}: stored {a!r} recomputed {b!r}")
    if (
        int(stored["evaluation"]["secondary"]["long_fixed_speed_champion_divergent_cells"])
        != independent["long_divergent"]
    ):
        problems.append("secondary long-horizon divergence count disagrees")
    return {"agrees": not problems, "problems": problems, "independent": independent}
