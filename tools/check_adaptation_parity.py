#!/usr/bin/env python3
"""Diagnostic only: replay round-3 Q2 through both historical and v2 designs.

Uses already observed identities. It cannot create confirmation evidence.
CHAMPION_0 is deliberate: round 3 predates Champion 1's reach-gated rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_1k.measurements import adaptation_difference_of_differences  # noqa: E402
from research.aaa_1k.seeds import derive_seed  # noqa: E402
from research.aaa_1k.selection import Configuration  # noqa: E402
from research.aaa_1k_v2.arms import CHAMPION_0, CHAMPION_1  # noqa: E402
from research.aaa_1k_v2.capability import adaptation  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_1k_evaluation_round3.json"
CHAMPION_1_REPLAY = ROOT / "docs/evidence/aaa1k_loop_0006/round3_champion_1.json"
FIELDS = {
    "changed_online": ("changed", "online_mae"),
    "changed_frozen": ("changed", "frozen_mae"),
    "control_online": ("control", "online_mae"),
    "control_frozen": ("control", "frozen_mae"),
}


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    trials = evidence["capability_vector"]["dimensions"]["q2_adaptation"]["trials"]
    config = Configuration(**evidence["configuration"])
    inits = sorted({int(t["model_seed_index"]) for t in trials})
    environments = sorted({int(t["environment_index"]) for t in trials})
    seeds = [next(int(t["seed"]) for t in trials if t["environment_index"] == e) for e in environments]
    if len(trials) != len(inits) * len(environments):
        raise RuntimeError("historical Q2 crossing is incomplete")
    batched = adaptation(CHAMPION_0, [derive_seed("model_init", i) for i in inits], seeds)
    if len(batched) != len(trials):
        raise RuntimeError("batched Q2 crossing is incomplete")
    maximum = dict.fromkeys((*FIELDS, "difference_of_differences"), 0.0)
    for stored, newer in zip(trials, batched, strict=True):
        if newer["failed"] or str(stored["seed"]) not in newer["stream_id"]:
            raise RuntimeError("batched trial failed or identity order changed")
        historical = adaptation_difference_of_differences(
            config, seed=int(stored["seed"]), model_seed_index=int(stored["model_seed_index"])
        )
        for field, (branch, measure) in FIELDS.items():
            values = (stored[branch][measure], historical[branch][measure], newer[field])
            maximum[field] = max(maximum[field], max(values) - min(values))
        effects = (
            stored["adaptation_effect"],
            historical["adaptation_effect"],
            newer["difference_of_differences"],
        )
        maximum["difference_of_differences"] = max(
            maximum["difference_of_differences"], max(effects) - min(effects)
        )
    result = {
        "purpose": "diagnostic parity on previously observed round-3 identities; not confirmation",
        "trials": len(trials),
        "historical_model": "Champion 0 own-prediction target rule",
        "max_absolute_primitive_difference": maximum,
        "historical_mean_effect": sum(t["adaptation_effect"] for t in trials) / len(trials),
        "batched_mean_effect": sum(t["difference_of_differences"] for t in batched) / len(batched),
    }
    replay = json.loads(CHAMPION_1_REPLAY.read_text())
    replay_trials = replay["capability_vector"]["dimensions"]["q2_adaptation"]["trials"]
    champion1 = adaptation(CHAMPION_1, [derive_seed("model_init", i) for i in inits], seeds)
    if len(replay_trials) != len(champion1):
        raise RuntimeError("Champion 1 replay crossing is incomplete")
    replay_max = max(
        max(
            *(abs(old[branch][measure] - new[field]) for field, (branch, measure) in FIELDS.items()),
            abs(old["adaptation_effect"] - new["difference_of_differences"]),
        )
        for old, new in zip(replay_trials, champion1, strict=True)
    )
    result["champion1_replay_max_absolute_primitive_difference"] = replay_max
    print(json.dumps(result, indent=2, allow_nan=False))
    if any(value > 1e-12 for value in maximum.values()) or replay_max > 1e-12:
        raise SystemExit("adaptation designs differ on historical identities")


if __name__ == "__main__":
    main()
