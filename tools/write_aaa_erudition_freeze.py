#!/usr/bin/env python3
"""Write the ``aaa.erudition.v0`` confirmation freeze.

Run once, from a clean tree, after development and attack evidence are
committed and before any confirmation identity exists. Commit the result
alone; ``identity.admit`` then allows exactly the declared confirmation runs.
The file is never rewritten. A change after the freeze is a successor identity.

    python tools/write_aaa_erudition_freeze.py --model docs/evidence/aaa_erudition_v0/model/erudition.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_erudition import PROTOCOL_VERSION  # noqa: E402
from research.aaa_erudition.identity import FREEZE_SCHEMA, fingerprint  # noqa: E402
from research.aaa_erudition.language import load_profile  # noqa: E402

FREEZE = ROOT / "docs/evidence/aaa_erudition_v0/freeze.json"
INDICES = list(range(12))
ARMS = ["frozen/never", "lm_only/erudition", "wm_only/erudition", "joint/erudition", "joint/heuristic"]
SEED = 20261004

CONTRACTS = [
    {
        "name": "erudition_improves_juniper",
        "question": "Does the Erudition Model, adapting both components, reduce task failure against never adapting, without damaging tasks no shift touched?",
        "seed": SEED,
        "criteria": [
            {
                "name": "failure",
                "metric": "failure",
                "challenger": "joint/erudition",
                "reference": "frozen/never",
                "kind": "superior",
                "threshold": 0.0,
            },
            {
                "name": "retention",
                "metric": "retention_failure",
                "challenger": "joint/erudition",
                "reference": "frozen/never",
                "kind": "noninferior",
                "threshold": 0.05,
            },
        ],
    },
    {
        "name": "joint_over_language_model_only",
        "question": "Does joint adaptation beat the same Erudition Model restricted to the Language Model?",
        "seed": SEED + 1,
        "criteria": [
            {
                "name": "failure",
                "metric": "failure",
                "challenger": "joint/erudition",
                "reference": "lm_only/erudition",
                "kind": "superior",
                "threshold": 0.0,
            },
        ],
    },
    {
        "name": "joint_over_world_model_only",
        "question": "Does joint adaptation beat the same Erudition Model restricted to the World Model?",
        "seed": SEED + 2,
        "criteria": [
            {
                "name": "failure",
                "metric": "failure",
                "challenger": "joint/erudition",
                "reference": "wm_only/erudition",
                "kind": "superior",
                "threshold": 0.0,
            },
        ],
    },
    {
        "name": "learned_control_versus_rules",
        "question": "Is the learned controller no worse than the auditable rule set on failure, poisoning and false adaptation?",
        "seed": SEED + 3,
        "criteria": [
            {
                "name": "failure",
                "metric": "failure",
                "challenger": "joint/erudition",
                "reference": "joint/heuristic",
                "kind": "noninferior",
                "threshold": 0.03,
            },
            {
                "name": "poisoned",
                "metric": "poisoned",
                "challenger": "joint/erudition",
                "reference": "joint/heuristic",
                "kind": "noninferior",
                "threshold": 0.02,
            },
            {
                "name": "false_adaptation",
                "metric": "false_adaptation",
                "challenger": "joint/erudition",
                "reference": "joint/heuristic",
                "kind": "noninferior",
                "threshold": 0.005,
            },
        ],
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if FREEZE.exists():
        print("refusing: the freeze exists and is never rewritten", file=sys.stderr)
        return 2
    if subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"], check=False).returncode != 0:
        print("refusing: the tree has uncommitted changes", file=sys.stderr)
        return 2
    meta = json.loads(args.model.with_suffix(".json").read_text("utf-8"))
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if digest != meta["weights_sha256"]:
        print("refusing: the model does not match its metadata", file=sys.stderr)
        return 2
    profile = load_profile()["profile"]
    freeze = {
        "schema": FREEZE_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "parent_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "fingerprint": fingerprint(),
        "erudition_model": {
            "path": str(args.model.resolve().relative_to(ROOT)),
            "weights_sha256": digest,
            "trainable_parameters": meta["trainable_parameters"],
            "margin": meta["margin"],
        },
        "language_model": {
            "artifact": profile["artifact"],
            "runtime": profile["runtime"],
            "template": profile["template"],
        },
        "confirmation": {"split": "confirmation", "indices": INDICES, "arms": ARMS, "backend": "llama"},
        "contracts": CONTRACTS,
        "promotion_contract": "aaa.promotion.paired.v1",
    }
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    FREEZE.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"wrote {FREEZE.relative_to(ROOT)}; commit it alone before any confirmation run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
