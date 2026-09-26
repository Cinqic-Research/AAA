"""Write the opaque.v0 confirmation freeze manifest (run once, then commit before confirmation).

``python -m research.aaa_wm.opaque.write_freeze``

The manifest fixes, before any confirmation identity exists:

* the source fingerprint and specification hash (:mod:`.freeze`);
* the design: confirmation task count (slice-pure streams of 40), fresh initializations
  shared by every arm, the run and step budgets, and stream reset for online arms;
* the arms (candidate, references, causal controls, baselines, ceiling);
* the ``aaa.promotion.crossed.v1`` contracts, with a gatekeeping order;
* the hypotheses and what counts as WORLD_MODEL_SUCCESS for this phase;
* the resource budget.

Thresholds are practical. Against each comparator the candidate must cut the per-episode error
rate by at least 25% (the upper bound of the error ratio < 0.75): 20% for the training-learned
library control and 5% for online adaptation. On no slice may it be worse than the reference
(per-slice non-inferiority at 1.0). The thresholds were fixed after the development evaluate stage
and the red team, and before any confirmation identity existed.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import generator as gen
from .freeze import FREEZE_PATH, ROOT, SCHEMA, fingerprint

INITS = [100, 101, 102, 103, 104]
SEED = 20260927


def contract(name: str, reference: str, challenger: str, threshold: float, *, deterministic_reference: bool = False) -> dict:
    return {
        "name": name,
        "contract_id": "aaa.promotion.crossed.v1",
        "reference": reference,
        "challenger": challenger,
        "reference_deterministic": deterministic_reference,
        "initializations": INITS,
        "groups": "every slice, design crossed, series = the slice's 20 streams",
        "metric": "Jeffreys-smoothed per-cell error (errors + 0.5) / (40 + 1)",
        "criterion": {"name": "superior", "rule": "superior", "threshold": threshold},
        "threshold": threshold,
        "seed": SEED,
        "draws": 20000,
    }


def artifact_hashes() -> dict[str, str]:
    """SHA-256 of every learned artifact the confirmation will load (produced from the train split only)."""

    import hashlib

    from .experiment import ckpt_path, data_root

    out = {}
    for s in INITS:
        paths = [data_root() / "wms" / f"table_s{s}_p200000.json", ckpt_path("policy_aux", s, 20000), ckpt_path("wm", s, 20000)]
        for p in paths:
            if not p.exists():
                raise SystemExit(f"missing artifact {p}: produce it before the freeze")
            out[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def manifest(reference: str) -> dict:
    """``reference`` = the strongest learned non-world-model arm on the evaluate stage (chosen there)."""

    fp = fingerprint(ROOT)
    slices = gen.load_spec()["slices"]["blocks"]
    arms = [
        {"name": "wms:online", "role": "candidate", "initializations": INITS},
        {"name": "policy_aux", "role": "learned model-free reference (tool search)", "initializations": INITS, "train_steps": 20000},
        {"name": "policy_aux:plan", "role": "learned model-free reference inside the WM-S planner (controller-matched)", "initializations": INITS, "train_steps": 20000},
        {"name": "wms:online+empty", "role": "causal control: online learning, no training-learned library", "initializations": INITS},
        {"name": "wms:empty", "role": "causal control: interpreter only, no library knowledge", "initializations": [0]},
        {"name": "wms", "role": "adaptation control: frozen library model", "initializations": INITS},
        {"name": "wm:d1", "role": "black-box neural world model (tune-selected depth 1)", "initializations": INITS, "train_steps": 20000},
        {"name": "tells", "role": "strongest non-learned baseline (grammar shortcut, red team)", "initializations": [0]},
        {"name": "tool_gap", "role": "non-learned tool search", "initializations": [0]},
        {"name": "ceiling", "role": "true-library planner (not a baseline)", "initializations": [0]},
    ]
    contracts = [
        contract("C1_wm_over_learned_reference", reference, "wms:online", 0.75),
        contract("C2a_training_learned_library_is_causal", "wms:online+empty", "wms:online", 0.80),
        contract("C2b_library_model_is_causal", "wms:empty", "wms:online", 0.75, deterministic_reference=True),
        contract("C3_wm_over_best_nonlearned", "tells", "wms:online", 0.75, deterministic_reference=True),
        contract("C4_online_adaptation", "wms", "wms:online", 0.95),
        contract("C5_structured_over_blackbox_wm", "wm:d1", "wms:online", 0.75),
    ]
    for g in slices:
        c = contract(f"N_{g}_noninferior", reference, "wms:online", 1.0)
        c["groups"] = [g]
        c["criterion"] = {"name": "noninferior", "rule": "noninferior", "threshold": 1.0}
        contracts.append(c)
    return {
        "schema": SCHEMA,
        "protocol": gen.PROTOCOL,
        "source_fingerprint": fp["sha256"],
        "source_files": fp["files"],
        "spec_sha256": gen.spec_hash(),
        "artifacts_sha256": artifact_hashes(),
        "design": {
            "tasks": 4000,
            "stream_length": 40,
            "slices": slices,
            "streams_per_slice": 20,
            "initializations": INITS,
            "runs_budget": gen.load_spec()["budgets"]["runs"],
            "steps_budget": gen.load_spec()["budgets"]["steps"],
            "online_stream_reset": True,
            "confirmation_split": "confirmation indices [0, 4000); programs of pilot/train/development/attack excluded",
            "artifacts": "WM-S tables and neural checkpoints at the confirmation initializations were produced from the train split before this freeze; hashes above; confirm refuses mismatches",
        },
        "arms": arms,
        "contracts": contracts,
        "gatekeeping": ["C1", "C2a", "C2b", "C3", "N_*", "C5", "C4"],
        "hypotheses": {
            "H-OP1": "WM-S online improves end-to-end success over the strongest learned non-world-model reference (C1)",
            "H-OP1-causal": "the training-learned library model (C2a) and library knowledge at all (C2b) are causally necessary",
            "H-OP1-nonlearned": "the gain exceeds the strongest non-learned shortcut at the same run budget (C3)",
            "H-OP1-noregression": "WM-S is not worse than the reference on any slice (N_*)",
            "H-OP2": "WM-S beats the black-box neural world model (C5)",
            "H-OP3": "online adaptation of the library model improves over the frozen model (C4)",
            "success_rule": "WORLD_MODEL_SUCCESS (this phase, scoped claim) requires C1, C2a, C2b, C3 and every N_* to PROMOTE with agreeing implementations, attack results that do not explain the gain, and independent review; C4 and C5 scope the claim",
            "claim_scope": "an exact interpreter of visible Python plus a learned tabular model of an opaque library, used by a verified depth-2 planner, versus learned model-free and non-learned agents with the same two real runs",
        },
        "resource_budget": {"cpu_seconds_per_episode_wms_max": 2.0, "real_runs_per_episode": 2, "ram_cap_per_job_gb": 6},
        "written_before_confirmation": True,
    }


def main() -> int:
    path = ROOT / FREEZE_PATH
    if path.exists():
        raise SystemExit(f"{FREEZE_PATH} exists; a freeze is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    import sys

    path.write_text(json.dumps(manifest(sys.argv[1] if len(sys.argv) > 1 else "policy_aux:plan"), indent=1) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

_ = Path
