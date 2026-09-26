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

Thresholds are practical: the candidate must cut the per-episode error rate by at least 20%
relative to each comparator (the upper bound of the error ratio < 0.80). They were chosen for
practical meaning, not from confirmation data. Development variance (``evaluate`` stage) was
used only to check that the declared size resolves an effect of that magnitude.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import generator as gen
from .freeze import FREEZE_PATH, ROOT, SCHEMA, fingerprint

INITS = [100, 101, 102]
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


def manifest() -> dict:
    fp = fingerprint(ROOT)
    arms = [
        {"name": "wms:online", "role": "candidate", "initializations": INITS},
        {"name": "policy_aux", "role": "strongest learned non-world-model reference (tune)", "initializations": INITS, "train_steps": 20000},
        {"name": "wms:empty", "role": "causal control: same agent, learned library removed", "initializations": INITS},
        {"name": "wms", "role": "adaptation control: frozen library model", "initializations": INITS},
        {"name": "wm:d1", "role": "black-box neural world model (tune-selected depth 1)", "initializations": INITS, "train_steps": 20000},
        {"name": "tool_gap", "role": "strongest non-learned baseline", "initializations": [0]},
        {"name": "ceiling", "role": "true-library planner (not a baseline)", "initializations": [0]},
    ]
    contracts = [
        contract("C1_wm_over_learned_reference", "policy_aux", "wms:online", 0.80),
        contract("C2_learned_library_is_causal", "wms:empty", "wms:online", 0.80),
        contract("C3_wm_over_best_tool_baseline", "tool_gap", "wms:online", 0.80, deterministic_reference=True),
        contract("C4_online_adaptation", "wms", "wms:online", 0.90),
        contract("C5_structured_over_blackbox_wm", "wm:d1", "wms:online", 0.80),
    ]
    return {
        "schema": SCHEMA,
        "protocol": gen.PROTOCOL,
        "source_fingerprint": fp["sha256"],
        "source_files": fp["files"],
        "spec_sha256": gen.spec_hash(),
        "design": {
            "tasks": 4000,
            "stream_length": 40,
            "slices": gen.load_spec()["slices"]["blocks"],
            "streams_per_slice": 20,
            "initializations": INITS,
            "runs_budget": gen.load_spec()["budgets"]["runs"],
            "steps_budget": gen.load_spec()["budgets"]["steps"],
            "online_stream_reset": True,
            "confirmation_split": "confirmation indices [0, 4000); programs of pilot/train/development/attack excluded",
            "neural_training": "policy_aux and wm retrained from the frozen source at seeds 100-102 (train split only) before confirmation",
            "wms_tables": "learned at seeds 100-102 from 200,000 train programs (train split only)",
        },
        "arms": arms,
        "contracts": contracts,
        "gatekeeping": ["C1", "C2", "C3", "C5", "C4"],
        "hypotheses": {
            "H-OP1": "WM-S online improves end-to-end success over the strongest learned non-world-model reference (C1 PROMOTE)",
            "H-OP1-causal": "removing the learned library model destroys the gain (C2 PROMOTE)",
            "H-OP1-tool": "the gain exceeds the best non-learned tool baseline at the same run budget (C3 PROMOTE)",
            "H-OP2": "WM-S beats the black-box neural world model (C5 PROMOTE)",
            "H-OP3": "online adaptation of the library model improves over the frozen model (C4 PROMOTE)",
            "success_rule": "WORLD_MODEL_SUCCESS (this phase) requires C1, C2 and C3 PROMOTE with agreeing implementations, no slice where wms:online is worse than policy_aux, attack results that do not explain the gain, and independent review; C4/C5 are reported and scope the claim",
        },
        "resource_budget": {"cpu_seconds_per_episode_wms_max": 2.0, "real_runs_per_episode": 2, "ram_cap_per_job_gb": 6},
        "written_before_confirmation": True,
    }


def main() -> int:
    path = ROOT / FREEZE_PATH
    if path.exists():
        raise SystemExit(f"{FREEZE_PATH} exists; a freeze is never rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(), indent=1) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

_ = Path
