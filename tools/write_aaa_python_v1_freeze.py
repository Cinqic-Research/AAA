"""Write the ``aaa.python.v1`` confirmation freeze from committed development evidence.

The manifest is derived mechanically from the development stages:

* the encoder selected by the declared rule, and the 16-unit ``e1``/``e2`` arms
  with their tuned budgets (the encoder claim at matched capacity);
* the capacity arms (~1K, ~4K, ~10K) of the selected encoder and heads, with
  their tuned budgets (the scale claim);
* the baselines;
* the primary contrasts and the declared 1K-versus-10K rule;
* three ``aaa.promotion.crossed.v1`` contracts on the Jeffreys-smoothed per-cell
  error ratio (challenger / reference; lower is better), each over the five
  families as ``crossed`` groups of 30 streams and the ten fresh
  initializations 2000-2009, 20,000 draws:

  - ``encoder``: ``h16@e2`` against ``h16@e1``, superior at 0.95;
  - ``scale_over_1k``: 10K against 1K, superior at 0.95;
  - ``scale_over_4k``: 10K against 4K, superior at 0.97;

* the phase fingerprint and specification hash of the source that will run.

The freeze must be committed, with a clean tree, before ``confirm`` runs; the
fingerprint excludes the manifest itself. Run from a clean checkout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_python_v1 import spec as spec_module  # noqa: E402
from research.aaa_python_v1 import stages  # noqa: E402
from research.aaa_python_v1.experiment import DESIGN  # noqa: E402
from research.aaa_python_v1.freeze import FREEZE_PATH, FREEZE_SCHEMA  # noqa: E402
from research.aaa_python_v1.identity import fingerprint  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_python_v1"
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")


def _load(stage: str) -> dict:
    return json.loads((EVIDENCE / f"stage_{stage}.json").read_text(encoding="utf-8"))["stage"]


def _arm(payload: dict, name: str) -> dict:
    keep = (
        "encoder",
        "hidden",
        "output_head",
        "localize_head",
        "channels",
        "tool",
        "weight_decay",
        "momentum",
        "clip",
        "train_per_family",
        "learning_rate",
        "epochs",
    )
    return {"name": name, **{k: payload[k] for k in keep}}


def manifest() -> dict:
    encoders = _load("encoders")
    capacity = _load("capacity")
    encoder = stages.select_encoder(encoders["summary"])
    arms = [_arm(encoders["arms"][n], n) for n in ("h16@e1", "h16@e2")]
    arms += [_arm(capacity["arms"][f"{t}@{encoder}"], f"{t}@{encoder}") for t in ("1k", "4k", "10k")]
    shape = DESIGN["confirmation"]
    inits = list(range(shape["init_offset"], shape["init_offset"] + shape["initializations"]))
    groups = [{"name": f, "design": "crossed", "series": list(range(shape["streams"]))} for f in FAMILIES]

    def contract(name: str, reference: str, challenger: str, threshold: float, seed: int) -> dict:
        return {
            "name": name,
            "contract_id": "aaa.promotion.crossed.v1",
            "reference": reference,
            "challenger": challenger,
            "initializations": inits,
            "groups": groups,
            "criteria": [{"name": f"{name}_superior", "rule": "superior", "threshold": threshold}],
            "seed": seed,
            "draws": 20000,
            "metric": "Jeffreys-smoothed error per cell, (errors + 0.5) / (n + 1); ratio challenger/reference",
        }

    return {
        "schema": FREEZE_SCHEMA,
        "protocol": "aaa.python.v1",
        "phase_fingerprint": fingerprint(ROOT)["sha256"],
        "spec_sha256": spec_module.spec_hash(),
        "design": {
            "encoder": encoder,
            "selections": capacity["selections"],
            "primary": [
                ["h16@e2", "h16@e1", "frozen"],
                [f"10k@{encoder}", f"1k@{encoder}", "frozen"],
                [f"10k@{encoder}", f"4k@{encoder}", "frozen"],
            ],
            "capacity_rule": "docs/aaa_python_v1_research_brief.md, 'Capacity decision rule', applied unchanged",
            "confirmation": shape,
            "development_capacity_verdict": capacity["capacity_verdict"],
        },
        "arms": arms,
        "baselines": ["majority", "lookup", "v0_heuristic", "rules", "medoid", "visible_tests"],
        "contracts": [
            contract("encoder", "h16@e1", "h16@e2", 0.95, 20260926),
            contract("scale_over_1k", f"1k@{encoder}", f"10k@{encoder}", 0.95, 20260927),
            contract("scale_over_4k", f"4k@{encoder}", f"10k@{encoder}", 0.97, 20260928),
        ],
        "hypotheses": {
            "C1": "At matched trainable capacity, e2 has lower smoothed error than e1 (geometric ratio upper bound < 0.95).",
            "C2": "10K has lower smoothed error than 1K (upper bound < 0.95), and the declared per-family rule holds on fresh tasks.",
            "C3": "10K earns its size over 4K (upper bound < 0.97).",
            "scale_justified_requires": "C2 and C3 PROMOTE, the per-family rule SCALE_JUSTIFIED_PENDING_ADAPTATION_AND_PLASTICITY, and no development adaptation, forgetting or plasticity regression at 10K",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = manifest()
    text = json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if args.write:
        path = ROOT / FREEZE_PATH
        if path.exists():
            print("refused: a freeze exists and is never rewritten", file=sys.stderr)
            return 2
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
