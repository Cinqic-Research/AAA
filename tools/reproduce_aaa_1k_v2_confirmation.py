"""Rerun the aaa.1k.v2 confirmation and compare every cell with the committed artifact.

``confirm`` spends its identity blocks and cannot run twice. This tool reads the
committed registry, in which those blocks are already spent by the recorded
observer, and reruns ``run_confirmation`` on the same frozen arms. It writes
nothing to the repository: it compares the rerun with
``docs/evidence/aaa_1k_v2/confirmation.json`` and prints the result.

On the evidence platform (FLOWBOX, Zen 3, CPU float64), the reproduction
standard is bitwise equality of every primitive, capability record and
baseline. Elsewhere it is verdict-level (``AAA-173``), and the tool reports the
largest numerical drift instead.

    python tools/reproduce_aaa_1k_v2_confirmation.py --workers 8 [--report out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_1k_v2 import identities  # noqa: E402
from research.aaa_1k_v2.confirmation import run_confirmation  # noqa: E402
from research.aaa_1k_v2.evidence import sha256_file  # noqa: E402
from research.aaa_1k_v2.freeze import arm_from_dict  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_1k_v2"
VOLATILE = {"compute_seconds", "wall_seconds", "environment"}


def _walk(stored: Any, rerun: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(stored, dict) and isinstance(rerun, dict):
        if set(stored) != set(rerun):
            out["structure"].append(f"{path}: keys differ")
        for key in set(stored) & set(rerun):
            if not path and key in VOLATILE:
                continue
            _walk(stored[key], rerun[key], f"{path}/{key}", out)
    elif isinstance(stored, list) and isinstance(rerun, list):
        if len(stored) != len(rerun):
            out["structure"].append(f"{path}: length {len(stored)} != {len(rerun)}")
        for index, (a, b) in enumerate(zip(stored, rerun, strict=False)):
            _walk(a, b, f"{path}[{index}]", out)
    elif isinstance(stored, float) and isinstance(rerun, float):
        out["floats"] += 1
        if stored != rerun and not (math.isnan(stored) and math.isnan(rerun)):
            out["different_floats"] += 1
            scale = max(abs(stored), abs(rerun), 1e-300)
            drift = abs(stored - rerun) / scale
            if drift > out["max_relative_drift"]:
                out["max_relative_drift"], out["worst"] = drift, path
    elif stored != rerun:
        out["structure"].append(f"{path}: {stored!r} != {rerun!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--data-root", help="Monash data root (default $AAA_DATA_ROOT)")
    parser.add_argument(
        "--report", help="optional path for a JSON comparison report (outside the repository)"
    )
    args = parser.parse_args()

    stored = json.loads((EVIDENCE / "confirmation.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "freeze.json").read_text(encoding="utf-8"))
    manifest["_sha256"] = sha256_file(EVIDENCE / "freeze.json")
    frozen = manifest["frozen"]
    manifest.update(
        arms=frozen["arms"],
        capability_arms=frozen["capability_arms"],
        monash_arms=frozen["monash_arms"],
        challenger=frozen["challenger"],
    )
    arms = {name: arm_from_dict(entry["spec"]) for name, entry in frozen["arms"].items()}
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    started = time.time()
    rerun = run_confirmation(
        registry,
        manifest,
        arms,
        observer=stored["observer"],
        workers=args.workers,
        data_root=args.data_root,
        log=lambda m: print(m, flush=True),
    )
    comparison: dict[str, Any] = {
        "floats": 0,
        "different_floats": 0,
        "max_relative_drift": 0.0,
        "worst": None,
        "structure": [],
    }
    _walk(json.loads(json.dumps(stored)), json.loads(json.dumps(rerun)), "", comparison)
    comparison["bitwise"] = comparison["different_floats"] == 0 and not comparison["structure"]
    comparison["decision_equal"] = rerun["decision"] == stored["decision"]
    comparison["wall_seconds"] = time.time() - started
    if args.report:
        Path(args.report).write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(
        f"floats compared {comparison['floats']:,}; different {comparison['different_floats']:,}; "
        f"structural differences {len(comparison['structure'])}; bitwise {comparison['bitwise']}; "
        f"decision equal {comparison['decision_equal']}; max relative drift {comparison['max_relative_drift']:.3e}"
    )
    return 0 if comparison["decision_equal"] and not comparison["structure"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
