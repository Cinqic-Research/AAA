#!/usr/bin/env python3
"""Apply the ``aaa.promotion.crossed.v1`` successor to the retained ``aaa.1k.v2`` K5 groups.

This is a **descriptive** agreement check on already-observed evidence, not a
promotion decision. The frozen v2 confirmation had no challenger
(``NO_CHALLENGER``). Its K5 path is forbidden for promotion (``AAA-180``).
Nothing here changes a v2 file, verdict or report.

For each of the six non-champion arms that has Monash cells, the successor
contract declares all nineteen v2 K5 groups: NARMA-10/20, Mackey-Glass, the
twelve dysts systems, and the four Monash datasets restricted to series whose
MASE scale is defined (the v2 ``keep`` rule). ``monash:saugeen`` has one
series and is declared ``conditional_on_single_series``. The margin is the
frozen v2 ``external_margin`` (ratio 1.02). Because the arms were never
challengers and the identities are read from the retained evidence rather
than declared before observation, every result is labelled descriptive.

The check passes when both successor implementations agree for every arm, so
the verdict is neither ``DISAGREEMENT`` nor ``INVALID_EVIDENCE``. It also
reproduces ``AAA-180`` on the frozen code for contrast: without Monash,
frozen ``decide`` gives one K5 status; with Monash, frozen ``decide`` gives
``INCONCLUSIVE`` while the frozen recompute gives ``FAIL``.

    python tools/check_promotion_successor.py
"""

from __future__ import annotations

import copy
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aaa.promotion import Contract, Criterion, GroupDeclaration, adjudicate  # noqa: E402
from research.aaa_1k_v2 import identities  # noqa: E402
from research.aaa_1k_v2.confirmation import decide  # noqa: E402
from research.aaa_1k_v2.recompute import recompute  # noqa: E402
from research.aaa_1k_v2.tournament import external_metric  # noqa: E402

EVIDENCE = ROOT / "docs/evidence/aaa_1k_v2"
REFERENCE = "c1_champion1"
DIAGNOSTIC_SEED = 20260923


def _keep(confirmation: Mapping[str, Any]) -> dict[str, set[str]]:
    return {
        task: {
            sid
            for sid, record in records.items()
            if record["persistence"].get("forecast_mase") is not None
            and math.isfinite(record["persistence"]["forecast_mase"])
        }
        for task, records in confirmation["baselines"]["monash"].items()
    }


def successor_inputs(
    confirmation: Mapping[str, Any], manifest: Mapping[str, Any], arm: str
) -> tuple[Contract, list[dict[str, Any]]]:
    keep = _keep(confirmation)
    frozen = manifest["frozen"]
    cells: dict[str, list[Mapping[str, Any]]] = {}
    for name in (REFERENCE, arm):
        external = [c for c in confirmation["primitives"][name] if c["group"] in frozen["external_tasks"]]
        monash = [
            c
            for c in confirmation["monash_primitives"][name]
            if c["group"] in frozen["monash_tasks"] and c["series_id"] in keep[c["group"]]
        ]
        cells[name] = external + monash
    groups = []
    for task in [*frozen["external_tasks"], *frozen["monash_tasks"]]:
        series = sorted({c["series_id"] for c in cells[REFERENCE] if c["group"] == task})
        design = "crossed" if len(series) >= 2 else "conditional_on_single_series"
        groups.append(GroupDeclaration(task, design, tuple(series)))
    inits = tuple(sorted({c["init"] for c in cells[REFERENCE]}))
    contract = Contract(
        reference=REFERENCE,
        challenger=arm,
        initializations=inits,
        groups=tuple(groups),
        criteria=(
            Criterion(
                "K5_external_noninferiority", "noninferior", 1.0 + frozen["thresholds"]["external_margin"]
            ),
        ),
        seed=DIAGNOSTIC_SEED,
        notes={"status": "descriptive; identities read from retained evidence"},
    )
    records = []
    for name, rows in cells.items():
        for c in rows:
            value = None if c.get("failed") else c.get(external_metric(c["group"]))
            records.append(
                {
                    "arm": name,
                    "group": c["group"],
                    "init": c["init"],
                    "series": c["series_id"],
                    "value": value,
                }
            )
    return contract, records


def frozen_contrast(
    confirmation: Mapping[str, Any], manifest: Mapping[str, Any], registry: Mapping[str, Any], arm: str
) -> dict[str, str]:
    persistence = {
        r["stream_id"]: r["persistence"] for recs in confirmation["baselines"]["dot"].values() for r in recs
    }
    external = {sid: m for task in confirmation["baselines"]["external"].values() for sid, m in task.items()}
    common = {
        "persistence": persistence,
        "external_base": external,
        "bootstrap": identities.seeds_of(dict(registry), "v2-confirmation-bootstrap"),
        "families": confirmation["families"],
        "external": confirmation["external"],
    }
    without = decide(confirmation["primitives"], arm, **common)
    with_monash = decide(
        confirmation["primitives"],
        arm,
        **common,
        monash_primitives=confirmation["monash_primitives"],
        monash_keep=_keep(confirmation),
    )
    fake = copy.deepcopy(dict(manifest))
    fake["frozen"]["challenger"] = {"name": arm}
    independent = recompute(confirmation, fake, registry)
    return {
        "frozen_decide_as_called": without["statuses"]["K5_external_noninferiority"],
        "frozen_decide_with_monash": with_monash["statuses"]["K5_external_noninferiority"],
        "frozen_recompute": independent["statuses"]["K5_external_noninferiority"],
    }


def main() -> int:
    confirmation = json.loads((EVIDENCE / "confirmation.json").read_text(encoding="utf-8"))
    manifest = json.loads((EVIDENCE / "freeze.json").read_text(encoding="utf-8"))
    registry = identities.load_registry(ROOT / identities.REGISTRY_PATH)
    arms = sorted(a for a in confirmation["monash_primitives"] if a != REFERENCE)
    failures = []
    for arm in arms:
        contract, records = successor_inputs(confirmation, manifest, arm)
        result = adjudicate(contract, records)
        contrast = frozen_contrast(confirmation, manifest, registry, arm)
        p, i = result.get("primary", {}), result.get("independent", {})
        print(
            f"{arm}: successor {result['verdict']} (scope {contract.scope}; conditional {contract.conditional_groups}); "
            f"primary {p.get('geometric_ratio', float('nan')):.4f} [{p.get('lower') or float('nan'):.4f}, {p.get('upper') or float('nan'):.4f}], "
            f"independent [{i.get('lower') or float('nan'):.4f}, {i.get('upper') or float('nan'):.4f}] | "
            f"frozen v2 K5: as called {contrast['frozen_decide_as_called']}, "
            f"with Monash {contrast['frozen_decide_with_monash']}, recompute {contrast['frozen_recompute']}"
        )
        if result["verdict"] in {"DISAGREEMENT", "INVALID_EVIDENCE"}:
            failures.append(
                f"{arm}: {result['verdict']} {result.get('reason') or result.get('agreement_problems')}"
            )
    for failure in failures:
        print(f"FAILED: {failure}", file=sys.stderr)
    print("descriptive successor agreement on retained v2 K5 groups: " + ("FAILED" if failures else "AGREE"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
