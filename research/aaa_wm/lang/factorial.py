"""Summarize the language-conditioned factorial (decision system x language channel), development.

``python -m research.aaa_wm.lang.factorial $AAA_DATA_ROOT/lang/play_*.json``

Arms are keyed ``agent@channel``. Cells are initialization x stream (40 tasks), as in
:mod:`research.aaa_wm.opaque.summarize`. Contrasts are paired by initialization and stream.
The adapter seed *s* is paired with WM-S table seed *s*, and each neural policy with seed *s*.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from typing import Any

import numpy as np

from ..opaque.summarize import contrast, table


def load(paths: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], list[str], dict[str, float]]:
    arms: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    slices: list[str] | None = None
    exact: dict[str, list[float]] = defaultdict(list)
    for p in paths:
        d = json.load(open(p))
        ch = d["channel"]
        channel = "none" if ch == "none" else "rules" if ch == "rules" else "gold" if ch == "gold" else ("lm_random" if "_random_" in ch else "lm")
        if slices is None:
            slices = d["slices"]
        for agent, per in d["results"].items():
            for seed, r in per.items():
                arms[f"{agent}@{channel}"][seed] = np.array([c == "1" for c in r["bits"]], dtype=float)
        exact[channel].append(d["channel_exact"])
    assert slices is not None
    return dict(arms), slices, {k: float(np.mean(v)) for k, v in exact.items()}


CONTRASTS = [
    ("wms:online@lm", "wms:online@lm_random", "L1 pretraining, end to end (WM + LM vs WM + random-init LM)"),
    ("wms:online@lm", "wms:online@rules", "L2 language model over hand-written rules (with WM)"),
    ("wms:online@lm", "wms:online@none", "language channel over none (with WM)"),
    ("wms:online@lm", "policy_aux:plan@lm", "WM contribution given the LM (WM + LM vs policy-in-planner + LM)"),
    ("wms:online@lm", "policy_aux@lm", "full system vs model-free policy + LM"),
    ("wms:online@lm", "policy_aux:plan@rules", "full system vs policy-in-planner + rules"),
    ("policy_aux:plan@lm", "policy_aux:plan@rules", "LM over rules without WM"),
    ("wms:online@gold", "wms:online@lm", "remaining gap to perfect language understanding (with WM)"),
]


def main() -> int:
    arms, slices, exact = load(sys.argv[1:])
    seeds = sorted({s for b in arms.values() for s in b}, key=int)
    tab = table(arms, slices, seeds)
    print("channel exact extraction:", {k: round(v, 3) for k, v in exact.items()})
    for arm, row in sorted(tab.items(), key=lambda kv: -kv[1]["all"][0]):
        m, lo, hi = row["all"]
        print(f"{arm:32s} {m:.3f} [{lo:.3f}, {hi:.3f}]")
    out: dict[str, Any] = {"seeds": seeds, "exact": exact, "table": tab, "contrasts": {}}
    for a, b, label in CONTRASTS:
        if a in arms and b in arms:
            r = contrast(arms, slices, seeds, a, b)
            out["contrasts"][label] = {"a": a, "b": b, **r}
            print(f"{label}: {r['difference']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['sign']}")
    json.dump(out, open(sys.argv[1].rsplit("/", 1)[0] + "/factorial_summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
