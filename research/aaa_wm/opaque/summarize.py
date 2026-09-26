"""Summaries of opaque.v0 evaluation documents (development and, later, confirmation).

Cells are ``initialization x stream`` (a stream = one block of 40 consecutive, slice-pure tasks).
Reported per arm and slice: success rate with a crossed percentile bootstrap (initializations
and streams resampled), paired differences for declared contrasts, and, when requested, the
``aaa.promotion.crossed.v1`` adjudication of the Jeffreys-smoothed error ratio. Development
adjudications are *descriptive*: nothing is promoted by a development document.

    python -m research.aaa_wm.opaque.summarize EVAL.json [EVAL.json ...] --contrast A:B ...
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

BLOCK = 40


def load(paths: Sequence[str]) -> tuple[dict[str, dict[str, np.ndarray]], list[str]]:
    """arm -> seed -> bits array (tasks); slices per task (identical across documents)."""

    arms: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    slices: list[str] | None = None
    tasks: list[str] | None = None
    for p in paths:
        d = json.load(open(p))
        if tasks is None:
            tasks, slices = d["tasks"], d["slices"]
        elif d["tasks"] != tasks:
            raise ValueError(f"{p}: different task identities")
        for arm, per_seed in d["results"].items():
            for seed, r in per_seed.items():
                arms[arm][seed] = np.array([c == "1" for c in r["bits"]], dtype=float)
    assert slices is not None
    return dict(arms), slices


def cells(bits: dict[str, np.ndarray], slices: list[str], seeds: list[str], group: str) -> np.ndarray:
    """[seeds, streams] success rates for one slice (deterministic arms broadcast their one seed)."""

    idx = [i for i, s in enumerate(slices) if s == group]
    streams = [idx[k : k + BLOCK] for k in range(0, len(idx), BLOCK)]
    rows = []
    for seed in seeds:
        b = bits.get(seed, bits.get("0") if len(bits) == 1 else None)
        if b is None:
            raise KeyError(f"missing seed {seed}")
        rows.append([b[s].mean() for s in streams])
    return np.array(rows)


def crossed_ci(x: np.ndarray, draws: int = 4000, seed: int = 20260926) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    S, T = x.shape
    est = []
    for _ in range(draws):
        est.append(x[np.ix_(rng.integers(0, S, S), rng.integers(0, T, T))].mean())
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def table(arms: dict[str, dict[str, np.ndarray]], slices: list[str], seeds: list[str]) -> dict[str, dict[str, Any]]:
    groups = sorted(set(slices))
    out: dict[str, dict[str, Any]] = {}
    for arm, bits in arms.items():
        row: dict[str, Any] = {}
        per = [cells(bits, slices, seeds, g) for g in groups]
        row["all"] = crossed_ci(np.concatenate(per, axis=1))
        for g, c in zip(groups, per, strict=True):
            row[g] = crossed_ci(c)
        out[arm] = row
    return out


def contrast(arms: dict[str, dict[str, np.ndarray]], slices: list[str], seeds: list[str], a: str, b: str) -> dict[str, Any]:
    groups = sorted(set(slices))
    da = np.concatenate([cells(arms[a], slices, seeds, g) for g in groups], axis=1)
    db = np.concatenate([cells(arms[b], slices, seeds, g) for g in groups], axis=1)
    m, lo, hi = crossed_ci(da - db)
    return {"difference": m, "ci": [lo, hi], "sign": "POSITIVE" if lo > 0 else "NEGATIVE" if hi < 0 else "INCONCLUSIVE"}


def contract_records(arms: dict[str, dict[str, np.ndarray]], slices: list[str], seeds: list[str], arm: str) -> list[dict[str, Any]]:
    """Primitives for aaa.promotion.crossed.v1: Jeffreys-smoothed error per (init, stream) cell."""

    recs = []
    for g in sorted(set(slices)):
        c = cells(arms[arm], slices, seeds, g)
        for i, seed in enumerate(seeds):
            for s in range(c.shape[1]):
                errors = round((1 - c[i, s]) * BLOCK)
                recs.append({"arm": arm, "group": g, "init": seed, "series": s, "value": (errors + 0.5) / (BLOCK + 1)})
    return recs


def adjudicate_ratio(arms: dict[str, dict[str, np.ndarray]], slices: list[str], seeds: list[str], reference: str, challenger: str, threshold: float, seed: int = 20260926, *, rule: str = "superior", only: list[str] | None = None) -> dict[str, Any]:
    from aaa.promotion.adjudicate import adjudicate
    from aaa.promotion.contract import Contract, Criterion, GroupDeclaration

    groups = sorted(set(slices)) if only is None else sorted(only)
    n_streams = {g: sum(1 for s in slices if s == g) // BLOCK for g in groups}
    contract = Contract(
        reference=reference,
        challenger=challenger,
        initializations=tuple(seeds),
        groups=tuple(GroupDeclaration(g, "crossed", tuple(range(n_streams[g]))) for g in groups),
        criteria=(Criterion(rule, rule, threshold),),
        seed=seed,
        draws=20000,
    )
    recs = contract_records(arms, slices, seeds, reference) + contract_records(arms, slices, seeds, challenger)
    recs = [r for r in recs if r["group"] in groups]
    res = adjudicate(contract, recs)
    p = res.get("primary", {})
    q = res.get("independent", {})
    return {
        "verdict": res["verdict"],
        "ratio": p.get("geometric_ratio"),
        "interval": [p.get("lower"), p.get("upper")],
        "interval_status": p.get("interval_status"),
        "group_ratios": p.get("group_ratios"),
        "independent_ratio": q.get("geometric_ratio"),
        "independent_interval": [q.get("lower"), q.get("upper")],
        "agreement_problems": res.get("agreement_problems"),
        "threshold": threshold,
        "reason": res.get("reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--contrast", action="append", default=[])
    ap.add_argument("--ratio", action="append", default=[], help="REF|CHALLENGER|THRESHOLD (descriptive crossed.v1)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    arms, slices = load(args.evals)
    seeds = args.seeds.split(",") if args.seeds else sorted({s for b in arms.values() for s in b}, key=int)
    tab = table(arms, slices, seeds)
    groups = sorted(set(slices))
    print(f"{'arm':26s} {'all':>20s} " + " ".join(f"{g[:12]:>14s}" for g in groups))
    for arm, row in sorted(tab.items(), key=lambda kv: -kv[1]["all"][0]):
        m, lo, hi = row["all"]
        print(f"{arm:26s} {m:.3f} [{lo:.3f},{hi:.3f}] " + " ".join(f"{row[g][0]:14.3f}" for g in groups))
    out: dict[str, Any] = {"seeds": seeds, "table": tab, "contrasts": {}, "ratios": {}}
    for c in args.contrast:
        a, b = c.split("|")
        r = contrast(arms, slices, seeds, a, b)
        out["contrasts"][c] = r
        print(f"{a} - {b}: {r['difference']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['sign']}")
    for c in args.ratio:
        ref, ch, th = c.split("|")
        r = adjudicate_ratio(arms, slices, seeds, ref, ch, float(th))
        out["ratios"][c] = r
        print(f"ratio {ch}/{ref}: {r}")
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
