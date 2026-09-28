#!/usr/bin/env python3
"""Audit retained WM and language decisions without replaying spent tasks.

The confirmation implementations are frozen. This post-freeze gate validates the
stored primitive grid and recomputes every numeric decision field from its bits.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.aaa_wm.opaque.summarize import adjudicate_ratio  # noqa: E402

CASES = (
    ("opaque", 0, 4000, "docs/evidence/aaa_wm_opaque_v0"),
    ("language", 4000, 2000, "docs/evidence/aaa_wm_lang_v0"),
)
GROUPS = ("in_distribution", "two_fault", "novel_literals", "novel_grammar", "library_B")


def strict_load(path: Path) -> dict[str, Any]:
    def reject(token: str) -> Any:
        raise ValueError(f"non-standard JSON constant {token}")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate JSON key {key}")
            out[key] = value
        return out

    return json.loads(path.read_text(), parse_constant=reject, object_pairs_hook=unique)


def _close(a: Any, b: Any, *, width: float = 0.0) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) <= max(1e-12 * max(abs(a), abs(b)), width)
    )


def independent_point(results: dict[str, Any], slices: list[str], contract: dict[str, Any]) -> float:
    """Rebuild the declared geometric error ratio directly from stored bit strings."""

    groups = sorted(contract["groups"]) if isinstance(contract.get("groups"), list) else sorted(set(slices))
    logs = []
    for group in groups:
        positions = [i for i, label in enumerate(slices) if label == group]
        assert len(positions) % 40 == 0
        means = []
        for arm in (contract["reference"], contract["challenger"]):
            errors = []
            for seed in contract["initializations"]:
                per = results[arm]
                bits = per.get(str(seed), per.get("0"))["bits"]
                for offset in range(0, len(positions), 40):
                    errors.append(
                        (40 - sum(bits[i] == "1" for i in positions[offset : offset + 40]) + 0.5) / 41
                    )
            means.append(sum(errors) / len(errors))
        logs.append(math.log(means[1] / means[0]))
    return math.exp(sum(logs) / len(logs))


def independent_interval(
    results: dict[str, Any], slices: list[str], contract: dict[str, Any]
) -> tuple[float, float]:
    """Third crossed bootstrap directly from correctness bits, with a distinct RNG stream."""

    import numpy as np

    names = sorted(contract["groups"]) if isinstance(contract.get("groups"), list) else sorted(set(slices))
    seeds = [str(s) for s in contract["initializations"]]
    if len(seeds) < 2:
        raise ValueError("crossed interval needs multiple initializations")
    draws = 20000
    rng = np.random.default_rng([contract["seed"], 0xA11D17])
    rows = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    logs = np.zeros(draws)
    for group in names:
        positions = [i for i, label in enumerate(slices) if label == group]
        streams = [positions[i : i + 40] for i in range(0, len(positions), 40)]
        cells = {}
        for arm in (contract["reference"], contract["challenger"]):
            by_seed = results[arm]
            cells[arm] = np.asarray(
                [
                    [
                        (
                            40
                            - sum(by_seed.get(seed, by_seed.get("0"))["bits"][i] == "1" for i in stream)
                            + 0.5
                        )
                        / 41
                        for stream in streams
                    ]
                    for seed in seeds
                ]
            )
        cols = rng.integers(0, len(streams), size=(draws, len(streams)))
        ref = cells[contract["reference"]][rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
        cha = cells[contract["challenger"]][rows[:, :, None], cols[:, None, :]].mean(axis=(1, 2))
        logs += np.log(cha / ref)
    ratios = np.exp(logs / len(names))
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def verify(
    kind: str,
    start: int,
    count: int,
    folder: str,
    *,
    freeze: dict[str, Any] | None = None,
    doc: dict[str, Any] | None = None,
) -> list[str]:
    base = ROOT / folder
    m = strict_load(base / "freeze.json") if freeze is None else freeze
    d = strict_load(base / "confirmation.json") if doc is None else doc
    problems: list[str] = []
    if d.get("freeze") != m:
        problems.append("embedded freeze differs from committed freeze")
    if d.get("provenance", {}).get("dirty") is not False:
        problems.append("confirmation provenance is dirty")
    expected_commit = (
        "f994d91215c35c9c484b12ac417699e97b152315"
        if kind == "opaque"
        else "a8c088dc250961ce15f49be3f9761814bff2ada6"
    )
    if d.get("provenance", {}).get("commit") != expected_commit:
        problems.append("unexpected confirmation provenance commit")
    tasks, slices = d.get("tasks"), d.get("slices")
    if tasks != [f"aaa.python.opaque.v0:confirmation:{i}" for i in range(start, start + count)]:
        problems.append("missing, extra or out-of-order task identities")
    if slices != [GROUPS[(i // 40) % 5] for i in range(count)]:
        problems.append("missing, extra or out-of-order slice labels")
    if kind == "opaque":
        hashes = d.get("task_sha256")
        if (
            not isinstance(hashes, list)
            or len(hashes) != count
            or any(not isinstance(h, str) or re.fullmatch(r"[0-9a-f]{64}", h) is None for h in hashes)
        ):
            problems.append("task hash grid is incomplete or malformed")
        declared = {a["name"]: {str(s) for s in a["initializations"]} for a in m["arms"]}
    else:
        declared = {f"{a}@{b}": {str(s) for s in m["design"]["initializations"]} for a, b in m["arms"]}
    results = d.get("results", {})
    if set(results) != set(declared):
        problems.append("result arms differ from the freeze")
    for arm, seeds in declared.items():
        per = results.get(arm, {})
        if set(per) != seeds:
            problems.append(f"{arm}: missing or extra initialization")
        for seed, row in per.items():
            bits = row.get("bits") if isinstance(row, dict) else None
            if not isinstance(bits, str) or len(bits) != count or set(bits) - {"0", "1"}:
                problems.append(f"{arm}/{seed}: missing or invalid correctness bits")
            if kind == "opaque" and isinstance(row, dict):
                runs = row.get("runs")
                if (
                    not isinstance(runs, list)
                    or len(runs) != count
                    or any(type(x) is not int or not 0 <= x <= m["design"]["runs_budget"] for x in runs)
                ):
                    problems.append(f"{arm}/{seed}: invalid real-run counts")
    contracts = {c["name"]: c for c in m["contracts"]}
    if set(d.get("adjudications", {})) != set(contracts):
        problems.append("stored adjudication names differ from the freeze")
    if problems:
        return problems
    assert isinstance(slices, list)

    import numpy as np

    arms = {
        a: {s: np.fromiter((b == "1" for b in row["bits"]), dtype=float) for s, row in per.items()}
        for a, per in results.items()
    }
    for name, c in contracts.items():
        ordered_seeds = [str(s) for s in c["initializations"]]
        ref = c["reference"]
        if c.get("reference_deterministic"):
            local_ref = {s: arms[ref].get(s, arms[ref]["0"]) for s in ordered_seeds}
        else:
            local_ref = arms[ref]
        local = {ref: local_ref, c["challenger"]: arms[c["challenger"]]}
        fresh = adjudicate_ratio(
            local,
            slices,
            ordered_seeds,
            ref,
            c["challenger"],
            c["threshold"],
            seed=c["seed"],
            rule=c["criterion"]["rule"],
            only=c["groups"] if isinstance(c.get("groups"), list) else None,
        )
        stored = d["adjudications"][name]
        if not _close(stored.get("ratio"), independent_point(results, slices, c)):
            problems.append(f"{name}: independent primitive ratio differs")
        for key in ("verdict", "interval_status", "agreement_problems", "threshold", "reason"):
            if stored.get(key) != fresh.get(key):
                problems.append(f"{name}: {key} differs")
        for key in ("ratio", "independent_ratio"):
            if not _close(stored.get(key), fresh.get(key)):
                problems.append(f"{name}: {key} differs")
        if set(stored.get("group_ratios", {})) != set(fresh.get("group_ratios", {})):
            problems.append(f"{name}: group ratios incomplete")
        else:
            for group, value in fresh["group_ratios"].items():
                if not _close(stored["group_ratios"][group], value):
                    problems.append(f"{name}: {group} ratio differs")
        for key in ("interval", "independent_interval"):
            actual, expected = stored.get(key), fresh.get(key)
            if (
                not isinstance(actual, list)
                or not isinstance(expected, list)
                or len(actual) != 2
                or len(expected) != 2
            ):
                problems.append(f"{name}: {key} missing")
            else:
                tolerance = 0.05 * max(expected[1] - expected[0], 0.0)
                if any(not _close(x, y, width=tolerance) for x, y in zip(actual, expected, strict=True)):
                    problems.append(f"{name}: {key} differs")
        third = independent_interval(results, slices, c)
        for key in ("interval", "independent_interval"):
            stored_interval = stored.get(key)
            if isinstance(stored_interval, list) and len(stored_interval) == 2:
                width = 0.05 * max(stored_interval[1] - stored_interval[0], 0.0)
                if any(not _close(a, b, width=width) for a, b in zip(stored_interval, third, strict=True)):
                    problems.append(f"{name}: {key} differs from third bootstrap")
    return problems


def main() -> int:
    failed = False
    for kind, start, count, folder in CASES:
        problems = verify(kind, start, count, folder)
        for problem in problems:
            print(f"{kind}: {problem}", file=sys.stderr)
        print(f"{kind}: {count} retained tasks, {'FAIL' if problems else 'PASS'}")
        failed |= bool(problems)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
