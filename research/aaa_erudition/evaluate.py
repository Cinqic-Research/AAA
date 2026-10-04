"""Aggregate retained runs into metrics, descriptive tables and promotion decisions.

``--protocol`` names a frozen protocol JSON (see ``docs/juniper1/protocol.md``)
whose contracts are adjudicated with ``aaa.promotion.paired.v1``. Without
one, only descriptive results are produced. Every run is first passed through
``recompute``; a run that does not recompute is an error, never dropped.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contracts import canonical_json
from .metrics import stream_metrics
from .promotion import Contract, Criterion, adjudicate
from .recompute import confirmation_stream, load_run, recompute

RATE_METRICS = (
    "failure",
    "shifted_failure",
    "retention_failure",
    "false_adaptation",
    "misattribution",
    "missed_adaptation",
    "unresolved_deficiency",
    "poisoned",
    "unhelpful_adaptation",
)


def arm_name(run: dict[str, Any]) -> str:
    return f"{run['condition']}/{run['controller']}"


def collect(paths: list[Path], *, freeze: Path | None = None) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        run = load_run(path)
        stream = None
        if run["split"] == "confirmation":
            if freeze is None:
                raise RuntimeError(f"{path.name}: confirmation runs are evaluated only under their freeze")
            stream = confirmation_stream(run, freeze)
        recomputed = recompute(run, stream)
        metrics = stream_metrics(run)
        if abs(recomputed["failure"] - (metrics["failure"] or 0.0)) > 1e-12:
            raise RuntimeError(f"{path.name}: recomputed failure disagrees with the metric")
        metrics["arm"] = arm_name(run)
        metrics["cost"] = run["cost"]
        rows.append(metrics)
    return rows


def declared_streams(freeze: dict[str, Any]) -> list[str]:
    from .toolshift import ENVIRONMENT

    return sorted(f"{ENVIRONMENT}/confirmation/{i:05d}" for i in freeze["confirmation"]["indices"])


def decide(freeze: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Adjudicate the freeze's contracts over every declared stream; a missing run is missing evidence."""

    streams = declared_streams(freeze)
    present = {(r["stream"], r["arm"]) for r in rows}
    missing = [(s, a) for s in streams for a in freeze["confirmation"]["arms"] if (s, a) not in present]
    if missing:
        raise RuntimeError(f"declared confirmation runs are missing: {missing[:4]}")
    evidence = primitives(rows)
    return {c.name: adjudicate(c, evidence) for c in contracts_from(freeze, streams)}


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    table: dict[str, Any] = {}
    for arm, items in sorted(by_arm.items()):
        entry: dict[str, Any] = {"streams": len(items)}
        for metric in RATE_METRICS:
            values = [r[metric] for r in items if r[metric] is not None]
            entry[metric] = round(statistics.fmean(values), 4) if values else None
        for count in ("accepted", "rejected", "rollbacks", "oscillations"):
            entry[count] = sum(r[count] for r in items)
        calls: defaultdict[str, float] = defaultdict(float)
        for r in items:
            for k, v in r["lm_calls"].items():
                calls[k] += v
        entry["lm_calls_per_stream"] = {k: round(v / len(items), 1) for k, v in sorted(calls.items())}
        families: dict[str, list[float]] = defaultdict(list)
        for r in items:
            families[r["family"]].append(r["failure"])
        entry["failure_by_family"] = {f: round(statistics.fmean(v), 4) for f, v in sorted(families.items())}
        table[arm] = entry
    return table


def primitives(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for metric in RATE_METRICS:
            if row[metric] is not None:
                out.append(
                    {
                        "metric": metric,
                        "arm": row["arm"],
                        "stream": row["stream"],
                        "value": float(row[metric]),
                    }
                )
    return out


def contracts_from(protocol: dict[str, Any], streams: list[str]) -> list[Contract]:
    out = []
    for c in protocol["contracts"]:
        criteria = tuple(Criterion(**criterion) for criterion in c["criteria"])
        out.append(Contract(c["name"], tuple(streams), criteria, int(c["seed"])))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.evaluate")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--freeze", type=Path, help="the committed freeze; required for confirmation runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = collect(args.runs, freeze=args.freeze)
    report: dict[str, Any] = {
        "schema": "aaa.erudition.evaluation.v1",
        "descriptive": describe(rows),
        "streams": rows,
    }
    if args.freeze is not None:
        from .identity import committed_freeze

        report["decisions"] = decide(committed_freeze(args.freeze), rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report["descriptive"], indent=1))
    for name, decision in report.get("decisions", {}).items():
        print(name, decision["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
