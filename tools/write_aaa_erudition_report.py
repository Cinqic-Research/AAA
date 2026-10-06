#!/usr/bin/env python3
"""Generate ``docs/juniper1/development_report.md`` from retained ``aaa.erudition.v0`` evidence.

Every number in the report comes from a committed evidence file. ``--check``
fails if the committed report differs from what the evidence produces.

    python tools/write_aaa_erudition_report.py          # write
    python tools/write_aaa_erudition_report.py --check  # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/aaa_erudition_v0"
REPORT = ROOT / "docs/juniper1/development_report.md"
NARRATIVE = ROOT / "docs/juniper1/development_report.narrative.json"

METRICS = (
    ("failure", "failure"),
    ("shifted_failure", "shifted"),
    ("retention_failure", "retention"),
    ("false_adaptation", "false adapt."),
    ("misattribution", "misattrib."),
    ("missed_adaptation", "missed"),
    ("unresolved_deficiency", "unresolved"),
    ("poisoned", "poisoned"),
    ("unhelpful_adaptation", "unhelpful"),
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def arm_table(descriptive: dict[str, Any], arms: list[str] | None = None) -> str:
    header = (
        "| Arm | Streams | "
        + " | ".join(label for _, label in METRICS)
        + " | Accepted | Rejected | Rollbacks | Gate+extract calls/stream |"
    )
    rule = "|" + "---|" * (len(METRICS) + 6)
    rows = [header, rule]
    for arm in arms or sorted(descriptive):
        if arm not in descriptive:
            continue
        e = descriptive[arm]
        calls = e["lm_calls_per_stream"]
        rows.append(
            f"| `{arm}` | {e['streams']} | "
            + " | ".join(_fmt(e[key]) for key, _ in METRICS)
            + f" | {e['accepted']} | {e['rejected']} | {e['rollbacks']} | {calls['lm_gate'] + calls['lm_extract']:.1f} |"
        )
    return "\n".join(rows)


def family_table(descriptive: dict[str, Any], arms: list[str]) -> str:
    families = sorted(
        {f for arm in arms if arm in descriptive for f in descriptive[arm]["failure_by_family"]}
    )
    rows = [
        "| Family | " + " | ".join(f"`{a}`" for a in arms if a in descriptive) + " |",
        "|---|" + "---|" * sum(a in descriptive for a in arms),
    ]
    for family in families:
        rows.append(
            f"| {family} | "
            + " | ".join(
                _fmt(descriptive[a]["failure_by_family"].get(family)) for a in arms if a in descriptive
            )
            + " |"
        )
    return "\n".join(rows)


def decision_table(decisions: dict[str, Any]) -> str:
    rows = ["| Contract | Criterion | Point | 95% interval | Status | Verdict |", "|---|---|---|---|---|---|"]
    for name, decision in decisions.items():
        primary = decision.get("primary") or {"criteria": []}
        for c in primary["criteria"]:
            interval = "n/a" if c["lower"] is None else f"[{c['lower']:.3f}, {c['upper']:.3f}]"
            rows.append(
                f"| {name} | {c['criterion']} | {c['point']:.3f} | {interval} | {c['status']} | {decision['verdict']} |"
            )
        if not primary["criteria"]:
            rows.append(f"| {name} | — | — | — | — | {decision['verdict']} |")
    return "\n".join(rows)


def render() -> str:
    narrative = _load(NARRATIVE)
    parts = [narrative["intro"]]
    char = _load(EVIDENCE / "characterization/characterization.json")["summary"]
    consult = _load(EVIDENCE / "characterization/characterization_consult.json")["summary"]
    cases = {**char["cases"], **consult["cases"]}
    rows = [
        "| Situation | n | Acted | Abstained | Correct tank | Reached target | After World Model help |",
        "|---|---|---|---|---|---|---|",
    ]
    for case, c in cases.items():
        helped = ""
        if c.get("revision_n"):
            helped = f"{c['revision_reached']}/{c['revision_n']} after a consequence table"
        if c.get("consult_n"):
            helped = f"{c['consult_reached']}/{c['consult_n']} after a consultation"
        rows.append(
            f"| `{case}` | {c['n']} | {c['act']} | {c['abstain']} | {c['entity']} | {c['reached']} | {helped} |"
        )
    parts.append(narrative["characterization"] + "\n\n" + "\n".join(rows))
    parts.append(
        "Correction extraction, share correct by feedback form: "
        + ", ".join(f"form {k}: {v:.2f}" for k, v in char["extraction"].items())
        + "."
    )
    wm = _load(EVIDENCE / "wm_diagnostic.json")
    rows = [
        "| Dynamics / arm | Error on B after 2 / 6 / 10 | ±2 sd coverage after 10 | Error on A when it returns |",
        "|---|---|---|---|",
    ]
    for key, m in wm.items():
        if isinstance(m, dict):
            rows.append(
                f"| `{key}` | {m['error_B_after_2']:.3f} / {m['error_B_after_6']:.3f} / {m['error_B_after_10']:.3f} | {m['coverage_B_after_10']:.3f} | {m['error_A_on_return']:.3f} |"
            )
    parts.append(
        narrative["world_model"]
        + f"\n\n{wm['trials']} trials per row, seed {wm['seed']}.\n\n"
        + "\n".join(rows)
    )
    sweep_dir = EVIDENCE / "sweep"
    rows = [
        "| Model | Trainable parameters | Validation regret | Margin | `joint` failure | `lm_only` failure | `wm_only` failure | Gate+extract calls/stream (joint) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    baselines = None
    for meta_path in sorted(sweep_dir.glob("*.model.json")):
        name = meta_path.name.removesuffix(".model.json")
        if not (sweep_dir / f"{name}.evaluation.json").exists():
            continue  # an intermediate continuation policy, not a sweep member
        meta = _load(meta_path)
        evaluation = _load(sweep_dir / f"{name}.evaluation.json")["descriptive"]
        baselines = evaluation
        joint = evaluation["joint/erudition"]
        rows.append(
            f"| `{name}` | {meta['trainable_parameters']:,} | {min(meta['validation']['regret'].values()):.4f} | {meta['margin']} | "
            f"{joint['failure']:.3f} | {evaluation['lm_only/erudition']['failure']:.3f} | {evaluation['wm_only/erudition']['failure']:.3f} | "
            f"{joint['lm_calls_per_stream']['lm_gate'] + joint['lm_calls_per_stream']['lm_extract']:.1f} |"
        )
    parts.append(narrative["sweep"] + "\n\n" + "\n".join(rows))
    if baselines is not None:
        controls = [
            "frozen/never",
            "joint/always",
            "joint/heuristic",
            "lm_only/heuristic",
            "wm_only/heuristic",
        ]
        parts.append(narrative["sweep_controls"] + "\n\n" + arm_table(baselines, controls))
    selected = _load(EVIDENCE / "model/erudition.json")
    parts.append(
        narrative["selected"]
        + f"\n\nWeights SHA-256 `{selected['weights_sha256']}`; {selected['trainable_parameters']:,} trainable parameters; "
        f"decision margin {selected['margin']}; trained {selected['epochs']} epochs on {selected['streams']} simulated streams "
        f"({selected['train_examples']:,} examples) in {selected['train_seconds']:.0f} s on {selected['gpu'] or selected['device']}."
    )
    for stage in ("development", "attack"):
        path = EVIDENCE / stage / "evaluation.json"
        if path.exists():
            descriptive = _load(path)["descriptive"]
            arms = sorted(descriptive)
            parts.append(
                narrative[stage]
                + "\n\n"
                + arm_table(descriptive)
                + "\n\nFailure by family:\n\n"
                + family_table(descriptive, arms)
            )
    confirmation = EVIDENCE / "confirmation/evaluation.json"
    if confirmation.exists():
        evaluation = _load(confirmation)
        arms = sorted(evaluation["descriptive"])
        parts.append(
            narrative["confirmation"]
            + "\n\n"
            + decision_table(evaluation["decisions"])
            + "\n\n"
            + arm_table(evaluation["descriptive"])
            + "\n\nFailure by family:\n\n"
            + family_table(evaluation["descriptive"], arms)
        )
    else:
        parts.append("## Confirmation\n\nNot executed.")
    parts.append(narrative["failures"])
    return "\n\n".join(parts).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if REPORT.read_text("utf-8") != text:
            print("development_report.md does not match the evidence; regenerate it", file=sys.stderr)
            return 1
        print("development report matches the evidence")
        return 0
    REPORT.write_text(text, "utf-8")
    print(f"wrote {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
