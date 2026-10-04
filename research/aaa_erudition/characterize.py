"""Characterize Juniper LM 1.1 (gpt-oss-20b) behaviour in ToolShift situations.

Runs the real model on controlled situations built from *training-split*
names and aliases and records how often it reads references, notes, World
Model checks and feedback correctly. The rates parameterize the simulation
surrogate the Erudition Model trains against (``surrogate.py``). Nothing
here touches development, attack or confirmation identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Observation, WMPrediction, canonical_json
from .language import (
    ACT_TOOLS,
    AdapterState,
    Backend,
    CachedBackend,
    CallCache,
    LlamaServer,
    Note,
    act_messages,
    chat_request,
    consultation_messages,
    extract_correction,
    parse_proposal,
    revision_messages,
)
from .toolshift import Dynamics, load_spec, manual_dynamics

SCHEMA = "aaa.erudition.lm_characterization.v1"


CASES = (
    "name",
    "alias_unknown",
    "alias_noted",
    "decoy",
    "dyn_note",
    "dyn_observed",
    "revision",
    "alias_poisoned",
)


def _situations(count: int, seed: int, cases: tuple[str, ...] = CASES) -> list[dict[str, Any]]:
    spec = load_spec()
    names = spec["splits"]["train"]["names"]
    aliases = spec["splits"]["train"]["aliases"]
    g = np.random.default_rng(seed)
    rows = []
    shifted_pool = [Dynamics(5, 1, 3, 0), Dynamics(4, 0, 1, 2), Dynamics(6, 2, 4, 1), Dynamics(2, 1, 3, 2)]
    for case in cases:
        for i in range(count):
            picked = [str(x) for x in g.permutation(names)]
            workspace, decoy = picked[:4], picked[4]
            state = {name: int(g.integers(10, 90)) for name in workspace}
            alias_of = dict(zip(workspace, (str(a) for a in g.permutation(aliases)[:4]), strict=True))
            entity = str(g.choice(workspace))
            dynamics = (
                manual_dynamics()
                if case in ("name", "alias_unknown", "alias_noted", "decoy", "alias_poisoned")
                else shifted_pool[i % 4]
            )
            op = str(g.choice(["fill", "drain"]))
            for _ in range(50):
                n = int(g.integers(1, 7))
                value = dynamics.apply(op, state[entity], n, 0, 99)
                if 0 < value < 99 and value != state[entity]:
                    break
                op = "drain" if op == "fill" else "fill"
            reference = (
                alias_of[entity] if case.startswith("alias") else (decoy if case == "decoy" else entity)
            )
            template = str(g.choice(spec["templates"]))
            request = template.format(ref=reference, value=value)
            request = request[0].upper() + request[1:]
            notes: list[Note] = []
            if case == "alias_noted":
                notes.append(Note("alias", alias_of[entity], {"tank": entity}, "feedback", 2, ()))
            if case == "alias_poisoned":
                wrong = next(w for w in workspace if w != entity)
                notes.append(Note("alias", alias_of[entity], {"tank": wrong}, "feedback", 2, ()))
            if case == "dyn_note":
                notes.append(
                    Note(
                        "dynamics",
                        "fill",
                        {"rate": dynamics.fill_rate, "offset": dynamics.fill_bonus},
                        "wm",
                        8,
                        (),
                    )
                )
                notes.append(
                    Note(
                        "dynamics",
                        "drain",
                        {"rate": dynamics.drain_rate, "offset": dynamics.drain_fee},
                        "wm",
                        8,
                        (),
                    )
                )
            if case == "dyn_observed":
                for k in range(6):
                    o = ("fill", "drain")[k % 2]
                    t = workspace[k % 4]
                    m = int(g.integers(1, 6))
                    before = int(g.integers(20, 80))
                    after = dynamics.apply(o, before, m, 0, 99)
                    notes.append(
                        Note(
                            "observation",
                            f"obs{k}",
                            {"op": o, "tank": t, "n": m, "before": before, "after": after},
                            "experience",
                            1,
                            (),
                        )
                    )
            rows.append(
                {
                    "case": case,
                    "workspace": workspace,
                    "state": state,
                    "entity": None if case == "decoy" else entity,
                    "value": value,
                    "request": request,
                    "notes": [note.to_dict() for note in notes],
                    "dynamics": [
                        dynamics.fill_rate,
                        dynamics.fill_bonus,
                        dynamics.drain_rate,
                        dynamics.drain_fee,
                    ],
                    "reference": reference,
                }
            )
    return rows


def _score(
    row: dict[str, Any], kind: str, op: str | None, entity: str | None, amount: int | None
) -> dict[str, Any]:
    d = Dynamics(*row["dynamics"])
    correct_entity = entity == row["entity"]
    reached = None
    if kind == "act" and correct_entity and op and amount is not None and entity is not None:
        reached = d.apply(op, row["state"][entity], amount, 0, 99) == row["value"]
    return {"kind": kind, "entity_correct": correct_entity, "value_reached": reached}


def run(
    backend: Backend,
    count: int,
    seed: int,
    cases: tuple[str, ...] = CASES,
    extraction_count: int | None = None,
) -> dict[str, Any]:
    results = []
    for row in _situations(count, seed, cases):
        obs = Observation(
            episode_id=f"characterize/{row['case']}",
            step=0,
            workspace="train",
            state=row["state"],
            lower=0,
            upper=99,
            request=row["request"],
        )
        adapter = AdapterState(tuple(Note.from_dict(n) for n in row["notes"]))
        messages = act_messages(obs, adapter)
        response = backend.chat(chat_request(messages, ACT_TOOLS))
        proposal = parse_proposal(response, 0, backend.backend_id, adapter.digest(), tuple(row["state"]))
        manual = manual_dynamics()
        reachable = row["entity"] is not None and any(
            manual.apply(op, row["state"][row["entity"]], n, 0, 99) == row["value"]
            for op in ("fill", "drain")
            for n in range(1, 10)
        )
        entry = {
            "case": row["case"],
            "manual_reachable": reachable,
            "first": _score(row, proposal.kind, proposal.op, proposal.entity, proposal.amount),
            "confidence": proposal.confidence,
        }
        if row["case"] == "revision" and proposal.kind == "act" and proposal.entity == row["entity"]:
            d = Dynamics(*row["dynamics"])
            assert proposal.op and proposal.amount is not None and proposal.entity
            before = row["state"][proposal.entity]
            predicted = d.apply(proposal.op, before, proposal.amount, 0, 99)
            entry["revision_triggered"] = predicted != proposal.expected
            if predicted != proposal.expected:
                table = [(n, d.apply(proposal.op, before, n, 0, 99)) for n in range(1, 10)]
                pred = WMPrediction(
                    op=proposal.op,
                    entity=proposal.entity,
                    amount=proposal.amount,
                    before=before,
                    mean=float(predicted),
                    sd=0.3,
                    context="c",
                    wm_state="s",
                )
                second = backend.chat(
                    chat_request(revision_messages(messages, response, pred, table), ACT_TOOLS)
                )
                revised = parse_proposal(second, 1, backend.backend_id, adapter.digest(), tuple(row["state"]))
                entry["revised"] = _score(row, revised.kind, revised.op, revised.entity, revised.amount)
                entry["table_reaches_value"] = any(after == row["value"] for _, after in table)
        if row["case"] == "consult" and proposal.kind == "abstain":
            d = Dynamics(*row["dynamics"])
            behaviour = {"fill": (d.fill_rate, d.fill_bonus), "drain": (d.drain_rate, d.drain_fee)}
            second = backend.chat(
                chat_request(consultation_messages(messages, response, behaviour), ACT_TOOLS)
            )
            consulted = parse_proposal(second, 1, backend.backend_id, adapter.digest(), tuple(row["state"]))
            entry["consulted"] = _score(row, consulted.kind, consulted.op, consulted.entity, consulted.amount)
        results.append(entry)
    g = np.random.default_rng(seed + 1)
    names = load_spec()["splits"]["train"]["names"]
    aliases = load_spec()["splits"]["train"]["aliases"]
    extraction = []
    for i in range(count if extraction_count is None else extraction_count):
        workspace = [str(x) for x in g.permutation(names)[:4]]
        alias = str(g.choice(aliases))
        tank, acted = workspace[0], workspace[1]
        request = f"Bring {alias} to {int(g.integers(10, 90))}."
        forms = [
            (f"By '{alias}' I meant {tank}, not {acted}.", tank),
            (f"By '{alias}' I meant {tank}.", tank),
            ("Thanks, that's right.", None),
            (f"{acted} should be 40, but it is 37.", None),
        ]
        text, truth = forms[i % 4]
        got = extract_correction(backend, text, request, tuple(workspace))
        extraction.append(
            {
                "form": i % 4,
                "correct": (got[1] if got else None) == truth and (got is None or got[0] == alias),
            }
        )
    return {
        "schema": SCHEMA,
        "backend": backend.backend_id,
        "count": count,
        "seed": seed,
        "acts": results,
        "extraction": extraction,
    }


def summarize(evidence: dict[str, Any]) -> dict[str, Any]:
    by_case: dict[str, dict[str, Any]] = {}
    for entry in evidence["acts"]:
        case = by_case.setdefault(
            entry["case"],
            {
                "n": 0,
                "act": 0,
                "abstain": 0,
                "invalid": 0,
                "entity": 0,
                "reached": 0,
                "revision_n": 0,
                "revision_reached": 0,
                "revision_abstain": 0,
                "consult_n": 0,
                "consult_reached": 0,
                "consult_abstain": 0,
                "confidence": [],
            },
        )
        case["n"] += 1
        case[entry["first"]["kind"]] += 1
        case["entity"] += int(entry["first"]["entity_correct"])
        case["reached"] += int(bool(entry["first"]["value_reached"]))
        if entry["confidence"] is not None:
            case["confidence"].append(entry["confidence"])
        if "revised" in entry:
            case["revision_n"] += 1
            case["revision_reached"] += int(bool(entry["revised"]["value_reached"]))
            case["revision_abstain"] += int(entry["revised"]["kind"] == "abstain")
        if "consulted" in entry:
            case["consult_n"] += 1
            case["consult_reached"] += int(bool(entry["consulted"]["value_reached"]))
            case["consult_abstain"] += int(entry["consulted"]["kind"] == "abstain")
    for case in by_case.values():
        conf = case.pop("confidence")
        case["confidence_mean"] = float(np.mean(conf)) if conf else None
    forms: dict[int, list[bool]] = {}
    for entry in evidence["extraction"]:
        forms.setdefault(entry["form"], []).append(entry["correct"])
    return {"cases": by_case, "extraction": {str(k): sum(v) / len(v) for k, v in sorted(forms.items())}}


def fit_surrogate(evidence: list[dict[str, Any]], defaults: dict[str, Any]) -> dict[str, Any]:
    """Map measured rates to surrogate parameters. Anything not measured keeps its default and is listed."""

    acts = [e for ev in evidence for e in ev["acts"]]
    extraction = [e for ev in evidence for e in ev["extraction"]]
    cases: dict[str, list[dict[str, Any]]] = {}
    for entry in acts:
        cases.setdefault(entry["case"], []).append(entry)

    def share(rows: list[Any], test: Any) -> float | None:
        # Posterior mean under a uniform prior: 24 successes in 24 trials is not certainty.
        return (sum(1 for r in rows if test(r)) + 1) / (len(rows) + 2) if rows else None

    rates = dict(defaults)
    measured: dict[str, float | None] = {
        "invalid": share(acts, lambda e: e["first"]["kind"] == "invalid"),
        "name_misread": share(
            [e for e in cases.get("name", []) if e["first"]["kind"] == "act"],
            lambda e: not e["first"]["entity_correct"],
        ),
        "alias_note_follow": share(cases.get("alias_noted", []), lambda e: e["first"]["entity_correct"]),
        "alias_unknown_abstain": share(
            cases.get("alias_unknown", []), lambda e: e["first"]["kind"] == "abstain"
        ),
        "decoy_abstain": share(cases.get("decoy", []), lambda e: e["first"]["kind"] == "abstain"),
        "dynamics_note_use": share(cases.get("dyn_note", []), lambda e: bool(e["first"]["value_reached"])),
        "observation_infer": share(
            cases.get("dyn_observed", []), lambda e: bool(e["first"]["value_reached"])
        ),
        "arithmetic_slip": share(
            [e for e in cases.get("name", []) if e["first"]["entity_correct"]],
            lambda e: not e["first"]["value_reached"],
        ),
        "revision_follow": share(
            [e for e in cases.get("revision", []) if "revised" in e and e.get("table_reaches_value")],
            lambda e: bool(e["revised"]["value_reached"]),
        ),
        "unreachable_abstain": share(
            [e for c in ("revision", "consult") for e in cases.get(c, []) if not e["manual_reachable"]],
            lambda e: e["first"]["kind"] == "abstain",
        ),
        "revision_abstain": share(
            [e for e in cases.get("revision", []) if "revised" in e and not e["revised"]["value_reached"]],
            lambda e: e["revised"]["kind"] == "abstain",
        ),
        "consult_follow": share(
            [e for e in cases.get("consult", []) if "consulted" in e],
            lambda e: bool(e["consulted"]["value_reached"]),
        ),
        "extract_correct": share([e for e in extraction if e["form"] in (0, 1)], lambda e: e["correct"]),
        "extract_spurious": share([e for e in extraction if e["form"] in (2, 3)], lambda e: not e["correct"]),
    }
    unmeasured = []
    for key, value in measured.items():
        if value is None:
            unmeasured.append(key)
        else:
            rates[key] = round(value, 4)
    acted = [e["confidence"] for e in acts if e["confidence"] is not None and e["first"]["kind"] == "act"]
    abstained = [
        e["confidence"] for e in acts if e["confidence"] is not None and e["first"]["kind"] == "abstain"
    ]
    rates.pop("confidence_correct", None)
    rates.pop("confidence_wrong", None)
    for key, values in (("confidence_act", acted), ("confidence_abstain", abstained)):
        if len(values) > 1:
            rates[key] = [round(float(np.mean(values)), 5), round(float(np.std(values)), 5)]
        else:
            unmeasured.append(key)
    return {
        "rates": rates,
        "unmeasured": sorted(
            set(unmeasured) | (set(rates) - set(measured) - {"confidence_act", "confidence_abstain"})
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.characterize")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--url", default="http://127.0.0.1:18741")
    parser.add_argument("--key", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--extraction-count", type=int)
    parser.add_argument(
        "--fit", type=Path, nargs="+", help="fit the surrogate from these characterization files and exit"
    )
    args = parser.parse_args(argv)
    if args.fit:
        inputs = [json.loads(path.read_text("utf-8")) for path in args.fit]
        current = json.loads(args.output.read_text("utf-8")) if args.output.exists() else {"rates": {}}
        fitted = fit_surrogate(inputs, current["rates"])
        payload = {
            "schema": "aaa.erudition.lm_surrogate.v1",
            "source": [
                {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in args.fit
            ],
            "rates": fitted["rates"],
            "unmeasured_defaults": fitted["unmeasured"],
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0
    inner = None if args.replay else LlamaServer(args.url, args.key)
    backend = CachedBackend(CallCache(args.cache), inner, backend_id="replay")
    evidence = run(backend, args.count, args.seed, tuple(args.cases.split(",")), args.extraction_count)
    evidence["summary"] = summarize(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(evidence) + b"\n")
    print(json.dumps(evidence["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
