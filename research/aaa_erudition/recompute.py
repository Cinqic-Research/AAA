"""Independent recomputation of a retained run.

Trusts nothing the run stored about its own results. For every step it:
- rebuilds the episode from the stream identity;
- checks that the recorded observation is the one the environment produces;
- re-executes the recorded action and requires the identical outcome and
  feedback;
- re-derives success.

Every evaluator field the metrics read (events, regime, alias use and the
deficiency flags) is re-derived from the stream and from the state payloads
the run embeds. Each payload is checked against its content address. The flag
code here is written separately from the scorer in ``lifecycle`` and does not
import it.

The lineage is checked too:
- heads move only through accepted and rollback records whose parents chain;
- every episode ran on the state the lineage says was current;
- no arm adapted a component outside its menu;
- the final states match.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

from .contracts import Action, EpisodeRecord, sha256_json
from .language import AdapterState
from .toolshift import Dynamics, Episode, Stream, make_stream, manual_dynamics
from .world import WorldState

MENU_COMPONENTS = {"frozen": set(), "lm_only": {"lm"}, "wm_only": {"wm"}, "joint": {"lm", "wm"}}
GRID = [(op, n) for op in ("fill", "drain") for n in (1, 2, 3, 4, 5, 6)]


class RecomputeError(RuntimeError):
    pass


def _payload(run: dict[str, Any], state_id: str) -> dict[str, Any]:
    payload = run["states"].get(state_id)
    if payload is None:
        raise RecomputeError(f"state {state_id} is referenced but not embedded")
    kind, _, digest = state_id.partition(":")
    if sha256_json({"kind": kind, "payload": payload}) != digest:
        raise RecomputeError(f"embedded state {state_id} does not match its content address")
    embedded: dict[str, Any] = payload
    return embedded


def _persistent(stream: Stream, step: int) -> Dynamics:
    t = step
    while t > 0 and "glitch" in stream.schedule[t].events:
        t -= 1
    return stream.schedule[t].dynamics


def _flags(stream: Stream, step: int, lm: AdapterState, wm: WorldState) -> dict[str, Any]:
    regime = stream.schedule[step]
    truth = _persistent(stream, step)
    meant = {alias: tank for tank, alias in stream.aliases.items()}
    notes = {n.key: n.values["tank"] for n in lm.notes if n.kind == "alias"}
    manual = manual_dynamics()
    stated = {"fill": (manual.fill_rate, manual.fill_bonus), "drain": (manual.drain_rate, manual.drain_fee)}
    seen: dict[str, list[tuple[int, int, int]]] = {"fill": [], "drain": []}
    for note in lm.notes:
        if note.kind == "dynamics":
            stated[note.key] = (int(note.values["rate"]), int(note.values["offset"]))
        elif note.kind == "observation":
            v = note.values
            seen[str(v["op"])].append((int(v["n"]), int(v["before"]), int(v["after"])))
    belief = Dynamics(stated["fill"][0], stated["fill"][1], stated["drain"][0], stated["drain"][1])
    lm_wrong = 0
    wm_wrong = 0
    for op, n in GRID:
        right = truth.apply(op, 50, n, 0, 99)
        informed = belief.apply(op, 50, n, 0, 99) == right or (
            len(seen[op]) >= 2 and all(truth.apply(op, b, m, 0, 99) == a for m, b, a in seen[op])
        )
        lm_wrong += not informed
        mean, _ = wm.context().predict(op, n, 50, 0, 99)
        wm_wrong += round(mean) != right
    return {
        "alias_deficit": regime.alias_probability > 0 and any(notes.get(a) != t for a, t in meant.items()),
        "wrong_alias_notes": sum(meant.get(phrase) != tank for phrase, tank in notes.items()),
        "lm_dynamics_deficit": lm_wrong >= 2,
        "wm_deficit": wm_wrong >= 2,
    }


def recompute(run: dict[str, Any], stream: Stream | None = None) -> dict[str, Any]:
    stream = stream or make_stream(run["split"], run["index"])
    if stream.stream_id != run["stream"] or stream.family != run["family"]:
        raise RecomputeError("the run's stream identity does not rebuild")
    if len(run["records"]) != len(stream.schedule) or len(run["scores"]) != len(stream.schedule):
        raise RecomputeError("the run does not cover its stream")
    allowed = MENU_COMPONENTS.get(run["condition"])
    if allowed is None:
        raise RecomputeError(f"unknown condition {run['condition']!r}")
    heads: dict[str, str] = dict(run["initial_states"])
    moves: dict[int, list[dict[str, Any]]] = {}
    for record in run["lineage"]:
        component = record["component"]
        if record["kind"] == "rejected":
            if record["result_state"] != record["parent_state"]:
                raise RecomputeError("a rejection moved a state")
            continue
        if component not in allowed:
            raise RecomputeError(f"a {run['condition']} run changed the {component} component")
        if record["kind"] == "accepted" and (record.get("evaluation") or {}).get("verdict") != "ACCEPT":
            raise RecomputeError("an accepted record lacks an ACCEPT evaluation")
        moves.setdefault(record["step"], []).append(record)
    successes = []
    for step, payload in enumerate(run["records"]):
        record = EpisodeRecord.from_dict(payload)
        if (record.lm_state, record.wm_state) != (heads["lm"], heads["wm"]):
            raise RecomputeError(f"step {step}: the episode did not run on the lineage's current states")
        episode = Episode(stream, step)
        if record.observation != episode.observation:
            raise RecomputeError(f"step {step}: recorded observation differs from the environment")
        action = Action.from_dict(record.action.to_dict())
        if action.op == "invalid":
            if record.outcome.status != "error" or record.outcome.state_after != episode.state:
                raise RecomputeError(f"step {step}: an invalid proposal changed the workspace")
            outcome = record.outcome
        else:
            outcome = episode.execute(action)
            if outcome != record.outcome:
                raise RecomputeError(
                    f"step {step}: re-executing the recorded action gives a different outcome"
                )
        if episode.feedback(action, outcome) != record.feedback:
            raise RecomputeError(f"step {step}: recorded feedback differs from the environment's")
        final = record.proposals[-1]
        executed = (action.op, action.entity, action.amount)
        if final.kind == "act" and executed != (final.op, final.entity, final.amount):
            raise RecomputeError(f"step {step}: the host executed something other than the final proposal")
        if final.kind != "act" and action.op != final.kind:
            raise RecomputeError(f"step {step}: the executed action does not match a non-act proposal")
        stored = run["scores"][step]
        label = episode.label
        derived = {
            "success": episode.success(action, outcome),
            "events": list(label.regime.events),
            "language": label.regime.language,
            "dynamics": label.regime.dynamics.key,
            "alias_used": label.alias_used,
            "feasible": label.feasible,
            **_flags(
                stream,
                step,
                AdapterState.from_payload(_payload(run, heads["lm"])),
                WorldState.from_payload(_payload(run, heads["wm"])),
            ),
        }
        for key, value in derived.items():
            if stored.get(key) != value:
                raise RecomputeError(f"step {step}: stored {key} {stored.get(key)!r} recomputes to {value!r}")
        successes.append(derived["success"])
        for move in moves.get(step, []):
            if heads[move["component"]] != move["parent_state"]:
                raise RecomputeError("lineage parents do not chain")
            _payload(run, move["result_state"])
            heads[move["component"]] = move["result_state"]
    if heads != run["final_states"]:
        raise RecomputeError("final states do not match the lineage")
    post = successes[30:]
    return {
        "stream": run["stream"],
        "condition": run["condition"],
        "controller": run["controller"],
        "steps": len(successes),
        "failure": 1.0 - sum(post) / len(post),
        "accepted": sum(r["kind"] == "accepted" for r in run["lineage"]),
    }


def load_run(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        run: dict[str, Any] = json.loads(stream.read())
    return run


def confirmation_stream(run: dict[str, Any], freeze_path: Path) -> Stream:
    """Rebuild a confirmation stream for recomputation, only under its committed freeze."""

    from .identity import committed_freeze

    freeze = committed_freeze(freeze_path)
    if run["index"] not in freeze["confirmation"]["indices"]:
        raise RecomputeError("the run's stream is not declared in the freeze")
    return make_stream("confirmation", run["index"], admitted=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.recompute")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--freeze", type=Path, help="the committed freeze, for confirmation runs")
    args = parser.parse_args(argv)
    failures = 0
    for path in args.runs:
        try:
            run = load_run(path)
            stream = None
            if run["split"] == "confirmation":
                if args.freeze is None:
                    raise RecomputeError("confirmation runs recompute only under --freeze")
                stream = confirmation_stream(run, args.freeze)
            print(json.dumps(recompute(run, stream)))
        except RecomputeError as error:
            failures += 1
            print(f"RECOMPUTE FAILED {path.name}: {error}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
