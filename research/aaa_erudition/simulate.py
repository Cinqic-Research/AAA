"""Training data for the Erudition Model: counterfactual returns from simulation.

For a training-split stream run with the surrogate Language Model, the
simulator pauses at decision points, copies the whole system once per
allowed action, forces that action, and runs every copy forward ``HORIZON``
steps under a continuation policy. The measured return of each branch --
mean task success minus adaptation cost -- is the target for that action's
``q`` output. Branches share every random draw with each other (common
random numbers), so return differences are caused by the action.

Every step also yields a diagnosis label from the hidden regime. These labels
and the counterfactual returns are where privileged information enters, and
only as training targets.

Inside the simulator the surrogate knows one hidden fact per request: which tank
it refers to, so that an unknown alias is guessed correctly one time in four,
as a model guessing among four tanks would. That shapes simulated outcomes,
just as the true world shapes real ones. It never enters the controller's
inputs directly. In particular, the surrogate's confidence depends only on
whether it acts or abstains, as the real model's does.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Diagnosis
from .controllers import Heuristic
from .lifecycle import ACTION_NAMES, CONDITIONS, Controller, ControllerView, Decision, System
from .store import MemoryStore
from .surrogate import SurrogateLM
from .toolshift import Episode, Stream, make_stream

HORIZON = 25
POINTS_PER_STREAM = 10
CALL_COST = 0.002
CANDIDATE_COST = 0.01
TRAIN_CONDITIONS = ("lm_only", "wm_only", "joint")
PROBE = [(op, n) for op in ("fill", "drain") for n in range(1, 7)]


class Switch:
    """A controller that forces one action at one step and otherwise defers."""

    def __init__(self, continuation: Controller) -> None:
        self.continuation = continuation
        self.forced: tuple[int, str] | None = None
        self.name = continuation.name

    def decide(self, view: ControllerView) -> Decision:
        if self.forced is not None and view.step == self.forced[0]:
            return Decision(self.forced[1], {}, "forced")
        return self.continuation.decide(view)


def register_truth(surrogate: SurrogateLM, stream: Stream) -> None:
    for step in range(len(stream.schedule)):
        episode = Episode(stream, step)
        label = episode.label
        surrogate.register(episode.observation.request, episode.state, label.entity)


def diagnosis_label(system: System, stream: Stream, step: int, onset: dict[str, int]) -> int:
    """The hidden regime at ``step`` given what the system has learned so far."""

    regime = stream.schedule[step]
    alias_notes = {n.key: n.values["tank"] for n in system.lm.notes if n.kind == "alias"}
    alias_deficit = regime.alias_probability > 0 and any(
        alias_notes.get(alias) != entity for entity, alias in stream.aliases.items()
    )
    context = system.wm.context()
    misses = sum(
        round(context.predict(op, n, 50, 0, 99)[0]) != regime.dynamics.apply(op, 50, n, 0, 99)
        for op, n in PROBE
    )
    wm_deficit = misses >= 2
    for name, active in (("alias", alias_deficit), ("wm", wm_deficit)):
        if active and name not in onset:
            onset[name] = step
        if not active:
            onset.pop(name, None)
    if regime.corruption > 0:
        label = Diagnosis.FEEDBACK_UNRELIABLE
    elif regime.outage > 0:
        label = Diagnosis.TOOL_FAILURE
    elif "glitch" in regime.events:
        label = Diagnosis.TRANSIENT
    elif alias_deficit or wm_deficit:
        if min(onset.values()) > step - 3:
            label = Diagnosis.INSUFFICIENT_EVIDENCE
        elif alias_deficit and wm_deficit:
            label = Diagnosis.JOINT_DEFICIENCY
        elif alias_deficit:
            label = Diagnosis.LM_DEFICIENCY
        else:
            label = Diagnosis.WM_DEFICIENCY
    else:
        label = Diagnosis.STABLE
    return list(Diagnosis).index(label)


def _decision_points(stream: Stream, g: np.random.Generator) -> list[int]:
    onsets = [t for t in range(1, len(stream.schedule)) if stream.schedule[t] != stream.schedule[t - 1]]
    near = sorted(
        {min(len(stream.schedule) - HORIZON - 1, t + int(k)) for t in onsets for k in g.integers(0, 14, 2)}
    )
    random = [int(t) for t in g.integers(8, len(stream.schedule) - HORIZON - 1, POINTS_PER_STREAM)]
    points = sorted(set(near + random))
    return [t for t in points if 4 <= t < len(stream.schedule) - HORIZON][: POINTS_PER_STREAM + 6]


def _branch_return(system: System, start: int) -> float:
    for step in range(start, start + HORIZON + 1):
        system.run_episode(step)
    window = system.scores[start + 1 : start + HORIZON + 1]
    success = float(np.mean([s["success"] for s in window]))
    return success


def simulate_stream(index: int, continuation: str, model: str | None, seed: int) -> dict[str, Any]:
    g = np.random.default_rng([seed, index])
    stream = make_stream("train", index)
    condition = TRAIN_CONDITIONS[int(g.integers(len(TRAIN_CONDITIONS)))]
    surrogate = SurrogateLM()
    register_truth(surrogate, stream)
    policy = _policy(continuation, model)
    switch = Switch(policy)
    system = System(stream, surrogate, switch, condition, MemoryStore())
    allowed = [a for a in ACTION_NAMES if a in CONDITIONS[condition]]
    points = set(_decision_points(stream, g))
    examples = []
    diagnoses = []
    onset: dict[str, int] = {}
    for step in range(len(stream.schedule)):
        if step in points:
            returns = {}
            for action in allowed:
                branch = copy.deepcopy(system)
                assert isinstance(branch.controller, Switch)
                branch.controller.forced = (step, action)
                cost_before = branch.cost["lm_gate"] + branch.cost["lm_extract"]
                cand_before = branch.cost["lm_candidates"] + branch.cost["wm_candidates"]
                value = _branch_return(branch, step)
                calls = branch.cost["lm_gate"] + branch.cost["lm_extract"] - cost_before
                cands = branch.cost["lm_candidates"] + branch.cost["wm_candidates"] - cand_before
                returns[action] = value - CALL_COST * calls - CANDIDATE_COST * cands
            system.run_episode(step)
            examples.append(
                {
                    "step": step,
                    "features": system.steps[-1].features.tolist(),
                    "history": len(system.steps),
                    "returns": returns,
                }
            )
        else:
            system.run_episode(step)
        diagnoses.append(diagnosis_label(system, stream, step, onset))
    features = np.stack([s.features for s in system.steps])
    return {
        "index": index,
        "family": stream.family,
        "condition": condition,
        "continuation": continuation,
        "features": features.astype(np.float16).tolist(),
        "diagnoses": diagnoses,
        "examples": [{k: v for k, v in e.items() if k != "features"} for e in examples],
        "success": float(np.mean([s["success"] for s in system.scores])),
    }


def _policy(name: str, model: str | None) -> Controller:
    if name == "heuristic":
        return Heuristic()
    if name == "erudition":
        import torch

        from .erudition import EruditionController

        torch.set_num_threads(1)
        assert model is not None
        return EruditionController.load(Path(model))
    raise ValueError(name)


def _work(job: tuple[int, str, str | None, int]) -> dict[str, Any]:
    return simulate_stream(*job)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.simulate")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--continuation", choices=("heuristic", "erudition"), default="heuristic")
    parser.add_argument("--model", type=str)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    jobs = [(i, args.continuation, args.model, args.seed) for i in range(args.start, args.start + args.count)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    done = 0
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool, args.output.open("w") as out:
        # Ordered, so the output file is byte-reproducible from the command line.
        for result in pool.imap(_work, jobs, chunksize=1):
            out.write(json.dumps(result, separators=(",", ":")) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} streams in {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
