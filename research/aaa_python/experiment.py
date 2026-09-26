"""Development runs of ``aaa.python.v0``: every agent, through the causal boundary.

One run does four things, all on development, probe and training identities
(never confirmation, which v0 refuses to generate):

1. **Train.** For each declared initialization and representation, an
   ``OnlineLinear`` learner passes over the training pool through the
   :class:`~.episode.Environment` (predict, commit, reveal, learn). Baselines
   see the same training feedback.
2. **Evaluate.** Every ``initialization x stream`` cell of every family runs
   each arm on the same development tasks. From one trained state per
   initialization: ``online``, ``frozen``, ``feedback_disabled`` and
   ``reset_each_task``, plus online/frozen pairs for each alternative
   representation; and the ``uniform``, ``majority``, ``lookup`` and
   ``heuristic`` baselines, which do not adapt during evaluation.
3. **Adapt (matched design).** Per cell, one learner runs a shared
   in-distribution prefix online. Its state is cloned into four branches:
   online and frozen, each on a *changed* branch (novel program structure) and
   a *control* branch (more in-distribution programs). The estimand is the
   difference of differences of error rates, so the ordinary benefit of
   continued learning cancels.
4. **Retain.** A fixed probe bank, never trained on, is scored read-only from
   the post-prefix state and from each branch's online end state.

The run emits cell-level primitives (counts, calibration bins, per-class
recall counts) from which every summary number is recomputed
(:mod:`.recompute`), plus the full per-action records.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import PROTOCOL_VERSION, generator
from . import spec as spec_module
from .episode import Action, Environment, label_space
from .generator import Task
from .learners import Agent, Heuristic, Lookup, Majority, OnlineLinear, StateError, Uniform
from .rng import Stream, derive_seed
from .stats import balanced_accuracy, calibration, crossed_mean, grid, resolved_sign

SCHEMA = "aaa.python.v0.development.v1"
RECORD_SCHEMA = "aaa.python.v0.action_record.v1"
LEARNER_CONTROLS = ("online", "frozen", "feedback_disabled", "reset_each_task")
BASELINES = ("uniform", "majority", "lookup", "heuristic")


class RunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plan:
    initializations: int
    streams: int
    stream_length: int
    train_limit: int | None
    train_epochs: int
    prefix: int
    branch: int
    probe_bank: int
    representations: tuple[str, ...]
    default_representation: str
    adaptation_families: tuple[str, ...]
    draws: int
    quick: bool

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any], *, quick: bool = False) -> Plan:
        dev, learner = spec["development"], spec["learner"]
        plan = cls(
            initializations=dev["initializations"],
            streams=dev["streams"],
            stream_length=dev["stream_length"],
            train_limit=None,
            train_epochs=learner["train_epochs"],
            prefix=dev["adaptation"]["prefix"],
            branch=dev["adaptation"]["branch"],
            probe_bank=dev["probe_bank"],
            representations=tuple(learner["representations"]),
            default_representation=learner["default_representation"],
            adaptation_families=tuple(dev["adaptation"]["families"]),
            draws=spec["statistics"]["draws"],
            quick=False,
        )
        if quick:
            plan = replace(
                plan,
                initializations=2,
                streams=2,
                stream_length=6,
                train_limit=100,
                prefix=4,
                branch=5,
                probe_bank=10,
                draws=1000,
                quick=True,
            )
        return plan


# ------------------------------------------------------------------- tasks
def load_tasks(plan: Plan, spec: Mapping[str, Any]) -> dict[str, dict[str, list[Task]]]:
    families = list(spec["families"])
    train = {f: list(generator.pool("train", f))[: plan.train_limit] for f in families}
    # Development tasks come only from the declared pool: later splits (probe,
    # attack) are disjoint from exactly that pool and nothing beyond it.
    development = {f: list(generator.pool("development", f)) for f in families}
    probe = {f: generator.build("probe", f, range(plan.probe_bank), spec) for f in families}
    return {"train": train, "development": development, "probe": probe}


# ------------------------------------------------------------------- running
def run_task(agent: Agent, env: Environment, task: Task, *, learn: bool = True) -> dict[str, Any]:
    agent.begin_task()
    view = env.present(task)
    action = agent.act(view)
    env.commit(view, action)
    correct, feedback = env.reveal(view)
    updated = agent.learn(view, action, feedback) if learn else False
    return {
        "task_id": task.task_id,
        "family": task.family,
        "slice": task.slice,
        "answer": action.answer,
        "truth": task.answer,
        "confidence": float(action.confidence),
        "abstain": bool(action.abstain),
        "correct": bool(correct),
        "updated": bool(updated),
    }


def train(
    agent: Agent, tasks: Mapping[str, Sequence[Task]], epochs: int, order_seed: int, spec: Mapping[str, Any]
) -> None:
    pool = [task for family in sorted(tasks) for task in tasks[family]]
    env = Environment(spec)
    for epoch in range(epochs):
        for task in Stream(derive_seed(order_seed, "epoch", epoch)).shuffled(pool):
            run_task(agent, env, task)


def make_learner(spec: Mapping[str, Any], init: int, representation: str) -> OnlineLinear:
    learner_spec = spec["learner"]
    labels = {family: label_space(_label_probe(family), spec) for family in spec["families"]}
    return OnlineLinear(
        seed=derive_seed(PROTOCOL_VERSION, "init", init) % 2**63,
        representation=representation,
        dimensions=learner_spec["feature_dimensions"],
        learning_rate=learner_spec["learning_rate"],
        init_scale=learner_spec["init_scale"],
        abstain_below=learner_spec["abstain_below"],
        label_spaces=labels,
    )


def _label_probe(family: str) -> Task:
    return Task("probe", "train", family, 0, 0, "in_distribution", "none", "", candidates=("",) * 4)


def code_identity() -> str:
    """A digest of this package's Python source, so a checkpoint cannot resume under changed code (``AAA-198``)."""

    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def trained_learner(
    spec: Mapping[str, Any],
    plan: Plan,
    tasks: Mapping[str, Any],
    init: int,
    representation: str,
    checkpoints: Path | None,
) -> OnlineLinear:
    learner = make_learner(spec, init, representation)
    learner.remember_initial()
    # A plan alone does not identify the exam or the concrete training pool.
    checkpoint_identity = {
        "spec_sha256": spec_module.canonical_hash(spec),
        "train_pool_sha256": spec_module.canonical_hash(
            {family: [asdict(task) for task in tasks["train"][family]] for family in sorted(tasks["train"])}
        ),
        "code_sha256": code_identity(),
    }
    path = None if checkpoints is None else checkpoints / f"init-{init}-{representation}.json"
    if path is not None and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        # Compare JSON-normalized forms: a stored plan's tuples come back as lists.
        if saved.get("plan") != json.loads(json.dumps(asdict(plan))):
            raise RunError(f"{path}: checkpoint was produced by a different plan; refusing to resume")
        if saved.get("identity") != checkpoint_identity:
            raise RunError(f"{path}: checkpoint specification or training pool differs; refusing to resume")
        learner.load_state(saved["state"])
        if learner.state_hash() != saved["state_hash"]:
            raise RunError(f"{path}: checkpoint state does not match its recorded hash")
        return learner
    train(
        learner, tasks["train"], plan.train_epochs, derive_seed(PROTOCOL_VERSION, "train-order", init), spec
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "plan": asdict(plan),
            "identity": checkpoint_identity,
            "state": learner.state_dict(),
            "state_hash": learner.state_hash(),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    return learner


def fitted_baselines(tasks: Mapping[str, Sequence[Task]], spec: Mapping[str, Any]) -> dict[str, Agent]:
    fitted: dict[str, Agent] = {"majority": Majority(), "lookup": Lookup(), "heuristic": Heuristic()}
    env = Environment(spec)
    for agent in fitted.values():
        for family in sorted(tasks):
            for task in tasks[family]:
                run_task(agent, env, task)
    return fitted


def _arm_agent(arm: str, trained: OnlineLinear) -> tuple[Agent, bool]:
    """``(agent, feedback_enabled)`` for a learner arm, always a fresh clone of the trained state."""

    agent = trained.clone(arm)
    if arm.startswith("frozen"):
        agent.update_enabled = False
    if arm == "reset_each_task":
        agent.reset_each_task = True
    return agent, arm != "feedback_disabled"


def run_stream(
    agent: Agent, tasks: Sequence[Task], spec: Mapping[str, Any], *, feedback: bool, learn: bool
) -> list[dict[str, Any]]:
    env = Environment(spec, feedback_enabled=feedback)
    records = []
    for position, task in enumerate(tasks):
        record = run_task(agent, env, task, learn=learn)
        record["position"] = position
        records.append(record)
    return records


def adaptation_split(
    tasks: Sequence[Task], plan: Plan, stream: int, spec: Mapping[str, Any]
) -> dict[str, list[Task]]:
    held = tasks[plan.streams * plan.stream_length :]
    ind = [t for t in held if t.slice == "in_distribution"]
    novel = [t for t in held if t.slice == spec["development"]["adaptation"]["changed_slice"]]
    width = plan.prefix + plan.branch
    prefix = ind[stream * width : stream * width + plan.prefix]
    control = ind[stream * width + plan.prefix : (stream + 1) * width]
    changed = novel[stream * plan.branch : (stream + 1) * plan.branch]
    if len(prefix) != plan.prefix or len(control) != plan.branch or len(changed) != plan.branch:
        raise RunError("the development pool is too small for the declared adaptation design")
    return {"prefix": prefix, "control": control, "changed": changed}


def _error(records: Sequence[Mapping[str, Any]]) -> float:
    return 1.0 - sum(r["correct"] for r in records) / len(records)


def develop(
    spec: Mapping[str, Any] | None = None,
    *,
    quick: bool = False,
    provenance: Mapping[str, Any] | None = None,
    checkpoints: Path | None = None,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = spec or spec_module.load()
    plan = Plan.from_spec(spec, quick=quick)
    tasks = load_tasks(plan, spec)
    log(
        f"tasks: train {sum(map(len, tasks['train'].values()))}, development {sum(map(len, tasks['development'].values()))}"
    )
    baselines = fitted_baselines(tasks["train"], spec)
    records: list[dict[str, Any]] = []
    parameters: dict[str, Any] = {}
    trained_hashes: dict[str, str] = {}
    for init in range(plan.initializations):
        learners = {r: trained_learner(spec, plan, tasks, init, r, checkpoints) for r in plan.representations}
        for rep, learner in learners.items():
            trained_hashes[f"{init}:{rep}"] = learner.state_hash()
            parameters[rep] = learner.parameter_count()
        log(f"init {init}: trained {len(learners)} representations")
        default = learners[plan.default_representation]
        arms: list[tuple[str, OnlineLinear]] = [(a, default) for a in LEARNER_CONTROLS]
        for rep in plan.representations:
            if rep != plan.default_representation:
                arms += [(f"online@{rep}", learners[rep]), (f"frozen@{rep}", learners[rep])]
        for family in spec["families"]:
            for stream in range(plan.streams):
                chunk = tasks["development"][family][
                    stream * plan.stream_length : (stream + 1) * plan.stream_length
                ]
                cell = {"init": init, "stream": stream}
                for arm, source in arms:
                    agent, feedback = _arm_agent(arm, source)
                    for record in run_stream(agent, chunk, spec, feedback=feedback, learn=True):
                        records.append({**record, **cell, "arm": arm, "phase": "evaluate"})
                for name in BASELINES:
                    agent_b: Agent = (
                        Uniform(derive_seed(PROTOCOL_VERSION, "uniform", init, family, stream))
                        if name == "uniform"
                        else baselines[name]
                    )
                    for record in run_stream(agent_b, chunk, spec, feedback=True, learn=False):
                        records.append({**record, **cell, "arm": name, "phase": "evaluate"})
            if family not in plan.adaptation_families:
                continue
            for stream in range(plan.streams):
                split = adaptation_split(tasks["development"][family], plan, stream, spec)
                cell = {"init": init, "stream": stream}
                shared = default.clone("adapt:prefix")
                for record in run_stream(shared, split["prefix"], spec, feedback=True, learn=True):
                    records.append({**record, **cell, "arm": "adapt:prefix", "phase": "adapt"})
                ends: dict[str, OnlineLinear] = {}
                for branch in ("changed", "control"):
                    for mode in ("online", "frozen"):
                        agent = shared.clone(f"adapt:{branch}:{mode}")
                        agent.update_enabled = mode == "online"
                        arm = f"adapt:{branch}:{mode}"
                        for record in run_stream(agent, split[branch], spec, feedback=True, learn=True):
                            records.append({**record, **cell, "arm": arm, "phase": "adapt"})
                        if mode == "online":
                            ends[branch] = agent
                for when, state in (
                    ("before", shared),
                    ("after_changed", ends["changed"]),
                    ("after_control", ends["control"]),
                ):
                    reader = state.clone(f"probe:{when}")
                    reader.update_enabled = False
                    for record in run_stream(
                        reader, tasks["probe"][family], spec, feedback=False, learn=False
                    ):
                        records.append({**record, **cell, "arm": f"probe:{when}", "phase": "retain"})
    lookup = baselines["lookup"]
    assert isinstance(lookup, Lookup)
    cells = cells_from_records(records, spec)
    evidence = {
        "schema": SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "status": "development evidence: not confirmation, not a capability claim",
        "spec_sha256": spec_module.canonical_hash(spec),
        "provenance": dict(provenance or {}),
        "plan": asdict(plan),
        "parameters": parameters,
        "trained_state_hashes": trained_hashes,
        "records": {"count": len(records), "sha256": records_sha256(records)},
        "cells": cells,
        "summary": summarize(cells, plan, spec),
    }
    return evidence, records


# ------------------------------------------------------------------- primitives
def records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def cells_from_records(records: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Per ``(arm, family, init, stream)``: counts, calibration bins and per-class recall counts."""

    bins = spec["statistics"]["calibration_bins"]
    grouped: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (record["arm"], record["family"], record["init"], record["stream"])
        grouped.setdefault(key, []).append(record)
    cells = []
    for (arm, family, init, stream), rows in sorted(grouped.items()):
        bin_n, bin_conf, bin_correct = [0] * bins, [0.0] * bins, [0] * bins
        per_class: dict[str, list[int]] = {}
        for row in rows:
            b = min(int(row["confidence"] * bins), bins - 1)
            bin_n[b] += 1
            bin_conf[b] += row["confidence"]
            bin_correct[b] += int(row["correct"])
            counts = per_class.setdefault(json.dumps(row["truth"]), [0, 0])
            counts[0] += 1
            counts[1] += int(row["correct"])
        cells.append(
            {
                "arm": arm,
                "family": family,
                "init": init,
                "stream": stream,
                "n": len(rows),
                "correct": sum(int(r["correct"]) for r in rows),
                "abstained": sum(int(r["abstain"]) for r in rows),
                "updates": sum(int(r["updated"]) for r in rows),
                "brier_sum": float(sum((r["confidence"] - float(r["correct"])) ** 2 for r in rows)),
                "bins": {"n": bin_n, "confidence_sum": bin_conf, "correct": bin_correct},
                "per_class": dict(sorted(per_class.items())),
            }
        )
    return cells


def summarize(cells: Sequence[Mapping[str, Any]], plan: Plan, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Every reported number, from cell primitives only."""

    stats = spec["statistics"]
    seed, draws, confidence = stats["bootstrap_seed"], plan.draws, stats["confidence"]
    by: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for cell in cells:
        by.setdefault((cell["arm"], cell["family"]), []).append(cell)

    def accuracy_grid(arm: str, family: str) -> np.ndarray:
        rows = [{**c, "accuracy": c["correct"] / c["n"]} for c in by[(arm, family)]]
        return grid(rows, "accuracy")[0]

    out: dict[str, Any] = {"families": {}}
    for family in spec["families"]:
        arms = sorted({arm for arm, f in by if f == family and not arm.startswith(("adapt:", "probe:"))})
        entry: dict[str, Any] = {"arms": {}, "contrasts": {}}
        for arm in arms:
            rows = by[(arm, family)]
            n = sum(c["n"] for c in rows)
            pooled_bins = np.sum([c["bins"]["n"] for c in rows], axis=0)
            conf_sum = np.sum([c["bins"]["confidence_sum"] for c in rows], axis=0)
            correct_sum = np.sum([c["bins"]["correct"] for c in rows], axis=0)
            ece = float(
                sum(abs(cs - k) for cs, k, m in zip(conf_sum, correct_sum, pooled_bins, strict=True) if m) / n
            )
            classes: dict[str, list[int]] = {}
            for c in rows:
                for label, (total, hits) in c["per_class"].items():
                    counts = classes.setdefault(label, [0, 0])
                    counts[0] += total
                    counts[1] += hits
            entry["arms"][arm] = {
                "accuracy": crossed_mean(
                    accuracy_grid(arm, family), seed=seed, draws=draws, confidence=confidence
                ),
                "balanced_accuracy": float(np.mean([h / t for t, h in classes.values()])),
                "coverage": 1.0 - sum(c["abstained"] for c in rows) / n,
                "brier": sum(c["brier_sum"] for c in rows) / n,
                "expected_calibration_error": ece,
                "tasks": n,
                "updates": sum(c["updates"] for c in rows),
            }
            if arm == "lookup":
                # The lookup answers with confidence 1.0 exactly on an exact-memory hit.
                entry["arms"][arm]["lookup_hit_rate"] = float(pooled_bins[-1] / n)
        for left, right in [
            ("online", r) for r in ("frozen", "feedback_disabled", "reset_each_task", *BASELINES)
        ] + [(f"online@{r}", "online") for r in plan.representations if r != plan.default_representation]:
            if (left, family) in by and (right, family) in by:
                diff = accuracy_grid(left, family) - accuracy_grid(right, family)
                result = crossed_mean(diff, seed=seed, draws=draws, confidence=confidence)
                result["resolved_sign"] = resolved_sign(result)
                entry["contrasts"][f"{left} - {right}"] = result
        if family in plan.adaptation_families:
            entry["adaptation"] = _adaptation(by, family, seed, draws, confidence)
        out["families"][family] = entry
    return out


def _adaptation(
    by: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    family: str,
    seed: int,
    draws: int,
    confidence: float,
) -> dict[str, Any]:
    def error(arm: str) -> np.ndarray:
        rows = [{**c, "error": 1.0 - c["correct"] / c["n"]} for c in by[(arm, family)]]
        return grid(rows, "error")[0]

    changed = error("adapt:changed:frozen") - error("adapt:changed:online")
    control = error("adapt:control:frozen") - error("adapt:control:online")
    results: dict[str, Any] = {}
    for name, values in (
        ("online_advantage_changed", changed),
        ("online_advantage_control", control),
        ("difference_of_differences", changed - control),
        ("forgetting_after_changed", error("probe:after_changed") - error("probe:before")),
        ("forgetting_after_control", error("probe:after_control") - error("probe:before")),
    ):
        result = crossed_mean(values, seed=seed, draws=draws, confidence=confidence)
        result["resolved_sign"] = resolved_sign(result)
        results[name] = result
    return results


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    temporary.replace(path)


def read_records(path: Path) -> list[dict[str, Any]]:
    def refuse_constant(token: str) -> Any:
        raise RunError(f"non-standard JSON constant in records: {token}")

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line, parse_constant=refuse_constant) for line in handle if line.strip()]


def pooled_calibration(records: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]) -> dict[str, float]:
    return calibration(
        [r["confidence"] for r in records],
        [r["correct"] for r in records],
        spec["statistics"]["calibration_bins"],
    )


def pooled_balanced_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    return balanced_accuracy([r["answer"] for r in records], [r["truth"] for r in records])


__all__ = ["Action", "Plan", "RunError", "StateError", "develop"]
