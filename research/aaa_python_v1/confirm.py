"""``aaa.python.v1`` confirmation: fresh identities, observed once, under a committed freeze.

Nothing here runs without an :class:`~.freeze.Admission`, which re-verifies at
the moment of use that the freeze manifest is committed, the tree is clean and
the frozen phase fingerprint and specification equal the running source.

The freeze manifest declares, before any confirmation identity exists:

* the arms, each with its encoder, core width, heads and the training budget
  development selected (rate and epochs; nothing is re-tuned);
* the baselines scored on the same streams;
* the primary contrasts and the capacity rule (the same declared rule as
  development, applied to confirmation evidence);
* the ``aaa.promotion.crossed.v1`` contracts: reference and challenger arms,
  groups (the five families, each ``crossed`` over its 30 streams), fresh
  initializations, criteria, thresholds, seed and draws.

The contract metric is the per-cell Jeffreys-smoothed error rate
``(errors + 0.5) / (n + 1)``, declared in the freeze. It is strictly positive,
as ``crossed.v1`` requires (a raw error rate can be exactly zero, which the
contract refuses).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from aaa.promotion import Contract, Criterion, GroupDeclaration, adjudicate
from research.aaa_python.generator import Task
from research.aaa_python.rng import derive_seed

from . import PROTOCOL_VERSION, generator
from .baselines import BASELINES
from .experiment import (
    DESIGN,
    FAMILIES,
    ArmSpec,
    bits,
    frozen_pass,
    train,
    train_tasks,
    trained_agent,
    unbits,
    warm_pools,
)
from .freeze import Admission
from .stages import capacity_verdict
from .summarize import summarize

_TASKS: dict[str, list[Task]] = {}


def arm_from(payload: Mapping[str, Any]) -> ArmSpec:
    return ArmSpec(
        payload["name"],
        payload["encoder"],
        payload["hidden"],
        payload["output_head"],
        payload["localize_head"],
        tuple(payload["channels"]),
        payload["tool"],
        payload["weight_decay"],
        payload["momentum"],
        payload["clip"],
        payload["train_per_family"],
        payload["learning_rate"],
        payload["epochs"],
    )


def _streams(family: str) -> list[list[Task]]:
    shape = DESIGN["confirmation"]
    tasks = _TASKS[family]
    n, length = shape["streams"], shape["stream_length"]
    return [tasks[s * length : (s + 1) * length] for s in range(n)]


def _learner_job(arm: ArmSpec, init: int) -> dict[str, Any]:
    assert arm.learning_rate is not None and arm.epochs is not None
    agent = trained_agent(arm, init, arm.learning_rate, arm.epochs)
    out: dict[str, Any] = {
        "arm": arm.name,
        "init": init,
        "state_hash": agent.model.state_hash(),
        "families": {},
    }
    for family in FAMILIES:
        correct: list[bool] = []
        for stream in _streams(family):
            c, _ = frozen_pass(agent.clone(), stream)
            correct += c
        out["families"][family] = {"frozen": {"bits": bits(correct)}}
    return out


def _baseline_job(name: str, init: int) -> dict[str, Any]:
    from . import spec as spec_module

    agent = BASELINES[name](derive_seed(PROTOCOL_VERSION, "baseline", name, init) % 2**63)
    if hasattr(agent, "training"):
        train(agent, train_tasks(DESIGN["train_per_family"]), 1, init, spec_module.load())
        agent.training = False
    out: dict[str, Any] = {"arm": name, "init": init, "families": {}}
    for family in FAMILIES:
        correct: list[bool] = []
        for stream in _streams(family):
            c, _ = frozen_pass(agent, stream)
            correct += c
        out["families"][family] = {"frozen": {"bits": bits(correct)}}
    return out


def smoothed_errors(rows: Sequence[Mapping[str, Any]], arm: str) -> list[dict[str, Any]]:
    """``crossed.v1`` primitives: Jeffreys-smoothed error per ``(family, init, stream)`` cell."""

    length = DESIGN["confirmation"]["stream_length"]
    out = []
    for row in rows:
        if row["arm"] != arm:
            continue
        for family in FAMILIES:
            values = unbits(row["families"][family]["frozen"]["bits"])
            for s in range(DESIGN["confirmation"]["streams"]):
                errors = length - sum(values[s * length : (s + 1) * length])
                out.append(
                    {
                        "arm": arm,
                        "group": family,
                        "init": row["init"],
                        "series": s,
                        "value": (errors + 0.5) / (length + 1),
                    }
                )
    return out


def contract_from(declared: Mapping[str, Any]) -> Contract:
    return Contract(
        reference=declared["reference"],
        challenger=declared["challenger"],
        initializations=tuple(declared["initializations"]),
        groups=tuple(
            GroupDeclaration(g["name"], g["design"], tuple(g["series"])) for g in declared["groups"]
        ),
        criteria=tuple(Criterion(c["name"], c["rule"], c["threshold"]) for c in declared["criteria"]),
        seed=declared["seed"],
        draws=declared["draws"],
    )


def run(admission: Admission, *, workers: int) -> dict[str, Any]:
    if not isinstance(admission, Admission) or not admission.verify():
        raise generator.ConfirmationNotAdmitted("confirmation requires a verified admission")
    manifest = admission.manifest
    shape = DESIGN["confirmation"]
    warm_pools(attack=True)
    for family in FAMILIES:
        _TASKS[family] = generator.build(
            "confirmation",
            family,
            list(range(shape["streams"] * shape["stream_length"])),
            admission=admission,
        )
    arms = [arm_from(a) for a in manifest["arms"]]
    inits = list(range(shape["init_offset"], shape["init_offset"] + shape["initializations"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        learners = list(pool.map(_learner_job, *zip(*[(a, i) for a in arms for i in inits], strict=True)))
        baselines = list(
            pool.map(
                _baseline_job, *zip(*[(b, i) for b in manifest["baselines"] for i in inits], strict=True)
            )
        )
    stage: dict[str, Any] = {
        "stage": "confirmation",
        "arms": {a.name: {**a.to_json(), "parameters": a.parameters()} for a in arms},
        "evaluations": learners,
        "baselines": baselines,
        "task_hashes": {f: [t.source_sha256 for t in _TASKS[f]] for f in FAMILIES},
    }
    primary = [tuple(c) for c in manifest["design"]["primary"]]
    stage["primary"], stage["secondary"] = [list(c) for c in primary], []
    stage["summary"] = summarize(stage, primary, split="confirmation")
    stage["capacity_verdict"] = capacity_verdict(stage["summary"], manifest["design"]["encoder"])
    rows = learners
    stage["adjudications"] = {}
    for declared in manifest["contracts"]:
        contract = contract_from(declared)
        records = smoothed_errors(rows, contract.reference) + smoothed_errors(rows, contract.challenger)
        stage["adjudications"][declared["name"]] = adjudicate(contract, records)
    return stage
