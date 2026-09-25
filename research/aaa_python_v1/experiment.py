"""Development experiments of ``aaa.python.v1`` (never confirmation; see :mod:`.confirm`).

Every arm is a :class:`ArmSpec`: an encoder, a core width, heads and an
optimizer. All arms share, per initialization, the same initialization seed,
training order and evaluation tasks, so every contrast is paired within an
``initialization x stream`` cell.

Stages (each a function of the declared :data:`DESIGN` and an arm list):

``tune``      For each arm, a small grid of learning rate x training epochs is
              trained on the full training pool and scored *frozen* on the
              development ``tune`` range (indices 0-399). Tuning uses its own
              initializations (1000, 1001), never the evaluation ones. The
              criterion is declared: the mean over families of tune accuracy;
              ties go to fewer epochs, then the lower rate. Every arm gets its
              own best budget, so no arm is under-trained relative to another
              (the v0 lesson, ``AAA-197``).
``evaluate``  With the selected budget, ten initializations are trained and
              scored on 30 streams x 40 tasks of the ``evaluate`` range, both
              frozen and online (learning from feedback during the stream,
              from a fresh clone per stream).
``adapt``     A distribution switch: from one trained state per cell, online
              and frozen clones run a *changed* branch (``novel_structure``)
              and a *control* branch (``in_distribution``) from the ``adapt``
              range; the estimand is v0's difference of differences. The
              never-trained probe bank is scored before and after each
              online branch (forgetting).
``plasticity`` see :mod:`.plasticity`.

Per-task correctness bits and log-losses are retained, so every summary can be
recomputed from primitives (:mod:`.recompute`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from research.aaa_python.generator import Task
from research.aaa_python.learners import Agent
from research.aaa_python.rng import Stream, derive_seed

from . import PROTOCOL_VERSION, generator
from . import spec as spec_module
from .baselines import BASELINES
from .encoders import E2_CHANNELS
from .episode import ToolEnvironment
from .models import CoreAgent, CoreModel, ModelConfig
from .runner import run_task

SCHEMA = "aaa.python.v1.development.v1"
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")

DESIGN: dict[str, Any] = {
    "dimensions": 256,
    "train_per_family": 2400,
    "init_scale": 1.0,
    "tune": {
        "learning_rates": [0.03, 0.1, 0.3],
        "epochs": [1, 3, 8],
        "initializations": [1000, 1001],
        "criterion": "mean over families of frozen accuracy on the tune range; ties -> fewer epochs, then lower rate",
    },
    "evaluate": {"initializations": 10, "streams": 30, "stream_length": 40},
    "adapt": {
        "initializations": 10,
        "streams": 15,
        "branch": 20,
        "changed_slice": "novel_structure",
        "control_slice": "in_distribution",
        "range": [1600, 3400],
        "probe_bank": 100,
    },
    "statistics": {"draws": 4000, "confidence": 0.95, "seed": 20260924, "holm_alpha": 0.05},
}


@dataclass(frozen=True)
class ArmSpec:
    name: str
    encoder: str
    hidden: int
    output_head: str = "onehot"
    localize_head: str = "onehot"
    channels: tuple[str, ...] = E2_CHANNELS
    tool: bool = False
    weight_decay: float = 0.0
    momentum: float = 0.0
    clip: float = 0.0
    train_per_family: int = 2400
    learning_rate: float | None = None
    epochs: int | None = None

    def config(self, seed: int, learning_rate: float) -> ModelConfig:
        return ModelConfig(
            encoder=self.encoder,
            dimensions=DESIGN["dimensions"],
            hidden=self.hidden,
            output_head=self.output_head,
            localize_head=self.localize_head,
            learning_rate=learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            clip=self.clip,
            init_scale=DESIGN["init_scale"],
            seed=seed,
            encoder_channels=self.channels,
            tool_inputs=2 if self.tool else 0,
        )

    def parameters(self) -> dict[str, Any]:
        return CoreModel(self.config(0, 0.1)).accounting()

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channels"] = list(self.channels)
        return payload


class RunError(RuntimeError):
    pass


# ------------------------------------------------------------------- tasks
def layout(role: str) -> tuple[int, int]:
    start, stop = spec_module.load()["splits"]["development_layout"][role]
    return int(start), int(stop)


def train_tasks(per_family: int) -> dict[str, list[Task]]:
    return {f: list(generator.pool("train", f))[:per_family] for f in FAMILIES}


def dev_range(role: str, family: str) -> list[Task]:
    start, stop = layout(role)
    return list(generator.pool("development", family))[start:stop]


def eval_streams(family: str) -> list[list[Task]]:
    tasks = dev_range("evaluate", family)
    n, length = DESIGN["evaluate"]["streams"], DESIGN["evaluate"]["stream_length"]
    if len(tasks) < n * length:
        raise RunError("the evaluate range is too small for the declared streams")
    return [tasks[s * length : (s + 1) * length] for s in range(n)]


def init_seed(init: int) -> int:
    return derive_seed(PROTOCOL_VERSION, "init", init) % 2**63


def training_order(tasks: Mapping[str, Sequence[Task]], init: int, epoch: int) -> list[Task]:
    pool = [task for family in FAMILIES for task in tasks[family]]
    return Stream(derive_seed(PROTOCOL_VERSION, "train-order", init, epoch)).shuffled(pool)


# ------------------------------------------------------------------- running
def _score(agent: Agent, env: ToolEnvironment, task: Task, learn: bool) -> tuple[bool, float]:
    """One causal step; the log-loss of the truth is read *after* the commit, before learning."""

    agent.begin_task()
    view = env.present(task)
    if (
        getattr(agent, "uses_tool", False)
        and task.family == "repair"
        and not getattr(agent, "training", False)
    ):
        from .runner import tool_vector

        results = env.run_visible_tests(view)
        store = agent.tool_results  # type: ignore[attr-defined]
        store.clear()
        store[view.task_ref] = tool_vector(results) if isinstance(agent, CoreAgent) else results
    action = agent.act(view)
    env.commit(view, action)
    nll = float("nan")
    if isinstance(agent, CoreAgent):
        nll = -math.log(max(agent.probability_of(view, task.answer), 1e-12))
    correct, feedback = env.reveal(view)
    if learn:
        agent.learn(view, action, feedback)
    return bool(correct), nll


def train(
    agent: Agent, tasks: Mapping[str, Sequence[Task]], epochs: int, init: int, spec: Mapping[str, Any]
) -> None:
    env = ToolEnvironment(spec)
    for epoch in range(epochs):
        for task in training_order(tasks, init, epoch):
            run_task(agent, env, task)


def trained_agent(arm: ArmSpec, init: int, learning_rate: float, epochs: int) -> CoreAgent:
    spec = spec_module.load()
    agent = CoreAgent(CoreModel(arm.config(init_seed(init), learning_rate)), name=arm.name)
    agent.remember_initial()
    train(agent, train_tasks(arm.train_per_family), epochs, init, spec)
    return agent


def frozen_pass(agent: CoreAgent | Agent, tasks: Sequence[Task]) -> tuple[list[bool], list[float]]:
    spec = spec_module.load()
    env = ToolEnvironment(spec)
    if isinstance(agent, CoreAgent):
        agent.update_enabled = False
    out = [_score(agent, env, task, learn=False) for task in tasks]
    return [c for c, _ in out], [n for _, n in out]


def online_pass(agent: CoreAgent, tasks: Sequence[Task]) -> tuple[list[bool], list[float]]:
    spec = spec_module.load()
    env = ToolEnvironment(spec)
    clone = agent.clone(agent.name)
    clone.update_enabled = True
    out = [_score(clone, env, task, learn=True) for task in tasks]
    return [c for c, _ in out], [n for _, n in out]


def bits(values: Iterable[bool]) -> str:
    out = 0
    count = 0
    for i, v in enumerate(values):
        out |= int(bool(v)) << i
        count = i + 1
    return f"{count}:{out:x}"


def unbits(text: str) -> list[bool]:
    count, value = text.split(":")
    number = int(value, 16)
    return [bool((number >> i) & 1) for i in range(int(count))]


# ------------------------------------------------------------------- stages
def tune_job(arm: ArmSpec, learning_rate: float, epochs: int, init: int) -> dict[str, Any]:
    agent = trained_agent(arm, init, learning_rate, epochs)
    accuracy = {}
    for family in FAMILIES:
        correct, _ = frozen_pass(agent.clone(), dev_range("tune", family))
        accuracy[family] = sum(correct) / len(correct)
    return {
        "arm": arm.name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "init": init,
        "accuracy": accuracy,
    }


def select(arm: ArmSpec, rows: Sequence[Mapping[str, Any]]) -> tuple[float, int, list[dict[str, Any]]]:
    """The declared criterion over the tuning grid; fixed budgets are honoured when given."""

    table: dict[tuple[float, int], list[float]] = {}
    for row in rows:
        if row["arm"] != arm.name:
            continue
        table.setdefault((row["learning_rate"], row["epochs"]), []).append(
            float(np.mean(list(row["accuracy"].values())))
        )
    summary = [
        {
            "learning_rate": lr,
            "epochs": ep,
            "mean_tune_accuracy": float(np.mean(v)),
            "initializations": len(v),
        }
        for (lr, ep), v in sorted(table.items())
    ]
    if arm.learning_rate is not None and arm.epochs is not None:
        return arm.learning_rate, arm.epochs, summary
    if not summary:
        raise RunError(f"no tuning results for {arm.name}")
    best = max(summary, key=lambda r: (round(r["mean_tune_accuracy"], 12), -r["epochs"], -r["learning_rate"]))
    return float(best["learning_rate"]), int(best["epochs"]), summary


def evaluate_job(arm: ArmSpec, learning_rate: float, epochs: int, init: int) -> dict[str, Any]:
    agent = trained_agent(arm, init, learning_rate, epochs)
    result: dict[str, Any] = {
        "arm": arm.name,
        "init": init,
        "state_hash": agent.model.state_hash(),
        "families": {},
    }
    for family in FAMILIES:
        frozen_c, frozen_n, online_c, online_n = [], [], [], []
        for stream in eval_streams(family):
            c, n = frozen_pass(agent.clone(), stream)
            frozen_c += c
            frozen_n += n
            c, n = online_pass(agent, stream)
            online_c += c
            online_n += n
        result["families"][family] = {
            "frozen": {"bits": bits(frozen_c), "nll": [round(v, 6) for v in frozen_n]},
            "online": {"bits": bits(online_c), "nll": [round(v, 6) for v in online_n]},
        }
    diag = agent.model.diagnostics.gradient_norms
    result["training"] = {
        "updates": agent.model.updates,
        "clipped": agent.model.diagnostics.clipped,
        "mean_gradient_norm_last_1000": float(np.mean(diag[-1000:])) if diag else 0.0,
    }
    return result


def baseline_job(name: str, init: int) -> dict[str, Any]:
    spec = spec_module.load()
    agent = BASELINES[name](derive_seed(PROTOCOL_VERSION, "baseline", name, init) % 2**63)
    if hasattr(agent, "training"):
        train(agent, train_tasks(DESIGN["train_per_family"]), 1, init, spec)
        agent.training = False
    result: dict[str, Any] = {"arm": name, "init": init, "families": {}}
    for family in FAMILIES:
        correct: list[bool] = []
        for stream in eval_streams(family):
            c, _ = frozen_pass(agent, stream)
            correct += c
        result["families"][family] = {"frozen": {"bits": bits(correct)}}
    return result


def v0_instrument_job(init: int) -> dict[str, Any]:
    """v0's unmodified ``OnlineLinear`` at v0's declared budget, scored frozen on v1's streams (an anchor)."""

    from research.aaa_python import spec as v0_spec
    from research.aaa_python.experiment import make_learner
    from research.aaa_python.experiment import train as v0_train

    v0 = v0_spec.load()
    learner = make_learner(v0, init, v0["learner"]["default_representation"])
    tasks = train_tasks(v0["splits"]["pool_per_family"]["train"])
    v0_train(
        learner, tasks, v0["learner"]["train_epochs"], derive_seed("aaa.python.v0", "train-order", init), v0
    )
    learner.update_enabled = False
    result: dict[str, Any] = {"arm": "v0_instrument", "init": init, "families": {}}
    for family in FAMILIES:
        correct: list[bool] = []
        for stream in eval_streams(family):
            c, _ = frozen_pass(learner, stream)
            correct += c
        result["families"][family] = {"frozen": {"bits": bits(correct)}}
    result["parameters"] = learner.parameter_count()
    return result


def adapt_job(arm: ArmSpec, learning_rate: float, epochs: int, init: int) -> dict[str, Any]:
    agent = trained_agent(arm, init, learning_rate, epochs)
    design = DESIGN["adapt"]
    start, stop = design["range"]
    spec = spec_module.load()
    out: dict[str, Any] = {"arm": arm.name, "init": init, "families": {}}
    for family in FAMILIES:
        block = list(generator.pool("development", family))[start:stop]
        changed = [t for t in block if t.slice == design["changed_slice"]]
        control = [t for t in block if t.slice == design["control_slice"]]
        probe = list(generator.pool("probe", family))[: design["probe_bank"]]
        rows = []
        for stream in range(design["streams"]):
            b = design["branch"]
            branches = {
                "changed": changed[stream * b : (stream + 1) * b],
                "control": control[stream * b : (stream + 1) * b],
            }
            if any(len(v) != b for v in branches.values()):
                raise RunError("the adapt range is too small for the declared streams")
            row: dict[str, Any] = {"stream": stream}
            before, _ = frozen_pass(agent.clone(), probe)
            row["probe_before"] = bits(before)
            for branch, tasks in branches.items():
                c, _ = frozen_pass(agent.clone(), tasks)
                row[f"{branch}_frozen"] = bits(c)
                learner = agent.clone()
                learner.update_enabled = True
                env = ToolEnvironment(spec)
                c = [_score(learner, env, task, learn=True)[0] for task in tasks]
                row[f"{branch}_online"] = bits(c)
                after, _ = frozen_pass(learner.clone(), probe)
                row[f"probe_after_{branch}"] = bits(after)
            rows.append(row)
        out["families"][family] = rows
    return out


# ------------------------------------------------------------------- orchestration
def _job_key(function: Callable[..., Any], job: tuple[Any, ...]) -> str | None:
    """A resumption key bound to the running source; ``None`` disables resumption.

    Enabled only when ``AAA_DATA_ROOT`` is set. The key covers the phase
    fingerprint, the specification, the design and the job's arguments, so a
    result produced by different code or a different design is never reused.
    """

    if not os.environ.get("AAA_DATA_ROOT"):
        return None
    from .identity import fingerprint

    global _FINGERPRINT
    if _FINGERPRINT is None:
        _FINGERPRINT = fingerprint()["sha256"]
    arguments = [a.to_json() if isinstance(a, ArmSpec) else a for a in job]
    return digest([_FINGERPRINT, spec_module.spec_hash(), DESIGN, function.__name__, arguments])


_FINGERPRINT: str | None = None


def _resume_path(key: str) -> Any:
    from pathlib import Path

    return Path(os.environ["AAA_DATA_ROOT"]) / "resume" / "aaa_python_v1" / f"{key}.json"


def _map(
    function: Callable[..., dict[str, Any]], jobs: Sequence[tuple[Any, ...]], workers: int
) -> list[dict[str, Any]]:
    """Run jobs in order; finished jobs of an interrupted run are reused when their key matches."""

    keys = [_job_key(function, job) for job in jobs]
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    for i, key in enumerate(keys):
        if key is not None and _resume_path(key).is_file():
            stored = json.loads(_resume_path(key).read_text(encoding="utf-8"))
            if stored.get("key") == key:
                results[i] = stored["result"]
    pending = [i for i, r in enumerate(results) if r is None]

    def keep(i: int, result: dict[str, Any]) -> None:
        results[i] = result
        key = keys[i]
        if key is not None:
            path = _resume_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps({"key": key, "result": result}, allow_nan=False), encoding="utf-8"
            )
            temporary.replace(path)

    if workers <= 1:
        for i in pending:
            keep(i, function(*jobs[i]))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(function, *jobs[i]): i for i in pending}
            for future, i in futures.items():
                keep(i, future.result())
    return [r for r in results if r is not None]


def warm_pools() -> None:
    """Build (or load from the content-addressed cache) every pool a stage reads, once."""

    for family in FAMILIES:
        generator.pool("train", family)
        generator.pool("development", family)
        generator.pool("probe", family)


def run_stage(
    stage: str,
    arms: Sequence[ArmSpec],
    *,
    baselines: Sequence[str] = (),
    workers: int = max(1, (os.cpu_count() or 2) - 1),
    log: Callable[[str], None] = print,
    tuning: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    warm_pools()
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise RunError("arm names must be unique")
    tune_rows = list(tuning or [])
    need = [a for a in arms if not (a.learning_rate is not None and a.epochs is not None)]
    have = {r["arm"] for r in tune_rows}
    missing = [a for a in need if a.name not in have]
    if missing:
        grid = DESIGN["tune"]
        jobs = [
            (a, lr, ep, init)
            for a in missing
            for lr in grid["learning_rates"]
            for ep in grid["epochs"]
            for init in grid["initializations"]
        ]
        log(f"{stage}: tuning {len(missing)} arms, {len(jobs)} jobs")
        tune_rows += _map(tune_job, jobs, workers)
    selected: dict[str, dict[str, Any]] = {}
    for arm in arms:
        lr, ep, table = select(arm, tune_rows)
        selected[arm.name] = {"learning_rate": lr, "epochs": ep, "tuning": table}
        log(f"{stage}: {arm.name} lr={lr} epochs={ep}")
    evidence: dict[str, Any] = {
        "stage": stage,
        "arms": {a.name: {**a.to_json(), "parameters": a.parameters(), **selected[a.name]} for a in arms},
        "tuning_rows": tune_rows,
    }
    inits = range(DESIGN["evaluate"]["initializations"])
    jobs2 = [
        (a, selected[a.name]["learning_rate"], selected[a.name]["epochs"], i) for a in arms for i in inits
    ]
    log(f"{stage}: evaluating {len(jobs2)} jobs")
    evidence["evaluations"] = _map(evaluate_job, jobs2, workers)
    if baselines:
        evidence["baselines"] = _map(baseline_job, [(b, i) for b in baselines for i in inits], workers)
    return evidence


def run_adapt(
    arms: Sequence[ArmSpec], selected: Mapping[str, Mapping[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    warm_pools()
    inits = range(DESIGN["adapt"]["initializations"])
    jobs = [
        (a, selected[a.name]["learning_rate"], selected[a.name]["epochs"], i) for a in arms for i in inits
    ]
    return _map(adapt_job, jobs, workers)


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass
class Evidence:
    """A development evidence document under construction."""

    provenance: Mapping[str, Any]
    stages: dict[str, Any] = field(default_factory=dict)

    def document(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "protocol": PROTOCOL_VERSION,
            "status": "development evidence: not confirmation, not a capability claim",
            "spec_sha256": spec_module.spec_hash(),
            "design": DESIGN,
            "provenance": dict(self.provenance),
            "stages": self.stages,
        }


__all__ = [
    "DESIGN",
    "ArmSpec",
    "Evidence",
    "RunError",
    "adapt_job",
    "baseline_job",
    "bits",
    "evaluate_job",
    "replace",
    "run_adapt",
    "run_stage",
    "select",
    "trained_agent",
    "tune_job",
    "unbits",
]
