"""The ``aaa.python.v1`` plasticity diagnostic: does the learner keep its ability to learn?

Retention asks whether old performance survives; plasticity asks whether *new*
things can still be learned late in life. They are different failures (Dohare
et al. 2024; Lyle et al. 2023), and a learner that retains everything but stops
learning fails AAA's objective.

Design (declared in :data:`PLASTICITY`):

* **Life.** A learner of the given arm lives through ``life_epochs`` epochs of
  continual online training on the training pool, in the shared per-epoch
  order.
* **Probe.** At each checkpoint (``0`` is the freshly initialized learner, the
  fresh-learner control) a *clone* must learn a mapping it cannot already
  know: ``syntax`` with its two labels swapped and ``outcome`` with its six
  labels cyclically shifted. The inputs are ordinary programs; only the
  post-action feedback is permuted. The clone learns online for
  ``probe_passes`` passes over a learning block and is then scored, frozen, on a
  held-out block under the same permutation. Both blocks come from the
  development ``adapt`` role's plasticity range (indices 3400-3599), which no
  other stage uses.
* **Diagnostics** at every checkpoint, on inputs from the plasticity range: weight
  norms per block, the fraction of saturated hidden activations (|z| > 0.95),
  the fraction of dormant units (mean |z| below 0.05 across the probe), the
  effective rank of the activation matrix (exponential of the entropy of its
  normalized singular values), the mean gradient norm over the last epoch, and
  the number of clipped steps.

The primary estimand is the ratio of permuted-task accuracy after learning at
the last checkpoint to that at the first non-zero checkpoint, per
initialization. Loss of plasticity is declared (brief, H6) when the upper bound
of its 95% interval over initializations is below 0.9.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from research.aaa_python.generator import Task
from research.aaa_python.rng import derive_seed

from . import PROTOCOL_VERSION, generator
from . import spec as spec_module
from .episode import ToolEnvironment, view_of
from .experiment import ArmSpec, frozen_pass, init_seed, train_tasks, training_order
from .models import CoreAgent, CoreModel
from .runner import run_task

PLASTICITY: dict[str, Any] = {
    "initializations": 8,
    "life_epochs": 32,
    "checkpoints": [0, 1, 4, 16, 32],
    "families": ["syntax", "outcome"],
    "range": [3400, 3600],
    "learn_block": 100,
    "test_block": 100,
    "probe_passes": 2,
    "permutations": {
        "syntax": {"valid": "invalid", "invalid": "valid"},
        "outcome": {
            "ok": "ZeroDivisionError",
            "ZeroDivisionError": "NameError",
            "NameError": "TypeError",
            "TypeError": "IndexError",
            "IndexError": "ValueError",
            "ValueError": "ok",
        },
    },
    "loss_threshold": 0.9,
    "bootstrap": {"draws": 4000, "confidence": 0.95, "seed": 20260925},
}


def permuted(task: Task) -> Task:
    """The same program with its answer and revealed outcome relabelled by the declared permutation."""

    mapping = PLASTICITY["permutations"][task.family]
    label = mapping[task.answer]
    oracle = dict(task.oracle)
    if task.family == "syntax":
        oracle["status"] = "valid" if label == "valid" else "syntax_error"
    elif label == "ok":
        oracle.update(status="ok", exception=None)
    else:
        oracle.update(status="exception", exception=label)
    return replace(task, answer=label, oracle=oracle)


def blocks(family: str) -> tuple[list[Task], list[Task]]:
    start, stop = PLASTICITY["range"]
    tasks = list(generator.pool("development", family))[start:stop]
    learn, test = PLASTICITY["learn_block"], PLASTICITY["test_block"]
    if len(tasks) < learn + test:
        raise ValueError("the plasticity range is too small for the declared blocks")
    return [permuted(t) for t in tasks[:learn]], [permuted(t) for t in tasks[learn : learn + test]]


def probe(agent: CoreAgent, seed: int) -> dict[str, float]:
    """Permuted-mapping accuracy of a clone before and after learning the permuted block."""

    spec = spec_module.load()
    out: dict[str, float] = {}
    for family in PLASTICITY["families"]:
        learn, test = blocks(family)
        clone = agent.clone()
        before, _ = frozen_pass(clone.clone(), test)
        clone.update_enabled = True
        env = ToolEnvironment(spec)
        from research.aaa_python.rng import Stream

        for p in range(PLASTICITY["probe_passes"]):
            for task in Stream(derive_seed(PROTOCOL_VERSION, "plasticity-order", seed, family, p)).shuffled(
                learn
            ):
                run_task(clone, env, task)
        after, _ = frozen_pass(clone, test)
        out[f"{family}_before"] = sum(before) / len(before)
        out[f"{family}_after"] = sum(after) / len(after)
    out["after_mean"] = float(np.mean([out[f"{f}_after"] for f in PLASTICITY["families"]]))
    return out


def diagnostics(agent: CoreAgent) -> dict[str, Any]:
    model = agent.model
    spec = spec_module.load()
    norms = {name: float(np.linalg.norm(array)) for name, array in sorted(model.params.items())}
    result: dict[str, Any] = {
        "weight_norms": norms,
        "updates": model.updates,
        "clipped": model.diagnostics.clipped,
    }
    grads = model.diagnostics.gradient_norms
    result["mean_gradient_norm_recent"] = float(np.mean(grads[-2000:])) if grads else 0.0
    if model.config.hidden == 0:
        result.update(saturated_fraction=None, dormant_fraction=None, effective_rank=None)
        return result
    rows = []
    start, stop = PLASTICITY["range"]
    for family in ("syntax", "outcome", "output"):
        for position, task in enumerate(list(generator.pool("development", family))[start:stop][:100]):
            encoding = model.encoder.encode(view_of(task, position, spec))
            rows.append(model.core(encoding.program)[0])
    z = np.array(rows)
    result["saturated_fraction"] = float(np.mean(np.abs(z) > 0.95))
    result["dormant_fraction"] = float(np.mean(np.mean(np.abs(z), axis=0) < 0.05))
    singular = np.linalg.svd(z - z.mean(axis=0), compute_uv=False)
    weights = singular / singular.sum() if singular.sum() > 0 else singular
    entropy = -float(np.sum([w * math.log(w) for w in weights if w > 0]))
    result["effective_rank"] = math.exp(entropy)
    result["hidden_units"] = int(z.shape[1])
    return result


def life_job(arm: ArmSpec, learning_rate: float, init: int) -> dict[str, Any]:
    spec = spec_module.load()
    agent = CoreAgent(CoreModel(arm.config(init_seed(init), learning_rate)), name=arm.name)
    tasks = train_tasks(arm.train_per_family)
    env = ToolEnvironment(spec)
    checkpoints = sorted(PLASTICITY["checkpoints"])
    records = []
    for epoch in range(PLASTICITY["life_epochs"] + 1):
        if epoch in checkpoints:
            records.append(
                {
                    "epoch": epoch,
                    "probe": probe(agent, derive_seed("probe", init, epoch) % 2**63),
                    "diagnostics": diagnostics(agent),
                }
            )
        if epoch == PLASTICITY["life_epochs"]:
            break
        for task in training_order(tasks, init, epoch):
            run_task(agent, env, task)
    return {"arm": arm.name, "init": init, "learning_rate": learning_rate, "checkpoints": records}


def summarize_plasticity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per arm: the late/early learning ratio over initializations with a percentile bootstrap."""

    b = PLASTICITY["bootstrap"]
    checkpoints = sorted(PLASTICITY["checkpoints"])
    early, late = checkpoints[1], checkpoints[-1]
    out: dict[str, Any] = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["init"])
        by = [{c["epoch"]: c for c in r["checkpoints"]} for r in mine]
        ratio = np.array(
            [m[late]["probe"]["after_mean"] / max(m[early]["probe"]["after_mean"], 1e-9) for m in by]
        )
        fresh_gap = np.array([m[late]["probe"]["after_mean"] - m[0]["probe"]["after_mean"] for m in by])
        rng = np.random.default_rng(b["seed"])
        idx = rng.integers(0, len(ratio), size=(b["draws"], len(ratio)))
        tail = (1 - b["confidence"]) / 2

        def interval(values: np.ndarray, idx: np.ndarray = idx, tail: float = tail) -> dict[str, float]:
            means = values[idx].mean(axis=1)
            return {
                "mean": float(values.mean()),
                "lower": float(np.quantile(means, tail)),
                "upper": float(np.quantile(means, 1 - tail)),
            }

        curve = {
            str(c): {
                "after_mean": float(np.mean([m[c]["probe"]["after_mean"] for m in by])),
                **{
                    key: (
                        float(np.mean([m[c]["diagnostics"][key] for m in by]))
                        if by[0][c]["diagnostics"][key] is not None
                        else None
                    )
                    for key in (
                        "saturated_fraction",
                        "dormant_fraction",
                        "effective_rank",
                        "mean_gradient_norm_recent",
                    )
                },
                "core_weight_norm": float(
                    np.mean([m[c]["diagnostics"]["weight_norms"].get("core.W", float("nan")) for m in by])
                ),
            }
            for c in checkpoints
        }
        ratio_interval = interval(ratio)
        out[arm] = {
            "late_over_early_ratio": ratio_interval,
            "late_minus_fresh": interval(fresh_gap),
            "loss_of_plasticity": ratio_interval["upper"] < PLASTICITY["loss_threshold"],
            "curve": curve,
            "initializations": len(by),
        }
    return out
