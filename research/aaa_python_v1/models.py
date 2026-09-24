"""The ``aaa.python.v1`` learner family: a fixed encoder, an optional shared core, and task heads.

    TaskView --encoder (fixed, 0 params)--> x in R^D --core--> z --heads--> actions

**Core.** ``hidden = 0`` is a linear model (``z = x``). ``hidden = H > 0`` is
one shared ``tanh`` layer, ``z = tanh(W x + b)``, used by every family. ``H``
is the capacity variable of the pre-scale sweep: everything else is held
fixed while it changes.

**Heads** (a separate experimental variable, never changed together with the
encoder in the same comparison):

* ``onehot`` -- v0's formulation: a softmax over the family's labels (101
  printed integers for ``output``; 40 absolute line numbers for ``localize``,
  masked to the program's length);
* ``gauss`` (``output`` only) -- an ordinal, discretized Gaussian over the
  integers -50..50 with a predicted mean and log-scale (2 outputs), so nearby
  values share evidence;
* ``pointer`` (``localize`` only) -- one scorer shared across lines: each
  line's vector goes through the same core and the scores are softmaxed over
  the program's lines (Vasic et al. 2019), so a line is judged by its content
  and position rather than by an absolute index;
* ``repair`` always uses a candidate scorer with bandit feedback (a logistic
  update on the *chosen* candidate only), as in v0.

**Learning.** Online SGD on the negative log-likelihood of the target the
post-action feedback implies, one task at a time. Optional L2 weight decay,
heavy-ball momentum (whose velocity is counted as adaptive state) and
global-norm gradient clipping. No step ever sees an answer key.

**Accounting.** :meth:`CoreModel.accounting` counts trainable parameters by
block, optimizer state, the update counter and the remembered initial state
used by the reset control. Nothing is hidden in untracked structures: the
encoder has no parameters (see :func:`.encoders.accounting`).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from research.aaa_python.episode import Action, Feedback, TaskView
from research.aaa_python.learners import Agent, target_from_feedback, valid_labels

from .encoders import Encoder, Encoding, Sparse

STATE_SCHEMA = "aaa.python.v1.model_state.v1"
FAMILIES = ("syntax", "outcome", "output", "localize", "repair")
OUTPUT_SCALE = 10.0
OUTPUT_LOG_SCALE_INIT = math.log(8.0)


class StateError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    encoder: str
    dimensions: int
    hidden: int
    output_head: str = "onehot"
    localize_head: str = "onehot"
    learning_rate: float = 0.1
    weight_decay: float = 0.0
    momentum: float = 0.0
    clip: float = 0.0
    init_scale: float = 1.0
    seed: int = 0
    encoder_channels: tuple[str, ...] = ("alpha", "line", "flow")
    tool_inputs: int = 0

    def __post_init__(self) -> None:
        if self.output_head not in ("onehot", "gauss"):
            raise ValueError(f"unknown output head {self.output_head!r}")
        if self.localize_head not in ("onehot", "pointer"):
            raise ValueError(f"unknown localize head {self.localize_head!r}")
        for name in ("hidden", "dimensions", "seed", "tool_inputs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("learning_rate", "weight_decay", "momentum", "clip", "init_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.learning_rate <= 0 or self.momentum >= 1:
            raise ValueError("learning rate must be positive and momentum below 1")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["encoder_channels"] = list(self.encoder_channels)
        return payload


@dataclass
class Diagnostics:
    """Running learning-dynamics statistics used by the plasticity diagnostic."""

    gradient_norms: list[float] = field(default_factory=list)
    clipped: int = 0


def _labels(family: str) -> tuple[Any, ...]:
    if family == "syntax":
        return ("valid", "invalid")
    if family == "outcome":
        return ("ok", "ZeroDivisionError", "NameError", "TypeError", "IndexError", "ValueError")
    if family == "output":
        return tuple(range(-50, 51))
    if family == "localize":
        return tuple(range(1, 41))
    raise ValueError(family)


class CoreModel:
    """Parameters, forward passes and single-example gradient steps."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.encoder = Encoder(
            config.encoder,
            config.dimensions,
            tuple(config.encoder_channels) if config.encoder == "e2" else ("alpha", "line", "flow"),
        )
        rng = np.random.default_rng(config.seed)
        D, H = config.dimensions + config.tool_inputs, config.hidden
        Z = H if H > 0 else D
        params: dict[str, np.ndarray] = {}
        if H > 0:
            params["core.W"] = rng.normal(0.0, config.init_scale, size=(H, D))
            params["core.b"] = np.zeros(H)
        for family in ("syntax", "outcome"):
            params[f"{family}.W"] = np.zeros((len(_labels(family)), Z))
            params[f"{family}.b"] = np.zeros(len(_labels(family)))
        if config.output_head == "onehot":
            params["output.W"] = np.zeros((101, Z))
            params["output.b"] = np.zeros(101)
        else:
            params["output.W"] = np.zeros((2, Z))
            params["output.b"] = np.array([0.0, OUTPUT_LOG_SCALE_INIT])
        if config.localize_head == "onehot":
            params["localize.W"] = np.zeros((40, Z))
            params["localize.b"] = np.zeros(40)
        else:
            params["localize.W"] = np.zeros((1, Z))
        params["repair.W"] = np.zeros((1, Z))
        params["repair.b"] = np.zeros(1)
        self.params = params
        self.velocity: dict[str, np.ndarray] = (
            {k: np.zeros_like(v) for k, v in params.items()} if config.momentum > 0 else {}
        )
        self.updates = 0
        self.diagnostics = Diagnostics()

    # ------------------------------------------------------------------ accounting
    def accounting(self) -> dict[str, Any]:
        per_block = {name: int(array.size) for name, array in sorted(self.params.items())}
        trainable = sum(per_block.values())
        core = sum(v for k, v in per_block.items() if k.startswith("core."))
        heads = {
            family: sum(v for k, v in per_block.items() if k.startswith(f"{family}.")) for family in FAMILIES
        }
        optimizer = sum(int(v.size) for v in self.velocity.values())
        return {
            "trainable_parameters": trainable,
            "core": core,
            "heads": heads,
            "per_block": per_block,
            "optimizer_state": optimizer,
            "update_counter": 1,
            "adaptive_state_total": trainable + optimizer + 1,
            "encoder": {"trainable_parameters": 0, "adaptive_state": 0, "identity": self.encoder.identity},
            "remembered_initial_state_for_reset_control": trainable,
            "persistent_recurrent_state": 0,
            "replay_memory": 0,
        }

    # ------------------------------------------------------------------ forward
    def _dense_tool(self, tool: np.ndarray | None) -> np.ndarray | None:
        if self.config.tool_inputs == 0:
            return None
        if tool is None or tool.shape != (self.config.tool_inputs,):
            raise ValueError("this model requires tool inputs of the declared width")
        return tool

    def core(self, x: Sparse, tool: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """``z`` and, for a hidden core, the cached ``z`` again (tanh derivative uses it)."""

        D = self.config.dimensions
        extra = self._dense_tool(tool)
        if self.config.hidden == 0:
            z = np.zeros(D + self.config.tool_inputs)
            z[x.index] = x.value
            if extra is not None:
                z[D:] = extra
            return z, None
        W = self.params["core.W"]
        pre = W[:, x.index] @ x.value + self.params["core.b"]
        if extra is not None:
            pre = pre + W[:, D:] @ extra
        z = np.tanh(pre)
        return z, z

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = scores - scores.max()
        e = np.exp(scores)
        return e / e.sum()

    def distribution(
        self, view: TaskView, encoding: Encoding, tool: np.ndarray | None = None
    ) -> tuple[tuple[Any, ...], np.ndarray]:
        family = view.family
        if family == "repair":
            zs = [
                self.core(c, tool[i] if tool is not None else None)[0]
                for i, c in enumerate(encoding.candidates)
            ]
            scores = np.array([self.params["repair.W"][0] @ z for z in zs])
            return tuple(range(len(zs))), self._softmax(scores)
        if family == "localize" and self.config.localize_head == "pointer":
            labels = tuple(range(1, len(encoding.lines) + 1))
            scores = np.array([self.params["localize.W"][0] @ self.core(x)[0] for x in encoding.lines])
            return labels, self._softmax(scores)
        z, _ = self.core(encoding.program)
        if family == "output" and self.config.output_head == "gauss":
            labels = _labels("output")
            mean, log_scale = self.params["output.W"] @ z + self.params["output.b"]
            mu, sigma = OUTPUT_SCALE * mean, math.exp(max(-3.0, min(5.0, log_scale)))
            k = np.arange(-50, 51, dtype=float)
            return labels, self._softmax(-((k - mu) ** 2) / (2 * sigma**2))
        labels = valid_labels(view) if family == "localize" else _labels(family)
        all_labels = _labels(family)
        rows = [all_labels.index(label) for label in labels]
        scores = self.params[f"{family}.W"][rows] @ z + self.params[f"{family}.b"][rows]
        return tuple(labels), self._softmax(scores)

    # ------------------------------------------------------------------ learning
    def loss(
        self,
        view: TaskView,
        encoding: Encoding,
        target: Any,
        tool: np.ndarray | None = None,
        chosen: int | None = None,
    ) -> float:
        """The loss :meth:`gradients` differentiates (used by the finite-difference check)."""

        if view.family == "repair":
            assert chosen is not None
            z, _ = self.core(encoding.candidates[chosen], tool[chosen] if tool is not None else None)
            score = float(self.params["repair.W"][0] @ z + self.params["repair.b"][0])
            prob = 1.0 / (1.0 + math.exp(-score))
            return -math.log(prob) if target else -math.log(1.0 - prob)
        labels, p = self.distribution(view, encoding, tool)
        return -math.log(float(p[labels.index(target)]))

    def _backprop_core(
        self, grads: dict[str, np.ndarray], x: Sparse, z: np.ndarray, dz: np.ndarray, tool: np.ndarray | None
    ) -> None:
        if self.config.hidden == 0:
            return
        dpre = dz * (1.0 - z * z)
        grads["core.b"] += dpre
        grads["core.W"][:, x.index] += np.outer(dpre, x.value)
        if tool is not None:
            grads["core.W"][:, self.config.dimensions :] += np.outer(dpre, tool)

    def gradients(
        self,
        view: TaskView,
        encoding: Encoding,
        target: Any,
        tool: np.ndarray | None = None,
        chosen: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        """Gradients of the loss for one example; ``None`` when the target is outside the support."""

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        family = view.family
        if family == "repair":
            assert chosen is not None
            t = tool[chosen] if tool is not None else None
            z, _ = self.core(encoding.candidates[chosen], t)
            score = float(self.params["repair.W"][0] @ z + self.params["repair.b"][0])
            ds = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score)))) - float(target)
            grads["repair.W"][0] += ds * z
            grads["repair.b"][0] += ds
            self._backprop_core(grads, encoding.candidates[chosen], z, ds * self.params["repair.W"][0], t)
            return grads
        if family == "localize" and self.config.localize_head == "pointer":
            if not isinstance(target, int) or not 1 <= target <= len(encoding.lines):
                return None
            zs = [self.core(x)[0] for x in encoding.lines]
            v = self.params["localize.W"][0]
            p = self._softmax(np.array([v @ z for z in zs]))
            dscore = p.copy()
            dscore[target - 1] -= 1.0
            for x, z, d in zip(encoding.lines, zs, dscore, strict=True):
                grads["localize.W"][0] += d * z
                self._backprop_core(grads, x, z, d * v, None)
            return grads
        z, _ = self.core(encoding.program)
        if family == "output" and self.config.output_head == "gauss":
            if not isinstance(target, int) or not -50 <= target <= 50:
                return None
            mean, log_scale = self.params["output.W"] @ z + self.params["output.b"]
            clamped = max(-3.0, min(5.0, log_scale))
            mu, sigma = OUTPUT_SCALE * mean, math.exp(clamped)
            k = np.arange(-50, 51, dtype=float)
            p = self._softmax(-((k - mu) ** 2) / (2 * sigma**2))
            dlogit = p.copy()
            dlogit[target + 50] -= 1.0
            dmu = float(dlogit @ ((k - mu) / sigma**2))
            dlog = float(dlogit @ ((k - mu) ** 2 / sigma**2)) if -3.0 < log_scale < 5.0 else 0.0
            dout = np.array([OUTPUT_SCALE * dmu, dlog])
            grads["output.W"] += np.outer(dout, z)
            grads["output.b"] += dout
            self._backprop_core(grads, encoding.program, z, self.params["output.W"].T @ dout, None)
            return grads
        labels = valid_labels(view) if family == "localize" else _labels(family)
        if target not in labels:
            return None
        all_labels = _labels(family)
        rows = [all_labels.index(label) for label in labels]
        W = self.params[f"{family}.W"]
        p = self._softmax(W[rows] @ z + self.params[f"{family}.b"][rows])
        dscore = p.copy()
        dscore[labels.index(target)] -= 1.0
        grads[f"{family}.W"][rows] += np.outer(dscore, z)
        grads[f"{family}.b"][rows] += dscore
        self._backprop_core(grads, encoding.program, z, W[rows].T @ dscore, None)
        return grads

    def step(self, grads: Mapping[str, np.ndarray]) -> None:
        cfg = self.config
        norm = math.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        self.diagnostics.gradient_norms.append(norm)
        scale = 1.0
        if cfg.clip > 0 and norm > cfg.clip:
            scale = cfg.clip / norm
            self.diagnostics.clipped += 1
        for name, g in grads.items():
            update = scale * g
            if cfg.weight_decay > 0 and not name.endswith(".b"):
                update = update + cfg.weight_decay * self.params[name]
            if cfg.momentum > 0:
                self.velocity[name] = cfg.momentum * self.velocity[name] + update
                update = self.velocity[name]
            self.params[name] -= cfg.learning_rate * update
        self.updates += 1

    # ------------------------------------------------------------------ state
    def clone(self) -> CoreModel:
        other = CoreModel.__new__(CoreModel)
        other.config = self.config
        other.encoder = self.encoder
        other.params = {k: v.copy() for k, v in self.params.items()}
        other.velocity = {k: v.copy() for k, v in self.velocity.items()}
        other.updates = self.updates
        other.diagnostics = Diagnostics(list(self.diagnostics.gradient_norms), self.diagnostics.clipped)
        return other

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "config": self.config.to_json(),
            "updates": self.updates,
            "params": {k: v.tolist() for k, v in sorted(self.params.items())},
            "velocity": {k: v.tolist() for k, v in sorted(self.velocity.items())},
        }

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Replace all learned state, or raise :class:`StateError` and change nothing."""

        if not isinstance(state, Mapping) or state.get("schema") != STATE_SCHEMA:
            raise StateError("unknown model state schema")
        if state.get("config") != self.config.to_json():
            raise StateError("state was produced by a different model configuration")
        updates = state.get("updates")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise StateError("update counter must be a non-negative integer")
        loaded: dict[str, dict[str, np.ndarray]] = {}
        for group, reference in (("params", self.params), ("velocity", self.velocity)):
            stored = state.get(group)
            if not isinstance(stored, Mapping) or set(stored) != set(reference):
                raise StateError(f"{group}: missing or extra blocks")
            loaded[group] = {}
            for name, current in reference.items():
                try:
                    array = np.asarray(stored[name], dtype=float)
                except (TypeError, ValueError):
                    raise StateError(f"{group}.{name}: not numeric") from None
                if array.shape != current.shape or not np.all(np.isfinite(array)):
                    raise StateError(f"{group}.{name}: wrong shape or non-finite values")
                loaded[group][name] = array
        self.params, self.velocity, self.updates = loaded["params"], loaded["velocity"], updates

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(json.dumps(self.config.to_json(), sort_keys=True).encode("utf-8"))
        digest.update(str(self.updates).encode("ascii"))
        for group in (self.params, self.velocity):
            for name, array in sorted(group.items()):
                digest.update(name.encode("utf-8"))
                digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
        return digest.hexdigest()


class CoreAgent(Agent):
    """A :class:`CoreModel` behind v0's causal agent interface, with its controls.

    ``update_enabled = False`` is the frozen control; ``reset_each_task`` restores
    the remembered untrained state (parameters, optimizer state *and* counter)
    before every task. ``tool`` agents receive the repair tool's visible-test
    results before acting (see :mod:`.episode`).
    """

    adaptive = True

    def __init__(self, model: CoreModel, *, name: str = "online") -> None:
        self.model = model
        self.name = name
        self.update_enabled = True
        self.reset_each_task = False
        self._initial: CoreModel | None = None
        self.tool_results: dict[int, np.ndarray] = {}

    @property
    def uses_tool(self) -> bool:
        return self.model.config.tool_inputs > 0

    def remember_initial(self) -> None:
        self._initial = self.model.clone()

    def begin_task(self) -> None:
        if self.reset_each_task:
            if self._initial is None:
                raise StateError("reset_each_task requires a remembered untrained state")
            self.model = self._initial.clone()

    def _tool(self, view: TaskView) -> np.ndarray | None:
        if not self.uses_tool or view.family != "repair":
            return None
        return self.tool_results.get(view.task_ref)

    def act(self, view: TaskView) -> Action:
        encoding = self.model.encoder.encode(view)
        labels, p = self.model.distribution(view, encoding, self._tool(view))
        best = int(np.argmax(p))
        return Action(labels[best], float(p[best]))

    def probability_of(self, view: TaskView, label: Any) -> float:
        encoding = self.model.encoder.encode(view)
        labels, p = self.model.distribution(view, encoding, self._tool(view))
        return float(p[labels.index(label)]) if label in labels else 0.0

    def learn(self, view: TaskView, action: Action, feedback: Feedback | None) -> bool:
        if not self.update_enabled or feedback is None:
            return False
        encoding = self.model.encoder.encode(view)
        tool = self._tool(view)
        if view.family == "repair":
            results = feedback.fields.get("chosen_candidate_hidden_results")
            if results is None or action.abstain:
                return False
            grads = self.model.gradients(
                view, encoding, 1.0 if all(results) else 0.0, tool, int(action.answer)
            )
        else:
            target = target_from_feedback(view, feedback)
            if target is None:
                return False
            grads = self.model.gradients(view, encoding, target, tool)
        if grads is None:
            return False
        self.model.step(grads)
        return True

    def clone(self, name: str | None = None) -> CoreAgent:
        other = CoreAgent(self.model.clone(), name=name or self.name)
        other.update_enabled = self.update_enabled
        other.reset_each_task = self.reset_each_task
        other._initial = None if self._initial is None else self._initial.clone()
        return other


def hidden_for_budget(
    budget: int, *, encoder: str, dimensions: int, output_head: str, localize_head: str, tool_inputs: int = 0
) -> int:
    """The hidden width whose total trainable count is closest to ``budget`` (ties go to the smaller)."""

    best, best_gap = 1, None
    for hidden in range(1, 512):
        count = CoreModel(
            ModelConfig(encoder, dimensions, hidden, output_head, localize_head, tool_inputs=tool_inputs)
        ).accounting()["trainable_parameters"]
        gap = abs(count - budget)
        if best_gap is None or gap < best_gap:
            best, best_gap = hidden, gap
        if count > budget:
            break
    return best


def labels_of(family: str) -> tuple[Any, ...]:
    return _labels(family)


__all__ = ["CoreAgent", "CoreModel", "ModelConfig", "StateError", "hidden_for_budget", "labels_of"]
