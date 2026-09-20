"""The AAA-1K recurrent core: a hand-written 994-parameter GRU.

Why NumPy and not a framework
-----------------------------
At 994 parameters a framework buys nothing and costs transparency. Writing the
forward and backward passes out by hand gives exact parameter accounting, an
inspectable hidden state, gradients that can be checked against finite
differences, deterministic CPU execution, and no dependency growth. AAA already
depends on NumPy.

Frozen GRU convention
---------------------
For input ``x_t`` (3 values) and previous hidden state ``h_{t-1}`` (16 values)::

    z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)
    r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)
    n_t = tanh(W_n x_t + U_n (r_t * h_{t-1}) + b_n)
    h_t = z_t * h_{t-1} + (1 - z_t) * n_t
    o_t = W_o h_t + b_o

Note the update-gate polarity: ``z_t`` is the *keep* gate. This is the
convention used throughout the forward pass, the backward pass, the tests, the
documentation and the serialized state; nothing here mixes conventions.

One bias vector per gate, not PyTorch's two. Parameter count::

    3 * (I*H + H*H + H) = 3 * (3*16 + 16*16 + 16) = 3 * 320 = 960
    H*2 + 2             = 16*2 + 2                =            34
                                                     total    994

Outputs
-------
``o_t[0]``
    the predicted next displacement in public normalized units (see
    :mod:`research.aaa_1k.features`). The evaluator converts it to a position.
``o_t[1]``
    a raw score turned into a non-negative predicted error magnitude by
    softplus. This is a learned predictive-error-magnitude estimate. It is not
    a Bayesian posterior and is not calibrated by construction.

Learning
--------
Truncated backpropagation through time. On each step the gradient of *that
step's* loss is propagated back through at most ``tbptt_steps`` stored hidden
transitions. Each step's loss is therefore counted exactly once, and the
truncation is the only approximation. The stored activations are the ones the
model actually realized, so the gradient is exact for the realized trajectory
up to the truncation horizon; it is *not* the exact online gradient, because
the influence of parameters on the hidden state more than ``tbptt_steps`` in
the past is discarded (see Tallec & Ollivier 2017 for why that bias can
matter).

The feature vector recorded for a step is a constant as far as gradients are
concerned, even though its third component was built from an earlier
prediction. Differentiating through the model's own history of input
construction would create a gradient shortcut that has nothing to do with
predicting the world.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MODEL_FORMAT_VERSION = "aaa.1k.gru.v1"
ARCHITECTURE_ID = "AAA1KGRU-3x16x2"

INPUT_SIZE = 3
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2

GATE_NAMES = ("z", "r", "n")
PARAMETER_NAMES = (
    "W_z",
    "U_z",
    "b_z",
    "W_r",
    "U_r",
    "b_r",
    "W_n",
    "U_n",
    "b_n",
    "W_o",
    "b_o",
)
RECURRENT_PARAMETER_NAMES = ("U_z", "U_r", "U_n")


def expected_parameter_count(
    input_size: int = INPUT_SIZE, hidden_size: int = HIDDEN_SIZE, output_size: int = OUTPUT_SIZE
) -> int:
    """The declared parameter count, recomputed from the architecture formula.

    Kept separate from the arrays on purpose: the test suite compares this
    formula against the actual array sizes, so agreement is evidence rather
    than a restatement.
    """

    gates = 3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    head = hidden_size * output_size + output_size
    return gates + head


def sigmoid(value: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""

    out = np.empty_like(value, dtype=float)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def softplus(value: np.ndarray) -> np.ndarray:
    """Numerically stable ``log(1 + exp(v))``."""

    return np.logaddexp(np.zeros_like(value, dtype=float), np.asarray(value, dtype=float))


class InvalidModelState(ValueError):
    """Raised when model state violates a required invariant."""


@dataclass
class StepCache:
    """Everything the backward pass needs about one realized transition."""

    x: np.ndarray
    h_prev: np.ndarray
    z: np.ndarray
    r: np.ndarray
    n: np.ndarray
    hr: np.ndarray
    h: np.ndarray
    output: np.ndarray


def _glorot(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + columns))
    return rng.uniform(-limit, limit, size=(rows, columns))


class AAA1KGRU:
    """A 994-parameter gated recurrent predictor with a self-error head.

    Parameters
    ----------
    seed:
        Model-local initialization seed. Deliberately independent of every
        environment, benchmark and bootstrap seed namespace.
    learning_rate, tbptt_steps:
        Plain SGD step size and truncation horizon.
    error_loss_weight:
        ``lambda_error`` in ``L = L_prediction + lambda * L_error_estimate``.
    gradient_clip:
        Global gradient-norm clip, or ``None`` for no clipping. When enabled,
        activations of the clip are counted and reported; the threshold is a
        declared mechanism, not a quiet safety net.
    freeze_recurrent:
        Ablation. Leaves ``U_z``, ``U_r`` and ``U_n`` at their initial values.
    reset_state_every_step:
        Ablation. Clears the hidden state before every forward pass, turning
        the model into a stateless function of the current input.
    zero_error_input:
        Ablation. Forces input 3 (the previous signed prediction error) to
        zero at every step.
    """

    format_version = MODEL_FORMAT_VERSION
    architecture_id = ARCHITECTURE_ID

    def __init__(
        self,
        *,
        seed: int = 0,
        learning_rate: float = 0.01,
        tbptt_steps: int = 8,
        error_loss_weight: float = 0.25,
        gradient_clip: float | None = 1.0,
        freeze_recurrent: bool = False,
        reset_state_every_step: bool = False,
        zero_error_input: bool = False,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise InvalidModelState("seed must be an integer")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise InvalidModelState("learning_rate must be positive and finite")
        if isinstance(tbptt_steps, bool) or not isinstance(tbptt_steps, (int, np.integer)):
            raise InvalidModelState("tbptt_steps must be an integer")
        if int(tbptt_steps) < 1:
            raise InvalidModelState("tbptt_steps must be at least 1")
        if not math.isfinite(float(error_loss_weight)) or error_loss_weight < 0:
            raise InvalidModelState("error_loss_weight must be finite and non-negative")
        if gradient_clip is not None and (
            not math.isfinite(float(gradient_clip)) or float(gradient_clip) <= 0
        ):
            raise InvalidModelState("gradient_clip must be positive and finite, or None")

        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.tbptt_steps = int(tbptt_steps)
        self.error_loss_weight = float(error_loss_weight)
        self.gradient_clip = None if gradient_clip is None else float(gradient_clip)
        self.freeze_recurrent = bool(freeze_recurrent)
        self.reset_state_every_step = bool(reset_state_every_step)
        self.zero_error_input = bool(zero_error_input)

        if parameters is None:
            self.parameters = self._initialize(self.seed)
        else:
            self.parameters = self._adopt(parameters)

        self.hidden = np.zeros(HIDDEN_SIZE, dtype=float)
        self._caches: deque[StepCache] = deque(maxlen=self.tbptt_steps)
        self.update_count = 0
        self.forward_count = 0
        self.clip_events = 0
        self.nonfinite_events = 0

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @staticmethod
    def _initialize(seed: int) -> dict[str, np.ndarray]:
        """Deterministic initialization from a model-local RNG.

        Input and recurrent matrices use Glorot-uniform limits. All gate biases
        are zero. The output head starts at exactly zero, so an untrained model
        predicts persistence -- the same starting behaviour as AAA's existing
        linear learners, which makes "did it learn anything" a fair question
        from step one. The error head therefore starts at ``softplus(0) = ln 2``
        in normalized displacement units, which is large; it is trained down
        within the first few dozen updates and is reported rather than hidden.
        """

        rng = np.random.default_rng(seed)
        parameters: dict[str, np.ndarray] = {}
        for gate in GATE_NAMES:
            parameters[f"W_{gate}"] = _glorot(rng, HIDDEN_SIZE, INPUT_SIZE)
            parameters[f"U_{gate}"] = _glorot(rng, HIDDEN_SIZE, HIDDEN_SIZE)
            parameters[f"b_{gate}"] = np.zeros(HIDDEN_SIZE, dtype=float)
        parameters["W_o"] = np.zeros((OUTPUT_SIZE, HIDDEN_SIZE), dtype=float)
        parameters["b_o"] = np.zeros(OUTPUT_SIZE, dtype=float)
        return parameters

    @staticmethod
    def _adopt(parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
        expected = {
            **{f"W_{gate}": (HIDDEN_SIZE, INPUT_SIZE) for gate in GATE_NAMES},
            **{f"U_{gate}": (HIDDEN_SIZE, HIDDEN_SIZE) for gate in GATE_NAMES},
            **{f"b_{gate}": (HIDDEN_SIZE,) for gate in GATE_NAMES},
            "W_o": (OUTPUT_SIZE, HIDDEN_SIZE),
            "b_o": (OUTPUT_SIZE,),
        }
        unknown = sorted(set(parameters) - set(expected))
        missing = sorted(set(expected) - set(parameters))
        if unknown or missing:
            raise InvalidModelState(f"invalid parameter set: unknown={unknown}, missing={missing}")
        adopted: dict[str, np.ndarray] = {}
        for name, shape in expected.items():
            array = np.asarray(parameters[name], dtype=float)
            if array.shape != shape:
                raise InvalidModelState(f"parameter {name} must have shape {shape}, got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise InvalidModelState(f"parameter {name} contains non-finite values")
            adopted[name] = array.copy()
        return adopted

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        """Total trainable scalars, summed from the actual arrays."""

        return int(sum(int(array.size) for array in self.parameters.values()))

    def parameter_inventory(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"shape": list(self.parameters[name].shape), "count": int(self.parameters[name].size)}
            for name in PARAMETER_NAMES
        }

    def state_footprint(self) -> dict[str, int]:
        """Every adaptive scalar the model carries, counted by category.

        A "1K model" whose optimizer secretly holds another 100K values would
        deserve ridicule, so the optimizer state is counted separately and is
        zero here: plain SGD has none.
        """

        cache_scalars = sum(
            int(cache.x.size + cache.h_prev.size + cache.z.size + cache.r.size + cache.n.size)
            + int(cache.hr.size + cache.h.size + cache.output.size)
            for cache in self._caches
        )
        trainable = self.parameter_count()
        return {
            "trainable_parameters": trainable,
            "hidden_state_scalars": int(self.hidden.size),
            "optimizer_state_scalars": 0,
            "tbptt_buffer_scalars": int(cache_scalars),
            "total_adaptive_state_scalars": trainable + int(self.hidden.size) + int(cache_scalars),
        }

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def reset_state(self) -> None:
        """Begin a new independent episode: clear hidden state and TBPTT buffer."""

        self.hidden = np.zeros(HIDDEN_SIZE, dtype=float)
        self._caches.clear()

    def forward(self, inputs: Sequence[float], *, record: bool = True) -> np.ndarray:
        """Advance the hidden state by one step and return ``[displacement, raw_error]``.

        This is the only place the live hidden state advances. A frozen clone
        calls it exactly as an online model does, which is the point: freezing
        means the weights stop changing, not that the memory is destroyed.
        """

        x = np.asarray(inputs, dtype=float)
        if x.shape != (INPUT_SIZE,):
            raise InvalidModelState(f"input must have shape ({INPUT_SIZE},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise InvalidModelState("input contains non-finite values")
        if self.zero_error_input:
            x = x.copy()
            x[2] = 0.0
        if self.reset_state_every_step:
            self.hidden = np.zeros(HIDDEN_SIZE, dtype=float)
            self._caches.clear()

        p = self.parameters
        h_prev = self.hidden
        z = sigmoid(p["W_z"] @ x + p["U_z"] @ h_prev + p["b_z"])
        r = sigmoid(p["W_r"] @ x + p["U_r"] @ h_prev + p["b_r"])
        hr = r * h_prev
        n = np.tanh(p["W_n"] @ x + p["U_n"] @ hr + p["b_n"])
        h = z * h_prev + (1.0 - z) * n
        output = p["W_o"] @ h + p["b_o"]
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(output)):
            self.nonfinite_events += 1
            raise FloatingPointError("AAA1KGRU forward pass produced non-finite values")

        self.hidden = h
        self.forward_count += 1
        if record:
            self._caches.append(
                StepCache(x=x.copy(), h_prev=h_prev.copy(), z=z, r=r, n=n, hr=hr, h=h, output=output)
            )
        return output

    @staticmethod
    def split_output(output: Sequence[float]) -> tuple[float, float]:
        """Return ``(predicted normalized displacement, predicted error magnitude)``."""

        values = np.asarray(output, dtype=float)
        return float(values[0]), float(softplus(values[1:2])[0])

    # ------------------------------------------------------------------
    # loss and backward
    # ------------------------------------------------------------------
    def step_loss(self, output: Sequence[float], target_displacement: float) -> dict[str, float]:
        """Return the two loss terms and the total for one step.

        The auxiliary target is ``|o0 - d*|`` with stop-gradient semantics: the
        model is asked how wrong it expects to be, not allowed to move the
        target by differentiating through its own prediction.
        """

        values = np.asarray(output, dtype=float)
        signed = float(values[0]) - float(target_displacement)
        estimate = float(softplus(values[1:2])[0])
        prediction_loss = signed * signed
        error_target = abs(signed)
        error_loss = (estimate - error_target) ** 2
        return {
            "prediction_loss": prediction_loss,
            "error_loss": error_loss,
            "total_loss": prediction_loss + self.error_loss_weight * error_loss,
            "error_target": error_target,
            "error_estimate": estimate,
        }

    def _output_gradient(self, output: np.ndarray, target_displacement: float) -> np.ndarray:
        signed = float(output[0]) - float(target_displacement)
        estimate = float(softplus(output[1:2])[0])
        # stop-gradient on the auxiliary target
        error_target = abs(signed)
        grad = np.zeros(OUTPUT_SIZE, dtype=float)
        grad[0] = 2.0 * signed
        grad[1] = (
            self.error_loss_weight
            * 2.0
            * (estimate - error_target)
            * float(sigmoid(output[1:2])[0])  # d softplus / d raw
        )
        return grad

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        """Gradient of the most recent step's loss, truncated to the stored window.

        The hidden state at the start of the window is treated as a constant,
        which is exactly the truncation. Older caches are never consulted.
        """

        if not self._caches:
            raise InvalidModelState("backward() requires at least one recorded forward step")
        gradients = {name: np.zeros_like(array) for name, array in self.parameters.items()}
        p = self.parameters
        window = list(self._caches)
        last = window[-1]

        d_output = self._output_gradient(last.output, target_displacement)
        gradients["W_o"] += np.outer(d_output, last.h)
        gradients["b_o"] += d_output
        d_h = p["W_o"].T @ d_output

        for cache in reversed(window):
            d_z = d_h * (cache.h_prev - cache.n)
            d_n = d_h * (1.0 - cache.z)
            d_h_prev = d_h * cache.z

            d_a_n = d_n * (1.0 - cache.n * cache.n)
            gradients["W_n"] += np.outer(d_a_n, cache.x)
            gradients["U_n"] += np.outer(d_a_n, cache.hr)
            gradients["b_n"] += d_a_n
            d_hr = p["U_n"].T @ d_a_n

            d_r = d_hr * cache.h_prev
            d_h_prev = d_h_prev + d_hr * cache.r

            d_a_r = d_r * cache.r * (1.0 - cache.r)
            gradients["W_r"] += np.outer(d_a_r, cache.x)
            gradients["U_r"] += np.outer(d_a_r, cache.h_prev)
            gradients["b_r"] += d_a_r
            d_h_prev = d_h_prev + p["U_r"].T @ d_a_r

            d_a_z = d_z * cache.z * (1.0 - cache.z)
            gradients["W_z"] += np.outer(d_a_z, cache.x)
            gradients["U_z"] += np.outer(d_a_z, cache.h_prev)
            gradients["b_z"] += d_a_z
            d_h_prev = d_h_prev + p["U_z"].T @ d_a_z

            d_h = d_h_prev
        return gradients

    # ------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------
    @staticmethod
    def gradient_norm(gradients: Mapping[str, np.ndarray]) -> float:
        return float(math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values())))

    def apply_gradients(self, gradients: Mapping[str, np.ndarray]) -> dict[str, float]:
        """Plain SGD, with optional declared global-norm clipping.

        A non-finite gradient fails loudly. Divergence is evidence; it is never
        silently repaired by resetting the model.
        """

        norm = self.gradient_norm(gradients)
        if not math.isfinite(norm):
            self.nonfinite_events += 1
            raise FloatingPointError("AAA1KGRU produced a non-finite gradient")
        scale = 1.0
        if self.gradient_clip is not None and norm > self.gradient_clip:
            scale = self.gradient_clip / norm
            self.clip_events += 1
        for name, array in self.parameters.items():
            if self.freeze_recurrent and name in RECURRENT_PARAMETER_NAMES:
                continue
            array -= self.learning_rate * scale * gradients[name]
        if not all(np.all(np.isfinite(array)) for array in self.parameters.values()):
            self.nonfinite_events += 1
            raise FloatingPointError("AAA1KGRU update produced non-finite parameters")
        self.update_count += 1
        return {
            "gradient_norm": norm,
            "clip_scale": scale,
            "parameter_norm": float(
                math.sqrt(sum(float(np.sum(a * a)) for a in self.parameters.values()))
            ),
        }

    def learn(self, target_displacement: float) -> dict[str, float]:
        """Backward plus one SGD step for the most recent forward pass."""

        gradients = self.backward(target_displacement)
        return self.apply_gradients(gradients)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, float]:
        """Live numerical health, read-only."""

        last = self._caches[-1] if self._caches else None
        return {
            "hidden_max_abs": float(np.max(np.abs(self.hidden))) if self.hidden.size else 0.0,
            "hidden_saturated_fraction": float(np.mean(np.abs(self.hidden) > 0.99)),
            "update_gate_mean": float(np.mean(last.z)) if last is not None else float("nan"),
            "reset_gate_mean": float(np.mean(last.r)) if last is not None else float("nan"),
            "gate_saturated_fraction": (
                float(np.mean((last.z > 0.99) | (last.z < 0.01) | (last.r > 0.99) | (last.r < 0.01)))
                if last is not None
                else float("nan")
            ),
            "parameter_norm": float(math.sqrt(sum(float(np.sum(a * a)) for a in self.parameters.values()))),
            "update_count": float(self.update_count),
            "clip_events": float(self.clip_events),
        }

    def last_gates(self) -> dict[str, list[float]]:
        """Update and reset gate vectors from the most recent forward pass."""

        if not self._caches:
            return {"update_gate": [], "reset_gate": []}
        last = self._caches[-1]
        return {
            "update_gate": [float(value) for value in last.z],
            "reset_gate": [float(value) for value in last.r],
        }

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def state_dict(self, *, parent_model_id: str | None = None) -> dict[str, Any]:
        """Complete state, sufficient for an exact resume.

        Includes the trainable weights, the hidden state, the TBPTT buffer, all
        counters and the configuration. The previous signed prediction error is
        part of the *adapter* state and is serialized by
        :class:`research.aaa_1k.adapters.AAA1KPredictor`, which owns it.
        """

        return {
            "format_version": self.format_version,
            "architecture_id": self.architecture_id,
            "parameter_count": self.parameter_count(),
            "parent_model_id": parent_model_id,
            "config": {
                "seed": self.seed,
                "learning_rate": self.learning_rate,
                "tbptt_steps": self.tbptt_steps,
                "error_loss_weight": self.error_loss_weight,
                "gradient_clip": self.gradient_clip,
                "freeze_recurrent": self.freeze_recurrent,
                "reset_state_every_step": self.reset_state_every_step,
                "zero_error_input": self.zero_error_input,
            },
            "parameters": {name: self.parameters[name].tolist() for name in PARAMETER_NAMES},
            "hidden": self.hidden.tolist(),
            "tbptt_buffer": [
                {
                    "x": cache.x.tolist(),
                    "h_prev": cache.h_prev.tolist(),
                    "z": cache.z.tolist(),
                    "r": cache.r.tolist(),
                    "n": cache.n.tolist(),
                    "hr": cache.hr.tolist(),
                    "h": cache.h.tolist(),
                    "output": cache.output.tolist(),
                }
                for cache in self._caches
            ],
            "counters": {
                "update_count": self.update_count,
                "forward_count": self.forward_count,
                "clip_events": self.clip_events,
                "nonfinite_events": self.nonfinite_events,
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> AAA1KGRU:
        if state.get("format_version") != MODEL_FORMAT_VERSION:
            raise InvalidModelState(
                f"unsupported model format {state.get('format_version')!r}; expected {MODEL_FORMAT_VERSION!r}"
            )
        if state.get("architecture_id") != ARCHITECTURE_ID:
            raise InvalidModelState("checkpoint architecture does not match AAA1KGRU")
        config = dict(state["config"])
        model = cls(
            seed=int(config["seed"]),
            learning_rate=float(config["learning_rate"]),
            tbptt_steps=int(config["tbptt_steps"]),
            error_loss_weight=float(config["error_loss_weight"]),
            gradient_clip=(None if config["gradient_clip"] is None else float(config["gradient_clip"])),
            freeze_recurrent=bool(config["freeze_recurrent"]),
            reset_state_every_step=bool(config["reset_state_every_step"]),
            zero_error_input=bool(config["zero_error_input"]),
            parameters={name: np.asarray(value, dtype=float) for name, value in state["parameters"].items()},
        )
        hidden = np.asarray(state["hidden"], dtype=float)
        if hidden.shape != (HIDDEN_SIZE,) or not np.all(np.isfinite(hidden)):
            raise InvalidModelState("checkpoint hidden state is invalid")
        model.hidden = hidden
        model._caches = deque(
            (
                StepCache(
                    x=np.asarray(entry["x"], dtype=float),
                    h_prev=np.asarray(entry["h_prev"], dtype=float),
                    z=np.asarray(entry["z"], dtype=float),
                    r=np.asarray(entry["r"], dtype=float),
                    n=np.asarray(entry["n"], dtype=float),
                    hr=np.asarray(entry["hr"], dtype=float),
                    h=np.asarray(entry["h"], dtype=float),
                    output=np.asarray(entry["output"], dtype=float),
                )
                for entry in state["tbptt_buffer"]
            ),
            maxlen=model.tbptt_steps,
        )
        counters = dict(state["counters"])
        model.update_count = int(counters["update_count"])
        model.forward_count = int(counters["forward_count"])
        model.clip_events = int(counters["clip_events"])
        model.nonfinite_events = int(counters["nonfinite_events"])
        return model

    def clone(self) -> AAA1KGRU:
        """An independent deep copy: no shared mutable arrays, identical state."""

        return self.from_state_dict(json.loads(json.dumps(self.state_dict())))

    def state_hash(self) -> str:
        """SHA-256 over the canonical complete state."""

        payload = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path, *, parent_model_id: str | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.state_dict(parent_model_id=parent_model_id), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> AAA1KGRU:
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
