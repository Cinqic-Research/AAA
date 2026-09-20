"""Matched-capacity neural controls, so "it has neurons" never counts as a win.

Two controls share AAA-1K's inputs, outputs, loss, optimizer and evaluation
path, and differ in exactly one mechanism each:

``StatelessMLPControl`` -- 982 parameters, ``3 -> 28 -> 28 -> 2`` with tanh
    hidden layers. Same capacity, no persistent state. Isolates *memory*.

``VanillaRNNControl`` -- 954 parameters, ``3 -> 28 tanh recurrent -> 2``.
    Persistent state, no gates. Isolates *gating*.

Both expose the same interface as :class:`research.aaa_1k.model.AAA1KGRU`, so
the training loop, the serializer and the evaluator cannot tell them apart.
This mirrors the ablation design in Foucault & Meyniel (2021), which tested
mechanisms -- gating, lateral connections, recurrent weight training -- rather
than admiring a single architecture.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .model import INPUT_SIZE, OUTPUT_SIZE, InvalidModelState, sigmoid, softplus

ControlT = TypeVar("ControlT", bound="_NeuralControl")

MLP_FORMAT_VERSION = "aaa.1k.mlp.v1"
RNN_FORMAT_VERSION = "aaa.1k.rnn.v1"
MLP_ARCHITECTURE_ID = "StatelessMLP-3x28x28x2"
RNN_ARCHITECTURE_ID = "VanillaRNN-3x28x2"
MLP_HIDDEN = 28
RNN_HIDDEN = 28


def expected_mlp_parameter_count(hidden: int = MLP_HIDDEN) -> int:
    return INPUT_SIZE * hidden + hidden + hidden * hidden + hidden + hidden * OUTPUT_SIZE + OUTPUT_SIZE


def expected_rnn_parameter_count(hidden: int = RNN_HIDDEN) -> int:
    return INPUT_SIZE * hidden + hidden * hidden + hidden + hidden * OUTPUT_SIZE + OUTPUT_SIZE


def _glorot(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    limit = math.sqrt(6.0 / (rows + columns))
    return rng.uniform(-limit, limit, size=(rows, columns))


class _NeuralControl:
    """Shared loss, optimizer, accounting and serialization for both controls."""

    format_version = ""
    architecture_id = ""
    parameter_names: tuple[str, ...] = ()
    recurrent_parameter_names: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        seed: int,
        learning_rate: float,
        tbptt_steps: int,
        error_loss_weight: float,
        gradient_clip: float | None,
        zero_error_input: bool,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise InvalidModelState("seed must be an integer")
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise InvalidModelState("learning_rate must be positive and finite")
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
        self.zero_error_input = bool(zero_error_input)
        self.freeze_recurrent = False
        self.reset_state_every_step = False
        self.parameters = self._initialize(self.seed) if parameters is None else self._adopt(parameters)
        self.update_count = 0
        self.forward_count = 0
        self.clip_events = 0
        self.nonfinite_events = 0

    # -- construction ---------------------------------------------------
    @staticmethod
    def _initialize(seed: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        raise NotImplementedError

    def _adopt(self, parameters: Mapping[str, Any]) -> dict[str, np.ndarray]:
        expected = self._shapes()
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

    # -- accounting -----------------------------------------------------
    def parameter_count(self) -> int:
        return int(sum(int(array.size) for array in self.parameters.values()))

    def parameter_inventory(self) -> dict[str, dict[str, Any]]:
        return {
            name: {"shape": list(self.parameters[name].shape), "count": int(self.parameters[name].size)}
            for name in self.parameter_names
        }

    # -- loss -----------------------------------------------------------
    @staticmethod
    def split_output(output: ArrayLike) -> tuple[float, float]:
        values = np.asarray(output, dtype=float)
        return float(values[0]), float(softplus(values[1:2])[0])

    def step_loss(self, output: ArrayLike, target_displacement: float) -> dict[str, float]:
        values = np.asarray(output, dtype=float)
        signed = float(values[0]) - float(target_displacement)
        estimate = float(softplus(values[1:2])[0])
        error_target = abs(signed)
        error_loss = (estimate - error_target) ** 2
        return {
            "prediction_loss": signed * signed,
            "error_loss": error_loss,
            "total_loss": signed * signed + self.error_loss_weight * error_loss,
            "error_target": error_target,
            "error_estimate": estimate,
        }

    def _output_gradient(self, output: np.ndarray, target_displacement: float) -> np.ndarray:
        signed = float(output[0]) - float(target_displacement)
        estimate = float(softplus(output[1:2])[0])
        grad = np.zeros(OUTPUT_SIZE, dtype=float)
        grad[0] = 2.0 * signed
        grad[1] = self.error_loss_weight * 2.0 * (estimate - abs(signed)) * float(sigmoid(output[1:2])[0])
        return grad

    # -- optimization ---------------------------------------------------
    @staticmethod
    def gradient_norm(gradients: Mapping[str, np.ndarray]) -> float:
        return float(math.sqrt(sum(float(np.sum(value * value)) for value in gradients.values())))

    def apply_gradients(self, gradients: Mapping[str, np.ndarray]) -> dict[str, float]:
        norm = self.gradient_norm(gradients)
        if not math.isfinite(norm):
            self.nonfinite_events += 1
            raise FloatingPointError(f"{self.architecture_id} produced a non-finite gradient")
        scale = 1.0
        if self.gradient_clip is not None and norm > self.gradient_clip:
            scale = self.gradient_clip / norm
            self.clip_events += 1
        for name, array in self.parameters.items():
            array -= self.learning_rate * scale * gradients[name]
        if not all(np.all(np.isfinite(array)) for array in self.parameters.values()):
            self.nonfinite_events += 1
            raise FloatingPointError(f"{self.architecture_id} update produced non-finite parameters")
        self.update_count += 1
        return {
            "gradient_norm": norm,
            "clip_scale": scale,
            "parameter_norm": float(math.sqrt(sum(float(np.sum(a * a)) for a in self.parameters.values()))),
        }

    def learn(self, target_displacement: float) -> dict[str, float]:
        return self.apply_gradients(self.backward(target_displacement))

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        raise NotImplementedError

    # -- serialization --------------------------------------------------
    def _extra_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def state_dict(self, *, parent_model_id: str | None = None) -> dict[str, Any]:
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
                "freeze_recurrent": False,
                "reset_state_every_step": False,
                "zero_error_input": self.zero_error_input,
            },
            "parameters": {name: self.parameters[name].tolist() for name in self.parameter_names},
            "counters": {
                "update_count": self.update_count,
                "forward_count": self.forward_count,
                "clip_events": self.clip_events,
                "nonfinite_events": self.nonfinite_events,
            },
            **self._extra_state(),
        }

    def state_hash(self) -> str:
        payload = json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_state_dict(cls: type[ControlT], state: Mapping[str, Any]) -> ControlT:
        raise NotImplementedError

    def clone(self: ControlT) -> ControlT:
        return type(self).from_state_dict(json.loads(json.dumps(self.state_dict())))

    def state_footprint(self) -> dict[str, int]:
        raise NotImplementedError

    def diagnostics(self) -> dict[str, float]:
        raise NotImplementedError


@dataclass
class _MLPCache:
    x: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    output: np.ndarray


class StatelessMLPControl(_NeuralControl):
    """982-parameter feed-forward control: same capacity, no memory."""

    format_version = MLP_FORMAT_VERSION
    architecture_id = MLP_ARCHITECTURE_ID
    parameter_names = ("W1", "b1", "W2", "b2", "W_o", "b_o")

    def __init__(
        self,
        *,
        seed: int = 0,
        learning_rate: float = 0.01,
        tbptt_steps: int = 1,
        error_loss_weight: float = 0.25,
        gradient_clip: float | None = 1.0,
        zero_error_input: bool = False,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            seed=seed,
            learning_rate=learning_rate,
            tbptt_steps=tbptt_steps,
            error_loss_weight=error_loss_weight,
            gradient_clip=gradient_clip,
            zero_error_input=zero_error_input,
            parameters=parameters,
        )
        self._cache: _MLPCache | None = None

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "W1": (MLP_HIDDEN, INPUT_SIZE),
            "b1": (MLP_HIDDEN,),
            "W2": (MLP_HIDDEN, MLP_HIDDEN),
            "b2": (MLP_HIDDEN,),
            "W_o": (OUTPUT_SIZE, MLP_HIDDEN),
            "b_o": (OUTPUT_SIZE,),
        }

    @staticmethod
    def _initialize(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "W1": _glorot(rng, MLP_HIDDEN, INPUT_SIZE),
            "b1": np.zeros(MLP_HIDDEN, dtype=float),
            "W2": _glorot(rng, MLP_HIDDEN, MLP_HIDDEN),
            "b2": np.zeros(MLP_HIDDEN, dtype=float),
            "W_o": np.zeros((OUTPUT_SIZE, MLP_HIDDEN), dtype=float),
            "b_o": np.zeros(OUTPUT_SIZE, dtype=float),
        }

    def reset_state(self) -> None:
        self._cache = None

    def forward(self, inputs: ArrayLike, *, record: bool = True) -> np.ndarray:
        x = np.asarray(inputs, dtype=float)
        if x.shape != (INPUT_SIZE,):
            raise InvalidModelState(f"input must have shape ({INPUT_SIZE},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise InvalidModelState("input contains non-finite values")
        if self.zero_error_input:
            x = x.copy()
            x[2] = 0.0
        p = self.parameters
        a1 = np.tanh(p["W1"] @ x + p["b1"])
        a2 = np.tanh(p["W2"] @ a1 + p["b2"])
        output = p["W_o"] @ a2 + p["b_o"]
        if not np.all(np.isfinite(output)):
            self.nonfinite_events += 1
            raise FloatingPointError("StatelessMLPControl produced non-finite values")
        self.forward_count += 1
        if record:
            self._cache = _MLPCache(x=x.copy(), a1=a1, a2=a2, output=output)
        return output

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        if self._cache is None:
            raise InvalidModelState("backward() requires a recorded forward step")
        cache = self._cache
        p = self.parameters
        gradients = {name: np.zeros_like(array) for name, array in p.items()}
        d_output = self._output_gradient(cache.output, target_displacement)
        gradients["W_o"] += np.outer(d_output, cache.a2)
        gradients["b_o"] += d_output
        d_a2 = (p["W_o"].T @ d_output) * (1.0 - cache.a2 * cache.a2)
        gradients["W2"] += np.outer(d_a2, cache.a1)
        gradients["b2"] += d_a2
        d_a1 = (p["W2"].T @ d_a2) * (1.0 - cache.a1 * cache.a1)
        gradients["W1"] += np.outer(d_a1, cache.x)
        gradients["b1"] += d_a1
        return gradients

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def state_footprint(self) -> dict[str, int]:
        trainable = self.parameter_count()
        return {
            "trainable_parameters": trainable,
            "hidden_state_scalars": 0,
            "optimizer_state_scalars": 0,
            "tbptt_buffer_scalars": 0,
            "total_adaptive_state_scalars": trainable,
        }

    def diagnostics(self) -> dict[str, float]:
        return {
            "hidden_max_abs": float(np.max(np.abs(self._cache.a2))) if self._cache else 0.0,
            "parameter_norm": float(math.sqrt(sum(float(np.sum(a * a)) for a in self.parameters.values()))),
            "update_count": float(self.update_count),
            "clip_events": float(self.clip_events),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> StatelessMLPControl:
        if state.get("format_version") != MLP_FORMAT_VERSION:
            raise InvalidModelState("unsupported StatelessMLPControl checkpoint format")
        config = dict(state["config"])
        model = cls(
            seed=int(config["seed"]),
            learning_rate=float(config["learning_rate"]),
            tbptt_steps=int(config["tbptt_steps"]),
            error_loss_weight=float(config["error_loss_weight"]),
            gradient_clip=(None if config["gradient_clip"] is None else float(config["gradient_clip"])),
            zero_error_input=bool(config["zero_error_input"]),
            parameters={name: np.asarray(v, dtype=float) for name, v in state["parameters"].items()},
        )
        counters = dict(state["counters"])
        model.update_count = int(counters["update_count"])
        model.forward_count = int(counters["forward_count"])
        model.clip_events = int(counters["clip_events"])
        model.nonfinite_events = int(counters["nonfinite_events"])
        return model


@dataclass
class _RNNCache:
    x: np.ndarray
    h_prev: np.ndarray
    h: np.ndarray
    output: np.ndarray


class VanillaRNNControl(_NeuralControl):
    """954-parameter ungated recurrent control: memory without gates."""

    format_version = RNN_FORMAT_VERSION
    architecture_id = RNN_ARCHITECTURE_ID
    parameter_names = ("W", "U", "b", "W_o", "b_o")
    recurrent_parameter_names = ("U",)

    def __init__(
        self,
        *,
        seed: int = 0,
        learning_rate: float = 0.01,
        tbptt_steps: int = 8,
        error_loss_weight: float = 0.25,
        gradient_clip: float | None = 1.0,
        zero_error_input: bool = False,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            seed=seed,
            learning_rate=learning_rate,
            tbptt_steps=tbptt_steps,
            error_loss_weight=error_loss_weight,
            gradient_clip=gradient_clip,
            zero_error_input=zero_error_input,
            parameters=parameters,
        )
        self.hidden = np.zeros(RNN_HIDDEN, dtype=float)
        self._caches: deque[_RNNCache] = deque(maxlen=self.tbptt_steps)

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "W": (RNN_HIDDEN, INPUT_SIZE),
            "U": (RNN_HIDDEN, RNN_HIDDEN),
            "b": (RNN_HIDDEN,),
            "W_o": (OUTPUT_SIZE, RNN_HIDDEN),
            "b_o": (OUTPUT_SIZE,),
        }

    @staticmethod
    def _initialize(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            "W": _glorot(rng, RNN_HIDDEN, INPUT_SIZE),
            "U": _glorot(rng, RNN_HIDDEN, RNN_HIDDEN),
            "b": np.zeros(RNN_HIDDEN, dtype=float),
            "W_o": np.zeros((OUTPUT_SIZE, RNN_HIDDEN), dtype=float),
            "b_o": np.zeros(OUTPUT_SIZE, dtype=float),
        }

    def reset_state(self) -> None:
        self.hidden = np.zeros(RNN_HIDDEN, dtype=float)
        self._caches.clear()

    def forward(self, inputs: ArrayLike, *, record: bool = True) -> np.ndarray:
        x = np.asarray(inputs, dtype=float)
        if x.shape != (INPUT_SIZE,):
            raise InvalidModelState(f"input must have shape ({INPUT_SIZE},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise InvalidModelState("input contains non-finite values")
        if self.zero_error_input:
            x = x.copy()
            x[2] = 0.0
        p = self.parameters
        h_prev = self.hidden
        h = np.tanh(p["W"] @ x + p["U"] @ h_prev + p["b"])
        output = p["W_o"] @ h + p["b_o"]
        if not np.all(np.isfinite(output)):
            self.nonfinite_events += 1
            raise FloatingPointError("VanillaRNNControl produced non-finite values")
        self.hidden = h
        self.forward_count += 1
        if record:
            self._caches.append(_RNNCache(x=x.copy(), h_prev=h_prev.copy(), h=h, output=output))
        return output

    def backward(self, target_displacement: float) -> dict[str, np.ndarray]:
        if not self._caches:
            raise InvalidModelState("backward() requires at least one recorded forward step")
        p = self.parameters
        gradients = {name: np.zeros_like(array) for name, array in p.items()}
        window = list(self._caches)
        last = window[-1]
        d_output = self._output_gradient(last.output, target_displacement)
        gradients["W_o"] += np.outer(d_output, last.h)
        gradients["b_o"] += d_output
        d_h = p["W_o"].T @ d_output
        for cache in reversed(window):
            d_a = d_h * (1.0 - cache.h * cache.h)
            gradients["W"] += np.outer(d_a, cache.x)
            gradients["U"] += np.outer(d_a, cache.h_prev)
            gradients["b"] += d_a
            d_h = p["U"].T @ d_a
        return gradients

    def _extra_state(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden.tolist(),
            "tbptt_buffer": [
                {
                    "x": cache.x.tolist(),
                    "h_prev": cache.h_prev.tolist(),
                    "h": cache.h.tolist(),
                    "output": cache.output.tolist(),
                }
                for cache in self._caches
            ],
        }

    def state_footprint(self) -> dict[str, int]:
        cache_scalars = sum(int(c.x.size + c.h_prev.size + c.h.size + c.output.size) for c in self._caches)
        trainable = self.parameter_count()
        return {
            "trainable_parameters": trainable,
            "hidden_state_scalars": int(self.hidden.size),
            "optimizer_state_scalars": 0,
            "tbptt_buffer_scalars": int(cache_scalars),
            "total_adaptive_state_scalars": trainable + int(self.hidden.size) + int(cache_scalars),
        }

    def diagnostics(self) -> dict[str, float]:
        return {
            "hidden_max_abs": float(np.max(np.abs(self.hidden))),
            "hidden_saturated_fraction": float(np.mean(np.abs(self.hidden) > 0.99)),
            "parameter_norm": float(math.sqrt(sum(float(np.sum(a * a)) for a in self.parameters.values()))),
            "update_count": float(self.update_count),
            "clip_events": float(self.clip_events),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> VanillaRNNControl:
        if state.get("format_version") != RNN_FORMAT_VERSION:
            raise InvalidModelState("unsupported VanillaRNNControl checkpoint format")
        config = dict(state["config"])
        model = cls(
            seed=int(config["seed"]),
            learning_rate=float(config["learning_rate"]),
            tbptt_steps=int(config["tbptt_steps"]),
            error_loss_weight=float(config["error_loss_weight"]),
            gradient_clip=(None if config["gradient_clip"] is None else float(config["gradient_clip"])),
            zero_error_input=bool(config["zero_error_input"]),
            parameters={name: np.asarray(v, dtype=float) for name, v in state["parameters"].items()},
        )
        hidden = np.asarray(state["hidden"], dtype=float)
        if hidden.shape != (RNN_HIDDEN,) or not np.all(np.isfinite(hidden)):
            raise InvalidModelState("checkpoint hidden state is invalid")
        model.hidden = hidden
        model._caches = deque(
            (
                _RNNCache(
                    x=np.asarray(entry["x"], dtype=float),
                    h_prev=np.asarray(entry["h_prev"], dtype=float),
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
