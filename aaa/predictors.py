"""Inspectably small prediction rules used in the experiment."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


class Predictor:
    """Minimal predictor interface used by the temporal runner."""

    name: str = "predictor"
    update_enabled: bool = False

    def predict(self, history: Sequence[float]) -> float:
        raise NotImplementedError

    def update(self, history: Sequence[float], target_position: float) -> None:
        return None


def _check_history(history: Sequence[float], minimum: int) -> None:
    if len(history) < minimum:
        raise ValueError(f"predictor requires at least {minimum} observations")
    if not all(math.isfinite(float(value)) for value in history[-minimum:]):
        raise ValueError("history contains a non-finite observation")


def reflect_prediction(position: float, lower: float, upper: float) -> float:
    """Reflect a predicted position into bounds without evaluator metadata."""

    if not all(math.isfinite(value) for value in (position, lower, upper)):
        raise ValueError("prediction and bounds must be finite")
    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    for _ in range(10_000):
        if position > upper:
            position = upper - (position - upper)
        elif position < lower:
            position = lower + (lower - position)
        else:
            return float(position)
    raise ValueError("prediction reflection exceeded safety iteration limit")


class PersistencePredictor(Predictor):
    name = "persistence"

    def predict(self, history: Sequence[float]) -> float:
        if not history:
            raise ValueError("persistence requires at least one observation")
        _check_history(history, 1)
        return float(history[-1])


class ConstantMotionPredictor(Predictor):
    name = "constant_motion"

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < 2:
            raise ValueError("constant_motion requires two observations")
        _check_history(history, 2)
        return float(history[-1] + (history[-1] - history[-2]))


class OnlineLinearPredictor(Predictor):
    """Online linear displacement predictor.

    The feature vector is ``[1, x[t-3], x[t-2], x[t-1], x[t]]``. The model
    predicts the next displacement, which is added to ``x[t]`` to obtain the
    next-position prediction. Its loss is one half of squared displacement
    error, and the update is ordinary single-example gradient descent.
    """

    format_version = "aaa.linear_predictor.v1"

    def __init__(
        self,
        learning_rate: float = 0.08,
        *,
        name: str = "linear_online",
        update_enabled: bool = True,
        weights: Sequence[float] | None = None,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self.name = name
        self.update_enabled = bool(update_enabled)
        self.weights = np.zeros(5, dtype=float) if weights is None else np.asarray(weights, dtype=float).copy()
        if self.weights.shape != (5,):
            raise ValueError("weights must contain exactly five values")
        self.update_count = 0

    @staticmethod
    def features(history: Sequence[float]) -> np.ndarray:
        if len(history) < 4:
            raise ValueError("online linear predictor requires four observations")
        return np.asarray([1.0, *history[-4:]], dtype=float)

    def predict(self, history: Sequence[float]) -> float:
        features = self.features(history)
        displacement = float(np.dot(self.weights, features))
        return float(history[-1] + displacement)

    def update(self, history: Sequence[float], target_position: float) -> None:
        if not self.update_enabled:
            return
        features = self.features(history)
        target_displacement = float(target_position - history[-1])
        predicted_displacement = float(np.dot(self.weights, features))
        # gradient of 0.5 * (predicted_displacement - target)^2
        gradient = (predicted_displacement - target_displacement) * features
        self.weights -= self.learning_rate * gradient
        self.update_count += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "learning_rate": self.learning_rate,
            "history_length": 4,
            "target": "next_displacement",
            "weights": [float(value) for value in self.weights],
            "update_count": self.update_count,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        update_enabled: bool = False,
    ) -> "OnlineLinearPredictor":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(state, name=name, update_enabled=update_enabled)

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, object],
        *,
        name: str | None = None,
        update_enabled: bool = False,
    ) -> "OnlineLinearPredictor":
        if state.get("format_version") != cls.format_version:
            raise ValueError("unsupported linear predictor checkpoint format")
        if state.get("history_length") != 4 or state.get("target") != "next_displacement":
            raise ValueError("checkpoint metadata does not match the AAA predictor")
        model = cls(
            learning_rate=float(state["learning_rate"]),
            name=name or str(state.get("name", "linear_online")),
            update_enabled=update_enabled,
            weights=[float(value) for value in state["weights"]],
        )
        model.update_count = int(state.get("update_count", 0))
        return model


class OnlineRLSPredictor(Predictor):
    """Small normalized recursive-least-squares predictor.

    Its parameters start at zero; the useful velocity relationship is learned
    from observed displacement features. Boundary handling is a deterministic
    observation-format transform, not an evaluator-provided event label.
    """

    format_version = "aaa.rls_predictor.v1"
    name = "adaptive_rls"
    update_enabled = True

    def __init__(
        self,
        *,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        displacement_scale: float = 0.01,
        forgetting: float = 1.0,
        ridge: float = 1e-4,
        name: str = "adaptive_rls",
        update_enabled: bool = True,
        weights: Sequence[float] | None = None,
        covariance: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not all(math.isfinite(float(value)) for value in (lower_bound, upper_bound, displacement_scale, forgetting, ridge)):
            raise ValueError("RLS parameters must be finite")
        if upper_bound <= lower_bound or displacement_scale <= 0 or not 0 < forgetting <= 1 or ridge <= 0:
            raise ValueError("invalid RLS bounds, forgetting, or ridge")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.displacement_scale = float(displacement_scale)
        self.forgetting = float(forgetting)
        self.ridge = float(ridge)
        self.name = name
        self.update_enabled = bool(update_enabled)
        self.weights = np.zeros(3, dtype=float) if weights is None else np.asarray(weights, dtype=float).copy()
        self.covariance = (
            np.eye(3, dtype=float) / self.ridge
            if covariance is None
            else np.asarray(covariance, dtype=float).copy()
        )
        if self.weights.shape != (3,) or self.covariance.shape != (3, 3):
            raise ValueError("RLS state has an invalid shape")
        self.update_count = 0
        self._validate_state()

    def _validate_state(self) -> None:
        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self.covariance)):
            raise ValueError("RLS state must be finite")
        self.covariance = (self.covariance + self.covariance.T) / 2.0

    def features(self, history: Sequence[float]) -> np.ndarray:
        _check_history(history, 4)
        values = np.asarray(history[-4:], dtype=float)
        width = self.upper_bound - self.lower_bound
        return np.asarray(
            [1.0, (values[-1] - values[-2]) / self.displacement_scale,
             (values[-1] - (self.lower_bound + self.upper_bound) / 2) / width],
            dtype=float,
        )

    def _raw_predict(self, history: Sequence[float]) -> float:
        phi = self.features(history)
        width = self.upper_bound - self.lower_bound
        return float(history[-1] + width * np.dot(self.weights, phi))

    def predict(self, history: Sequence[float]) -> float:
        raw = self._raw_predict(history)
        return reflect_prediction(raw, self.lower_bound, self.upper_bound)

    def update(self, history: Sequence[float], target_position: float) -> None:
        if not self.update_enabled:
            return
        if not math.isfinite(float(target_position)):
            raise ValueError("target_position must be finite")
        phi = self.features(history)
        width = self.upper_bound - self.lower_bound
        target = (float(target_position) - float(history[-1])) / width
        gain_denominator = self.forgetting + float(phi @ self.covariance @ phi)
        if not math.isfinite(gain_denominator) or gain_denominator <= 0:
            raise FloatingPointError("invalid RLS gain denominator")
        gain = (self.covariance @ phi) / gain_denominator
        error = target - float(phi @ self.weights)
        self.weights += gain * error
        self.covariance = (self.covariance - np.outer(gain, phi @ self.covariance)) / self.forgetting
        self._validate_state()
        self.update_count += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "history_length": 4,
            "target": "next_displacement_normalized_by_interval_width",
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "displacement_scale": self.displacement_scale,
            "forgetting": self.forgetting,
            "ridge": self.ridge,
            "weights": self.weights.tolist(),
            "covariance": self.covariance.tolist(),
            "update_count": self.update_count,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path, *, name: str | None = None, update_enabled: bool = False) -> "OnlineRLSPredictor":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")), name=name, update_enabled=update_enabled)

    @classmethod
    def from_state_dict(cls, state: dict[str, object], *, name: str | None = None, update_enabled: bool = False) -> "OnlineRLSPredictor":
        if state.get("format_version") != cls.format_version:
            raise ValueError("unsupported RLS predictor checkpoint format")
        model = cls(
            lower_bound=float(state["lower_bound"]),
            upper_bound=float(state["upper_bound"]),
            displacement_scale=float(state.get("displacement_scale", 0.01)),
            forgetting=float(state["forgetting"]),
            ridge=float(state["ridge"]),
            name=name or str(state.get("name", cls.name)),
            update_enabled=update_enabled,
            weights=[float(value) for value in state["weights"]],
            covariance=[[float(value) for value in row] for row in state["covariance"]],
        )
        model.update_count = int(state.get("update_count", 0))
        return model
