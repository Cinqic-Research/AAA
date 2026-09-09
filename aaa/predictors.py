"""Inspectably small prediction rules used in the experiment."""

from __future__ import annotations

import json
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


class PersistencePredictor(Predictor):
    name = "persistence"

    def predict(self, history: Sequence[float]) -> float:
        if not history:
            raise ValueError("persistence requires at least one observation")
        return float(history[-1])


class ConstantMotionPredictor(Predictor):
    name = "constant_motion"

    def predict(self, history: Sequence[float]) -> float:
        if len(history) < 2:
            raise ValueError("constant_motion requires two observations")
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
        destination.write_text(json.dumps(self.state_dict(), indent=2) + "\n", encoding="utf-8")

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
