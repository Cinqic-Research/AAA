"""Inspectably small prediction rules used in the experiment.

Every predictor here consumes a tuple of past positions and nothing else.
Evaluator metadata (scenario, event flags, velocities, change schedules) never
reaches this module; see :mod:`aaa.experiment` for the temporal boundary.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Numerical tolerances used when validating serialized learner state. They are
# module constants so tests and the benchmark specification can reference the
# exact same values instead of restating magic numbers.
SYMMETRY_TOLERANCE = 1e-9
PSD_TOLERANCE = 1e-9
MAX_CONDITION_NUMBER = 1e12


class Predictor:
    """Minimal predictor interface used by the temporal runner."""

    name: str = "predictor"
    update_enabled: bool = False

    def predict(self, history: Sequence[float]) -> float:
        raise NotImplementedError

    def update(self, history: Sequence[float], target_position: float) -> None:
        """Baselines and frozen copies do nothing here, by design."""

        del history, target_position


def _check_history(history: Sequence[float], minimum: int) -> None:
    if len(history) < minimum:
        raise ValueError(f"predictor requires at least {minimum} observations")
    if not all(math.isfinite(float(value)) for value in history[-minimum:]):
        raise ValueError("history contains a non-finite observation")


def unfold_observation(observed: float, reference: float, lower: float, upper: float) -> float:
    """Invert the public reflection map, choosing the branch nearest ``reference``.

    The reflection map is programmed public knowledge of the observation
    format. Applying it to a prediction while regressing on the *folded*
    observation is internally inconsistent: at a wall contact the observed
    displacement is not the displacement the linear law produced. This inverts
    the same public map, so the learner's update target lives in the same
    unfolded coordinate as its own linear prediction.

    ``reference`` must be the learner's own raw (unreflected) prediction, which
    is causally available. No evaluator bounce label is involved: the branch is
    chosen purely by proximity.
    """

    if upper <= lower:
        raise ValueError("upper bound must exceed lower bound")
    if not all(math.isfinite(float(value)) for value in (observed, reference, lower, upper)):
        raise ValueError("unfolding inputs must be finite")
    width = upper - lower
    # Candidate pre-images of ``observed`` under repeated reflection, ordered
    # outward from the interval. Two folds either side is far beyond anything
    # the declared dynamics can produce in one transition.
    candidates = [observed]
    for fold in range(1, 3):
        candidates.append(upper + (fold - 1) * 2 * width + (upper - observed))
        candidates.append(lower - (fold - 1) * 2 * width - (observed - lower))
    return float(min(candidates, key=lambda value: abs(value - reference)))


def reflect_prediction(position: float, lower: float, upper: float) -> float:
    """Reflect a predicted position into bounds without evaluator metadata.

    This is *programmed public knowledge* about the observation format, not a
    learned capability. Any predictor may use it; the benchmark therefore
    offers a reflected constant-motion baseline so the boundary policy is not
    confused with learning.
    """

    if not all(math.isfinite(float(value)) for value in (position, lower, upper)):
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
    """``x[t+1] = x[t]``."""

    name = "persistence"

    def predict(self, history: Sequence[float]) -> float:
        _check_history(history, 1)
        return float(history[-1])


class ConstantMotionPredictor(Predictor):
    """``x[t+1] = x[t] + (x[t] - x[t-1])`` with no boundary knowledge."""

    name = "constant_motion"

    def __init__(self, *, name: str = "constant_motion") -> None:
        self.name = name

    def predict(self, history: Sequence[float]) -> float:
        _check_history(history, 2)
        return float(history[-1] + (history[-1] - history[-2]))


class ReflectedConstantMotionPredictor(ConstantMotionPredictor):
    """Constant motion plus the *exact same* public reflection policy.

    This is the like-for-like baseline for any candidate that is allowed to
    reflect its own prediction. Without it, a boundary transform is easily
    mistaken for learned bounce anticipation.
    """

    name = "constant_motion_reflected"

    def __init__(
        self, *, lower_bound: float, upper_bound: float, name: str = "constant_motion_reflected"
    ) -> None:
        super().__init__(name=name)
        if not all(math.isfinite(float(value)) for value in (lower_bound, upper_bound)):
            raise ValueError("bounds must be finite")
        if upper_bound <= lower_bound:
            raise ValueError("upper bound must exceed lower bound")
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)

    def predict(self, history: Sequence[float]) -> float:
        raw = super().predict(history)
        return reflect_prediction(raw, self.lower_bound, self.upper_bound)


class OnlineLinearPredictor(Predictor):
    """Legacy online linear displacement predictor (historical v1 learner).

    The feature vector is ``[1, x[t-3], x[t-2], x[t-1], x[t]]``. The model
    predicts the next displacement, which is added to ``x[t]`` to obtain the
    next-position prediction. Its loss is one half of squared displacement
    error, and the update is ordinary single-example gradient descent.

    It is retained unchanged in behaviour as a named diagnostic comparison so
    the historical v1 result stays reproducible.
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
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = float(learning_rate)
        self.name = name
        self.update_enabled = bool(update_enabled)
        self.weights = (
            np.zeros(5, dtype=float) if weights is None else np.asarray(weights, dtype=float).copy()
        )
        if self.weights.shape != (5,):
            raise ValueError("weights must contain exactly five values")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("weights must be finite")
        self.update_count = 0

    @staticmethod
    def features(history: Sequence[float]) -> np.ndarray:
        _check_history(history, 4)
        return np.asarray([1.0, *history[-4:]], dtype=float)

    def predict(self, history: Sequence[float]) -> float:
        features = self.features(history)
        displacement = float(np.dot(self.weights, features))
        return float(history[-1] + displacement)

    def update(self, history: Sequence[float], target_position: float) -> None:
        if not self.update_enabled:
            return
        if not math.isfinite(float(target_position)):
            raise ValueError("target_position must be finite")
        features = self.features(history)
        target_displacement = float(target_position - history[-1])
        predicted_displacement = float(np.dot(self.weights, features))
        # gradient of 0.5 * (predicted_displacement - target)^2
        gradient = (predicted_displacement - target_displacement) * features
        self.weights -= self.learning_rate * gradient
        if not np.all(np.isfinite(self.weights)):
            raise FloatingPointError("legacy linear update produced non-finite weights")
        self.update_count += 1

    def state_dict(self) -> dict[str, Any]:
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
        _atomic_json_write(Path(path), self.state_dict())

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        name: str | None = None,
        update_enabled: bool = False,
    ) -> OnlineLinearPredictor:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(state, name=name, update_enabled=update_enabled)

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        name: str | None = None,
        update_enabled: bool = False,
    ) -> OnlineLinearPredictor:
        if state.get("format_version") != cls.format_version:
            raise ValueError("unsupported linear predictor checkpoint format")
        if state.get("history_length") != 4 or state.get("target") != "next_displacement":
            raise ValueError("checkpoint metadata does not match the AAA predictor")
        for field in ("learning_rate", "weights"):
            if field not in state:
                raise ValueError(f"linear checkpoint is missing required field {field!r}")
        model = cls(
            learning_rate=float(state["learning_rate"]),
            name=name or str(state.get("name", "linear_online")),
            update_enabled=update_enabled,
            weights=[float(value) for value in state["weights"]],
        )
        model.update_count = int(state.get("update_count", 0))
        return model


class InvalidLearnerState(ValueError):
    """Raised when learner state violates a required numerical invariant."""


def validate_covariance(
    covariance: np.ndarray,
    *,
    symmetry_tolerance: float = SYMMETRY_TOLERANCE,
    psd_tolerance: float = PSD_TOLERANCE,
    max_condition_number: float = MAX_CONDITION_NUMBER,
) -> dict[str, float]:
    """Validate a covariance matrix and return its numerical diagnostics.

    The check is deliberately loud: an invalid state raises instead of being
    silently symmetrized, clipped, or reset. A numerical failure has to become
    visible evidence, not a quiet correction.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise InvalidLearnerState("covariance must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise InvalidLearnerState("covariance contains non-finite entries")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    asymmetry = float(np.max(np.abs(matrix - matrix.T))) if matrix.size else 0.0
    if asymmetry > symmetry_tolerance * scale:
        raise InvalidLearnerState(f"covariance is not symmetric (max asymmetry {asymmetry:.3e})")
    symmetric = (matrix + matrix.T) / 2.0
    try:
        eigenvalues = np.linalg.eigvalsh(symmetric)
    except np.linalg.LinAlgError as error:  # pragma: no cover - hardware/LAPACK specific
        raise InvalidLearnerState(f"covariance eigenvalues did not converge: {error}") from error
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    if minimum < -psd_tolerance * scale:
        raise InvalidLearnerState(f"covariance is not positive semidefinite (min eigenvalue {minimum:.3e})")
    if maximum <= 0.0:
        raise InvalidLearnerState("covariance has no positive eigenvalue")
    condition = maximum / minimum if minimum > 0 else math.inf
    if not math.isfinite(condition) or condition > max_condition_number:
        raise InvalidLearnerState(
            f"covariance condition number {condition:.3e} exceeds {max_condition_number:.3e}"
        )
    return {
        "min_eigenvalue": minimum,
        "max_eigenvalue": maximum,
        "condition_number": float(condition),
        "trace": float(np.trace(symmetric)),
        "max_asymmetry": asymmetry,
    }


def _rank_one_inflate(factor: np.ndarray, direction: np.ndarray, weight: float) -> np.ndarray:
    """Return ``S'`` with ``S' S'^T == S S^T + weight * v v^T``.

    Computed by an LQ factorization of the augmented factor, so the result
    stays an exact square-root factor rather than a re-decomposed covariance.
    """

    if weight <= 0:
        return factor
    augmented = np.hstack([factor, math.sqrt(weight) * direction.reshape(-1, 1)])
    upper = np.linalg.qr(augmented.T, mode="r")
    return np.asarray(upper).T


def _cholesky_factor(covariance: np.ndarray) -> np.ndarray:
    """Return a square factor ``S`` with ``S @ S.T == covariance``."""

    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 0.0, None)
    return vectors @ np.diag(np.sqrt(clipped))


class OnlineRLSPredictor(Predictor):
    """Square-root recursive least squares with trace-bounded forgetting.

    Motivation
    ----------
    The ordinary covariance-form RLS recursion ``P <- (P - k phi^T P) / lambda``
    is algebraically correct but numerically fragile with ``lambda < 1``. When
    the regressor is weakly exciting (a stationary stream, a repeated input, a
    nearly constant displacement) the ``1 / lambda`` inflation is applied in
    directions that receive no information, ``P`` grows without bound, symmetry
    and positive semidefiniteness are lost to rounding, and the state finally
    overflows. This is the classical *covariance windup* failure described in
    the adaptive-control literature (Åström & Wittenmark, *Adaptive Control*,
    2nd ed., ch. 3; Ljung & Söderström, *Theory and Practice of Recursive
    Identification*, ch. 2).

    Two established remedies are applied together:

    ``square-root propagation``
        The state is the factor ``S`` with ``P = S @ S.T`` (Potter's rank-one
        measurement update). Positive semidefiniteness and symmetry hold by
        construction and the effective condition number is the square root of
        the covariance-form condition number.
    ``trace-bounded forgetting``
        The inflation is applied only while ``trace(P)`` stays below
        ``trace_bound``. When the bound would be exceeded the step uses
        ``lambda = 1`` instead. Suspended steps are *counted and reported*
        (:attr:`forgetting_suspensions`) rather than silently absorbed.
    ``directional forgetting`` (``forgetting_mode="directional"``)
        Ordinary exponential forgetting inflates ``P`` in *every* direction,
        including directions the incoming data says nothing about. On a
        weakly exciting stream the estimate then drifts along the unexcited
        null directions even though it never diverges. Directional forgetting
        instead adds uncertainty only along ``P phi``, the direction the new
        measurement is about to inform:

        ``P <- P + ((1 - lambda) / lambda) * (P phi)(P phi)^T / (phi^T P phi)``

        followed by the ordinary measurement downdate with no global
        inflation. Unexcited directions keep whatever they had learned. See
        Kulhavy & Karny, "Tracking of slowly varying parameters by directional
        forgetting" (IFAC 1984) and Cao & Schwartz, "A directional forgetting
        algorithm based on the decomposition of the information matrix"
        (*Automatica* 36(11), 2000).
    ``dead-zone updating`` (optional, ``dead_zone > 0``)
        A step whose a-priori normalized error is at or below ``dead_zone`` is
        skipped entirely: no weight change and no covariance inflation. This is
        the standard dead-zone modification from robust adaptive control
        (Ioannou & Sun, *Robust Adaptive Control*, sec. 8.4). It exists because
        exponential forgetting on an already-solved, weakly exciting stream
        makes the estimate drift; with a dead zone the model stops updating
        once it predicts well and resumes the moment it is surprised. The
        trigger is the model's **own** scored prediction error, which is
        causally available; no evaluator event label is involved.
        Skipped steps are counted in :attr:`dead_zone_skips`.
    ``self-triggered forgetting`` (optional, ``detector_multiplier > 0``)
        Forgetting is applied only on steps the learner itself judges
        surprising: when its own a-priori normalized error exceeds
        ``max(detector_multiplier * ewma(|error|), detector_floor)``. On a
        stream it already predicts well the model keeps ``lambda = 1`` and does
        not drift; when the world's law changes, the sustained surprise turns
        forgetting on and the model re-identifies. The trigger is the model's
        own scored prediction error only. The evaluator never signals a change,
        an event, a scenario or a mode switch. Detected steps are counted in
        :attr:`detected_surprises`.

    With ``forgetting == 1`` and no suspension the recursion is exactly ridge
    regularized batch least squares, which the test suite verifies against an
    independent SVD/`lstsq` solution.
    """

    format_version = "aaa.rls_predictor.v2"
    name = "adaptive_rls"
    update_enabled = True
    FEATURE_SETS = ("displacement_position", "displacement_only")
    FORGETTING_MODES = ("exponential", "directional")

    def __init__(
        self,
        *,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        displacement_scale: float = 0.01,
        forgetting: float = 1.0,
        forgetting_mode: str = "exponential",
        ridge: float = 1e-4,
        feature_set: str = "displacement_position",
        reflect: bool = True,
        unfold_target: bool = False,
        skip_after_reflected_prediction: bool = False,
        trace_bound: float = 1e5,
        symmetry_tolerance: float = SYMMETRY_TOLERANCE,
        psd_tolerance: float = PSD_TOLERANCE,
        max_condition_number: float = MAX_CONDITION_NUMBER,
        dead_zone: float = 0.0,
        detector_multiplier: float = 0.0,
        detector_floor: float = 1e-6,
        detector_decay: float = 0.05,
        name: str = "adaptive_rls",
        update_enabled: bool = True,
        weights: Sequence[float] | None = None,
        covariance: Sequence[Sequence[float]] | None = None,
        sqrt_factor: Sequence[Sequence[float]] | None = None,
        update_count: int = 0,
        forgetting_suspensions: int = 0,
        dead_zone_skips: int = 0,
        detected_surprises: int = 0,
        reflection_skips: int = 0,
        previous_prediction_reflected: bool = False,
        error_ewma: float = 0.0,
    ) -> None:
        numeric = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "displacement_scale": displacement_scale,
            "forgetting": forgetting,
            "ridge": ridge,
            "trace_bound": trace_bound,
            "dead_zone": dead_zone,
            "detector_multiplier": detector_multiplier,
            "detector_floor": detector_floor,
            "detector_decay": detector_decay,
            "symmetry_tolerance": symmetry_tolerance,
            "psd_tolerance": psd_tolerance,
            "max_condition_number": max_condition_number,
        }
        for key, value in numeric.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"RLS parameter {key} must be finite")
        if upper_bound <= lower_bound:
            raise ValueError("upper_bound must exceed lower_bound")
        if displacement_scale <= 0:
            raise ValueError("displacement_scale must be positive")
        if not 0 < forgetting <= 1:
            raise ValueError("forgetting must satisfy 0 < lambda <= 1")
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        if trace_bound <= 0:
            raise ValueError("trace_bound must be positive")
        if dead_zone < 0:
            raise ValueError("dead_zone must be non-negative")
        if detector_multiplier < 0 or detector_floor < 0:
            raise ValueError("detector parameters must be non-negative")
        if not 0 < detector_decay <= 1:
            raise ValueError("detector_decay must satisfy 0 < decay <= 1")
        if symmetry_tolerance < 0 or psd_tolerance < 0:
            raise ValueError("covariance tolerances must be non-negative")
        if max_condition_number <= 1:
            raise ValueError("max_condition_number must exceed 1")
        self.symmetry_tolerance = float(symmetry_tolerance)
        self.psd_tolerance = float(psd_tolerance)
        self.max_condition_number = float(max_condition_number)
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {self.FEATURE_SETS}")
        if forgetting_mode not in self.FORGETTING_MODES:
            raise ValueError(f"forgetting_mode must be one of {self.FORGETTING_MODES}")
        if min(int(update_count), int(forgetting_suspensions), int(dead_zone_skips)) < 0:
            raise ValueError("counters must be non-negative")

        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.displacement_scale = float(displacement_scale)
        self.forgetting = float(forgetting)
        self.forgetting_mode = str(forgetting_mode)
        self.ridge = float(ridge)
        self.feature_set = str(feature_set)
        self.reflect = bool(reflect)
        self.unfold_target = bool(unfold_target)
        self.skip_after_reflected_prediction = bool(skip_after_reflected_prediction)
        self.trace_bound = float(trace_bound)
        self.dead_zone = float(dead_zone)
        self.detector_multiplier = float(detector_multiplier)
        self.detector_floor = float(detector_floor)
        self.detector_decay = float(detector_decay)
        self.name = name
        self.update_enabled = bool(update_enabled)
        self.dimension = 3 if feature_set == "displacement_position" else 2

        self.weights = (
            np.zeros(self.dimension, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float).copy()
        )
        if self.weights.shape != (self.dimension,):
            raise ValueError("RLS weight vector has an invalid shape")
        if not np.all(np.isfinite(self.weights)):
            raise ValueError("RLS weights must be finite")

        if sqrt_factor is not None:
            factor = np.asarray(sqrt_factor, dtype=float)
            if factor.shape != (self.dimension, self.dimension):
                raise ValueError("RLS square-root factor has an invalid shape")
            if not np.all(np.isfinite(factor)):
                raise ValueError("RLS square-root factor must be finite")
            self._factor = factor.copy()
            self.diagnostics = self._validate(self.covariance)
        else:
            if covariance is None:
                initial = np.eye(self.dimension, dtype=float) / self.ridge
            else:
                initial = np.asarray(covariance, dtype=float)
                if initial.shape != (self.dimension, self.dimension):
                    raise ValueError("RLS covariance has an invalid shape")
            self.diagnostics = self._validate(initial)
            self._factor = _cholesky_factor(initial)
        self.update_count = int(update_count)
        self.forgetting_suspensions = int(forgetting_suspensions)
        self.dead_zone_skips = int(dead_zone_skips)
        self.detected_surprises = int(detected_surprises)
        self.reflection_skips = int(reflection_skips)
        self.previous_prediction_reflected = bool(previous_prediction_reflected)
        if not math.isfinite(float(error_ewma)) or error_ewma < 0:
            raise ValueError("error_ewma must be finite and non-negative")
        self.error_ewma = float(error_ewma)

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def covariance(self) -> np.ndarray:
        """Reconstructed ``P = S S^T``; always symmetric and PSD."""

        product = self._factor @ self._factor.T
        return (product + product.T) / 2.0

    def features(self, history: Sequence[float]) -> np.ndarray:
        _check_history(history, 4)
        values = np.asarray(history[-4:], dtype=float)
        width = self.upper_bound - self.lower_bound
        displacement = (values[-1] - values[-2]) / self.displacement_scale
        if self.feature_set == "displacement_only":
            return np.asarray([1.0, displacement], dtype=float)
        centered = (values[-1] - (self.lower_bound + self.upper_bound) / 2) / width
        return np.asarray([1.0, displacement, centered], dtype=float)

    def raw_predict(self, history: Sequence[float]) -> float:
        """Prediction before any boundary policy is applied."""

        phi = self.features(history)
        width = self.upper_bound - self.lower_bound
        return float(history[-1] + width * float(np.dot(self.weights, phi)))

    def predict(self, history: Sequence[float]) -> float:
        raw = self.raw_predict(history)
        if not math.isfinite(raw):
            raise FloatingPointError("RLS produced a non-finite prediction")
        if not self.reflect:
            return raw
        return reflect_prediction(raw, self.lower_bound, self.upper_bound)

    def update(self, history: Sequence[float], target_position: float) -> None:
        if not self.update_enabled:
            return
        if not math.isfinite(float(target_position)):
            raise ValueError("target_position must be finite")
        phi = self.features(history)
        width = self.upper_bound - self.lower_bound
        raw = self.raw_predict(history)

        # A window whose displacement feature straddles a wall is not a sample
        # of the linear law. Unfolding the *target* makes the wall transition
        # itself an ordinary sample, but the very next window still carries a
        # folded difference as its displacement feature, and fitting that one
        # sample is what damages a continuously updating instance.
        #
        # The trigger is the learner's own previous raw prediction having
        # needed reflection to stay in bounds -- public knowledge of the
        # observation format applied to its own output. No evaluator bounce
        # label, scenario or event flag is involved.
        straddling = self.previous_prediction_reflected
        self.previous_prediction_reflected = bool(
            self.reflect and raw != reflect_prediction(raw, self.lower_bound, self.upper_bound)
        )
        if self.skip_after_reflected_prediction and straddling:
            self.reflection_skips += 1
            return

        effective_target = float(target_position)
        if self.unfold_target:
            effective_target = unfold_observation(effective_target, raw, self.lower_bound, self.upper_bound)
        target = (effective_target - float(history[-1])) / width
        innovation = target - float(phi @ self.weights)
        if self.dead_zone > 0.0 and abs(innovation) <= self.dead_zone:
            # Already predicting inside the declared dead zone: nothing to
            # learn from this sample, and nothing to forget either.
            self.dead_zone_skips += 1
            return

        # Self-triggered forgetting: decide *before* folding this sample into
        # the running error scale, so the current surprise cannot mask itself.
        surprised = True
        if self.detector_multiplier > 0.0:
            scale = max(self.error_ewma, self.detector_floor)
            surprised = abs(innovation) > self.detector_multiplier * scale
            if surprised:
                self.detected_surprises += 1
        # The quiescent error scale is tracked from *unsurprising* steps only.
        # Folding the surprise itself into the scale would let a genuine law
        # change silence the detector after a handful of transitions, which is
        # exactly when sustained forgetting is needed.
        if not surprised or self.detector_multiplier <= 0.0:
            self.error_ewma = (1.0 - self.detector_decay) * self.error_ewma + self.detector_decay * abs(
                innovation
            )

        # Trace-bounded forgetting: only inflate while the covariance trace
        # stays inside the declared bound. Suspensions are recorded.
        trace = float(np.sum(self._factor * self._factor))
        lam = self.forgetting if surprised else 1.0
        if lam < 1.0 and trace / lam > self.trace_bound:
            lam = 1.0
            self.forgetting_suspensions += 1

        f = self._factor.T @ phi
        beta = float(f @ f)
        if self.forgetting_mode == "directional" and lam < 1.0:
            if beta > 0:
                self._factor = _rank_one_inflate(
                    self._factor, (self._factor @ f) / math.sqrt(beta), (1.0 - lam) / lam
                )
                f = self._factor.T @ phi
                beta = float(f @ f)
            # No global inflation in directional mode: unexcited directions
            # keep the information they already hold.
            lam = 1.0
        alpha = lam + beta
        if not math.isfinite(alpha) or alpha <= 0:
            raise FloatingPointError("invalid RLS gain denominator")
        error = innovation
        if beta > 0:
            gain = (self._factor @ f) / alpha
            self.weights = self.weights + gain * error
            gamma = (1.0 - math.sqrt(lam / alpha)) / beta
            self._factor = (self._factor - gamma * np.outer(self._factor @ f, f)) / math.sqrt(lam)
        else:
            # Zero regressor projection: no information, pure inflation only.
            self._factor = self._factor / math.sqrt(lam)

        if not np.all(np.isfinite(self.weights)) or not np.all(np.isfinite(self._factor)):
            raise FloatingPointError("RLS update produced non-finite state")
        self.update_count += 1

    def _validate(self, covariance: np.ndarray) -> dict[str, float]:
        return validate_covariance(
            covariance,
            symmetry_tolerance=self.symmetry_tolerance,
            psd_tolerance=self.psd_tolerance,
            max_condition_number=self.max_condition_number,
        )

    def check_state(self) -> dict[str, float]:
        """Recompute and store covariance diagnostics; raise if invalid."""

        self.diagnostics = self._validate(self.covariance)
        return self.diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "name": self.name,
            "history_length": 4,
            "target": "next_displacement_normalized_by_interval_width",
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "displacement_scale": self.displacement_scale,
            "forgetting": self.forgetting,
            "forgetting_mode": self.forgetting_mode,
            "ridge": self.ridge,
            "feature_set": self.feature_set,
            "reflect": self.reflect,
            "unfold_target": self.unfold_target,
            "skip_after_reflected_prediction": self.skip_after_reflected_prediction,
            "trace_bound": self.trace_bound,
            "dead_zone": self.dead_zone,
            "detector_multiplier": self.detector_multiplier,
            "detector_floor": self.detector_floor,
            "detector_decay": self.detector_decay,
            "weights": [float(value) for value in self.weights],
            # ``sqrt_factor`` is the authoritative state: P = S @ S.T. Saving
            # and reloading it is exact, so a checkpoint hash is stable across
            # a save/load round trip. ``covariance`` is the derived, human
            # inspectable view of the same state.
            "sqrt_factor": [[float(value) for value in row] for row in self._factor],
            "covariance": [[float(value) for value in row] for row in self.covariance],
            "update_count": self.update_count,
            "forgetting_suspensions": self.forgetting_suspensions,
            "dead_zone_skips": self.dead_zone_skips,
            "reflection_skips": self.reflection_skips,
            "previous_prediction_reflected": self.previous_prediction_reflected,
            "detected_surprises": self.detected_surprises,
            "error_ewma": self.error_ewma,
        }

    def save(self, path: str | Path) -> None:
        _atomic_json_write(Path(path), self.state_dict())

    @classmethod
    def load(
        cls, path: str | Path, *, name: str | None = None, update_enabled: bool = False
    ) -> OnlineRLSPredictor:
        return cls.from_state_dict(
            json.loads(Path(path).read_text(encoding="utf-8")), name=name, update_enabled=update_enabled
        )

    REQUIRED_STATE_FIELDS = (
        "lower_bound",
        "upper_bound",
        "displacement_scale",
        "forgetting",
        "ridge",
        "feature_set",
        "weights",
        "covariance",
    )

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        name: str | None = None,
        update_enabled: bool = False,
        symmetry_tolerance: float = SYMMETRY_TOLERANCE,
        psd_tolerance: float = PSD_TOLERANCE,
        max_condition_number: float = MAX_CONDITION_NUMBER,
    ) -> OnlineRLSPredictor:
        """Rebuild a model from serialized state.

        Numerical tolerances are *validation policy*, not learner state, so
        they are supplied by the caller (the benchmark passes the declared
        ``spec.tolerances``) and deliberately never enter the checkpoint hash.
        """

        if state.get("format_version") != cls.format_version:
            raise ValueError(
                f"unsupported RLS predictor checkpoint format {state.get('format_version')!r}; "
                f"expected {cls.format_version!r}"
            )
        missing = [field for field in cls.REQUIRED_STATE_FIELDS if field not in state]
        if missing:
            raise ValueError(f"RLS checkpoint is missing required fields: {sorted(missing)}")
        if state.get("history_length") != 4:
            raise ValueError("RLS checkpoint history_length must be 4")
        return cls(
            lower_bound=float(state["lower_bound"]),
            upper_bound=float(state["upper_bound"]),
            displacement_scale=float(state["displacement_scale"]),
            forgetting=float(state["forgetting"]),
            forgetting_mode=str(state.get("forgetting_mode", "exponential")),
            ridge=float(state["ridge"]),
            feature_set=str(state["feature_set"]),
            reflect=bool(state.get("reflect", True)),
            unfold_target=bool(state.get("unfold_target", False)),
            skip_after_reflected_prediction=bool(state.get("skip_after_reflected_prediction", False)),
            trace_bound=float(state.get("trace_bound", 1e5)),
            dead_zone=float(state.get("dead_zone", 0.0)),
            detector_multiplier=float(state.get("detector_multiplier", 0.0)),
            detector_floor=float(state.get("detector_floor", 1e-6)),
            detector_decay=float(state.get("detector_decay", 0.05)),
            name=name or str(state.get("name", cls.name)),
            update_enabled=update_enabled,
            symmetry_tolerance=symmetry_tolerance,
            psd_tolerance=psd_tolerance,
            max_condition_number=max_condition_number,
            weights=[float(value) for value in state["weights"]],
            covariance=[[float(value) for value in row] for row in state["covariance"]],
            sqrt_factor=(
                None
                if state.get("sqrt_factor") is None
                else [[float(value) for value in row] for row in state["sqrt_factor"]]
            ),
            update_count=int(state.get("update_count", 0)),
            forgetting_suspensions=int(state.get("forgetting_suspensions", 0)),
            dead_zone_skips=int(state.get("dead_zone_skips", 0)),
            reflection_skips=int(state.get("reflection_skips", 0)),
            previous_prediction_reflected=bool(state.get("previous_prediction_reflected", False)),
            detected_surprises=int(state.get("detected_surprises", 0)),
            error_ewma=float(state.get("error_ewma", 0.0)),
        )

    def clone(self, *, name: str, update_enabled: bool) -> OnlineRLSPredictor:
        return self.from_state_dict(
            self.state_dict(),
            name=name,
            update_enabled=update_enabled,
            symmetry_tolerance=self.symmetry_tolerance,
            psd_tolerance=self.psd_tolerance,
            max_condition_number=self.max_condition_number,
        )


def batch_least_squares(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> np.ndarray:
    """Independent stable reference solution for the ``forgetting == 1`` case.

    Solves ``(X^T X + ridge I) w = X^T y`` through an SVD of the augmented
    system rather than by re-running the recursive formula, so the RLS test is
    a genuine cross-check and not a restatement of the implementation.
    """

    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.shape[0]:
        raise ValueError("features must be 2-D and align with a 1-D target vector")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    columns = matrix.shape[1]
    augmented = np.vstack([matrix, math.sqrt(ridge) * np.eye(columns)])
    extended = np.concatenate([target, np.zeros(columns)])
    solution, *_ = np.linalg.lstsq(augmented, extended, rcond=None)
    return solution


def _atomic_json_write(destination: Path, value: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
