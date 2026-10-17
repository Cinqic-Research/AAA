"""The adaptable World Model successor: ``aaa.world.cbl.v0``.

A *context library* of Bayesian linear-Gaussian dynamics models. Each
context is one hypothesis about how the tools currently behave: for every
operation a Normal-Inverse-Gamma posterior over the coefficients of a small
generic basis ``[1, n]`` for the change a call produces, giving a Student-t
predictive with calibrated uncertainty from a handful of observations.

Why this shape (see docs/juniper1/decisions.md, D-WM):

* adaptation adds or switches contexts instead of overwriting one network,
  so a version that returns later is recalled rather than relearned -- the
  interference ``aaa.python.opaque.v0`` measured when WM-S adapted to
  library B;
* the posterior is exact and cheap, so every candidate is an immutable,
  hash-identified state that can be evaluated in isolation and rolled back;
* misfit is visible: dynamics outside the basis (the capped "novel" family)
  inflate the predictive variance rather than hiding behind a confident
  point estimate.

The World Model never decides when it adapts. It exposes mechanisms
(``new_context``, ``update_in_place``, ``recall``) that the Erudition
lifecycle invokes, evaluates and commits.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .contracts import WMPrediction, sha256_json

WM_VERSION = "aaa.world.cbl.v0"
FEATURES = ("bias", "amount")
PRIOR_PRECISION = 0.01
PRIOR_A = 1.0
PRIOR_B = 0.25
MAX_STORED = 48

Transition = tuple[str, int, int, int]
"""(op, amount, before, after) as observed by the host after a successful call."""


def _phi(amount: int) -> np.ndarray:
    return np.array([1.0, float(amount)])


@dataclasses.dataclass(frozen=True)
class Posterior:
    """Normal-Inverse-Gamma posterior over the coefficients of one operation."""

    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    a: float
    b: float
    n: int

    @staticmethod
    def prior() -> Posterior:
        d = len(FEATURES)
        precision = tuple(tuple(PRIOR_PRECISION if i == j else 0.0 for j in range(d)) for i in range(d))
        return Posterior(tuple(0.0 for _ in range(d)), precision, PRIOR_A, PRIOR_B, 0)

    def update(self, rows: list[tuple[int, int]]) -> Posterior:
        """Conjugate update with ``(amount, delta)`` observations."""

        if not rows:
            return self
        m0 = np.array(self.mean)
        l0 = np.array(self.precision)
        x = np.stack([_phi(amount) for amount, _ in rows])
        y = np.array([float(delta) for _, delta in rows])
        ln = l0 + x.T @ x
        mn = np.linalg.solve(ln, l0 @ m0 + x.T @ y)
        an = self.a + len(rows) / 2.0
        bn = self.b + 0.5 * float(y @ y + m0 @ l0 @ m0 - mn @ ln @ mn)
        bn = max(bn, 1e-9)
        return Posterior(
            tuple(float(v) for v in mn),
            tuple(tuple(float(v) for v in row) for row in ln),
            float(an),
            float(bn),
            self.n + len(rows),
        )

    def predictive(self, amount: int) -> tuple[float, float, float]:
        """Student-t predictive for the change: (mean, scale, degrees of freedom)."""

        phi = _phi(amount)
        cov = np.linalg.inv(np.array(self.precision))
        mean = float(phi @ np.array(self.mean))
        scale2 = (self.b / self.a) * (1.0 + float(phi @ cov @ phi))
        return mean, math.sqrt(scale2), 2.0 * self.a

    def to_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> Posterior:
        return Posterior(
            tuple(float(v) for v in payload["mean"]),
            tuple(tuple(float(v) for v in row) for row in payload["precision"]),
            float(payload["a"]),
            float(payload["b"]),
            int(payload["n"]),
        )


def _student_logpdf(x: float, mean: float, scale: float, dof: float) -> float:
    z = (x - mean) / scale
    return (
        math.lgamma((dof + 1) / 2)
        - math.lgamma(dof / 2)
        - 0.5 * math.log(dof * math.pi)
        - math.log(scale)
        - (dof + 1) / 2 * math.log1p(z * z / dof)
    )


@dataclasses.dataclass(frozen=True)
class Context:
    context_id: str
    created_step: int
    posteriors: dict[str, Posterior]
    transitions: tuple[Transition, ...]

    def fit(self, transitions: list[Transition], lower: int, upper: int) -> Context:
        posteriors = dict(self.posteriors)
        for op in posteriors:
            rows = [
                (amount, after - before)
                for (o, amount, before, after) in transitions
                # Results clipped at a declared bound are censored, not observations of the change.
                if o == op and lower < after < upper
            ]
            posteriors[op] = posteriors[op].update(rows)
        stored = (self.transitions + tuple(transitions))[-MAX_STORED:]
        return dataclasses.replace(self, posteriors=posteriors, transitions=stored)

    def predict(self, op: str, amount: int, before: int, lower: int, upper: int) -> tuple[float, float]:
        mean, scale, dof = self.posteriors[op].predictive(amount)
        sd = scale * math.sqrt(dof / (dof - 2.0)) if dof > 2.0 else scale * 10.0
        return float(min(upper, max(lower, before + mean))), sd

    def loglik(self, transitions: list[Transition], lower: int, upper: int) -> float:
        total = 0.0
        for op, amount, before, after in transitions:
            if not lower < after < upper:
                continue
            mean, scale, dof = self.posteriors[op].predictive(amount)
            total += _student_logpdf(after - before, mean, scale, dof)
        return total


@dataclasses.dataclass(frozen=True)
class WorldState:
    """One immutable World Model state: the library and which context is active."""

    contexts: tuple[Context, ...]
    active: str
    lower: int
    upper: int

    @staticmethod
    def initial(ops: tuple[str, ...], warmup: list[Transition], lower: int, upper: int) -> WorldState:
        empty = Context("ctx-0", -1, {op: Posterior.prior() for op in ops}, ())
        return WorldState((empty.fit(warmup, lower, upper),), "ctx-0", lower, upper)

    def context(self, context_id: str | None = None) -> Context:
        wanted = context_id or self.active
        for context in self.contexts:
            if context.context_id == wanted:
                return context
        raise KeyError(wanted)

    def predict(self, op: str, entity: str, amount: int, before: int, state_id: str) -> WMPrediction:
        mean, sd = self.context().predict(op, amount, before, self.lower, self.upper)
        return WMPrediction(
            op=op,
            entity=entity,
            amount=amount,
            before=before,
            mean=mean,
            sd=sd,
            context=self.active,
            wm_state=state_id,
        )

    def behaviour(self, max_sd: float) -> dict[str, tuple[int, int]]:
        """The active context's confident estimate of each operation, as (rate, offset) magnitudes."""

        out = {}
        for op, posterior in self.context().posteriors.items():
            _, scale, _ = posterior.predictive(3)
            if posterior.n < 3 or scale > max_sd:
                continue
            sign = 1 if op == "fill" else -1
            rate, offset = round(sign * posterior.mean[1]), round(sign * posterior.mean[0])
            if rate > 0:
                out[op] = (rate, offset)
        return out

    def surprise(self, transition: Transition) -> dict[str, float]:
        """Evidence for the Erudition Model: how well each hypothesis explains one transition."""

        active = self.context().loglik([transition], self.lower, self.upper)
        others = [
            c.loglik([transition], self.lower, self.upper)
            for c in self.contexts
            if c.context_id != self.active
        ]
        prior = Context("prior", -1, {op: Posterior.prior() for op in self.context().posteriors}, ())
        return {
            "active": active,
            "best_other": max(others) if others else active,
            "prior": prior.loglik([transition], self.lower, self.upper),
        }

    # ---- adaptation mechanisms (each returns a new state; nothing mutates) ----

    def new_context(self, transitions: list[Transition], step: int) -> WorldState:
        ops = tuple(self.context().posteriors)
        context_id = f"ctx-{len(self.contexts)}"
        fresh = Context(context_id, step, {op: Posterior.prior() for op in ops}, ()).fit(
            transitions, self.lower, self.upper
        )
        return dataclasses.replace(self, contexts=(*self.contexts, fresh), active=context_id)

    def update_in_place(self, transitions: list[Transition]) -> WorldState:
        updated = self.context().fit(transitions, self.lower, self.upper)
        contexts = tuple(updated if c.context_id == self.active else c for c in self.contexts)
        return dataclasses.replace(self, contexts=contexts)

    def recall(self, transitions: list[Transition]) -> WorldState:
        """Switch to the stored context that best explains recent evidence."""

        best = max(self.contexts, key=lambda c: c.loglik(transitions, self.lower, self.upper))
        return dataclasses.replace(self, active=best.context_id)

    # ---- serialization -----------------------------------------------------

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": WM_VERSION,
            "active": self.active,
            "lower": self.lower,
            "upper": self.upper,
            "contexts": [
                {
                    "context_id": c.context_id,
                    "created_step": c.created_step,
                    "posteriors": {op: p.to_payload() for op, p in sorted(c.posteriors.items())},
                    "transitions": [list(t) for t in c.transitions],
                }
                for c in self.contexts
            ],
        }

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> WorldState:
        if payload.get("version") != WM_VERSION:
            raise ValueError(f"not a {WM_VERSION} state")
        contexts = tuple(
            Context(
                str(c["context_id"]),
                int(c["created_step"]),
                {op: Posterior.from_payload(p) for op, p in c["posteriors"].items()},
                tuple((str(t[0]), int(t[1]), int(t[2]), int(t[3])) for t in c["transitions"]),
            )
            for c in payload["contexts"]
        )
        state = WorldState(contexts, str(payload["active"]), int(payload["lower"]), int(payload["upper"]))
        state.context()
        return state

    def digest(self) -> str:
        return sha256_json(self.to_payload())


def prediction_error_rate(state: WorldState, transitions: list[Transition]) -> float | None:
    """Fraction of transitions whose rounded prediction misses the observed value."""

    usable = [t for t in transitions if t]
    if not usable:
        return None
    misses = 0
    for op, amount, before, after in usable:
        mean, _ = state.context().predict(op, amount, before, state.lower, state.upper)
        misses += int(round(mean) != after)
    return misses / len(usable)
