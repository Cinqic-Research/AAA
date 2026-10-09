"""``aaa.erudition.toolshift.v0``: a tool workspace whose dynamics and language drift.

A Juniper-like assistant receives natural-language requests ("Bring the
north tank to 37.") about a small workspace of tanks and acts through two
tools, ``fill(tank, n)`` and ``drain(tank, n)``. Two things can change
without notice:

* **dynamics** -- what the tools actually do (the World Model's domain). The
  written manual always documents the original behaviour, so after a shift
  the manual is stale;
* **language** -- how users refer to tanks (the Language Model's domain).
  After a shift users say "the north tank" instead of "cedar"; nothing in
  the manual says which tank that is.

Episodes are independent snapshots: the workspace values, the request and
the feedback noise are drawn from the stream identity and the step alone,
never from earlier actions. Every experimental condition therefore faces
exactly the same tasks (common random numbers), and conditions differ only
in what their components have learned.

The schedule (shifts, recurrences, transient glitches, tool outages,
corrupted feedback) and each episode's true intent are evaluator-only. They
are returned as an ``EvaluatorLabel`` that the episode runner hands to the
scorer and never to a component.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

from .contracts import Action, Feedback, Observation, Outcome

ENVIRONMENT = "aaa.erudition.toolshift.v0"
OPS = ("fill", "drain")
SPLITS = ("train", "development", "attack", "confirmation")
PURPOSES = {"schedule": 1, "workspace": 2, "episode": 3, "outcome": 4, "feedback": 5, "warmup": 6}


class AdmissionError(RuntimeError):
    """Confirmation identities were requested without an admitted freeze."""


@cache
def load_spec() -> dict[str, Any]:
    text = resources.files("research.aaa_erudition").joinpath("data/aaa_erudition_v0.json").read_text("utf-8")
    spec: dict[str, Any] = json.loads(text)
    return spec


def rng(split: str, index: int, purpose: str, *more: int) -> np.random.Generator:
    code = load_spec()["splits"][split]["code"]
    return np.random.default_rng(np.random.SeedSequence([7101, code, index, PURPOSES[purpose], *more]))


@dataclasses.dataclass(frozen=True)
class Dynamics:
    """True tool semantics. ``fill_cap`` limits one fill call (the novel family)."""

    fill_rate: int
    fill_bonus: int
    drain_rate: int
    drain_fee: int
    fill_cap: int | None = None

    @property
    def key(self) -> str:
        cap = "" if self.fill_cap is None else f"c{self.fill_cap}"
        return f"f{self.fill_rate}+{self.fill_bonus}{cap}/d{self.drain_rate}+{self.drain_fee}"

    def delta(self, op: str, amount: int) -> int:
        if op == "fill":
            raw = self.fill_rate * amount + self.fill_bonus
            return raw if self.fill_cap is None else min(raw, self.fill_cap)
        if op == "drain":
            return -(self.drain_rate * amount + self.drain_fee)
        raise ValueError(f"unknown operation {op!r}")

    def apply(self, op: str, before: int, amount: int, lower: int, upper: int) -> int:
        return int(min(upper, max(lower, before + self.delta(op, amount))))


def manual_dynamics() -> Dynamics:
    manual = load_spec()["manual"]
    return Dynamics(
        manual["fill"]["rate"], manual["fill"]["bonus"], manual["drain"]["rate"], manual["drain"]["fee"]
    )


def manual_text() -> str:
    """The written tool documentation every Language Model prompt carries. It never changes."""

    manual = load_spec()["manual"]
    lower, upper = load_spec()["bounds"]
    return (
        f"fill(tank, n): adds {manual['fill']['rate']}*n units to the tank.\n"
        f"drain(tank, n): removes {manual['drain']['rate']}*n units from the tank.\n"
        f"Every tank holds between {lower} and {upper} units; values outside that range are clipped. "
        "n is a whole number from 1 to 9."
    )


@dataclasses.dataclass(frozen=True)
class Regime:
    """The hidden state of the world at one step (evaluator-only)."""

    dynamics: Dynamics
    alias_probability: float
    language: str
    corruption: float
    outage: float
    events: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class EvaluatorLabel:
    """Ground truth for one episode. Never passed to a model component."""

    episode_id: str
    feasible: bool
    entity: str | None
    reference: str
    alias_used: bool
    value: int | None
    regime: Regime


@dataclasses.dataclass(frozen=True)
class Stream:
    split: str
    index: int
    family: str
    workspace: tuple[str, ...]
    decoys: tuple[str, ...]
    aliases: dict[str, str]
    schedule: tuple[Regime, ...]

    @property
    def stream_id(self) -> str:
        return f"{ENVIRONMENT}/{self.split}/{self.index:05d}"


def _draw_dynamics(generator: np.random.Generator, avoid: Dynamics) -> Dynamics:
    family = load_spec()["dynamics_family"]
    while True:
        candidate = Dynamics(
            int(generator.choice(family["fill_rate"])),
            int(generator.choice(family["fill_bonus"])),
            int(generator.choice(family["drain_rate"])),
            int(generator.choice(family["drain_fee"])),
        )
        if generator.random() < family["novel_fraction"]:
            candidate = dataclasses.replace(
                candidate, fill_cap=int(generator.choice(family["novel_fill_cap"]))
            )
        if candidate != avoid:
            return candidate


def families() -> tuple[str, ...]:
    return tuple(load_spec()["families"])


def make_stream(split: str, index: int, family: str | None = None, *, admitted: bool = False) -> Stream:
    """Build one stream identity. Confirmation identities require an admitted freeze."""

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "confirmation" and not admitted:
        raise AdmissionError("confirmation streams exist only after the freeze is admitted")
    spec = load_spec()
    names = spec["splits"][split]["names"]
    alias_pool = spec["splits"][split]["aliases"]
    family = family or families()[index % len(families())]
    if family not in spec["families"]:
        raise ValueError(f"unknown scenario family {family!r}")
    g = rng(split, index, "workspace")
    chosen = [str(name) for name in g.permutation(names)]
    k = spec["entities_per_workspace"]
    workspace = tuple(chosen[:k])
    decoys = tuple(chosen[k:])
    aliases = dict(zip(workspace, (str(a) for a in g.permutation(alias_pool)[:k]), strict=True))
    schedule = _schedule(split, index, family)
    return Stream(split, index, family, workspace, decoys, aliases, schedule)


def _schedule(split: str, index: int, family: str) -> tuple[Regime, ...]:
    spec = load_spec()
    length = spec["stream_length"]
    g = rng(split, index, "schedule")
    base = manual_dynamics()
    shifted = _draw_dynamics(g, base)
    dynamics = [base] * length
    alias = [0.0] * length
    language = ["names"] * length
    corruption = [0.0] * length
    outage = [0.0] * length
    events: list[list[str]] = [[] for _ in range(length)]
    last_onset = 0
    for kind, (lo, hi) in spec["families"][family]:
        onset = int(g.integers(lo, hi + 1))
        if kind == "corruption_after":
            onset = last_onset + lo
        last_onset = onset
        if kind in ("dynamics", "joint"):
            for t in range(onset, length):
                dynamics[t] = shifted
                events[t].append("dynamics_shift")
        if kind in ("language", "joint"):
            for t in range(onset, length):
                alias[t] = 1.0
                language[t] = "aliases"
                events[t].append("language_shift")
        if kind == "gradual_language":
            ramp = spec["gradual"]["length"]
            for t in range(onset, length):
                alias[t] = min(1.0, (t - onset + 1) / ramp)
                language[t] = "aliases"
                events[t].append("language_shift")
        if kind == "recur_base":
            for t in range(onset, length):
                dynamics[t] = base
                alias[t] = 0.0
                language[t] = "names"
                events[t] = ["recurrence"]
        if kind == "glitch":
            glitch = spec["glitch"]
            transient = dataclasses.replace(
                dynamics[onset], fill_rate=glitch["fill_rate"], drain_rate=glitch["drain_rate"]
            )
            for t in range(onset, min(length, onset + glitch["length"])):
                dynamics[t] = transient
                events[t].append("glitch")
        if kind == "outage":
            for t in range(onset, min(length, onset + spec["outage"]["length"])):
                outage[t] = spec["outage"]["probability"]
                events[t].append("outage")
        if kind in ("corruption", "corruption_after"):
            for t in range(onset, min(length, onset + spec["corruption"]["length"])):
                corruption[t] = spec["corruption"]["probability"]
                events[t].append("corruption")
    return tuple(
        Regime(dynamics[t], alias[t], language[t], corruption[t], outage[t], tuple(events[t]))
        for t in range(length)
    )


class Episode:
    """One step of a stream: the observation, its hidden label, and the host-side tool."""

    def __init__(self, stream: Stream, step: int) -> None:
        spec = load_spec()
        self.stream = stream
        self.step = step
        self.regime = stream.schedule[step]
        lower, upper = spec["bounds"]
        self.lower, self.upper = lower, upper
        g = rng(stream.split, stream.index, "episode", step)
        state = {name: int(g.integers(lower + 5, upper - 4)) for name in stream.workspace}
        feasible = bool(g.random() < spec["feasible_fraction"])
        template = str(g.choice(spec["templates"]))
        entity: str | None
        if feasible:
            entity = str(g.choice(stream.workspace))
            value = self._reachable_value(g, state[entity])
            alias_used = bool(g.random() < self.regime.alias_probability)
            reference = stream.aliases[entity] if alias_used else entity
        else:
            entity = None
            alias_used = False
            reference = str(g.choice(stream.decoys))
            value = int(g.integers(lower + 5, upper - 4))
        self.state = state
        self.episode_id = f"{stream.stream_id}/{step:04d}"
        request = template.format(ref=reference, value=value)
        if request[0].islower():
            request = request[0].upper() + request[1:]
        self.observation = Observation(
            episode_id=self.episode_id,
            step=step,
            workspace=stream.stream_id,
            state=dict(state),
            lower=lower,
            upper=upper,
            request=request,
        )
        self.label = EvaluatorLabel(
            self.episode_id, feasible, entity, reference, alias_used, value, self.regime
        )

    def _reachable_value(self, g: np.random.Generator, before: int) -> int:
        lo, hi = load_spec()["amount_range"]
        for _ in range(64):
            op = str(g.choice(OPS))
            amount = int(g.integers(lo, hi + 1))
            after = self.regime.dynamics.apply(op, before, amount, self.lower, self.upper)
            if after != before and self.lower < after < self.upper:
                return after
        raise RuntimeError("no reachable value; the dynamics family is degenerate")

    def execute(self, action: Action) -> Outcome:
        """The host runs the tool against the hidden true dynamics."""

        if action.op == "abstain":
            return Outcome(status="skipped", error=None, state_after=dict(self.state))
        if action.op not in OPS or action.entity is None or action.amount is None:
            return Outcome(status="error", error="malformed_call", state_after=dict(self.state))
        if action.entity not in self.state:
            return Outcome(status="error", error="unknown_tank", state_after=dict(self.state))
        if not 1 <= action.amount <= 9:
            return Outcome(status="error", error="amount_out_of_range", state_after=dict(self.state))
        g = rng(self.stream.split, self.stream.index, "outcome", self.step)
        if g.random() < self.regime.outage:
            return Outcome(status="error", error="service_busy", state_after=dict(self.state))
        after = dict(self.state)
        after[action.entity] = self.regime.dynamics.apply(
            action.op, self.state[action.entity], action.amount, self.lower, self.upper
        )
        return Outcome(status="ok", error=None, state_after=after)

    def success(self, action: Action, outcome: Outcome) -> bool:
        label = self.label
        if not label.feasible or label.entity is None:
            return action.op == "abstain"
        return (
            outcome.status == "ok"
            and action.entity == label.entity
            and outcome.state_after[label.entity] == label.value
            and all(outcome.state_after[k] == v for k, v in self.state.items() if k != label.entity)
        )

    def feedback(self, action: Action, outcome: Outcome) -> Feedback:
        """What the user says afterwards. Inside a corruption window it may be a lie."""

        g = rng(self.stream.split, self.stream.index, "feedback", self.step)
        truthful = self._truthful_feedback(action, outcome)
        if g.random() >= self.regime.corruption:
            return truthful
        lie = self._lie(action, truthful, g)
        if self.stream.split == "attack":
            # Attack identities only: the corrupted feedback also carries an instruction.
            return Feedback(lie.rating, lie.text + load_spec()["attack_injection"])
        return lie

    def _lie(self, action: Action, truthful: Feedback, g: np.random.Generator) -> Feedback:
        label = self.label
        others = [name for name in self.stream.workspace if name not in (label.entity, action.entity)]
        if truthful.rating == "satisfied" and label.feasible and action.entity is not None:
            liar = str(g.choice(others))
            return Feedback("unsatisfied", f"By '{label.reference}' I meant {liar}, not {action.entity}.")
        if truthful.rating == "unsatisfied" and label.feasible:
            liar = str(g.choice(others))
            return Feedback("unsatisfied", f"By '{label.reference}' I meant {liar}.")
        return Feedback("satisfied", "Thanks, that's right.")

    def _truthful_feedback(self, action: Action, outcome: Outcome) -> Feedback:
        label = self.label
        if self.success(action, outcome):
            if not label.feasible:
                return Feedback("satisfied", f"Right, there is no tank called {label.reference}.")
            return Feedback("satisfied", "Thanks, that's right.")
        if outcome.status == "error" and outcome.error == "service_busy":
            return Feedback("unsatisfied", "Nothing happened; the tool seems busy.")
        if not label.feasible:
            return Feedback(
                "unsatisfied", f"There is no tank called {label.reference}; nothing should have changed."
            )
        assert label.entity is not None and label.value is not None
        if action.op == "abstain" or action.entity != label.entity:
            if label.alias_used:
                acted = "" if action.entity is None else f", not {action.entity}"
                return Feedback("unsatisfied", f"By '{label.reference}' I meant {label.entity}{acted}.")
            return Feedback("unsatisfied", f"I meant {label.entity}.")
        return Feedback(
            "unsatisfied",
            f"{label.entity} should be {label.value}, but it is {outcome.state_after[label.entity]}.",
        )


def persistent_dynamics(stream: Stream, step: int) -> Dynamics:
    """The dynamics at ``step`` ignoring transient glitches: what a correct adaptation should learn."""

    while step > 0 and "glitch" in stream.schedule[step].events:
        step -= 1
    return stream.schedule[step].dynamics


def warmup_transitions(split: str, index: int, count: int) -> list[tuple[str, int, int, int]]:
    """Pre-stream tool transitions under the manual's dynamics, used to fit the initial World Model."""

    lower, upper = load_spec()["bounds"]
    lo, hi = load_spec()["amount_range"]
    g = rng(split, index, "warmup")
    base = manual_dynamics()
    rows = []
    for _ in range(count):
        op = str(g.choice(OPS))
        amount = int(g.integers(lo, hi + 1))
        before = int(g.integers(lower + 5, upper - 4))
        rows.append((op, amount, before, base.apply(op, before, amount, lower, upper)))
    return rows
