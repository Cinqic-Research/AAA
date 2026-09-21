"""Named experimental arms, built identically wherever they appear.

An arm is a name plus a deterministic construction rule from an
initialization seed. Keeping every arm in one registry means a diagnostic, a
development comparison, an attack and the confirmation run cannot quietly
build "the champion" in four slightly different ways.

``CHAMPION_CONFIGURATION`` is the reviewed AAA-1K selection. It is not read
from a file here so that the confirmation code path has no file dependency
that could change after the freeze; the loop's champion verification and its
tests assert that it equals the committed development-selection evidence.

Arms whose name starts with ``probe:`` are *diagnostic probes*: they exist to
test a hypothesis and are never eligible to become a challenger.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from research.aaa_1k.agents import (
    ConstantMotionAgent,
    DeadReckoningAgent,
    PersistenceAgent,
    RLSAgent,
    WindowedLinearFitAgent,
)
from research.aaa_1k.controls import StatelessMLPControl, VanillaRNNControl
from research.aaa_1k.model import AAA1KGRU

from .harness import InstrumentedAgent

CHAMPION_CONFIGURATION: dict[str, Any] = {
    "learning_rate": 0.03,
    "tbptt_steps": 4,
    "error_loss_weight": 0.25,
    "gradient_clip": 10.0,
}
UNGATED_CONFIGURATION: dict[str, Any] = {
    "learning_rate": 0.01,
    "tbptt_steps": 4,
    "error_loss_weight": 0.25,
    "gradient_clip": None,
}
STATELESS_CONFIGURATION: dict[str, Any] = dict(UNGATED_CONFIGURATION)


def gru(seed: int, **overrides: Any) -> AAA1KGRU:
    return AAA1KGRU(seed=seed, **{**CHAMPION_CONFIGURATION, **overrides})


def gru_with_keep_bias(seed: int, keep_bias: float, **overrides: Any) -> AAA1KGRU:
    """The champion GRU with its keep-gate bias initialized to ``keep_bias``.

    Only the *initial value* of ``b_z`` changes: same equations, same 994
    parameters, same optimizer, same state. Everything else -- every other
    parameter's initial value included -- is drawn exactly as the champion
    draws it from the same seed, so the two differ in one number per unit.
    """

    model = gru(seed, **overrides)
    model.parameters["b_z"][:] = float(keep_bias)
    return model


@dataclass(frozen=True)
class ArmSpec:
    name: str
    build: Callable[[int], Any]
    kind: str
    parameters: int | None = None
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)


def _neural(model: Any, name: str, *, update_enabled: bool = True) -> InstrumentedAgent:
    return InstrumentedAgent(model, name=name, update_enabled=update_enabled)


def _registry() -> dict[str, ArmSpec]:
    specs: list[ArmSpec] = [
        ArmSpec(
            "gru", lambda s: _neural(gru(s), "gru"), "champion", 994, "Champion 0", CHAMPION_CONFIGURATION
        ),
        ArmSpec(
            "gru_frozen",
            lambda s: _neural(gru(s), "gru_frozen", update_enabled=False),
            "champion_ablation",
            994,
            "Champion 0 with weight updates disabled (hidden state still runs)",
        ),
        ArmSpec(
            "gru_state_reset",
            lambda s: _neural(gru(s, reset_state_every_step=True), "gru_state_reset"),
            "champion_ablation",
            994,
            "Champion 0 with hidden state cleared before every step",
        ),
        ArmSpec(
            "gru_no_error_input",
            lambda s: _neural(gru(s, zero_error_input=True), "gru_no_error_input"),
            "champion_ablation",
            994,
            "Champion 0 with input 3 forced to zero",
        ),
        ArmSpec(
            "probe:gru_lr0.01",
            lambda s: _neural(gru(s, learning_rate=0.01), "probe:gru_lr0.01"),
            "probe",
            994,
            "Champion 0 at the ungated control's learning rate",
        ),
        ArmSpec(
            "probe:gru_T16",
            lambda s: _neural(gru(s, tbptt_steps=16), "probe:gru_T16"),
            "probe",
            994,
            "Champion 0 with a 16-step truncation horizon",
        ),
        ArmSpec(
            "probe:gru_keep_bias_2",
            lambda s: _neural(gru_with_keep_bias(s, 2.0), "probe:gru_keep_bias_2"),
            "probe",
            994,
            "Champion 0 with the keep-gate bias initialized at +2 (initial keep ~0.88)",
        ),
        ArmSpec(
            "rnn28",
            lambda s: _neural(VanillaRNNControl(seed=s, **UNGATED_CONFIGURATION), "rnn28"),
            "control",
            954,
            "the Q4 ungated control on its own rule-selected configuration",
            UNGATED_CONFIGURATION,
        ),
        ArmSpec(
            "probe:rnn28_lr0.03",
            lambda s: _neural(
                VanillaRNNControl(seed=s, **{**UNGATED_CONFIGURATION, "learning_rate": 0.03}),
                "probe:rnn28_lr0.03",
            ),
            "probe",
            954,
            "the ungated control at the gated model's learning rate",
        ),
        ArmSpec(
            "probe:rnn16",
            lambda s: _neural(
                VanillaRNNControl(seed=s, hidden_size=16, **UNGATED_CONFIGURATION), "probe:rnn16"
            ),
            "probe",
            354,
            "the ungated control at the gated model's hidden width",
        ),
        ArmSpec(
            "mlp",
            lambda s: _neural(StatelessMLPControl(seed=s, **STATELESS_CONFIGURATION), "mlp"),
            "control",
            982,
            "the stateless matched-capacity control",
            STATELESS_CONFIGURATION,
        ),
        ArmSpec("rls_online", lambda _s: RLSAgent(name="rls_online"), "baseline", 3),
        ArmSpec("persistence", lambda _s: PersistenceAgent(), "baseline", 0),
        ArmSpec(
            "constant_motion_reflected",
            lambda _s: ConstantMotionAgent(name="constant_motion_reflected", reflect=True),
            "baseline",
            0,
        ),
        ArmSpec("dead_reckoning", lambda _s: DeadReckoningAgent(), "baseline", 0),
        ArmSpec("linear_fit", lambda _s: WindowedLinearFitAgent(), "baseline", 0),
    ]
    return {spec.name: spec for spec in specs}


ARMS: dict[str, ArmSpec] = _registry()


def register(spec: ArmSpec) -> None:
    """Add a challenger arm. Names are unique for the lifetime of the process."""

    if spec.name in ARMS:
        raise ValueError(f"arm {spec.name!r} is already registered")
    ARMS[spec.name] = spec


@dataclass(frozen=True)
class ArmSet:
    """A picklable factory: builds the named arms for one initialization seed."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = [name for name in self.names if name not in ARMS]
        if unknown:
            raise ValueError(f"unknown arms {unknown}")
        if len(set(self.names)) != len(self.names):
            raise ValueError("arm names must be unique")

    def __call__(self, seed: int) -> Sequence[Any]:
        agents = [ARMS[name].build(seed) for name in self.names]
        for name, agent in zip(self.names, agents, strict=True):
            if agent.name != name:
                raise RuntimeError(f"arm {name} built an agent named {agent.name}")
        return agents


def initial_parameters_equal_except(first: Any, second: Any, allowed: Sequence[str]) -> bool:
    """True when two models' initial arrays differ only in ``allowed`` names."""

    for name, array in first.parameters.items():
        if name in allowed:
            continue
        if not np.array_equal(array, second.parameters[name]):
            return False
    return True
