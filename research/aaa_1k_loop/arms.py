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


def gru_with_gate_biases(
    seed: int, *, keep_bias: float = 0.0, reset_bias: float = 0.0, **overrides: Any
) -> AAA1KGRU:
    """The champion GRU with its keep- and/or reset-gate biases initialized off zero.

    Only *initial values* of ``b_z`` and ``b_r`` change: same equations, same
    994 parameters, same optimizer, same state. Every other parameter's
    initial value is drawn exactly as the champion draws it from the same seed.
    """

    model = gru(seed, **overrides)
    model.parameters["b_z"][:] = float(keep_bias)
    model.parameters["b_r"][:] = float(reset_bias)
    return model


def gru_with_keep_bias(seed: int, keep_bias: float, **overrides: Any) -> AAA1KGRU:
    return gru_with_gate_biases(seed, keep_bias=keep_bias, **overrides)


SPLIT_KEEP_BIAS = tuple([-2.0] * 8 + [2.0] * 8)
"""Candidate c3's keep-gate bias: units 0-7 fast (-2, keep ~0.12), units 8-15 holding (+2, keep ~0.88)."""


def gru_with_split_keep_bias(seed: int, **overrides: Any) -> AAA1KGRU:
    """The champion GRU with the keep-gate bias initialized to :data:`SPLIT_KEEP_BIAS`.

    Glorot draws are exchangeable across units, so which half is fast is
    arbitrary; the first half is. Only the initial ``b_z`` differs from the
    champion's.
    """

    model = gru(seed, **overrides)
    model.parameters["b_z"][:] = np.asarray(SPLIT_KEEP_BIAS, dtype=float)
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


def _candidate(
    name: str, bias: float | None, *, update_enabled: bool = True, **overrides: Any
) -> Callable[[int], Any]:
    """``bias=None`` builds the split keep-bias candidate (c3)."""

    def build(seed: int) -> InstrumentedAgent:
        model = (
            gru_with_split_keep_bias(seed, **overrides)
            if bias is None
            else gru_with_gate_biases(seed, keep_bias=bias, **overrides)
        )
        return _neural(model, name, update_enabled=update_enabled)

    return build


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
            "probe:gru_keep_bias_-2",
            lambda s: _neural(gru_with_gate_biases(s, keep_bias=-2.0), "probe:gru_keep_bias_-2"),
            "probe",
            994,
            "keep-gate bias initialized at -2 (initial keep ~0.12): less leak, more recurrent drive",
        ),
        ArmSpec(
            "probe:gru_reset_bias_2",
            lambda s: _neural(gru_with_gate_biases(s, reset_bias=2.0), "probe:gru_reset_bias_2"),
            "probe",
            994,
            "reset-gate bias initialized at +2 (initial reset ~0.88): more recurrent drive into the candidate",
        ),
        ArmSpec(
            "probe:gru_keep_-2_reset_2",
            lambda s: _neural(
                gru_with_gate_biases(s, keep_bias=-2.0, reset_bias=2.0), "probe:gru_keep_-2_reset_2"
            ),
            "probe",
            994,
            "both: the gated core starts close to an ungated tanh recurrence",
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
    for bias in (-1.0, -2.0):
        base = f"cand:keep_bias_{bias:g}"
        specs.append(
            ArmSpec(
                base,
                _candidate(base, bias),
                "candidate",
                994,
                f"challenger candidate: champion with keep-gate bias initialized at {bias:g}",
                {**CHAMPION_CONFIGURATION, "keep_bias_init": bias},
            )
        )
        specs.append(
            ArmSpec(
                f"{base}:frozen",
                _candidate(f"{base}:frozen", bias, update_enabled=False),
                "candidate_ablation",
                994,
            )
        )
        specs.append(
            ArmSpec(
                f"{base}:state_reset",
                _candidate(f"{base}:state_reset", bias, reset_state_every_step=True),
                "candidate_ablation",
                994,
            )
        )
        specs.append(
            ArmSpec(
                f"{base}:no_error_input",
                _candidate(f"{base}:no_error_input", bias, zero_error_input=True),
                "candidate_ablation",
                994,
            )
        )
    base = "cand:keep_bias_split"
    specs.append(
        ArmSpec(
            base,
            _candidate(base, None),
            "candidate",
            994,
            "challenger candidate: keep-gate bias initialized -2 on units 0-7 and +2 on units 8-15",
            {**CHAMPION_CONFIGURATION, "keep_bias_init": list(SPLIT_KEEP_BIAS)},
        )
    )
    for suffix, options in (
        ("frozen", {"update_enabled": False}),
        ("state_reset", {"reset_state_every_step": True}),
        ("no_error_input", {"zero_error_input": True}),
    ):
        specs.append(
            ArmSpec(
                f"{base}:{suffix}", _candidate(f"{base}:{suffix}", None, **options), "candidate_ablation", 994
            )
        )
    specs.append(
        ArmSpec(
            "probe:gru_keep_bias_2:occlusion",
            lambda s: _neural(gru_with_keep_bias(s, 2.0), "probe:gru_keep_bias_2:occlusion"),
            "probe",
            994,
            "keep-bias +2 on every unit, to test whether holding units carry occlusion",
        )
    )
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
