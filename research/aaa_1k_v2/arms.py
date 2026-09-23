"""Named arms of the final AAA-1K tournament, built identically everywhere.

An :class:`ArmSpec` fixes the core (kind and sizes), the feature set, the
target rule, the learning rule, the TBPTT horizon, the initialization options
and the hyperparameters. Development selection may only choose among values
listed in the phase protocol; once selected, an arm's specification is
recorded in the development artifact and frozen before confirmation.

Every candidate's trainable parameter count is recomputed from its arrays
(:func:`audit`) and must not exceed :data:`research.aaa_1k_v2.PARAMETER_CAP`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from . import PARAMETER_CAP
from .cores import Core, build_core
from .engine import CellConfig

TEMPLATE_KINDS = ("reference", "candidate", "control", "ablation", "probe")


@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str
    core: tuple[tuple[str, Any], ...]
    features: str = "v1"
    target_rule: str = "reach_gated"
    rule: str = "live"
    tbptt_steps: int = 4
    config: CellConfig = field(default_factory=lambda: CellConfig(learning_rate=0.03))
    init: tuple[tuple[str, Any], ...] = ()
    update_enabled: bool = True
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in TEMPLATE_KINDS:
            raise ValueError(f"unknown arm kind {self.kind!r}")

    def build_core(self) -> Core:
        return build_core(**dict(self.core))

    def init_options(self) -> dict[str, Any]:
        return dict(self.init)

    def with_config(self, **changes: Any) -> ArmSpec:
        return replace(self, config=replace(self.config, **changes))

    def variant(self, name: str, kind: str = "ablation", **changes: Any) -> ArmSpec:
        return replace(self, name=name, kind=kind, **changes)

    def to_dict(self) -> dict[str, Any]:
        core = self.build_core()
        return {
            "name": self.name,
            "kind": self.kind,
            "core": core.describe(),
            "features": self.features,
            "target_rule": self.target_rule,
            "rule": self.rule,
            "tbptt_steps": self.tbptt_steps if self.rule != "rtrl" else None,
            "config": self.config.to_dict(),
            "init": dict(self.init),
            "update_enabled": self.update_enabled,
            "description": self.description,
        }


def core_spec(
    kind: str, inputs: int, hidden: int, outputs: int = 2, **extra: Any
) -> tuple[tuple[str, Any], ...]:
    return tuple(
        sorted({"kind": kind, "inputs": inputs, "hidden": hidden, "outputs": outputs, **extra}.items())
    )


CHAMPION_1 = ArmSpec(
    name="c1_champion1",
    kind="reference",
    core=core_spec("gru", 3, 16),
    features="v1",
    target_rule="reach_gated",
    rule="live",
    tbptt_steps=4,
    config=CellConfig(learning_rate=0.03, gradient_clip=10.0, error_loss_weight=0.25),
    description="Champion 1 exactly: AAA1KGRU-3x16x2, reach-gated unfolding, live TBPTT-4, SGD 0.03, clip 10",
)
"""Champion 1's frozen specification (``docs/evidence/aaa1k_loop_0006/champion_1.json``)."""

CHAMPION_0 = replace(
    CHAMPION_1,
    name="c0_champion0",
    kind="control",
    target_rule="own",
    description="Champion 0: identical except own-prediction unfolding (the M2 frame lock)",
)

CHRONO_TMAX = 64


def chrono_keep_bias(hidden: int, seed: int, t_max: int = CHRONO_TMAX) -> list[float]:
    """Chrono initialization (Tallec & Ollivier 2018) in AAA's keep-gate polarity.

    With keep gate ``z`` the characteristic timescale is ``1 / (1 - z)``;
    drawing it uniformly from ``[1, t_max - 1]`` gives ``b_z = log(tau - 1)``
    clipped below at ``log(1e-3)`` for ``tau = 1``. The draw uses its own
    generator so the Glorot draws are unchanged.
    """

    import numpy as np

    rng = np.random.default_rng([seed, 17])
    tau = rng.uniform(1.0, t_max - 1.0, size=hidden)
    return [math.log(max(t - 1.0, 1e-3)) for t in tau]


CANDIDATE_TEMPLATES: dict[str, ArmSpec] = {
    "gru_v1_keep-2": replace(
        CHAMPION_1,
        name="gru_v1_keep-2",
        kind="candidate",
        init=(("keep_bias", -2.0),),
        description=(
            "Champion 1 with the keep-gate bias initialized at -2 (M1 hypothesis: the zero-bias operating point "
            "cannot track the coarse family's sub-quantum phase); loop candidate c2 re-attacked on Champion 1's rule"
        ),
    ),
    "gru_v2": ArmSpec(
        name="gru_v2",
        kind="candidate",
        core=core_spec("gru", 4, 15),
        features="v2",
        description="GRU with the explicit observed flag (R-10), 15 hidden units to stay under the cap",
    ),
    "elman_v2": ArmSpec(
        name="elman_v2",
        kind="candidate",
        core=core_spec("elman", 4, 28),
        features="v2",
        description="ungated tanh recurrence at the widest state the cap allows, with the observed flag",
    ),
    "mgu_v2": ArmSpec(
        name="mgu_v2",
        kind="candidate",
        core=core_spec("mgu", 4, 19),
        features="v2",
        description="minimal gated unit (one gate) with the observed flag",
    ),
    "lru_v2": ArmSpec(
        name="lru_v2",
        kind="candidate",
        core=core_spec("lru", 4, 24, readout_hidden=13),
        features="v2",
        rule="rtrl",
        tbptt_steps=0,
        description=(
            "complex-diagonal linear recurrence, |lambda|<1 by construction, exact untruncated online gradient "
            "(RTRL with diagonal traces), tanh readout; long-memory/stability hypothesis"
        ),
    ),
}

CONTROL_TEMPLATES: dict[str, ArmSpec] = {
    "mlp_v2": ArmSpec(
        name="mlp_v2",
        kind="control",
        core=core_spec("mlp", 4, 27),
        features="v2",
        rule="live",
        tbptt_steps=1,
        description="stateless matched-capacity control with the observed flag",
    ),
}


def ablations(arm: ArmSpec) -> dict[str, ArmSpec]:
    """One-mechanism ablations sharing the parent's configuration exactly."""

    out = {
        f"{arm.name}:state_reset": arm.variant(
            f"{arm.name}:state_reset", config=replace(arm.config, reset_state_every_step=True)
        ),
        f"{arm.name}:no_error_input": arm.variant(
            f"{arm.name}:no_error_input", config=replace(arm.config, zero_input=(2,))
        ),
    }
    if arm.features == "v2":
        out[f"{arm.name}:no_observed_flag"] = arm.variant(
            f"{arm.name}:no_observed_flag", config=replace(arm.config, zero_input=(3,))
        )
    if arm.build_core().outputs == 2:
        out[f"{arm.name}:no_error_head_loss"] = arm.variant(
            f"{arm.name}:no_error_head_loss", config=replace(arm.config, error_loss_weight=0.0)
        )
    return out


ADAPTER_DIAGNOSTIC_COUNTERS = ("trained_steps", "skipped_targets", "mirror_steps", "gated_steps")


def adapter_state_scalars() -> int:
    """Per-cell interaction state the dot adapter carries, counted from a live adapter.

    Diagnostic counters are excluded; they never influence a prediction or update.
    """

    from aaa.compute import resolve_backend

    from .adapters import DotAdapter

    adapter = DotAdapter(resolve_backend("cpu"), 1)
    return sum(
        1
        for key, value in vars(adapter).items()
        if hasattr(value, "shape") and value.shape == (1,) and key not in ADAPTER_DIAGNOSTIC_COUNTERS
    )


def audit(arm: ArmSpec, tbptt_steps: int | None = None) -> dict[str, Any]:
    """Recount parameters from arrays and account every adaptive-state category."""

    import numpy as np

    core = arm.build_core()
    arrays = core.init_cell(0, **arm.init_options())
    counted = int(sum(int(np.prod(array.shape)) for array in arrays.values()))
    declared = core.parameter_count()
    horizon = arm.tbptt_steps if tbptt_steps is None else tbptt_steps
    cache = core.cache_size() * (horizon if arm.rule != "rtrl" else 0)
    return {
        "arm": arm.name,
        "trainable_parameters": counted,
        "declared_from_shapes": declared,
        "agree": counted == declared,
        "within_cap": counted <= PARAMETER_CAP,
        "hidden_state_scalars": core.state_size(),
        "optimizer_state_scalars": 0,
        "tbptt_buffer_scalars": cache,
        "eligibility_trace_scalars": core.trace_size(),
        "adapter_state_scalars": adapter_state_scalars(),
        "total_adaptive_state_scalars": counted + core.state_size() + cache + core.trace_size(),
        "total_including_adapter": counted
        + core.state_size()
        + cache
        + core.trace_size()
        + adapter_state_scalars(),
    }


def audit_all(arms: Mapping[str, ArmSpec]) -> list[dict[str, Any]]:
    return [audit(arm) for arm in arms.values()]
