"""Iteration 0002's intervention: a bounded previous-signed-error input.

Input 3 is ``(revealed - scored prediction) / displacement_scale``. It is
unbounded, and diagnosis round 3 showed it closing a runaway loop: once the
model's errors grow, input 3 grows (to ~214 in diverged cells against ~4 in
stable ones), saturates the state and produces larger errors (H16). With the
input zeroed the runaway nearly disappears (5 of 160 long coarse cells
against 60).

The intervention replaces input 3 by ``clip(e, -K, K)``. Whenever ``|e| <= K``
the agent is *bitwise identical* to the champion: same equations, same 994
parameters, same initialization, same optimizer and hyperparameters, same
state footprint. Only the error the agent feeds itself is capped.

Why this is not a gradient trick: the clipped quantity is an *input*, built
from a public constant and an observation the agent already received. No
evaluator quantity is involved, and the training target is untouched.

The bound travels with the agent. :class:`BoundedErrorAgent` overrides
``branch`` (AAA-1K's base returns a plain :class:`NeuralAgent`, which would
silently drop the bound in every online/frozen branch, adaptation trunk and
retention probe), serializes ``error_input_bound`` into its state, and refuses
to load a state saved under a different bound. The bound therefore also enters
the complete interaction-state hash that decides branch identity.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from research.aaa_1k.agents import NeuralAgent

from .harness import InstrumentedAgent

PUBLIC_BOUND = 8.0
"""``K`` for candidate c4, fixed by public arithmetic rather than by outcomes.

A 4-step gap at the maximum speed (0.32) with no velocity memory at all
leaves a revealed error of about 4 x 0.32 x 0.02 = 0.0256, i.e. 6.4 units of
``displacement_scale`` (0.004). A speed change of factor 1.65 at maximum speed
is under 1.1 units, and one coarse quantum is 1.25 units. K = 8 sits above
every error these benchmarks can legitimately produce in one step; values
beyond it only occur once predictions have already come loose.
"""

TIGHT_BOUND = 4.0
"""``K`` for candidate c5: the median per-cell maximum of ``|input 3|`` in the
champion's *stable* long-horizon cells in diagnosis round 3 (3.85), rounded up.
A tighter cap that clips legitimate occlusion-reveal errors."""


class BoundedErrorAgent(InstrumentedAgent):
    """An instrumented AAA-1K agent whose previous-error input is clipped to ``[-K, K]``."""

    def __init__(self, model: Any, *, name: str, error_input_bound: float, **options: Any) -> None:
        bound = float(error_input_bound)
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("error_input_bound must be positive and finite")
        super().__init__(model, name=name, **options)
        self.error_input_bound = bound

    def accept_observation(self, observation: float | None) -> None:
        super().accept_observation(observation)
        bound = self.error_input_bound
        self.previous_signed_error = max(-bound, min(bound, self.previous_signed_error))

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["error_input_bound"] = self.error_input_bound
        return state

    def load_state(self, state: Mapping[str, Any]) -> None:
        if state.get("error_input_bound") != self.error_input_bound:
            raise ValueError(
                f"checkpoint error_input_bound {state.get('error_input_bound')!r} does not match "
                f"this agent's {self.error_input_bound!r}"
            )
        previous = state.get("previous_signed_error")
        if previous is not None and abs(float(previous)) > self.error_input_bound:
            raise ValueError("checkpoint previous_signed_error exceeds the declared bound")
        super().load_state({key: value for key, value in state.items() if key != "error_input_bound"})

    def branch(self, *, name: str, update_enabled: bool) -> NeuralAgent:
        clone = BoundedErrorAgent(
            self.model.clone(),
            name=name,
            error_input_bound=self.error_input_bound,
            scales=self.scales,
            update_enabled=update_enabled,
            reflect=self.reflect,
            unfold_target=self.unfold_target,
        )
        clone.load_state(self.state_dict())
        clone.name = name
        clone.update_enabled = update_enabled
        return clone
