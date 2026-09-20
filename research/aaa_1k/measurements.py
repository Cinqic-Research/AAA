"""The two measurements that the first AAA-1K round got wrong.

Round 1 reported an adaptation result and a retention result. An adversarial
probe showed that neither measured what its name said:

* the online-versus-frozen advantage was 95% reproduced by branching at a point
  where nothing happened, so it measured continued learning (`AAA-153`);
* every arm returned to regime A better than it left it, which reads as "no
  forgetting" but is confounded with having had three times as much experience
  by then (`AAA-154`).

Both are design defects, not code defects, so both are fixed by designs rather
than by adjusting the wording of a claim.

Difference-of-differences adaptation
------------------------------------
Two streams, bit-identical until a declared step, after which one changes speed
and the other does not. One model is driven through the shared prefix, so the
trunk state at the branch is *the same object* for both members -- not merely
statistically similar. Then::

    advantage(variant) = frozen_error(variant) - online_error(variant)
    adaptation         = advantage(changed) - advantage(control)

Everything the two members have in common cancels, including the ordinary
benefit of continuing to learn. What is left is the part of the online arm's
advantage that exists *because the world changed*.

Retention against a frozen probe bank
-------------------------------------
A fixed bank of held-out regime-A episodes, generated once and never trained
on. At the end of each segment of the A/B/A stream the learner is cloned,
frozen, and run over the whole bank with its hidden state reset at the start of
each probe, so what is measured is what the *weights* know about regime A.

    forgetting     = probe(after B)  - probe(after A1)
    reacquisition  = probe(after A2) - probe(after A1)

Because the probe bank is identical at all three checkpoints, "more total
experience" cannot flatter the later measurements: they are the same questions,
asked again.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .agents import NeuralAgent
from .model import AAA1KGRU
from .runner import run_online_frozen_branch, run_stream
from .seeds import derive_seed
from .selection import Configuration
from .streams import Stream, aba_stream, motion_compat_stream, paired_change_streams

MEASUREMENT_SCHEMA = "aaa.1k.measurements.v1"

PROBE_BANK_SIZE = 8
PROBE_STEPS = 80


def _fresh_model(configuration: Configuration, model_seed_index: int) -> AAA1KGRU:
    return AAA1KGRU(seed=derive_seed("model_init", model_seed_index), **configuration.model_kwargs())


# ----------------------------------------------------------------------
# difference-of-differences adaptation
# ----------------------------------------------------------------------
def adaptation_difference_of_differences(
    configuration: Configuration,
    *,
    seed: int,
    model_seed_index: int = 0,
    steps: int = 240,
    change_step: int = 120,
    name: str = "aaa1k_gru",
) -> dict[str, Any]:
    """One paired trial: online-versus-frozen on a changed and an unchanged world."""

    changed, control = paired_change_streams(seed, steps=steps, change_step=change_step)
    outcomes: dict[str, dict[str, float]] = {}
    trunk_hashes: dict[str, str] = {}
    for label, stream in (("changed", changed), ("control", control)):
        trunk = NeuralAgent(_fresh_model(configuration, model_seed_index), name=name)
        outcome = run_online_frozen_branch(stream, trunk, branch_index=change_step)
        outcomes[label] = outcome.post_branch_advantage()
        trunk_hashes[label] = outcome.clone_state_hash

    # The prefixes are bit-identical and the model is identical, so the state at
    # the branch must be too. If it is not, the pairing is broken and the whole
    # difference-of-differences is meaningless, so this is checked rather than
    # assumed.
    if trunk_hashes["changed"] != trunk_hashes["control"]:
        raise RuntimeError(
            "paired trial diverged before the branch: the changed and control trunks "
            "must reach an identical model state"
        )

    adaptation = outcomes["changed"]["absolute_advantage"] - outcomes["control"]["absolute_advantage"]
    return {
        "seed": int(seed),
        "model_seed_index": int(model_seed_index),
        "change_step": change_step,
        "trunk_state_hash": trunk_hashes["changed"],
        "trunks_matched": True,
        "changed": outcomes["changed"],
        "control": outcomes["control"],
        "adaptation_effect": adaptation,
        "continued_learning_effect": outcomes["control"]["absolute_advantage"],
    }


# ----------------------------------------------------------------------
# retention against a frozen probe bank
# ----------------------------------------------------------------------
def probe_bank(*, size: int = PROBE_BANK_SIZE, steps: int = PROBE_STEPS) -> list[Stream]:
    """A fixed bank of held-out regime-A episodes.

    Drawn once from the benchmark-generation namespace, never trained on, and
    identical at every checkpoint of every stream. That is what makes the three
    measurements comparable.
    """

    return [
        motion_compat_stream(
            "bouncing", derive_seed("benchmark_generation", 1000 + index), steps=steps, change_step=None
        )
        for index in range(size)
    ]


def _probe(agent: NeuralAgent, bank: list[Stream], *, label: str) -> float:
    """Mean error of a frozen clone over the whole bank, hidden state reset."""

    errors: list[float] = []
    for index, stream in enumerate(bank):
        clone = agent.branch(name=f"probe_{label}_{index}", update_enabled=False)
        result = run_stream(stream, [clone], begin_episode=True)
        errors.append(result.mean_absolute_error(clone.name))
        if clone.model.update_count != agent.model.update_count:
            raise RuntimeError("a probe clone updated its weights; the probe is not read-only")
    return float(np.mean(errors))


def retention_trial(
    configuration: Configuration,
    *,
    seed: int,
    model_seed_index: int = 0,
    segment_steps: int = 200,
    bank: list[Stream] | None = None,
    name: str = "aaa1k_gru",
) -> dict[str, Any]:
    """Run one A/B/A stream, probing regime-A ability at each segment boundary."""

    bank = bank if bank is not None else probe_bank()
    stream = aba_stream(seed, segment_steps=segment_steps)
    agent = NeuralAgent(_fresh_model(configuration, model_seed_index), name=name)

    checkpoints = (segment_steps, 2 * segment_steps, 3 * segment_steps)
    probes: dict[str, float] = {}
    segment_errors: dict[str, float] = {}
    labels = ("after_A1", "after_B", "after_A2")
    start = 0
    for label, stop in zip(labels, checkpoints, strict=True):
        result = run_stream(stream, [agent], start=start, stop=stop, begin_episode=(start == 0))
        segment_errors[label] = result.mean_absolute_error(name)
        probes[label] = _probe(agent, bank, label=label)
        start = stop

    return {
        "seed": int(seed),
        "model_seed_index": int(model_seed_index),
        "segment_steps": segment_steps,
        "probe_bank_size": len(bank),
        "probe_error": probes,
        "in_stream_error": segment_errors,
        "forgetting": probes["after_B"] - probes["after_A1"],
        "reacquisition_gap": probes["after_A2"] - probes["after_A1"],
        "recovery_from_B": probes["after_B"] - probes["after_A2"],
        "updates": int(agent.model.update_count),
    }
