"""Adaptation mechanisms: how a candidate state is produced from evidence.

A mechanism is a pure function of the parent state and recorded evidence. It
never decides *whether* to adapt; the Erudition controller does that, and
the gate decides whether the result is kept. Every mechanism here is
available to every condition that is allowed to adapt the component it
targets.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict

from .language import AdapterState, Backend, Note, extract_correction
from .toolshift import manual_dynamics
from .world import Transition, WorldState

LM_MECHANISMS = (
    "lm.alias_from_feedback",
    "lm.dynamics_from_experience",
    "lm.dynamics_from_wm",
    "lm.retract_dynamics",
)
WM_MECHANISMS = ("wm.new_context", "wm.update_in_place", "wm.recall")

ALIAS_MIN_SUPPORT = 2
ALIAS_MIN_AGREEMENT = 2 / 3
OBSERVATION_NOTES = 6
WM_NOTE_MAX_SD = 1.0


@dataclasses.dataclass(frozen=True)
class Correction:
    episode_id: str
    phrase: str
    tank: str


def corrections_from_feedback(
    backend: Backend, evidence: list[tuple[str, str, str]], workspace: tuple[str, ...]
) -> list[Correction]:
    """Language Model extraction of (phrase, tank) from unsatisfied feedback, structurally verified."""

    found = []
    for episode_id, request, feedback in evidence:
        extracted = extract_correction(backend, feedback, request, workspace)
        if extracted is not None:
            found.append(Correction(episode_id, *extracted))
    return found


def alias_from_feedback(parent: AdapterState, corrections: list[Correction]) -> AdapterState:
    """Keep a phrase -> tank note only when independent corrections agree."""

    by_phrase: dict[str, list[Correction]] = defaultdict(list)
    for correction in corrections:
        by_phrase[correction.phrase].append(correction)
    learned: dict[str, Note] = {}
    for phrase, items in sorted(by_phrase.items()):
        tank, count = Counter(c.tank for c in items).most_common(1)[0]
        if count >= ALIAS_MIN_SUPPORT and count / len(items) >= ALIAS_MIN_AGREEMENT:
            evidence = tuple(sorted(c.episode_id for c in items if c.tank == tank))
            learned[phrase] = Note("alias", phrase, {"tank": tank}, "feedback", count, evidence)
    kept = tuple(n for n in parent.notes if not (n.kind == "alias" and n.key in learned))
    return AdapterState(kept + tuple(learned[p] for p in sorted(learned)))


def dynamics_from_experience(
    parent: AdapterState, transitions: list[tuple[str, str, Transition, str]]
) -> AdapterState:
    """Show the model recent observed tool results and let it infer the behaviour.

    ``transitions`` holds (episode_id, tank, transition, stated_kind) rows; only
    the most recent ``OBSERVATION_NOTES`` are rendered.
    """

    recent = transitions[-OBSERVATION_NOTES:]
    observed = tuple(
        Note(
            "observation",
            f"{episode_id}",
            {"op": t[0], "tank": tank, "n": t[1], "before": t[2], "after": t[3]},
            "experience",
            1,
            (episode_id,),
        )
        for episode_id, tank, t, _ in recent
    )
    kept = tuple(n for n in parent.notes if n.kind not in ("observation", "dynamics"))
    return AdapterState(kept + observed)


def dynamics_from_wm(parent: AdapterState, world: WorldState) -> AdapterState:
    """Distil the World Model's active context into dynamics notes where it departs from the manual.

    A note is written only for operations the World Model is confident about;
    where it agrees with the manual, any earlier dynamics note is withdrawn.
    """

    context = world.context()
    notes = [
        Note(
            "dynamics",
            op,
            {"rate": rate, "offset": offset},
            "wm",
            context.posteriors[op].n,
            (context.context_id,),
        )
        for op, (rate, offset) in sorted(departures_from_manual(world).items())
    ]
    kept = tuple(n for n in parent.notes if n.kind not in ("dynamics", "observation"))
    return AdapterState(kept + tuple(notes))


def departures_from_manual(world: WorldState) -> dict[str, tuple[int, int]]:
    manual = manual_dynamics()
    documented = {
        "fill": (manual.fill_rate, manual.fill_bonus),
        "drain": (manual.drain_rate, manual.drain_fee),
    }
    return {op: v for op, v in world.behaviour(WM_NOTE_MAX_SD).items() if v != documented[op]}


def retract_dynamics(parent: AdapterState) -> AdapterState:
    return AdapterState(tuple(n for n in parent.notes if n.kind not in ("dynamics", "observation")))
