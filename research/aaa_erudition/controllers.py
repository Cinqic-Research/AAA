"""Control policies that compete with the Erudition Model.

These are the matched controls: they see exactly the evidence features the
Erudition Model sees, choose from exactly the same condition-restricted
menu, and go through the same gate. They differ only in how the decision is
made.

* ``NeverAdapt`` -- the frozen system;
* ``AlwaysAdapt`` -- adapts on a fixed timer without looking at evidence, the
  "change equals improvement" strawman the gate must contain;
* ``Heuristic`` -- an auditable rule set written before any development
  result, the simple policy the literature says a learned controller must beat
  (docs/juniper1/literature.md, learned controllers).
"""

from __future__ import annotations

from .lifecycle import FEATURES, ControllerView, Decision

INDEX = {name: i for i, name in enumerate(FEATURES)}


class NeverAdapt:
    name = "never"

    def decide(self, _view: ControllerView) -> Decision:
        return Decision("wait", {}, self.name)


class AlwaysAdapt:
    name = "always"
    PERIOD = 10

    def decide(self, view: ControllerView) -> Decision:
        if view.step == 0 or view.step % self.PERIOD:
            return Decision("wait", {}, self.name)
        for action in ("joint.new_alias", "wm.update", "lm.alias", "lm.experience"):
            if action in view.allowed:
                if (
                    action == "lm.alias"
                    and "lm.experience" in view.allowed
                    and (view.step // self.PERIOD) % 2
                ):
                    action = "lm.experience"
                return Decision(action, {}, self.name)
        return Decision("wait", {}, self.name)


class Heuristic:
    """Windowed thresholds on the evidence features.

    WM: at least half of the World Model's predictions missed among three or
    more executed calls in the last twelve steps, outside an outage -> recall a
    stored context if one explains the evidence better, otherwise a new context.
    LM: two or more recent feedback messages name a tank other than the one
    acted on -> learn aliases; in LM-only conditions, repeated
    self-inconsistency -> show the model recent tool results. A request that
    is not repeated for ``RETRY`` steps.

    Revised once in development after the first real-model pilot (training
    stream 0): the WM window grew from six to twelve steps, because a model that
    abstains executes too few calls for a six-step window, and the retry
    back-off was added after the original rules re-requested a rejected
    mechanism every cooldown (each costs Language Model calls). Both changes make the
    control stronger, not weaker.
    """

    name = "heuristic"
    RETRY = 10

    def __init__(self) -> None:
        self.last: dict[str, int] = {}

    def decide(self, view: ControllerView) -> Decision:
        f = view.features
        recent = f[-12:]
        acted = recent[recent[:, INDEX["outcome_ok"]] == 1.0]
        busy = recent[-6:, INDEX["error_busy"]].sum()
        wm_signal = len(acted) >= 3 and acted[:, INDEX["wm_miss"]].mean() >= 0.5 and busy == 0
        recall = wm_signal and acted[:, INDEX["surprise_other_gain"]].mean() > 0.3
        inconsistent = len(acted) >= 3 and (1.0 - acted[:, INDEX["consistent"]]).mean() >= 0.5 and busy == 0
        lm_signal = f[-8:, INDEX["feedback_names_other"]].sum() >= 2
        diagnosis = {"wm_signal": float(wm_signal), "lm_signal": float(lm_signal), "recall": float(recall)}

        def pick(*options: str) -> Decision:
            for option in options:
                if option in view.allowed and view.step - self.last.get(option, -99) >= self.RETRY:
                    self.last[option] = view.step
                    return Decision(option, diagnosis, self.name)
            return Decision("wait", diagnosis, self.name)

        if wm_signal and lm_signal:
            return pick(
                "joint.recall" if recall else "joint.new_alias",
                "wm.recall" if recall else "wm.new",
                "lm.alias",
            )
        if wm_signal:
            return pick(
                "joint.recall" if recall else "joint.new",
                "wm.recall" if recall else "wm.new",
                "lm.experience",
            )
        if lm_signal:
            return pick("lm.alias")
        if inconsistent:
            return pick("lm.experience")
        return Decision("wait", diagnosis, self.name)
