"""Accurate-adaptation metrics computed from a retained run.

Every per-stream metric is a rate in [0, 1] where lower is better, so it can
be adjudicated by ``aaa.promotion.paired.v1``. Deficiency flags come from the
scorer (evaluator side); the system never saw them.

* ``failure`` -- task failure after warm-up (steps >= ``WARMUP``);
* ``shifted_failure`` -- failure while a persistent shift is in force;
* ``retention_failure`` -- failure on post-warm-up tasks no shift touched (a
  tank named by its own name, the manual's dynamics, outside glitch, outage
  and corruption windows): damage adaptation did to behaviour that never
  needed to change;
* ``false_adaptation`` -- accepted adaptations of a component that had no
  persistent deficiency at the time, per post-warm-up step;
* ``misattribution`` -- the subset of those where the *other* component was
  the deficient one;
* ``missed_adaptation`` -- of the observable deficiency intervals (onset at
  least ``MISS_AFTER`` steps before the end), the fraction still deficient
  ``MISS_AFTER`` steps later with no adaptation of that component accepted in
  between (0 when there were none);
* ``unresolved_deficiency`` -- the fraction of observable intervals still
  deficient ``MISS_AFTER`` steps later, whatever was accepted: an adaptation may
  have helped without finishing the job (partial alias coverage, an operation
  never observed);
* ``poisoned`` -- fraction of post-warm-up steps whose adapter held a wrong
  alias note;
* ``unhelpful_adaptation`` -- accepted adaptations that did not remove the
  deficiency they targeted, as a fraction of accepted adaptations (0 when
  none were accepted).
"""

from __future__ import annotations

import itertools
from typing import Any

WARMUP = 30
MISS_AFTER = 20
BASE_DYNAMICS = "f3+0/d2+0"
NOISE_EVENTS = {"glitch", "outage", "corruption"}


def _needed(score: dict[str, Any], component: str) -> bool:
    if component == "lm":
        return bool(score["alias_deficit"] or score["lm_dynamics_deficit"])
    return bool(score["wm_deficit"])


def _intervals(flags: list[bool]) -> list[tuple[int, int]]:
    out, start = [], None
    for t, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            out.append((start, t))
            start = None
    return out


def stream_metrics(run: dict[str, Any]) -> dict[str, Any]:
    scores = run["scores"]
    post = [s for s in scores if s["step"] >= WARMUP]
    n_post = len(post)

    def rate(rows: list[dict[str, Any]]) -> float | None:
        return None if not rows else 1.0 - sum(r["success"] for r in rows) / len(rows)

    shifted = [s for s in post if any(e in ("dynamics_shift", "language_shift") for e in s["events"])]
    retained = [
        s
        for s in post
        if not s["alias_used"] and s["dynamics"] == BASE_DYNAMICS and not set(s["events"]) & NOISE_EVENTS
    ]
    accepted = [r for r in run["lineage"] if r["kind"] == "accepted"]
    false = misattributed = unhelpful = 0
    for record in accepted:
        t = record["step"]
        component = record["component"]
        other = "wm" if component == "lm" else "lm"
        if not _needed(scores[t], component):
            false += 1
            misattributed += int(_needed(scores[t], other))
        nxt = scores[t + 1] if t + 1 < len(scores) else None
        if (
            nxt is None
            or _needed(nxt, component)
            or (component == "lm" and nxt["wrong_alias_notes"] > scores[t]["wrong_alias_notes"])
        ):
            unhelpful += 1
    intervals = []
    for component in ("lm", "wm"):
        flags = [_needed(s, component) for s in scores]
        # Observable: enough of the stream remains to see whether it was handled within MISS_AFTER steps.
        intervals += [(component, a, b) for a, b in _intervals(flags) if a + MISS_AFTER <= len(scores)]
    lasting = [(c, a, b) for c, a, b in intervals if b - a >= MISS_AFTER]
    unresolved = len(lasting)
    missed = sum(
        1
        for component, a, _ in lasting
        if not any(r["component"] == component and a <= r["step"] < a + MISS_AFTER for r in accepted)
    )
    poisoned = [s for s in post if s["wrong_alias_notes"] > 0]
    oscillations = 0
    for component in ("lm", "wm"):
        moves = [r for r in run["lineage"] if r["component"] == component and r["kind"] == "accepted"]
        for a, b in itertools.pairwise(moves):
            if b["result_state"] == a["parent_state"] and b["step"] - a["step"] <= 30:
                oscillations += 1
    return {
        "stream": run["stream"],
        "family": run["family"],
        "condition": run["condition"],
        "controller": run["controller"],
        "failure": rate(post),
        "shifted_failure": rate(shifted),
        "retention_failure": rate(retained),
        "false_adaptation": false / n_post,
        "misattribution": misattributed / n_post,
        "missed_adaptation": missed / len(intervals) if intervals else 0.0,
        "unresolved_deficiency": unresolved / len(intervals) if intervals else 0.0,
        "poisoned": len(poisoned) / n_post,
        "unhelpful_adaptation": unhelpful / len(accepted) if accepted else 0.0,
        "accepted": len(accepted),
        "rejected": sum(r["kind"] == "rejected" for r in run["lineage"]),
        "rollbacks": sum(r["kind"] == "rollback" for r in run["lineage"]),
        "oscillations": oscillations,
        "deficiency_intervals": len(intervals),
        "lm_calls": {k: run["cost"][k] for k in ("lm_act", "lm_revision", "lm_extract", "lm_gate")},
    }
