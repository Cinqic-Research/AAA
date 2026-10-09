"""A behavioural surrogate of Juniper LM 1.1 for simulation only.

The Erudition Model cannot be trained against gpt-oss-20b directly: training
needs hundreds of thousands of simulated episodes and counterfactual
branches, and the real model answers about one request every four seconds.
The surrogate reproduces the *rates* at which the real model reads names,
aliases, notes, World Model checks and feedback correctly, as measured on
training-split situations by ``characterize.py``. Its parameters are data
(``data/lm_surrogate.json``) and carry the characterization's digest.

The surrogate is told the hidden truth of each simulated request through
``register`` so it can, for example, guess an unknown alias correctly one
time in four. That truth never reaches the Erudition Model, whose inputs
are the same evidence features the real system produces. Whether a
controller trained on the surrogate transfers to the real model is measured
on development streams, not assumed.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

from .contracts import sha256_json
from .toolshift import Dynamics, manual_dynamics

ALIAS_NOTE = re.compile(r"When a user says '([^']+)', they mean the tank (\w+)")
DYN_NOTE = re.compile(r"(fill|drain)\(tank, n\) currently \w+ about (\d+)\*n(?: ([+-]) (\d+))?")
OBS_NOTE = re.compile(r"Observed earlier: (fill|drain)\((\w+), (\d+)\) took \w+ from (\d+) to (\d+)")
TABLE = re.compile(r"n=(\d+) -> (\d+)")


@cache
def load_parameters() -> dict[str, Any]:
    text = resources.files("research.aaa_erudition").joinpath("data/lm_surrogate.json").read_text("utf-8")
    params: dict[str, Any] = json.loads(text)
    return params


class SurrogateLM:
    backend_id = "surrogate:aaa.erudition.v0"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.p = (params or load_parameters())["rates"]
        self.truth: dict[str, str | None] = {}

    def register(self, request: str, state: dict[str, int], entity: str | None) -> None:
        """Record which tank a simulated request refers to (None for a nonexistent tank).

        Keyed by request and workspace state, since the same sentence can recur. This is
        the only hidden fact the surrogate uses: it lets an unknown alias be guessed
        correctly one time in four, as a model guessing among four tanks would.
        """

        self.truth[_key(request, state)] = entity

    def _rng(self, request: dict[str, Any]) -> np.random.Generator:
        return np.random.default_rng(int(sha256_json(request)[:15], 16))

    @staticmethod
    def _reply(name: str, args: dict[str, Any], confidence: float) -> dict[str, Any]:
        call = {"id": "call-0", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        logprobs = {"content": [{"token": "x", "logprob": confidence}]}
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "logprobs": logprobs}
            ]
        }

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        g = self._rng(request)
        tools = {tool["function"]["name"] for tool in request["tools"]}
        if "record_correction" in tools:
            return self._extract(request["messages"][1]["content"], g)
        return self._act(request["messages"], g)

    def _confidence(self, g: np.random.Generator, kind: str) -> float:
        # By proposal kind only. On the real model a wrong action is as confident as a right one
        # (characterization: -0.0003 both); conditioning on correctness would hand the controller
        # a label no real run has.
        mu, sd = self.p["confidence_act"] if kind == "act" else self.p["confidence_abstain"]
        return float(min(0.0, g.normal(mu, sd)))

    def _extract(self, text: str, g: np.random.Generator) -> dict[str, Any]:
        request = text.split('Request: "')[1].split('"')[0]
        feedback = text.split('Feedback: "')[1].rsplit('"', 1)[0]
        match = re.search(r"By '([^']+)' I meant (\w+)", feedback)
        if match and g.random() < self.p["extract_correct"]:
            return self._reply("record_correction", {"phrase": match.group(1), "tank": match.group(2)}, -0.01)
        if not match and g.random() < self.p["extract_spurious"]:
            words = re.findall(r"\b([a-z]+)\b", feedback.lower())
            return self._reply(
                "record_correction",
                {"phrase": request.split()[1].lower(), "tank": words[0] if words else ""},
                -0.5,
            )
        return self._reply("no_correction", {}, -0.01)

    def _act(self, messages: list[dict[str, Any]], g: np.random.Generator) -> dict[str, Any]:
        p = self.p
        text = messages[1]["content"]
        state = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", text.split("Tanks now:")[1].split("\n")[0])}
        request = text.split('User request: "')[1].rsplit('"', 1)[0]
        entity = self.truth.get(_key(request, state))
        if g.random() < p["invalid"]:
            return self._reply(
                "act", {"op": "fill", "tank": "?", "n": 1, "expected": 0}, self._confidence(g, "act")
            )
        lowered = request.lower()
        named = next((t for t in state if re.search(rf"\b{t}\b", lowered)), None)
        notes = dict(ALIAS_NOTE.findall(text))
        noted = next((tank for phrase, tank in notes.items() if phrase in lowered), None)
        target_value = int(re.findall(r"(\d+)", request)[-1])
        consulted = len(messages) > 2 and "World model consultation" in messages[-1]["content"]
        if len(messages) > 2 and not consulted:
            return self._revise(messages, target_value, g)
        if named is not None:
            tank = named if g.random() >= p["name_misread"] else str(g.choice(list(state)))
        elif noted is not None:
            tank = noted if g.random() < p["alias_note_follow"] else str(g.choice(list(state)))
        elif entity is not None:
            if consulted or g.random() < p["alias_unknown_abstain"]:
                return self._reply("abstain", {"reason": "unknown tank"}, self._confidence(g, "abstain"))
            tank = str(g.choice(list(state)))
        else:
            if g.random() < p["decoy_abstain"]:
                return self._reply("abstain", {"reason": "no such tank"}, self._confidence(g, "abstain"))
            tank = str(g.choice(list(state)))
        belief = manual_dynamics()
        dyn_notes = DYN_NOTE.findall(messages[-1]["content"] if consulted else text)
        if dyn_notes:
            if g.random() < p["consult_follow" if consulted else "dynamics_note_use"]:
                fill, drain = belief.fill_rate, belief.fill_bonus
                d_rate, d_fee = belief.drain_rate, belief.drain_fee
                for op, rate, sign, off in dyn_notes:
                    offset = int(off or 0) * (-1 if sign == "-" else 1)
                    if op == "fill":
                        fill, drain = int(rate), offset
                    else:
                        d_rate, d_fee = int(rate), offset
                belief = Dynamics(fill, drain, d_rate, d_fee)
        elif OBS_NOTE.search(text) and g.random() < p["observation_infer"]:
            belief = infer_from_observations(text, belief)
        before = state[tank]
        options = [
            (op, n)
            for op in ("fill", "drain")
            for n in range(1, 10)
            if belief.apply(op, before, n, 0, 99) == target_value
        ]
        if not options:
            if g.random() < p["unreachable_abstain"]:
                return self._reply(
                    "abstain", {"reason": "cannot reach the value"}, self._confidence(g, "abstain")
                )
            op = "fill" if target_value > before else "drain"
            n = max(1, min(9, round(abs(target_value - before) / 3)))
        else:
            op, n = options[0]
        if g.random() < p["arithmetic_slip"]:
            n = max(1, min(9, n + int(g.choice([-1, 1]))))
        return self._reply(
            "act", {"op": op, "tank": tank, "n": n, "expected": target_value}, self._confidence(g, "act")
        )

    def _revise(self, messages: list[dict[str, Any]], value: int, g: np.random.Generator) -> dict[str, Any]:
        p = self.p
        prior = json.loads(messages[-2]["tool_calls"][0]["function"]["arguments"])
        table = {int(after): int(n) for n, after in TABLE.findall(messages[-1]["content"])}
        if value in table and g.random() < p["revision_follow"]:
            return self._reply(
                "act",
                {"op": prior["op"], "tank": prior["tank"], "n": table[value], "expected": value},
                self._confidence(g, "act"),
            )
        if g.random() < p["revision_abstain"]:
            return self._reply("abstain", {"reason": "check disagrees"}, self._confidence(g, "abstain"))
        return self._reply("act", prior, self._confidence(g, "act"))


def _key(request: str, state: dict[str, int]) -> str:
    return request + "|" + ",".join(f"{k}={v}" for k, v in sorted(state.items()))


def infer_from_observations(text: str, prior: Dynamics) -> Dynamics:
    """What a model could infer from the observation notes it was shown: a least-squares line per operation."""

    rates = {"fill": (prior.fill_rate, prior.fill_bonus), "drain": (prior.drain_rate, prior.drain_fee)}
    for op in rates:
        rows = [
            (int(n), int(after) - int(before)) for o, _, n, before, after in OBS_NOTE.findall(text) if o == op
        ]
        if not rows:
            continue
        sign = 1 if op == "fill" else -1
        if len({n for n, _ in rows}) < 2:
            n, delta = rows[0]
            rates[op] = (max(1, round(sign * delta / n)), 0)
            continue
        slope, intercept = np.polyfit([n for n, _ in rows], [d for _, d in rows], 1)
        rates[op] = (max(1, round(sign * slope)), round(sign * intercept))
    return Dynamics(*rates["fill"], *rates["drain"])
