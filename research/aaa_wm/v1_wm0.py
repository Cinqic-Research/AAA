"""WM-0 on ``aaa.python.v1`` repair: a development diagnostic, not a promotable experiment.

Question (declared in ``docs/wm_program/diagnosis.md`` before this ran): with the
visible-test tool **disabled at decision time**, does a small action-conditioned
consequence model -- one that predicts, for a candidate edit ``a`` and a visible
test ``j``, whether the patched function passes ``j`` -- choose repairs better than
(a) v1's direct bandit scorer and (b) a direct scorer trained on the same tool
observations as a single scalar per candidate?

Experience (identical for every learned arm). Each training task goes through
:class:`~research.aaa_python_v1.episode.ToolEnvironment`: present -> the logged
visible-test tool (training only) -> act -> commit -> reveal -> learn. The tool
observation (pass/fail per candidate per visible test) and the bandit feedback
(hidden-test results of the chosen candidate only) are the only learning signals.
At evaluation the tool is never called and no learning happens.

Arms (all share the ``e2`` candidate encoding, one ``tanh`` core, the hidden-pass head):

``ref``        hidden-pass head only (v1's formulation; ignores tool observations)
``direct``     + a scalar head: "all visible tests pass" per candidate (same data as ``wm``)
``direct_t``   ``direct`` whose scalar head also sees the pooled visible-test features: the same
               *information* as ``wm`` at decision time, without per-test consequence structure
``wm``         + a per-test consequence head conditioned on the test (input, expected) features
``wm_nohead``  the trained ``wm`` with its consequence term disabled at decision time
``wm_shuffle`` the trained ``wm`` with the test/action pairing shuffled at decision time
``wm_random``  the ``wm`` architecture, untrained (random core, zero heads)

Decision rule for ``wm``: argmax over candidates of
``log sigma(hidden(a)) + sum_j log sigma(pass_j(a))``; for ``direct``:
``log sigma(hidden(a)) + log sigma(allpass(a))``; for ``ref``: ``hidden(a)``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from research.aaa_python.episode import Action
from research.aaa_python.generator import Task
from research.aaa_python.rng import Stream, derive_seed
from research.aaa_python_v1 import generator
from research.aaa_python_v1 import spec as spec_module
from research.aaa_python_v1.encoders import Encoder, Sparse, sparse
from research.aaa_python_v1.episode import ToolEnvironment

D = 256
TEST_D = 64


def test_features(inp: int, expected: int) -> list[str]:
    """What a visible test says, as bucketed facts (no execution)."""

    def b(v: int) -> str:
        return str(max(-12, min(12, v)))

    return [
        f"t:in:{b(inp)}",
        f"t:exp:{b(expected)}",
        f"t:gap:{b(expected - inp)}",
        f"t:sign:{(expected > inp) - (expected < inp)}",
    ]


@dataclass
class Params:
    W: np.ndarray  # H x (D + TEST_D)
    b: np.ndarray
    h: np.ndarray  # hidden-pass head (H + 1)
    s: np.ndarray  # scalar all-visible-pass head (H + 1), used by "direct"
    t: np.ndarray  # per-test consequence head (H + 1), used by "wm"

    def count(self, arm: str) -> int:
        n = self.W.size + self.b.size + self.h.size
        if arm in ("direct", "direct_t"):
            n += self.s.size
        if arm == "wm":
            n += self.t.size
        return int(n)


def init_params(hidden: int, seed: int) -> Params:
    rng = np.random.default_rng(seed)
    return Params(
        rng.normal(0.0, 1.0, size=(hidden, D + TEST_D)),
        np.zeros(hidden),
        np.zeros(hidden + 1),
        np.zeros(hidden + 1),
        np.zeros(hidden + 1),
    )


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _core(p: Params, x: Sparse, t: Sparse | None) -> np.ndarray:
    pre = p.W[:, x.index] @ x.value + p.b
    if t is not None:
        pre = pre + p.W[:, D + t.index] @ t.value
    return np.tanh(pre)


def _logit(head: np.ndarray, z: np.ndarray) -> float:
    return float(head[:-1] @ z + head[-1])


def _step(p: Params, head: str, x: Sparse, t: Sparse | None, target: float, lr: float) -> float:
    z = _core(p, x, t)
    w = getattr(p, head)
    q = _sig(_logit(w, z))
    d = q - target
    dz = d * w[:-1]
    w[:-1] -= lr * d * z
    w[-1] -= lr * d
    dpre = dz * (1.0 - z * z)
    p.b -= lr * dpre
    p.W[:, x.index] -= lr * np.outer(dpre, x.value)
    if t is not None:
        p.W[:, D + t.index] -= lr * np.outer(dpre, t.value)
    return -math.log(max(q if target else 1 - q, 1e-12))


class WM0Agent:
    """An agent for the v1 causal boundary. ``arm`` fixes what it learns from and how it decides."""

    def __init__(self, arm: str, hidden: int, seed: int, lr: float) -> None:
        self.arm, self.lr = arm, lr
        self.p = init_params(hidden, seed)
        self.encoder = Encoder("e2", D, ("alpha", "line", "flow"))
        self.stream = Stream(seed)
        self.mode = "wm" if arm.startswith("wm") else arm

    def _tests(self, view: Any) -> list[Sparse]:
        return [sparse(test_features(i, o), TEST_D) for i, o in view.visible_tests]

    def _pooled(self, view: Any) -> Sparse:
        feats: list[str] = []
        for i, o in view.visible_tests:
            feats += test_features(i, o)
        return sparse(feats, TEST_D)

    def scores(self, view: Any, *, shuffle: bool = False, disable_consequence: bool = False) -> np.ndarray:
        enc = self.encoder.encode(view).candidates
        tests = self._tests(view)
        if (
            shuffle
        ):  # break the action <-> consequence pairing: score candidate k with candidate k+1's features
            enc = enc[1:] + enc[:1]
        out = []
        for k, x in enumerate(self.encoder.encode(view).candidates):
            z = _core(self.p, x, None)
            s = math.log(max(_sig(_logit(self.p.h, z)), 1e-12))
            if self.mode == "direct":
                s += math.log(max(_sig(_logit(self.p.s, z)), 1e-12))
            if self.mode == "direct_t":
                xa = enc[k]
                s += math.log(max(_sig(_logit(self.p.s, _core(self.p, xa, self._pooled(view)))), 1e-12))
            if self.mode == "wm" and not disable_consequence:
                xa = enc[k]
                for t in tests:
                    s += math.log(max(_sig(_logit(self.p.t, _core(self.p, xa, t))), 1e-12))
            out.append(s)
        return np.array(out)

    def act(self, view: Any, **kw: Any) -> Action:
        s = self.scores(view, **kw)
        best = int(np.argmax(s))
        return Action(best, 1.0)

    def learn(self, view: Any, action: Action, feedback: Any, tool: Sequence[Sequence[bool]] | None) -> None:
        enc = self.encoder.encode(view).candidates
        results = feedback.fields.get("chosen_candidate_hidden_results") if feedback is not None else None
        if results is not None:
            _step(self.p, "h", enc[int(action.answer)], None, 1.0 if all(results) else 0.0, self.lr)
        if tool is None or self.mode == "ref":
            return
        tests = self._tests(view)
        for k, x in enumerate(enc):
            if self.mode == "direct":
                _step(self.p, "s", x, None, 1.0 if all(tool[k]) else 0.0, self.lr)
            elif self.mode == "direct_t":
                _step(self.p, "s", x, self._pooled(view), 1.0 if all(tool[k]) else 0.0, self.lr)
            else:
                for j, t in enumerate(tests):
                    _step(self.p, "t", x, t, 1.0 if tool[k][j] else 0.0, self.lr)


class CachedToolEnvironment(ToolEnvironment):
    """``ToolEnvironment`` whose tool answers come from a precomputed cache of the *same* function.

    The cache is built by :func:`build_tool_cache` with the real sandboxed tool
    (:func:`research.aaa_python_v1.episode.visible_test_results`); boundary checks and
    logging are unchanged, and a missing entry falls back to the real tool.
    """

    cache: dict[str, list[list[bool]]] = {}

    def run_visible_tests(self, view: Any) -> tuple[tuple[bool, ...], ...]:
        from research.aaa_python.episode import BoundaryError

        if self._current is None or view is not self._view:
            raise BoundaryError("the tool may be used only on the presented task")
        if self._action is not None:
            raise BoundaryError("the tool may be used only before the action is committed")
        hit = self.cache.get(_tool_key(view))
        if hit is None:
            return super().run_visible_tests(view)
        self._log("tool")
        return tuple(tuple(bool(b) for b in row) for row in hit)


def _tool_key(view: Any) -> str:
    import hashlib

    return hashlib.sha256(
        repr((view.source, view.repair_line, view.candidates, view.visible_tests)).encode()
    ).hexdigest()


def build_tool_cache(tasks: Sequence[Task], path: Path) -> None:
    from research.aaa_python_v1.episode import view_of, visible_test_results

    spec = spec_module.load()
    cache = json.loads(path.read_text()) if path.exists() else {}
    added = 0
    for task in tasks:
        view = view_of(task, 0, spec)
        key = _tool_key(view)
        if key not in cache:
            cache[key] = [list(r) for r in visible_test_results(view)]
            added += 1
    if added:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cache))
        tmp.replace(path)
    CachedToolEnvironment.cache = cache


def train_agent(agent: WM0Agent, tasks: Sequence[Task], epochs: int, seed: int, spec: Any) -> None:
    env = CachedToolEnvironment(spec)
    for epoch in range(epochs):
        order = Stream(derive_seed("aaa.wm.v1_wm0", "order", seed, epoch)).shuffled(list(tasks))
        for task in order:
            view = env.present(task)
            tool = env.run_visible_tests(view)  # training experience: logged pre-action tool
            action = agent.act(view)
            env.commit(view, action)
            _, feedback = env.reveal(view)
            agent.learn(view, action, feedback, tool)


def evaluate(agent: WM0Agent, tasks: Sequence[Task], spec: Any, **kw: Any) -> list[bool]:
    env = ToolEnvironment(spec, feedback_enabled=False)
    out = []
    for task in tasks:
        view = env.present(task)
        action = agent.act(view, **kw)
        env.commit(view, action)
        correct, _ = env.reveal(view)
        out.append(bool(correct))
    return out


def tool_accuracy_of_consequence_head(agent: WM0Agent, tasks: Sequence[Task]) -> float:
    """Intrinsic diagnostic: per-test pass prediction accuracy against the real tool."""

    from research.aaa_python_v1.episode import view_of, visible_test_results

    spec = spec_module.load()
    right = total = 0
    for task in tasks:
        view = view_of(task, 0, spec)
        truth = visible_test_results(view)
        enc = agent.encoder.encode(view).candidates
        for k, x in enumerate(enc):
            for j, t in enumerate(agent._tests(view)):
                pred = _sig(_logit(agent.p.t, _core(agent.p, x, t))) > 0.5
                right += pred == truth[k][j]
                total += 1
    return right / total


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=11)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--role", default="tune", choices=("tune", "evaluate"))
    ap.add_argument("--arms", default="ref,direct,direct_t,wm")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    spec = spec_module.load()
    train = list(generator.pool("train", "repair"))
    start, stop = spec["splits"]["development_layout"][args.role]
    dev = list(generator.pool("development", "repair"))[start:stop]
    cache_path = Path(os.environ["AAA_DATA_ROOT"]) / "wm0" / "tool_cache.json"
    build_tool_cache(train + dev, cache_path)
    started = time.time()
    rows: dict[str, list[list[bool]]] = {}
    intrinsic = []
    for seed in range(args.seeds):
        s = derive_seed("aaa.wm.v1_wm0", "init", seed) % 2**63
        arms = args.arms.split(",")
        agents = {arm: WM0Agent(arm, args.hidden, s, args.lr) for arm in arms}
        for agent in agents.values():
            train_agent(agent, train, args.epochs, seed, spec)
        for arm in arms:
            rows.setdefault(arm, []).append(evaluate(agents[arm], dev, spec))
        if "wm" in agents:
            rows.setdefault("wm_nohead", []).append(
                evaluate(agents["wm"], dev, spec, disable_consequence=True)
            )
            rows.setdefault("wm_shuffle", []).append(evaluate(agents["wm"], dev, spec, shuffle=True))
            rows.setdefault("wm_random", []).append(
                evaluate(WM0Agent("wm", args.hidden, s, args.lr), dev, spec)
            )
            intrinsic.append(tool_accuracy_of_consequence_head(agents["wm"], dev[:200]))
        print(seed, {k: round(float(np.mean(v[-1])), 3) for k, v in rows.items()}, intrinsic[-1:], flush=True)
    doc = {
        "schema": "aaa.wm.v1_wm0.result.v1",
        "status": "development diagnostic",
        "args": vars(args),
        "identities": {
            "train": "v1 train/repair (2400)",
            "dev": f"v1 development/repair {args.role} [{start},{stop})",
        },
        "parameters": {arm: init_params(args.hidden, 0).count(arm) for arm in args.arms.split(",")},
        "bits": {k: ["".join("1" if b else "0" for b in r) for r in v] for k, v in rows.items()},
        "accuracy": {k: float(np.mean(v)) for k, v in rows.items()},
        "consequence_head_test_accuracy": intrinsic,
        "seconds": time.time() - started,
        "python": sys.version,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["accuracy"], indent=1))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
