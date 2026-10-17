"""Development diagnostic behind decision D-WM: which dynamics learner should the World Model use?

Every arm sees the same transition sequences drawn from training-split
dynamics: version A (the manual), then a shifted version B, then A again.
The *timing* of context changes is given to every arm that has contexts,
so the comparison isolates the learner from the controller.

Arms:
* ``library_bayes`` -- the selected design: one Bayesian linear context per version;
* ``single_bayes``  -- one Bayesian model updated continually (no contexts);
* ``single_mlp``    -- one ensemble of five small MLPs updated online;
* ``library_mlp``   -- an MLP ensemble per context, copied from the previous one at a switch.

Reported per arm: held-out error on B after k transitions of B (sample
efficiency), error on A right after it returns (retention), the share of
held-out results inside the +/-2 sd interval (calibration), and the same on
capped "novel" dynamics outside the Bayesian basis.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from .toolshift import OPS, Dynamics, manual_dynamics, warmup_transitions
from .world import Context, Posterior, Transition

KS = (2, 4, 6, 10)


def _draw(g: np.random.Generator, dynamics: Dynamics, count: int) -> list[Transition]:
    rows = []
    for _ in range(count):
        op = str(g.choice(OPS))
        n = int(g.integers(1, 7))
        before = int(g.integers(5, 95))
        rows.append((op, n, before, dynamics.apply(op, before, n, 0, 99)))
    return rows


class BayesArm:
    def __init__(self, library: bool) -> None:
        self.library = library
        self.contexts: dict[str, Context] = {}
        self.active = "A"

    def fresh(self) -> Context:
        return Context("c", -1, {op: Posterior.prior() for op in OPS}, ())

    def switch(self, name: str) -> None:
        if self.library:
            self.active = name
            self.contexts.setdefault(name, self.fresh())

    def observe(self, rows: list[Transition]) -> None:
        key = self.active if self.library else "single"
        self.contexts[key] = self.contexts.get(key, self.fresh()).fit(rows, 0, 99)

    def predict(self, op: str, n: int, before: int) -> tuple[float, float]:
        key = self.active if self.library else "single"
        return self.contexts[key].predict(op, n, before, 0, 99)


class MLPArm:
    def __init__(self, library: bool, seed: int) -> None:
        import torch

        self.torch = torch
        torch.manual_seed(seed)
        self.library = library
        self.models: dict[str, Any] = {}
        self.active = "A"
        self.models["A"] = self._new()

    def _new(self) -> Any:
        nn = self.torch.nn
        members = [
            nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))
            for _ in range(5)
        ]
        return {"members": members, "opt": [self.torch.optim.Adam(m.parameters(), lr=1e-2) for m in members]}

    def _x(self, rows: list[tuple[str, int, int]]) -> Any:
        return self.torch.tensor(
            [[float(op == "fill"), float(op == "drain"), n / 9, before / 99] for op, n, before in rows]
        )

    def switch(self, name: str) -> None:
        if not self.library:
            return
        if name not in self.models:
            import copy

            base = copy.deepcopy(self.models[self.active])
            base["opt"] = [self.torch.optim.Adam(m.parameters(), lr=1e-2) for m in base["members"]]
            self.models[name] = base
        self.active = name

    def observe(self, rows: list[Transition]) -> None:
        model = self.models[self.active if self.library else "A"]
        x = self._x([(op, n, b) for op, n, b, _ in rows])
        y = self.torch.tensor([[(a - b) / 30] for _, _, b, a in rows], dtype=self.torch.float32)
        for member, opt in zip(model["members"], model["opt"], strict=True):
            for _ in range(300):
                loss = ((member(x) - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def predict(self, op: str, n: int, before: int) -> tuple[float, float]:
        model = self.models[self.active if self.library else "A"]
        with self.torch.no_grad():
            outs = np.array([float(m(self._x([(op, n, before)]))[0, 0]) * 30 for m in model["members"]])
        return float(min(99, max(0, before + outs.mean()))), float(outs.std() + 0.5)


def _score(arm: Any, held: list[Transition]) -> tuple[float, float]:
    errors, covered = 0, 0
    for op, n, before, after in held:
        mean, sd = arm.predict(op, n, before)
        errors += int(round(mean) != after)
        covered += int(abs(mean - after) <= 2 * sd + 0.5)
    return errors / len(held), covered / len(held)


def run(trials: int, seed: int) -> dict[str, Any]:
    g = np.random.default_rng(seed)
    family = [
        Dynamics(f, b, d, e) for f in (2, 4, 5, 6) for b in (0, 1, 2) for d in (1, 3, 4) for e in (0, 1, 2)
    ]
    results: dict[str, dict[str, list[float]]] = {}
    for trial in range(trials):
        shifted = family[int(g.integers(len(family)))]
        novel = Dynamics(
            shifted.fill_rate, shifted.fill_bonus, shifted.drain_rate, shifted.drain_fee, fill_cap=10
        )
        for label, b_dyn in (("in_family", shifted), ("novel", novel)):
            warm = [tuple(r) for r in warmup_transitions("train", trial, 24)]
            stream_b = _draw(g, b_dyn, max(KS))
            held_b = _draw(g, b_dyn, 20)
            held_a = _draw(g, manual_dynamics(), 20)
            for name in ("library_bayes", "single_bayes", "single_mlp", "library_mlp"):
                arm: Any = (
                    BayesArm(name.startswith("library"))
                    if name.endswith("bayes")
                    else MLPArm(name.startswith("library"), trial)
                )
                arm.switch("A")
                arm.observe(warm)
                arm.switch("B")
                seen = 0
                row = results.setdefault(f"{label}/{name}", {})
                for k in KS:
                    arm.observe(stream_b[seen:k])
                    seen = k
                    err, cover = _score(arm, held_b)
                    row.setdefault(f"error_B_after_{k}", []).append(err)
                    row.setdefault(f"coverage_B_after_{k}", []).append(cover)
                arm.switch("A")
                err_a, _ = _score(arm, held_a)
                row.setdefault("error_A_on_return", []).append(err_a)
    return {
        key: {metric: round(float(np.mean(v)), 4) for metric, v in sorted(metrics.items())}
        for key, metrics in sorted(results.items())
    } | {"trials": trials, "seed": seed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.aaa_erudition.wm_diagnostic")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.trials, args.seed), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
