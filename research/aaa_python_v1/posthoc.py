"""A post-hoc diagnostic, labelled as such: does ~4K adapt, forget and stay plastic like ~10K?

Added **after** the declared adaptation and plasticity stages showed that 10K
adapts to novel structure in four families where 1K adapts in one, and that
experienced learners (10K more than 1K) learn a conflicting mapping worse than
fresh ones. The declared stages measured only 1K and 10K; the scale question
("would 4K produce the same benefit?") needs 4K. Nothing here enters a declared
verdict. It re-runs 1K and 10K too, so the committed stage evidence is also
checked for determinism.

Paired contrasts:

* adaptation: the difference-of-differences grid of one size minus another's,
  cell by cell (same initialization index and stream), crossed bootstrap;
* plasticity: per initialization index, (late - fresh) of one size minus
  another's, percentile bootstrap over initializations.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from . import stages
from .experiment import DESIGN, run_adapt, unbits
from .plasticity import PLASTICITY, life_job, summarize_plasticity
from .stats import crossed, resolved_sign
from .summarize import summarize_adapt

FAMILIES = ("syntax", "outcome", "output", "localize", "repair")


def _did(rows: list[Mapping[str, Any]], arm: str, family: str) -> np.ndarray:
    mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["init"])
    streams = DESIGN["adapt"]["streams"]
    out = np.empty((len(mine), streams))
    for i, row in enumerate(mine):
        for s, cell in enumerate(row["families"][family]):

            def err(key: str, cell: Mapping[str, Any] = cell) -> float:
                bits = unbits(cell[key])
                return 1.0 - sum(bits) / len(bits)

            out[i, s] = (err("changed_frozen") - err("changed_online")) - (
                err("control_frozen") - err("control_online")
            )
    return out


def run_posthoc(previous: dict[str, Any], workers: int) -> dict[str, Any]:
    encoder = stages.select_encoder(previous["encoders"]["stage"]["summary"])
    capacity = previous["capacity"]["stage"]
    sizes = stages.selected_sizes(capacity, encoder)
    arms = [stages._fixed(sizes[t], t) for t in ("1k", "4k", "10k")]
    selected = {a.name: {"learning_rate": a.learning_rate, "epochs": a.epochs} for a in arms}
    adapt_rows = run_adapt(arms, selected, workers=workers)
    jobs = [(a, a.learning_rate, i) for a in arms for i in range(PLASTICITY["initializations"])]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        life_rows = list(pool.map(life_job, *zip(*jobs, strict=True)))
    st = DESIGN["statistics"]
    contrasts: dict[str, Any] = {}
    for left, right in (("10k", "4k"), ("10k", "1k"), ("4k", "1k")):
        for family in FAMILIES:
            diff = _did(adapt_rows, left, family) - _did(adapt_rows, right, family)
            result = crossed(diff, seed=st["seed"], draws=st["draws"], confidence=st["confidence"])
            result["resolved_sign"] = resolved_sign(result)
            contrasts[f"adaptation DiD {family}: {left} - {right}"] = result
        gap = {}
        for arm in (left, right):
            mine = sorted((r for r in life_rows if r["arm"] == arm), key=lambda r: r["init"])
            gap[arm] = np.array(
                [
                    {c["epoch"]: c for c in r["checkpoints"]}[max(PLASTICITY["checkpoints"])]["probe"][
                        "after_mean"
                    ]
                    - {c["epoch"]: c for c in r["checkpoints"]}[0]["probe"]["after_mean"]
                    for r in mine
                ]
            )
        diff1 = gap[left] - gap[right]
        rng = np.random.default_rng(st["seed"])
        idx = rng.integers(0, len(diff1), size=(st["draws"], len(diff1)))
        means = diff1[idx].mean(axis=1)
        lower, upper = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        contrasts[f"plasticity late-minus-fresh: {left} - {right}"] = {
            "mean": float(diff1.mean()),
            "lower": lower,
            "upper": upper,
            "interval_status": "MEASURED",
            "resolved_sign": "POSITIVE" if lower > 0 else "NEGATIVE" if upper < 0 else "INCONCLUSIVE",
        }
    return {
        "stage": "posthoc",
        "arms": {a.name: {**a.to_json(), "parameters": a.parameters()} for a in arms},
        "adapt_rows": adapt_rows,
        "adapt_summary": summarize_adapt(adapt_rows),
        "life_rows": life_rows,
        "plasticity_summary": summarize_plasticity(life_rows),
        "contrasts": contrasts,
    }
