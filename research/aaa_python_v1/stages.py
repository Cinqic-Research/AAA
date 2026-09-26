"""The declared development stages of ``aaa.python.v1`` and their selection rules.

Each stage names its arms and its primary contrasts (Holm-adjusted together).
Later stages take earlier stages' *selections* through the declared rules in
this module, never through a hand choice:

* :func:`select_encoder` -- the encoder with the highest mean (over families)
  frozen evaluation accuracy behind the 16-unit core;
* :func:`select_heads` -- per family, the alternative head when its frozen
  accuracy is at least the one-hot head's minus 0.01 (ties favour the head
  with fewer parameters).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .encoders import E2_CHANNELS
from .experiment import FAMILIES, ArmSpec
from .models import hidden_for_budget

CORE = 16
BUDGETS = {"1k": 1000, "2k": 2000, "4k": 4000, "10k": 10000, "20k": 20000}
HEAD_TOLERANCE = 0.01


def _drop(channel: str) -> tuple[str, ...]:
    return tuple(c for c in E2_CHANNELS if c != channel)


def encoder_arms() -> list[ArmSpec]:
    arms = [ArmSpec(f"lin@{e}", e, 0) for e in ("e0", "e1", "e2")]
    arms += [ArmSpec(f"h{CORE}@{e}", e, CORE) for e in ("e0", "e1", "e2")]
    arms += [ArmSpec(f"h{CORE}@e2-{c}", "e2", CORE, channels=_drop(c)) for c in E2_CHANNELS]
    return arms


ENCODER_PRIMARY = [
    (f"h{CORE}@e2", f"h{CORE}@e1", "frozen"),
    (f"h{CORE}@e1", f"h{CORE}@e0", "frozen"),
    ("lin@e2", "lin@e1", "frozen"),
]
ENCODER_SECONDARY = [(f"h{CORE}@e2", f"h{CORE}@e2-{c}", "frozen") for c in E2_CHANNELS] + [
    (f"h{CORE}@e2", "lin@e2", "frozen"),
    (f"h{CORE}@e2", f"h{CORE}@e2", "online-frozen"),
    (f"h{CORE}@e1", f"h{CORE}@e1", "online-frozen"),
]


def _frozen_mean(summary: Mapping[str, Any], arm: str) -> float:
    return float(np.mean([summary["arms"][arm][f]["frozen"]["mean"] for f in FAMILIES]))


def select_encoder(summary: Mapping[str, Any]) -> str:
    scores = {e: _frozen_mean(summary, f"h{CORE}@{e}") for e in ("e0", "e1", "e2")}
    return max(scores, key=lambda e: (scores[e], e))


def head_arms(encoder: str) -> list[ArmSpec]:
    return [
        ArmSpec(f"h{CORE}@{encoder}:{o}/{loc}", encoder, CORE, output_head=o, localize_head=loc)
        for o in ("onehot", "gauss")
        for loc in ("onehot", "pointer")
    ]


def head_primary(encoder: str) -> list[tuple[str, str, str]]:
    base = f"h{CORE}@{encoder}:onehot/onehot"
    return [
        (f"h{CORE}@{encoder}:gauss/onehot", base, "frozen"),
        (f"h{CORE}@{encoder}:onehot/pointer", base, "frozen"),
    ]


def select_heads(summary: Mapping[str, Any], encoder: str) -> tuple[str, str]:
    base = summary["arms"][f"h{CORE}@{encoder}:onehot/onehot"]
    gauss = summary["arms"][f"h{CORE}@{encoder}:gauss/onehot"]
    pointer = summary["arms"][f"h{CORE}@{encoder}:onehot/pointer"]
    output = (
        "gauss"
        if gauss["output"]["frozen"]["mean"] >= base["output"]["frozen"]["mean"] - HEAD_TOLERANCE
        else "onehot"
    )
    localize = (
        "pointer"
        if pointer["localize"]["frozen"]["mean"] >= base["localize"]["frozen"]["mean"] - HEAD_TOLERANCE
        else "onehot"
    )
    return output, localize


def capacity_arms(encoder: str, output: str, localize: str) -> list[ArmSpec]:
    arms = []
    for tag, budget in BUDGETS.items():
        hidden = hidden_for_budget(
            budget, encoder=encoder, dimensions=256, output_head=output, localize_head=localize
        )
        arms.append(ArmSpec(f"{tag}@{encoder}", encoder, hidden, output_head=output, localize_head=localize))
    for tag in ("1k", "10k"):
        base = next(a for a in arms if a.name == f"{tag}@{encoder}")
        arms.append(
            ArmSpec(f"{tag}@{encoder}:n600", encoder, base.hidden, output, localize, train_per_family=600)
        )
    return arms


def capacity_primary(encoder: str) -> list[tuple[str, str, str]]:
    return [
        (f"10k@{encoder}", f"1k@{encoder}", "frozen"),
        (f"10k@{encoder}", f"4k@{encoder}", "frozen"),
    ]


def capacity_secondary(encoder: str) -> list[tuple[str, str, str]]:
    return [
        (f"2k@{encoder}", f"1k@{encoder}", "frozen"),
        (f"4k@{encoder}", f"2k@{encoder}", "frozen"),
        (f"20k@{encoder}", f"10k@{encoder}", "frozen"),
        (f"10k@{encoder}", f"1k@{encoder}", "online"),
        (f"1k@{encoder}", f"1k@{encoder}", "online-frozen"),
        (f"10k@{encoder}", f"10k@{encoder}", "online-frozen"),
        (f"1k@{encoder}", f"1k@{encoder}:n600", "frozen"),
        (f"10k@{encoder}", f"10k@{encoder}:n600", "frozen"),
    ]


def capacity_verdict(summary: Mapping[str, Any], encoder: str) -> dict[str, Any]:
    """The declared 1K-versus-10K rule (research brief), applied to development evidence."""

    material, earns, harmed = [], [], []
    for family in FAMILIES:
        d1 = summary["contrasts"][f"{family}: 10k@{encoder} - 1k@{encoder} [frozen]"]
        d4 = summary["contrasts"][f"{family}: 10k@{encoder} - 4k@{encoder} [frozen]"]
        if d1["mean"] >= 0.03 and d1["holm"]["resolved_after_holm"] and d1["mean"] > 0:
            material.append(family)
        if d4["mean"] >= 0.02 and d4["holm"]["resolved_after_holm"] and d4["mean"] > 0:
            earns.append(family)
        if d1["interval_status"] == "MEASURED" and d1["upper"] < -0.02:
            harmed.append(family)
    if not material:
        verdict = "SCALE_NOT_JUSTIFIED"
    elif len(material) >= 3 and len(earns) >= 2 and not harmed:
        verdict = "SCALE_JUSTIFIED_PENDING_ADAPTATION_AND_PLASTICITY"
    else:
        verdict = "MIXED"
    return {"materially_improved": material, "earns_over_4k": earns, "harmed": harmed, "verdict": verdict}


# ------------------------------------------------------------------ later stages
def selected_sizes(capacity: Mapping[str, Any], encoder: str) -> dict[str, dict[str, Any]]:
    """The capacity stage's arms with their tuned budgets, by size tag."""

    arms = capacity["arms"]
    return {tag: arms[f"{tag}@{encoder}"] for tag in ("1k", "4k", "10k") if f"{tag}@{encoder}" in arms}


def _fixed(arm: Mapping[str, Any], name: str, **changes: Any) -> ArmSpec:
    base = {
        "encoder": arm["encoder"],
        "hidden": arm["hidden"],
        "output_head": arm["output_head"],
        "localize_head": arm["localize_head"],
        "channels": tuple(arm["channels"]),
        "learning_rate": arm["learning_rate"],
        "epochs": arm["epochs"],
    }
    base.update(changes)
    return ArmSpec(name, **base)


def optimization_arms(capacity: Mapping[str, Any], encoder: str) -> list[ArmSpec]:
    """One change at a time at the tuned 1K and 10K budgets (rate and epochs held)."""

    sizes = selected_sizes(capacity, encoder)
    arms = []
    for tag in ("1k", "10k"):
        arm = sizes[tag]
        arms += [
            _fixed(arm, f"{tag}:sgd"),
            _fixed(arm, f"{tag}:wd", weight_decay=1e-4),
            _fixed(arm, f"{tag}:momentum", momentum=0.9, learning_rate=arm["learning_rate"] * 0.1),
            _fixed(arm, f"{tag}:clip", clip=1.0),
        ]
    return arms


def optimization_primary() -> list[tuple[str, str, str]]:
    return [(f"{t}:{v}", f"{t}:sgd", "frozen") for t in ("1k", "10k") for v in ("wd", "momentum", "clip")]


def tool_arms(capacity: Mapping[str, Any], encoder: str) -> list[ArmSpec]:
    """Each selected size with the visible-test tool inputs, tuned afresh (the input differs)."""

    sizes = selected_sizes(capacity, encoder)
    arms = []
    for tag, arm in sizes.items():
        arms.append(_fixed(arm, f"{tag}:no_tool"))
        arms.append(
            ArmSpec(
                f"{tag}:tool",
                arm["encoder"],
                arm["hidden"],
                arm["output_head"],
                arm["localize_head"],
                tuple(arm["channels"]),
                tool=True,
            )
        )
    return arms


def tool_primary() -> list[tuple[str, str, str]]:
    return [(f"{t}:tool", f"{t}:no_tool", "frozen") for t in ("1k", "4k", "10k")] + [
        (f"{t}:tool", "visible_tests", "frozen") for t in ("1k", "4k", "10k")
    ]


def adapt_or_plasticity_arms(capacity: Mapping[str, Any], encoder: str) -> list[ArmSpec]:
    sizes = selected_sizes(capacity, encoder)
    return [_fixed(sizes[tag], tag) for tag in ("1k", "10k")]


def budget_trigger(capacity: Mapping[str, Any], encoder: str) -> bool:
    """The declared rule: any capacity size at 32 epochs with a 16-to-32 tune gain above 0.01."""

    for name, arm in capacity["arms"].items():
        if not name.endswith(f"@{encoder}") or arm["epochs"] != 32:
            continue
        table = {(r["learning_rate"], r["epochs"]): r["mean_tune_accuracy"] for r in arm["tuning"]}
        lr = arm["learning_rate"]
        if (lr, 16) in table and table[(lr, 32)] - table[(lr, 16)] > 0.01:
            return True
    return False


def budget_arms(capacity: Mapping[str, Any], encoder: str) -> list[ArmSpec]:
    sizes = selected_sizes(capacity, encoder)
    arms = []
    for tag in ("1k", "10k"):
        arms.append(_fixed(sizes[tag], f"{tag}:ep{sizes[tag]['epochs']}"))
        arms.append(_fixed(sizes[tag], f"{tag}:ep64", epochs=64))
    return arms


def budget_primary(capacity: Mapping[str, Any], encoder: str) -> list[tuple[str, str, str]]:
    sizes = selected_sizes(capacity, encoder)
    return [
        ("10k:ep64", "1k:ep64", "frozen"),
        ("1k:ep64", f"1k:ep{sizes['1k']['epochs']}", "frozen"),
        ("10k:ep64", f"10k:ep{sizes['10k']['epochs']}", "frozen"),
    ]
