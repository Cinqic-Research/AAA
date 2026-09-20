"""Render AAA-1K evidence as a report that states exactly what was measured.

Every claim in the output is tied to a number in the evidence file, and every
claim carries its own boundary. The renderer never upgrades a measurement into
a capability: "the online copy beat its frozen twin" is not "the model adapts",
and neither is "the model understands".
"""

from __future__ import annotations

import math
from typing import Any

CLAIM_LADDER = (
    ("implementation_works", "the code executes correctly and deterministically"),
    ("neural_learning_occurred", "weights changed in a useful direction on some measured stream"),
    ("hidden_state_helps", "persistent recurrence beat matched state-reset and stateless controls"),
    ("online_adaptation_helps", "continued learning beat an identical frozen-weight clone"),
    ("retention_exists", "learning a new regime did not completely erase an old one"),
    ("error_estimation_informative", "predicted error magnitude tracked realized error"),
    ("baseline_competitiveness", "the model beat the specified simple alternatives"),
    ("generalization", "held-out trajectories, regimes or families support a generalization claim"),
)


def _verdict(record: dict[str, Any]) -> str:
    """Interval-based verdict, using the same four statuses the AAA core uses."""

    if record.get("interval_status") != "MEASURED":
        return "INSUFFICIENT_EVIDENCE"
    low, high = record["ci_low"], record["ci_high"]
    if not (math.isfinite(low) and math.isfinite(high)):
        return "INSUFFICIENT_EVIDENCE"
    if low > 0:
        return "POSITIVE"
    if high < 0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def _effect(record: dict[str, Any]) -> str:
    if record.get("interval_status") != "MEASURED":
        return f"{record.get('mean_difference', float('nan')):+.2e} (no interval)"
    return (
        f"{record['mean_difference']:+.2e} "
        f"[{record['ci_low']:+.2e}, {record['ci_high']:+.2e}] "
        f"({record['favours_first']}/{record['streams']} streams)"
    )


def _table(rows: list[list[str]], header: list[str]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_report(evidence: dict[str, Any], selection: dict[str, Any], identity: dict[str, Any]) -> str:
    dims = evidence["capability_vector"]["dimensions"]
    lines: list[str] = []
    add = lines.append

    add("# AAA-1K: a 994-parameter recurrent predictive core")
    add("")
    add(
        "AAA-1K is a research seed, not Juniper and not an agent. It is one primitive: a "
        "persistent gated recurrent predictor that learns online, keeps a hidden state, and "
        "estimates how wrong it expects to be. Everything below describes what was measured "
        "on a moving dot."
    )
    add("")

    add("## Identity")
    add("")
    add(
        "\n".join(
            _table(
                [
                    ["phase", f"`{identity['phase_version']}`"],
                    ["scientific fingerprint", f"`{identity['sha256']}`"],
                    ["files covered", str(identity["file_count"])],
                    ["trainable parameters", str(evidence["model_accounting"]["parameter_count"])],
                    [
                        "selected configuration",
                        f"lr={evidence['configuration']['learning_rate']:g}, "
                        f"T={evidence['configuration']['tbptt_steps']}, "
                        f"lambda={evidence['configuration']['error_loss_weight']:g}",
                    ],
                    ["replicas per family", str(evidence["replicas_per_family"])],
                    ["evaluation streams", str(len(evidence["streams"]))],
                ],
                ["Item", "Value"],
            )
        )
    )
    add("")

    add("## Parameter and state accounting")
    add("")
    footprint = evidence["model_accounting"]["state_footprint"]
    add(
        "\n".join(
            _table(
                [[key.replace("_", " "), str(value)] for key, value in footprint.items()],
                ["Scalar category", "Count"],
            )
        )
    )
    add("")
    add(
        "The optimizer is plain SGD and holds no state, so the adaptive footprint is the "
        "parameters, the 16-value hidden state, and the truncation buffer. There is no hidden "
        "second model."
    )
    add("")

    add("## Development selection")
    add("")
    probe = selection["stage_zero_divergence_probe"]
    boundary = probe["lowest_diverging_learning_rate"]
    add(
        f"The unclipped divergence probe put the boundary at a learning rate of "
        f"{boundary}. The declared stability margin therefore restricted the search to "
        f"{selection['eligible_learning_rates']}, which **eliminated the best-performing "
        f"configurations on development data**. That cost is real and is reported rather than "
        f"quietly avoided: the rule was declared before the numbers existed, and a rule that "
        f"never binds is not a rule."
    )
    add("")
    eliminated = [record for record in selection["stage_one_eliminated"] if record["stable"]]
    if eliminated:
        best_eliminated = min(eliminated, key=lambda record: record["mean_mae"])
        chosen = min(
            (
                record
                for record in selection["stage_one"]
                if record["key"].startswith(
                    f"lr={selection['selected']['learning_rate']:g};T={selection['selected']['tbptt_steps']}"
                )
            ),
            key=lambda record: record["mean_mae"],
        )
        add(
            f"Best eliminated configuration: `{best_eliminated['key']}` at "
            f"{best_eliminated['mean_mae']:.3e} development error. Selected: "
            f"`{chosen['key']}` at {chosen['mean_mae']:.3e}, a "
            f"{(chosen['mean_mae'] / best_eliminated['mean_mae'] - 1):.0%} development penalty "
            f"paid for the margin."
        )
        add("")
    add(f"Auxiliary-weight decision: {selection['error_weight_decision']}.")
    add("")

    add("## Capability vector")
    add("")
    add(
        "Reported as separate dimensions. There is deliberately no combined score: one number "
        "is the most efficient way to hide a failure inside a success."
    )
    add("")

    rows = []
    for key, question in (
        ("q1_online_learning", "Q1 online learning"),
        ("q3_hidden_state", "Q3 hidden state"),
        ("q4_gating", "Q4 gating"),
        ("q2_adaptation", "Q2 online vs frozen"),
    ):
        record = dims[key]
        primary = record.get("overall") or record.get("versus_stateless_mlp") or record.get("all_families")
        rows.append([question, _verdict(primary), _effect(primary)])
    add("\n".join(_table(rows, ["Question", "Verdict", "Effect (normalized error) [95% CI]"])))
    add("")

    add("### Q1 -- can it learn online?")
    q1 = dims["q1_online_learning"]
    add("")
    add(f"Overall: **{_verdict(q1['overall'])}**, {_effect(q1['overall'])}.")
    add("")
    add(
        "\n".join(
            _table(
                [
                    [family, _verdict(record), _effect(record)]
                    for family, record in sorted(q1["by_family"].items())
                ],
                ["Family", "Verdict", "First quarter minus last quarter"],
            )
        )
    )
    add("")

    add("### Q2 -- does continued learning help after a change?")
    q2 = dims["q2_adaptation"]
    add("")
    if q2.get("branches"):
        add(f"Overall: **{_verdict(q2['overall'])}**, {_effect(q2['overall'])}.")
        add("")
        add(
            f"Every branch started from a clone whose complete model-state hash matched its twin: "
            f"`clone_hashes_matched = {q2['clone_hashes_matched']}`. The frozen arm kept running "
            f"its recurrence and its previous-error input; only its weights stopped moving."
        )
        add("")
        add(
            "\n".join(
                _table(
                    [
                        [family, _verdict(record), _effect(record)]
                        for family, record in sorted(q2["by_family"].items())
                    ],
                    ["Family", "Verdict", "Frozen minus online"],
                )
            )
        )
    else:
        add("`NOT_VERIFIED`: no stream in this run carried a declared change point.")
    add("")

    add("### Q3 -- is persistent recurrent state worth anything?")
    q3 = dims["q3_hidden_state"]
    add("")
    add(
        "\n".join(
            _table(
                [
                    [
                        "vs state-reset ablation, memory families",
                        _verdict(q3["versus_state_reset"]),
                        _effect(q3["versus_state_reset"]),
                    ],
                    [
                        "vs stateless MLP, memory families",
                        _verdict(q3["versus_stateless_mlp"]),
                        _effect(q3["versus_stateless_mlp"]),
                    ],
                    [
                        "vs state-reset, all families",
                        _verdict(q3["versus_state_reset_all_families"]),
                        _effect(q3["versus_state_reset_all_families"]),
                    ],
                    [
                        "vs stateless MLP, all families",
                        _verdict(q3["versus_stateless_mlp_all_families"]),
                        _effect(q3["versus_stateless_mlp_all_families"]),
                    ],
                ],
                ["Comparison", "Verdict", "Control minus AAA-1K"],
            )
        )
    )
    occluded = q3.get("occluded_steps_only", {})
    if occluded.get("versus_state_reset"):
        add("")
        add("Restricted to steps whose target was never shown to the agent:")
        add("")
        add(
            "\n".join(
                _table(
                    [
                        [
                            "vs state-reset",
                            _verdict(occluded["versus_state_reset"]),
                            _effect(occluded["versus_state_reset"]),
                        ],
                        [
                            "vs stateless MLP",
                            _verdict(occluded["versus_stateless_mlp"]),
                            _effect(occluded["versus_stateless_mlp"]),
                        ],
                    ],
                    ["Comparison", "Verdict", "Control minus AAA-1K"],
                )
            )
        )
    add("")

    add("### Q4 -- do the gates earn their parameters?")
    q4 = dims["q4_gating"]
    add("")
    add(q4["note"] + ".")
    add("")
    add(
        "\n".join(
            _table(
                [
                    ["all families", _verdict(q4["all_families"]), _effect(q4["all_families"])],
                    ["memory families", _verdict(q4["memory_families"]), _effect(q4["memory_families"])],
                ],
                ["Scope", "Verdict", "Ungated RNN minus AAA-1K"],
            )
        )
    )
    add("")

    add("### Q5 -- what survives A, then B, then A again?")
    q5 = dims["q5_retention"]
    add("")
    if q5.get("status") == "NOT_VERIFIED":
        add("`NOT_VERIFIED`: " + q5["reason"] + ".")
    else:
        gaps = q5["reacquisition_gap"]["values"]
        add(q5["reacquisition_gap"]["definition"] + ".")
        add("")
        add(
            "\n".join(
                _table(
                    [
                        [
                            name,
                            f"{q5['segments']['A1'][name]['mean']:.3e}",
                            f"{q5['segments']['B'][name]['mean']:.3e}",
                            f"{q5['segments']['A2'][name]['mean']:.3e}",
                            f"{gaps[name]['mean']:+.3e}",
                        ]
                        for name in gaps
                    ],
                    ["Arm", "A1 tail", "B tail", "A2 tail", "A2 - A1"],
                )
            )
        )
    add("")

    add("### Q6 -- can it estimate its own error?")
    q6 = dims["q6_error_calibration"]
    add("")
    if q6.get("status") != "MEASURED":
        add(f"`{q6.get('status')}`: {q6.get('reason')}.")
    else:
        add(
            "\n".join(
                _table(
                    [
                        [
                            "rank correlation (Spearman)",
                            f"{q6['spearman']['mean']:.3f}",
                            f"{q6['spearman']['median']:.3f}",
                            f"{q6['spearman']['min']:.3f}",
                            f"{q6['spearman']['max']:.3f}",
                        ],
                        [
                            "linear correlation (Pearson)",
                            f"{q6['pearson']['mean']:.3f}",
                            f"{q6['pearson']['median']:.3f}",
                            f"{q6['pearson']['min']:.3f}",
                            f"{q6['pearson']['max']:.3f}",
                        ],
                        [
                            "calibration slope",
                            f"{q6['slope']['mean']:.3f}",
                            f"{q6['slope']['median']:.3f}",
                            f"{q6['slope']['min']:.3f}",
                            f"{q6['slope']['max']:.3f}",
                        ],
                        [
                            "bias (predicted minus realized)",
                            f"{q6['bias']['mean']:.3f}",
                            f"{q6['bias']['median']:.3f}",
                            f"{q6['bias']['min']:.3f}",
                            f"{q6['bias']['max']:.3f}",
                        ],
                    ],
                    ["Measure", "Mean", "Median", "Min", "Max"],
                )
            )
        )
        add("")
        add(
            f"{q6['streams_with_monotone_bins']} of {q6['streams_measured']} streams had a "
            f"monotone quintile calibration table. {q6['caveat']}."
        )
    add("")

    add("### Q7 -- does it beat the simple alternatives?")
    q7 = dims["q7_baseline_competitiveness"]
    add("")
    add(q7["decision_rule"] + ".")
    add("")
    families = sorted(q7["by_family"])
    baselines = sorted(next(iter(q7["by_family"].values())))
    add(
        "\n".join(
            _table(
                [
                    [family] + [_verdict(q7["by_family"][family][baseline]) for baseline in baselines]
                    for family in families
                ],
                ["Family", *baselines],
            )
        )
    )
    add("")
    wins = {family: names for family, names in q7["wins_by_family"].items() if names}
    if wins:
        add("Baselines AAA-1K beat outright, by family:")
        add("")
        for family, names in sorted(wins.items()):
            add(f"- `{family}`: {', '.join(sorted(names))}")
    else:
        add("AAA-1K did not beat any listed baseline outright on any family.")
    add("")

    add("## Numerical stability and cost")
    add("")
    stability = dims["numerical_stability"]
    cost = dims["compute_cost"]
    add(
        "\n".join(
            _table(
                [
                    ["gradient-clip activations", str(stability["total_clip_events"])],
                    ["non-finite events", str(stability["total_nonfinite_events"])],
                    ["trained steps", str(stability["total_trained_steps"])],
                    ["targets skipped as unavailable", str(stability["total_skipped_targets"])],
                    ["scored transitions", str(cost["scored_transitions"])],
                    ["transitions per second", f"{cost['transitions_per_second']:.0f}"],
                    ["wall seconds", f"{cost['wall_seconds']:.1f}"],
                    ["arms per stream", str(cost["arms"])],
                ],
                ["Measure", "Value"],
            )
        )
    )
    add("")
    add(
        "Gradient clipping is a declared mechanism with a declared threshold, not a silent "
        "safety net, so its activation count is part of the result."
    )
    add("")

    add("## What these numbers do and do not support")
    add("")
    add(
        "\n".join(
            _table(
                [[key.replace("_", " "), description] for key, description in CLAIM_LADDER],
                ["Claim", "What it would mean"],
            )
        )
    )
    add("")
    add(
        "None of these implies intelligence, general autonomy, causal understanding, physical "
        "understanding, AGI or consciousness. This is a 994-parameter network predicting where a "
        "dot goes next."
    )
    add("")
    return "\n".join(lines) + "\n"
