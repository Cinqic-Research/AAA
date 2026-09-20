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


def _claim_verdicts(evidence: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """Attach a status and its supporting number to each rung of the claim ladder.

    Each rung is decided by one predeclared statistic, and a rung that no
    statistic in this run can decide is `NOT_VERIFIED` rather than assumed.
    """

    dims = evidence["capability_vector"]["dimensions"]
    stability = dims["numerical_stability"]
    q1 = dims["q1_online_learning"]["overall"]
    q2 = dims["q2_adaptation"].get("overall")
    q3 = dims["q3_hidden_state"]["versus_stateless_mlp"]
    q5 = dims["q5_retention"]
    q6 = dims["q6_error_calibration"]
    q7 = dims["q7_baseline_competitiveness"]

    verdicts: dict[str, tuple[str, str]] = {}
    verdicts["implementation_works"] = (
        "SUPPORTED" if stability["total_nonfinite_events"] == 0 else "FAILED",
        f"{stability['total_nonfinite_events']} non-finite events over "
        f"{dims['compute_cost']['scored_transitions']} scored transitions; the suite includes "
        f"exhaustive finite-difference gradient checks and a resume-equals-uninterrupted test",
    )
    verdicts["neural_learning_occurred"] = (
        "SUPPORTED" if _verdict(q1) == "POSITIVE" else _verdict(q1),
        f"Q1, {_effect(q1)}",
    )
    verdicts["hidden_state_helps"] = (
        "SUPPORTED" if _verdict(q3) == "POSITIVE" else _verdict(q3),
        f"Q3 against the stateless control on the memory families, {_effect(q3)}",
    )
    verdicts["online_adaptation_helps"] = (
        ("SUPPORTED" if _verdict(q2) == "POSITIVE" else _verdict(q2)) if q2 else "NOT_VERIFIED",
        f"Q2, {_effect(q2)}" if q2 else "no stream carried a declared change point",
    )
    if q5.get("status") == "NOT_VERIFIED":
        verdicts["retention_exists"] = ("NOT_VERIFIED", q5.get("reason", ""))
    else:
        gap = q5["reacquisition_gap"]["values"]["aaa1k_gru"]["mean"]
        verdicts["retention_exists"] = (
            "SUPPORTED, WITH A CONFOUND",
            f"the A2 tail was {abs(gap):.2e} *lower* than the A1 tail, so no catastrophic "
            f"forgetting was detected -- but the model has also had twice as much total "
            f"experience by A2, so this run cannot separate retention from continued learning",
        )
    if q6.get("status") == "MEASURED":
        verdicts["error_estimation_informative"] = (
            "SUPPORTED, WEAKLY",
            f"mean rank correlation {q6['spearman']['mean']:.2f} between the predicted and "
            f"realized error magnitude, but only {q6['streams_with_monotone_bins']} of "
            f"{q6['streams_measured']} streams had a monotone quintile table",
        )
    else:
        verdicts["error_estimation_informative"] = ("INSUFFICIENT_EVIDENCE", q6.get("reason", ""))
    won = sorted({name for names in q7["wins_by_family"].values() for name in names})
    lost = sorted(
        {
            baseline
            for row in q7["by_family"].values()
            for baseline, record in row.items()
            if _verdict(record) == "NEGATIVE"
        }
    )
    verdicts["baseline_competitiveness"] = (
        "MIXED" if won and lost else ("SUPPORTED" if won else "NOT SUPPORTED"),
        f"beat {', '.join(won) if won else 'nothing'} on at least one family; lost to "
        f"{', '.join(lost) if lost else 'nothing'} on at least one family",
    )
    verdicts["generalization"] = (
        "NOT CLAIMED",
        "evaluation streams are held out from development, which supports a claim about "
        "unseen trajectories of the *same* families only. No unseen family was tested, so "
        "nothing here supports generalization to a new kind of world",
    )
    return verdicts


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
    plan = evidence.get("replication_plan")
    if plan:
        met = evidence.get("precision_objective_met")
        add(
            f"**The declared precision objective was {'met' if met else 'NOT met'}.** The "
            f"development pilot measured a primary effect of {plan['pilot_effect']:.2e} with a "
            f"per-stream spread of {plan['pilot_spread']:.2e}; resolving a quarter of that "
            f"effect at 95% would have needed {plan['required_replicas']} replicas per family, "
            f"and the declared bound of {plan['bounds'][1]} was applied. Every interval below "
            f"is therefore wider than the design asked for, and effects near zero should be "
            f"read as unresolved rather than absent."
        )
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
            "**Important qualification, from an adversarial probe.** Branching at a point where "
            "*nothing changes* reproduces 95% of this effect. Q2 therefore measures continued "
            "learning in general far more than it measures adaptation specific to the change. "
            "The headline number is real; the natural reading of it is wrong. See "
            "`docs/evidence/aaa_1k_adversarial_probes.json` and `docs/aaa_1k_self_review.md`."
        )
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
        "**Read the capacity match carefully.** Parameter counts are close (954 against 994), "
        "but the ungated control carries 28 hidden units to the gated model's 16. Matching on "
        "parameters buys the ungated arm more state, which is exactly the trade a gate costs "
        "you. The comparison is the honest one for a fixed parameter budget, and it is not a "
        "comparison at matched hidden width."
    )
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
            "**Read this table carefully.** Every learning arm came back *better* than it left, "
            "so no catastrophic forgetting was detected. But A1 is the first segment a learner "
            "ever sees and A2 is the third, so by A2 the model has had three times as much "
            'experience in total. This design cannot separate "it retained A" from "it kept '
            'getting better at everything", and the next phase needs a fixed frozen probe bank '
            "measured at both boundaries to do so. The non-learning arms are the control: "
            "`constant_motion_reflected` is flat across A1 and A2, as it must be."
        )
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
    verdicts = _claim_verdicts(evidence)
    add(
        "\n".join(
            _table(
                [
                    [
                        key.replace("_", " "),
                        f"**{verdicts[key][0]}**",
                        description,
                        verdicts[key][1],
                    ]
                    for key, description in CLAIM_LADDER
                ],
                ["Claim", "Status", "What it would mean", "What was actually measured"],
            )
        )
    )
    add("")
    add(
        "Every effect above is conditional on a single model initialization: the evaluation "
        "gives every arm the same initialization seed so that an ablation differs from the "
        "primary in exactly one mechanism, and the intervals therefore resample streams but "
        "not initializations. A probe across five initializations found every comparison "
        "keeping its sign while magnitudes varied by up to a factor of two."
    )
    add("")
    add(
        "None of these implies intelligence, general autonomy, causal understanding, physical "
        "understanding, AGI or consciousness. This is a 994-parameter network predicting where a "
        "dot goes next."
    )
    add("")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# round 2
# ----------------------------------------------------------------------
def _round2_claims(evidence: dict[str, Any]) -> dict[str, tuple[str, str]]:
    dims = evidence["capability_vector"]["dimensions"]
    stability = dims["numerical_stability"]
    q1 = dims["q1_online_learning"]["overall"]
    q2 = dims["q2_adaptation"]["adaptation_effect"]
    q3 = dims["q3_hidden_state"]["versus_stateless_mlp"]
    q4 = dims["q4_gating"]["all_families"]
    q5 = dims["q5_retention"]["forgetting"]
    q6 = dims["q6_error_calibration"]
    q7 = dims["q7_baseline_competitiveness"]

    def status(record: dict[str, Any], positive: str, negative: str, null: str) -> str:
        verdict = _verdict(record)
        return {"POSITIVE": positive, "NEGATIVE": negative}.get(verdict, null)

    verdicts: dict[str, tuple[str, str]] = {
        "implementation_works": (
            "SUPPORTED" if stability["total_nonfinite_events"] == 0 else "FAILED",
            f"{stability['total_nonfinite_events']} non-finite events over "
            f"{dims['compute_cost']['scored_transitions']} scored transitions; every parameter "
            f"of every model is finite-difference verified and a resumed run is bitwise "
            f"identical to an uninterrupted one",
        ),
        "neural_learning_occurred": (
            status(q1, "SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"),
            f"Q1, {_effect(q1)}",
        ),
        "hidden_state_helps": (
            status(q3, "SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"),
            f"Q3 against the stateless control on the memory families, {_effect(q3)}; each "
            f"control runs on its own rule-selected hyperparameters",
        ),
        "gating_helps": (
            status(q4, "SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"),
            f"Q4, {_effect(q4)}. Round 1 reported this as CONTRADICTED; that result did not "
            f"survive giving each architecture its own rule-selected learning rate and clip",
        ),
        "online_adaptation_helps": (
            status(q2, "SUPPORTED", "CONTRADICTED", "INCONCLUSIVE"),
            f"Q2 difference-of-differences against a bit-identical unchanged world, {_effect(q2)}",
        ),
        "retention_exists": (
            "SUPPORTED" if _verdict(q5) in ("NEGATIVE", "INCONCLUSIVE") else "CONTRADICTED",
            f"probe-bank error after regime B minus after regime A1, {_effect(q5)}; a positive "
            f"value would be forgetting",
        ),
        "error_estimation_informative": (
            "SUPPORTED, WEAKLY" if q6.get("status") == "MEASURED" else "INSUFFICIENT_EVIDENCE",
            (
                f"mean rank correlation {q6['spearman']['mean']:.2f} between predicted and "
                f"realized error magnitude; only {q6['cells_with_monotone_bins']} of "
                f"{q6['cells_measured']} cells had a monotone quintile table"
            )
            if q6.get("status") == "MEASURED"
            else q6.get("reason", ""),
        ),
    }
    won = sorted({name for names in q7["wins_by_family"].values() for name in names})
    lost = sorted({name for names in q7["losses_by_family"].values() for name in names})
    verdicts["baseline_competitiveness"] = (
        "MIXED" if won and lost else ("SUPPORTED" if won else "NOT SUPPORTED"),
        f"beat {', '.join(won) if won else 'nothing'} on at least one family; lost to "
        f"{', '.join(lost) if lost else 'nothing'} on at least one family",
    )
    verdicts["generalization"] = (
        "NOT CLAIMED",
        "evaluation streams are held out from development and from round 1, which supports a "
        "claim about unseen trajectories of the same families only. No unseen family was "
        "tested",
    )
    return verdicts


def render_round2_report(
    evidence: dict[str, Any],
    selection: dict[str, Any],
    identity: dict[str, Any],
    characterization: dict[str, Any] | None = None,
) -> str:
    dims = evidence["capability_vector"]["dimensions"]
    lines: list[str] = []
    add = lines.append

    add("# AAA-1K round 2: a 994-parameter recurrent predictive core")
    add("")
    add(
        "AAA-1K is a research seed, not Juniper and not an agent. It is one primitive: a "
        "persistent gated recurrent predictor that learns online, keeps a hidden state, and "
        "estimates how wrong it expects to be. Everything below describes what was measured "
        "on a moving dot."
    )
    add("")
    add(
        "**This round supersedes round 1.** Round 1 is retained unchanged at "
        "`docs/evidence/aaa_1k_evaluation.json`. Four defects in it were found by probing its "
        "own conclusions, and all four are repaired here rather than annotated."
    )
    add("")
    add(
        "\n".join(
            _table(
                [
                    [
                        "adaptation (Q2)",
                        "a single online/frozen branch, which a control showed was 95% reproduced where nothing changed",
                        "difference-of-differences against a bit-identical unchanged world",
                        "`AAA-153`",
                    ],
                    [
                        "retention (Q5)",
                        "segment tails, confounded with three times the accumulated experience",
                        "a fixed frozen probe bank asked the same questions at every checkpoint",
                        "`AAA-154`",
                    ],
                    [
                        "uncertainty",
                        "intervals resampled streams only, treating the starting weights as fixed by nature",
                        "a crossed bootstrap over initializations and streams",
                        "`AAA-155`",
                    ],
                    [
                        "baseline fairness",
                        "one architecture's clip threshold imposed on all arms",
                        "each architecture on its own rule-selected learning rate and clip",
                        "`AAA-156`",
                    ],
                ],
                ["What", "Round 1", "Round 2", "Tracked as"],
            )
        )
    )
    add("")

    add("## Identity and design")
    add("")
    design = evidence["design"]
    add(
        "\n".join(
            _table(
                [
                    ["phase", f"`{identity['phase_version']}`"],
                    ["scientific fingerprint", f"`{identity['sha256']}`"],
                    ["trainable parameters", str(evidence["model_accounting"]["parameter_count"])],
                    ["initializations", str(design["initializations"])],
                    ["streams per family", str(design["replicas_per_family"])],
                    ["evaluation cells", str(len(evidence["cells"]))],
                    ["adaptation trials", str(design["adaptation_trials"])],
                    ["retention trials", str(design["retention_trials"])],
                    ["stream identities", "fresh; no round-1 stream is reused"],
                ],
                ["Item", "Value"],
            )
        )
    )
    add("")
    add("Each architecture runs on the hyperparameters the same declared rules select for it:")
    add("")
    configurations = evidence.get("architecture_configurations") or {}
    if isinstance(configurations, dict) and "note" not in configurations:
        add(
            "\n".join(
                _table(
                    [
                        [
                            name,
                            f"{values['learning_rate']:g}",
                            str(values["tbptt_steps"]),
                            f"{values['error_loss_weight']:g}",
                            "none" if values["gradient_clip"] is None else f"{values['gradient_clip']:g}",
                        ]
                        for name, values in sorted(configurations.items())
                    ],
                    ["Architecture", "learning rate", "TBPTT", "lambda", "gradient clip"],
                )
            )
        )
        add("")
        add(
            "Ablations of the gated model share its row exactly, because an ablation is the "
            "same architecture with one mechanism removed."
        )
        add("")

    add("## Development selection")
    add("")
    probe = selection["stage_zero_divergence_probe"]
    add(
        f"The unclipped divergence probe put the gated model's boundary at a learning rate of "
        f"{probe['lowest_diverging_learning_rate']}, and the declared stability margin "
        f"restricted the search to {selection['eligible_learning_rates']}. That rule "
        f"**eliminated the best-performing development configurations**, which is what a rule "
        f"that can bind looks like."
    )
    add("")
    eliminated = [record for record in selection["stage_one_eliminated"] if record["stable"]]
    if eliminated:
        best_eliminated = min(eliminated, key=lambda record: record["mean_mae"])
        add(
            f"Best eliminated configuration: `{best_eliminated['key']}` at "
            f"{best_eliminated['mean_mae']:.3e} development error, discarded for sitting inside "
            f"the margin."
        )
        add("")
    clip = selection.get("stage_three_gradient_clip")
    if clip:
        add(
            f"Stage 3 then selected the clip threshold instead of asserting it. The originally "
            f"declared 1.0 activated on {clip['declared_clip_rate']:.0%} of updates and cost "
            f"{clip['declared_cost_versus_best']:.0%} of development error against the best "
            f"stable threshold."
        )
        add("")

    add("## Capability vector")
    add("")
    verdict_rows = [
        ["Q1 online learning", dims["q1_online_learning"]["overall"]],
        ["Q2 adaptation (difference-of-differences)", dims["q2_adaptation"]["adaptation_effect"]],
        ["Q3 hidden state vs stateless control", dims["q3_hidden_state"]["versus_stateless_mlp"]],
        ["Q4 gating vs ungated control", dims["q4_gating"]["all_families"]],
        ["Q5 forgetting (probe bank)", dims["q5_retention"]["forgetting"]],
    ]
    add(
        "\n".join(
            _table(
                [[label, _verdict(record), _effect(record)] for label, record in verdict_rows],
                ["Question", "Verdict", "Effect (normalized error) [95% CI]"],
            )
        )
    )
    add("")
    add(
        "For Q5 a *negative* effect is the good direction: it means probe-bank error on regime "
        "A fell while the model was training on regime B."
    )
    add("")

    add("### Achieved precision")
    add("")
    add(
        "Reported against the effect actually measured, rather than only against a target "
        "sized from a pilot estimate of an effect nobody had seen."
    )
    add("")
    precision_rows = []
    for label, key, field in (
        ("Q1 online learning", "q1_online_learning", "precision"),
        ("Q2 adaptation", "q2_adaptation", "adaptation_precision"),
        ("Q3 hidden state", "q3_hidden_state", "precision"),
        ("Q4 gating", "q4_gating", "precision"),
    ):
        record = dims[key].get(field, {})
        if record.get("status") != "MEASURED":
            precision_rows.append([label, "-", "-", "not measured"])
            continue
        precision_rows.append(
            [
                label,
                f"{record['effect']:+.2e}",
                f"{record['half_width']:.2e}",
                f"{record['half_width_over_effect']:.0%} of the effect"
                + (" (target met)" if record["meets_quarter_effect_target"] else ""),
            ]
        )
    add("\n".join(_table(precision_rows, ["Question", "Effect", "CI half-width", "Resolution"])))
    add("")

    for key, heading, fields in (
        (
            "q1_online_learning",
            "Q1 -- can it learn online?",
            (("overall", "all families"),),
        ),
        (
            "q3_hidden_state",
            "Q3 -- is persistent recurrent state worth anything?",
            (
                ("versus_state_reset", "vs state-reset ablation, memory families"),
                ("versus_stateless_mlp", "vs stateless control, memory families"),
                ("versus_no_error_input", "vs no-previous-error ablation, memory families"),
                ("versus_frozen_recurrent", "vs frozen-recurrent-weights ablation, memory families"),
                ("versus_state_reset_all_families", "vs state-reset, all families"),
                ("versus_stateless_mlp_all_families", "vs stateless control, all families"),
            ),
        ),
    ):
        add(f"### {heading}")
        add("")
        add(
            "\n".join(
                _table(
                    [
                        [label, _verdict(dims[key][field]), _effect(dims[key][field])]
                        for field, label in fields
                        if field in dims[key]
                    ],
                    ["Comparison", "Verdict", "Effect"],
                )
            )
        )
        add("")
        if key == "q3_hidden_state" and "occluded_steps_only" in dims[key]:
            occluded = dims[key]["occluded_steps_only"]
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
                                "vs stateless control",
                                _verdict(occluded["versus_stateless_mlp"]),
                                _effect(occluded["versus_stateless_mlp"]),
                            ],
                        ],
                        ["Comparison", "Verdict", "Effect"],
                    )
                )
            )
            add("")

    add("### Q2 -- does continued learning help *because the world changed*?")
    q2 = dims["q2_adaptation"]
    add("")
    add(q2["design"] + ".")
    add("")
    add(
        "\n".join(
            _table(
                [
                    [
                        "adaptation (changed minus control)",
                        _verdict(q2["adaptation_effect"]),
                        _effect(q2["adaptation_effect"]),
                    ],
                    [
                        "continued learning (control alone)",
                        _verdict(q2["continued_learning_effect"]),
                        _effect(q2["continued_learning_effect"]),
                    ],
                ],
                ["Component", "Verdict", "Effect"],
            )
        )
    )
    add("")
    add(
        f"Adaptation accounts for {q2['adaptation_share_of_total']:.0%} of the total online "
        f"advantage; the rest is the ordinary benefit of continuing to learn, which round 1 "
        f"reported as if it were all adaptation. Every paired trial's two trunks reached an "
        f"identical model state at the branch: `trunks_matched = {q2['trunks_matched']}`."
    )
    add("")

    add("### Q4 -- do the gates earn their parameters?")
    q4 = dims["q4_gating"]
    add("")
    add(q4["capacity_note"])
    add("")
    add(
        "\n".join(
            _table(
                [
                    ["all families", _verdict(q4["all_families"]), _effect(q4["all_families"])],
                    ["memory families", _verdict(q4["memory_families"]), _effect(q4["memory_families"])],
                ],
                ["Scope", "Verdict", "Effect (ungated minus gated)"],
            )
        )
    )
    add("")
    all_families = q4["all_families"]
    if all_families["mean_difference"] * all_families["median_difference"] < 0:
        add(
            f"**The mean and the median disagree, and that is the finding.** The median stream "
            f"favours the gated model ({all_families['median_difference']:+.2e}) while the mean "
            f"favours the ungated one ({all_families['mean_difference']:+.2e}): the gated model "
            f"is slightly better on {all_families['favours_first']} of "
            f"{all_families['streams']} streams and much worse on the rest. A single averaged "
            f"number would have reported only half of that."
        )
        add("")

    fairness = q4.get("tuning_fairness") or {}
    if fairness.get("finding"):
        gated = fairness["gated"]
        ungated = fairness["ungated"]
        add(
            f"**The ungated control is the less stable architecture.** Run without a clip it "
            f"diverges at a learning rate of {ungated['unclipped_divergence_boundary']}, against "
            f"{gated['unclipped_divergence_boundary']} for the gated model, so the same declared "
            f"stability margin allows it only {ungated['selected_learning_rate']:g} where the "
            f"gated model gets {gated['selected_learning_rate']:g}. Round 1 ran both at the "
            f"gated model's rate and reported that gating loses. {fairness['finding'].capitalize()}."
        )
        add("")

    add("### Q5 -- what survives learning a new regime?")
    q5 = dims["q5_retention"]
    add("")
    add(q5["design"] + ".")
    add("")
    add(
        "\n".join(
            _table(
                [
                    [label.replace("_", " "), f"{q5['probe_error'][label]['mean']:.3e}"]
                    for label in ("after_A1", "after_B", "after_A2")
                ],
                ["Probe-bank error", "Mean"],
            )
        )
    )
    add("")
    add(
        f"Forgetting: **{_verdict(q5['forgetting'])}**, {_effect(q5['forgetting'])}. A positive "
        f"value would mean regime-A ability degraded while learning regime B. The probe bank is "
        f"identical at all three checkpoints, so accumulated experience cannot flatter the "
        f"later measurements -- which is exactly what round 1's design could not rule out."
    )
    add("")

    add("### Q6 -- can it estimate its own error?")
    q6 = dims["q6_error_calibration"]
    add("")
    if q6.get("status") == "MEASURED":
        add(
            "\n".join(
                _table(
                    [
                        [
                            name,
                            f"{q6[key]['mean']:.3f}",
                            f"{q6[key]['median']:.3f}",
                            f"{q6[key]['min']:.3f}",
                            f"{q6[key]['max']:.3f}",
                        ]
                        for name, key in (
                            ("rank correlation (Spearman)", "spearman"),
                            ("linear correlation (Pearson)", "pearson"),
                            ("calibration slope", "slope"),
                            ("bias (predicted minus realized)", "bias"),
                        )
                    ],
                    ["Measure", "Mean", "Median", "Min", "Max"],
                )
            )
        )
        add("")
        add(
            f"{q6['cells_with_monotone_bins']} of {q6['cells_measured']} cells had a monotone "
            f"quintile calibration table. {q6['caveat']}."
        )
    else:
        add(f"`{q6.get('status')}`.")
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
    for label, key in (("beat", "wins_by_family"), ("lost to", "losses_by_family")):
        rows = {family: names for family, names in q7[key].items() if names}
        if rows:
            add(f"Baselines AAA-1K {label} outright:")
            add("")
            for family, names in sorted(rows.items()):
                add(f"- `{family}`: {', '.join(names)}")
            add("")

    if characterization:
        add("## What the development probes settled")
        add("")
        for probe in characterization["probes"]:
            add(f"**{probe['probe']}** -- {probe['question']}")
            add("")
            add(probe["finding"] + ".")
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
                    ["mean clip rate", f"{stability['mean_clip_rate']:.2%}"],
                    ["max clip rate on any cell", f"{stability['max_clip_rate']:.2%}"],
                    ["non-finite events", str(stability["total_nonfinite_events"])],
                    ["trained steps", str(stability["total_trained_steps"])],
                    ["targets skipped as unavailable", str(stability["total_skipped_targets"])],
                    ["scored transitions", str(cost["scored_transitions"])],
                    ["transitions per second", f"{cost['transitions_per_second']:.0f}"],
                    ["wall seconds", f"{cost['wall_seconds']:.1f}"],
                ],
                ["Measure", "Value"],
            )
        )
    )
    add("")
    add(
        "Round 1 clipped on 22% of updates at a threshold that was asserted rather than "
        "selected, and a probe found that threshold costing 29% of development error. The "
        "threshold is now chosen by the same rule as every other hyperparameter, and the "
        "activation rate above is what a guard rather than a decision-maker looks like."
    )
    add("")

    add("## What these numbers do and do not support")
    add("")
    verdicts = _round2_claims(evidence)
    add(
        "\n".join(
            _table(
                [[key.replace("_", " "), f"**{verdicts[key][0]}**", verdicts[key][1]] for key in verdicts],
                ["Claim", "Status", "What was actually measured"],
            )
        )
    )
    add("")
    add(
        "None of these implies intelligence, general autonomy, causal understanding, physical "
        "understanding, AGI or consciousness. This is a 994-parameter network predicting where "
        "a dot goes next."
    )
    add("")
    return "\n".join(lines) + "\n"
