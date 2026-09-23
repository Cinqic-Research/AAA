"""The preregistered protocol of ``aaa.1k.v2``, as code.

Everything here was committed before any development, attack or confirmation
identity of this phase was observed (see the git history of this file and
``docs/aaa_1k_v2_benchmark_protocol.md``). Changing any value after an
identity has been observed makes the affected stage a new stage with new
identities; it is never an edit.

Stages
------
``development``  hyperparameter selection per candidate on development
                 identities, then a screen of each selected candidate against
                 Champion 1 that designates at most one challenger;
``attack``       fresh identities, attack-only families, perturbed
                 initializations and a cross-backend check; every criterion
                 must pass;
``freeze``       challenger, protocol, thresholds, identities, backend and
                 source fingerprint committed before confirmation;
``confirmation`` observed once on fresh identities, all families including the
                 held-out ones; decision recomputed independently.
``capacity``     post-confirmation exploratory capacity sweep; cannot change
                 the confirmation decision.
"""

from __future__ import annotations

from typing import Any

from . import stress
from .external.dysts_subset import select_systems
from .external.monash import DATASETS as MONASH
from .external.tasks import calibration_namespace

PROTOCOL_VERSION = "aaa.1k.v2.protocol.v1"

# ----------------------------------------------------------------------
# candidate budget and hyperparameter grid (development)
# ----------------------------------------------------------------------
CANDIDATES = ("gru_v1_retuned", "gru_v1_keep-2", "gru_v2", "elman_v2", "mgu_v2", "lru_v2")
"""Precommitted candidate budget: six. Champion 1 participates unchanged as the reference."""

LEARNING_RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
TBPTT_OPTIONS = ((1, "live"), (4, "live"), (4, "replay"), (16, "live"), (16, "replay"))
GRADIENT_CLIPS = (None, 1.0, 10.0)
ERROR_LOSS_WEIGHT = 0.25
MARGIN_STEPS = 2
"""An eligible learning rate sits at least two grid positions below the lowest rate at which an unclipped
configuration (T=4 live, or RTRL) failed or diverged on any development cell."""

TIE_MARGIN = 0.02
"""Selection-score ties within 2% prefer, in order: the lower learning rate, the live rule, the shorter
horizon, the larger clip threshold (None counts as largest), i.e. the simpler and more conservative option."""

DIVERGENCE_FACTOR = 2.0
"""A dot cell diverged if it failed or its MAE exceeds twice persistence's on the same stream (loop rule)."""

# ----------------------------------------------------------------------
# identities
# ----------------------------------------------------------------------
INITS = {
    "development": 4,
    "attack": 5,
    "confirmation": 8,
    "diagnostic": 5,
    "capacity": 5,
    "qualification": 4,
    "scratch": 2,
}
DOT_STREAMS = {
    "development": 12,
    "attack": 8,
    "confirmation": 16,
    "diagnostic": 8,
    "capacity": 8,
    "qualification": 4,
    "scratch": 2,
}
EXTERNAL_STREAMS = {
    "development": {"narma10": 3, "narma20": 3, "mackey_glass17": 3, "dysts": 2},
    "attack": {"narma10": 3, "narma20": 3, "mackey_glass17": 3, "dysts": 2},
    "confirmation": {"narma10": 8, "narma20": 8, "mackey_glass17": 8, "dysts": 4},
    "capacity": {"narma10": 3, "narma20": 3, "mackey_glass17": 3, "dysts": 0},
    "scratch": {"narma10": 1, "narma20": 1, "mackey_glass17": 1, "dysts": 1},
    "qualification": {"narma10": 1, "narma20": 0, "mackey_glass17": 1, "dysts": 0},
}
STAGE_FAMILIES = {
    "development": tuple(stress.families("development")),
    "attack": tuple(stress.families("development")) + tuple(stress.families("attack")),
    "confirmation": tuple(stress.families()),
}
V1_FAMILIES = tuple(name for name in stress.FAMILIES if name.startswith("v1_"))
STRESS_FAMILIES = tuple(name for name in stress.FAMILIES if not name.startswith("v1_"))
HELD_OUT_FAMILIES = tuple(stress.families("held_out"))
PROBE_BANK = 8


def external_tasks() -> list[str]:
    return ["narma10", "narma20", "mackey_glass17", *[f"dysts:{s['system']}" for s in select_systems()]]


def monash_tasks() -> list[str]:
    return [f"monash:{name}" for name in MONASH]


def _block(role: str, namespace: str, count: int, purpose: str) -> dict[str, Any]:
    return {
        "block_id": f"v2-{role}-{namespace}",
        "role": role,
        "namespace": namespace,
        "start": 0,
        "count": count,
        "purpose": purpose,
    }


def declared_blocks() -> list[dict[str, Any]]:
    """Every identity block this phase will use, declared up front."""

    blocks: list[dict[str, Any]] = []
    for role in (
        "development",
        "attack",
        "confirmation",
        "diagnostic",
        "capacity",
        "qualification",
        "scratch",
    ):
        blocks.append(_block(role, "init", INITS[role], f"{role} model initializations"))
        families = STAGE_FAMILIES.get(role, tuple(stress.FAMILIES))
        for family in families:
            blocks.append(_block(role, f"env.{family}", DOT_STREAMS[role], f"{role} streams of {family}"))
        for task, count in EXTERNAL_STREAMS.get(role, {}).items():
            if task == "dysts":
                if count:
                    for system in select_systems():
                        name = system["system"]
                        blocks.append(
                            _block(role, f"ext.dysts-{name}", count, f"{role} initial conditions of {name}")
                        )
            elif count:
                blocks.append(_block(role, f"ext.{task}", count, f"{role} realizations of {task}"))
        blocks.append(_block(role, "bootstrap", 64, f"{role} bootstrap generator seeds"))
    for role in ("confirmation",):
        blocks.append(_block(role, "probe.v1_aba", PROBE_BANK, "retention probe bank (regime A episodes)"))
        blocks.append(
            _block(role, "env.paired_change", DOT_STREAMS[role], "paired-change adaptation streams")
        )
        blocks.append(_block(role, "env.retention", DOT_STREAMS[role], "A/B/A retention training streams"))
    blocks.append(_block("development", "external.dysts-selection", 1, "dysts subset selection draw"))
    for task in ["narma10", "narma20", "mackey_glass17", *[f"dysts:{s['system']}" for s in select_systems()]]:
        blocks.append(
            _block("development", calibration_namespace(task), 4, f"normalization calibration for {task}")
        )
    return blocks


def all_declared_blocks() -> list[dict[str, Any]]:
    """The plan's blocks plus the diagnostic blocks declared with the diagnostics stage."""

    from .diagnostics import DIAGNOSTIC_BLOCKS

    return [*declared_blocks(), *(dict(block) for block in DIAGNOSTIC_BLOCKS)]


def block_seeds_for(
    registry: dict[str, Any], role: str, namespace: str, *, observer: str | None = None
) -> list[int]:
    """Seeds of ``v2-<role>-<namespace>``, refusing the wrong use.

    Selection roles must be selection-eligible; a confirmation block may only
    be read by the run that already marked it spent (``observer``).
    """

    from .identities import IdentityError, find, require, seeds_of

    block_id = f"v2-{role}-{namespace}"
    if role in ("development", "attack", "diagnostic"):
        require(registry, block_id, purpose="selection")
    elif role == "confirmation":
        block = find(registry, block_id)
        if block["status"] != "spent" or block.get("observed_by") != observer:
            raise IdentityError(
                f"{block_id} is {block['status']} by {block.get('observed_by')!r}, not spent by {observer!r}"
            )
    else:
        require(registry, block_id, purpose=role)
    return seeds_of(registry, block_id)


# ----------------------------------------------------------------------
# development screen, attack and confirmation criteria
# ----------------------------------------------------------------------
BOOTSTRAP_DRAWS = 4000
CONFIDENCE = 0.95
NONINFERIORITY_MARGIN = 0.02
"""v1 regression families: relative-MAE upper bound must not exceed +2% (loop screen rule)."""

STRESS_IMPROVEMENT = 0.95
"""Confirmation K3: geometric-mean relative MAE over stress families must be at most 0.95 (a >= 5% overall
improvement) with its upper bound below 1.0. 'Slightly lower average error' does not promote."""

REGRESSION_GUARD = 0.05
"""No stress family may show a statistically resolved regression beyond +5% (relative lower bound > 0.05)."""

EXTERNAL_MARGIN = 0.02
"""External suite: geometric-mean relative error upper bound must not exceed 1.02."""

SCREEN = {
    "S1_stability": "no failed or diverged development cell (already an eligibility rule)",
    "S2_v1_noninferiority": f"every v1 family: crossed-bootstrap relative-MAE upper bound <= +{NONINFERIORITY_MARGIN:.0%}",
    "S3_stress_superiority": "development stress families: geometric-mean relative MAE upper bound < 1.0",
    "S4_external_noninferiority": f"narma10, narma20, mackey_glass17: geometric-mean relative prequential error upper bound <= {1 + EXTERNAL_MARGIN}",
    "designation": "the passing candidate with the lowest S3 point estimate becomes the challenger; none -> no challenger",
}

ATTACK = {
    "A1_stability": "challenger: zero failed cells, and diverged cells <= Champion 1's on the same attack cells",
    "A2_v1_noninferiority": "as S2, on attack identities",
    "A3_stress_superiority": "as S3, over development and attack-only stress families on attack identities",
    "A4_regression_guard": f"no stress family with relative-MAE lower bound > +{REGRESSION_GUARD:.0%}",
    "A5_external_noninferiority": "as S4 over narma10, narma20, mackey_glass17 and the twelve dysts systems",
    "A6_backend_verdicts": "A1-A3 recomputed from a CUDA re-run of the attack v1 families give identical verdicts",
    "A7_perturbed_initialization": "challenger with recurrent/input initial weights scaled x1.5 passes A1",
}

CONFIRMATION = {
    "K1_stability": "challenger: zero failed cells and diverged cells <= Champion 1's",
    "K2_v1_noninferiority": f"every v1 family non-inferior at +{NONINFERIORITY_MARGIN:.0%}",
    "K3_stress_improvement": f"geometric-mean relative MAE over all 19 stress families <= {STRESS_IMPROVEMENT} with upper bound < 1.0",
    "K4_regression_guard": f"no stress family with relative-MAE lower bound > +{REGRESSION_GUARD:.0%}",
    "K5_external_noninferiority": f"external suite geometric-mean relative error upper bound <= {1 + EXTERNAL_MARGIN}",
    "decision": "PROMOTE if every criterion PASSES; REJECT if any FAILS; otherwise INCONCLUSIVE (Champion 1 retained)",
    "multiplicity": "intersection-union: promotion requires every criterion, so no alpha is spent per criterion",
}

CONFIRMATION_BACKEND = "cpu"
"""Frozen in advance: confirmation runs on the CPU (NumPy float64), whose exact reproduction is established
on the Zen 3 evidence platform; the CUDA path is qualified separately and used for development."""

DEVELOPMENT_BACKEND = "cpu"
"""Development runs on the CPU with 8 single-threaded workers. Measured on FLOWBOX, its jobs (one core type
and one TBPTT horizon per lockstep batch, about 1-2 thousand cells) run faster on the parallel CPU path than
on the RTX 2060, which overtakes it only above roughly 2,000-4,000 lockstep cells
(``docs/aaa_1k_v2_compute_report.md``). CUDA is used where it is qualified and useful: backend parity,
the compute qualification, and attack criterion A6 (cross-backend verdict equivalence)."""
