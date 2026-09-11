"""Candidate training and the frozen learning-progress probe.

Learning is measured the only way that distinguishes it from a parameterless
analytic rule: frozen checkpoints are taken at increasing cumulative update
budgets and scored on **one fixed probe bank** that never changes between
checkpoints. A rule with no parameters has no budgets and produces a flat
curve; a learner that is actually identifying coefficients produces a
decreasing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence

import numpy as np

from ..config import WorldConfig
from ..experiment import TrialIdentity, run_episode
from ..metrics import normalized_errors
from ..predictors import OnlineRLSPredictor
from .families import build_environment, make_candidate, clone_candidate, straight_training_state, training_world, EpisodePlan
from .seeds import probe_seed, training_seed
from .spec import BenchmarkSpec


@dataclass
class TrainedReplica:
    replica: int
    model: OnlineRLSPredictor
    lineage: str
    seeds: tuple[int, ...]
    checkpoints: dict[int, dict[str, object]]
    diagnostics: dict[str, object]


def _training_plan(spec: BenchmarkSpec, lineage: str, replica: int, episode: int) -> EpisodePlan:
    world = training_world(spec)
    seed = training_seed(spec.randomness.root_seed, lineage, replica, episode)
    rng = np.random.default_rng(seed)
    position, velocity = straight_training_state(rng, world)
    return EpisodePlan(
        family="training",
        branch="training",
        replica=replica,
        episode=episode,
        stratum="training",
        environment_seed=seed,
        world=world,
        scenario="straight",
        initial_position=position,
        initial_velocity=velocity,
    )


def train_replica(spec: BenchmarkSpec, lineage: str, replica: int, *, role: str = "development") -> TrainedReplica:
    """Train one replica and snapshot frozen state at every declared budget."""

    model = make_candidate(spec, name="candidate_online", update_enabled=True)
    budgets = set(spec.training.learning_probe_budgets)
    checkpoints: dict[int, dict[str, object]] = {}
    seeds: list[int] = []
    if 0 in budgets:
        checkpoints[0] = model.state_dict()
    for episode in range(spec.training.episodes_per_replica):
        plan = _training_plan(spec, lineage, replica, episode)
        seeds.append(plan.environment_seed)
        identity = TrialIdentity(
            trial_id=f"training:{lineage}:r{replica:03d}:e{episode:04d}",
            role=role,
            family="training",
            scenario="straight",
            environment_seed=plan.environment_seed,
            replica_id=replica,
            episode=episode,
            branch="training",
            training_seed_lineage=(),
            stratum="training",
            update_mode="online",
        )
        run_episode(build_environment(plan), [model], identity, learn=True)
        if episode + 1 in budgets:
            checkpoints[episode + 1] = model.state_dict()
    diagnostics = dict(model.check_state())
    diagnostics["update_count"] = model.update_count
    diagnostics["forgetting_suspensions"] = model.forgetting_suspensions
    return TrainedReplica(
        replica=replica,
        model=model,
        lineage=lineage,
        seeds=tuple(seeds),
        checkpoints=checkpoints,
        diagnostics=diagnostics,
    )


def probe_bank(spec: BenchmarkSpec) -> list[EpisodePlan]:
    """The fixed development probe episodes used at every learning checkpoint."""

    world = training_world(spec)
    plans: list[EpisodePlan] = []
    for episode in range(spec.training.learning_probe_episodes):
        seed = probe_seed(spec.randomness.root_seed, spec.training.learning_probe_seed_offset, episode)
        rng = np.random.default_rng(seed)
        position, velocity = straight_training_state(rng, world)
        plans.append(
            EpisodePlan(
                family="learning_probe",
                branch="probe",
                replica=0,
                episode=episode,
                stratum="probe",
                environment_seed=seed,
                world=world,
                scenario="straight",
                initial_position=position,
                initial_velocity=velocity,
            )
        )
    return plans


def measure_learning_curve(
    spec: BenchmarkSpec, replicas: Sequence[TrainedReplica], *, role: str = "development"
) -> dict[str, object]:
    """Score every frozen budget checkpoint on the identical probe bank."""

    plans = probe_bank(spec)
    budgets = sorted(spec.training.learning_probe_budgets)
    per_budget: dict[str, dict[str, list[float]]] = {str(budget): {} for budget in budgets}
    for trained in replicas:
        for budget in budgets:
            state = trained.checkpoints.get(budget)
            if state is None:
                raise RuntimeError(f"replica {trained.replica} is missing a checkpoint at budget {budget}")
            values: list[float] = []
            for plan in plans:
                model = OnlineRLSPredictor.from_state_dict(
                    state, name="candidate_frozen", update_enabled=False
                )
                identity = TrialIdentity(
                    trial_id=f"learning_probe:b{budget:04d}:r{trained.replica:03d}:e{plan.episode:04d}",
                    role=role,
                    family="learning_probe",
                    scenario="straight",
                    environment_seed=plan.environment_seed,
                    replica_id=trained.replica,
                    episode=plan.episode,
                    branch=f"budget-{budget}",
                    stratum="probe",
                    update_mode="frozen",
                )
                records = run_episode(build_environment(plan), [model], identity, learn=False)
                values.append(mean(normalized_errors(records, "candidate_frozen")))
                after = model.state_dict()
                if after["weights"] != state["weights"] or after["update_count"] != state["update_count"]:
                    raise AssertionError("frozen probe evaluation mutated the checkpoint")
            per_budget[str(budget)][str(trained.replica)] = values
    curve = {
        str(budget): {
            "budget": budget,
            "replica_episode_values": per_budget[str(budget)],
            "mean": float(np.mean([v for values in per_budget[str(budget)].values() for v in values])),
        }
        for budget in budgets
    }
    return {
        "probe_episodes": len(plans),
        "probe_seeds": [plan.environment_seed for plan in plans],
        "budgets": budgets,
        "curve": curve,
    }
