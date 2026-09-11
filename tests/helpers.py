"""Shared construction helpers for the AAA test suite."""

from __future__ import annotations

from typing import Sequence

from aaa.experiment import StepRecord, TrialIdentity


def identity(
    *,
    trial_id: str = "test:trial",
    role: str = "development",
    family: str = "test",
    scenario: str = "changed",
    environment_seed: int = 2,
    replica_id: int = 0,
    episode: int = 0,
    branch: str = "main",
    stratum: str = "unstratified",
    update_mode: str = "frozen",
    confirmation_batch: str | None = None,
    training_seed_lineage: tuple[int, ...] = (),
    checkpoint_hash: str | None = None,
) -> TrialIdentity:
    return TrialIdentity(
        trial_id=trial_id,
        role=role,
        family=family,
        scenario=scenario,
        environment_seed=environment_seed,
        replica_id=replica_id,
        episode=episode,
        branch=branch,
        stratum=stratum,
        update_mode=update_mode,
        confirmation_batch=confirmation_batch,
        training_seed_lineage=training_seed_lineage,
        checkpoint_hash=checkpoint_hash,
    )


def record(
    step: int,
    error: float,
    *,
    bounced: bool = False,
    changed: bool = False,
    predictors: Sequence[str] = ("p",),
    actual: float = 0.4,
    width: float = 1.0,
    walls: tuple[str, ...] = (),
    ident: TrialIdentity | None = None,
    errors: dict[str, float] | None = None,
) -> StepRecord:
    """One synthetic scored transition with controllable per-predictor error.

    ``error`` is the **raw** signed error in world units; the normalized field
    is derived from it exactly as :mod:`aaa.experiment` does.
    """

    values = errors or {name: error for name in predictors}
    return StepRecord(
        identity=ident or identity(),
        step=step,
        target_step=step + 1,
        history=(0.1, 0.2, 0.3, 0.4),
        current_observation=0.4,
        actual_next_position=actual,
        bounced=bounced or bool(walls),
        changed=changed,
        bounce_walls=walls or (("upper",) if bounced else ()),
        predictions={
            name: {
                "raw": actual + value,
                "scored": actual + value,
                "absolute_error": abs(value),
                "normalized_absolute_error": abs(value) / width,
                "signed_error": value,
            }
            for name, value in values.items()
        },
        updates_enabled={name: False for name in values},
        interval_width=width,
    )
