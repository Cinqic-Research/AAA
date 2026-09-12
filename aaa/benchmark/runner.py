"""Benchmark v2.1 orchestration.

The runner — not only the CLI — enforces confirmation invariants. A Python
caller cannot weaken confirmation minimums, reuse a spent confirmation batch,
run confirmation from a dirty tree, or skip the freeze manifest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..experiment import TrialIdentity, run_episode
from ..predictors import OnlineRLSPredictor
from .evidence import (
    ExperimentRegistry,
    TrialRecord,
    dependency_lock,
    git_metadata,
    hardware_metadata,
    json_dump,
    sha256_file,
    sha256_text,
    verify_checksums,
    verify_records,
    write_jsonl_gz,
)
from .families import (
    CHANGED_LAW_PREDICTOR_NAMES,
    MOTION_PREDICTOR_NAMES,
    ONLINE_PREDICTOR_NAMES,
    EpisodePlan,
    build_environment,
    clone_candidate,
    make_candidate,
    plan_motion_family,
    run_always_online_family,
    run_changed_law_family,
    run_motion_family,
    training_world,
)
from .gates import GateContext, evaluate_gates
from .manifest import check_manifest, load_manifest
from .recompute import compare_results, rebuild_collectors, recovery_config
from .report import write_report
from .seeds import (
    CONFIRMATION_ROLES,
    DEFAULT_GOLDEN_CASES,
    ROLES,
    ConfirmationBatchRegistry,
    golden_seed_fixture,
    lineage_for,
    purpose_for,
)
from .spec import BenchmarkSpec, load_spec, spec_hash
from .training import TrainedReplica, measure_learning_curve, train_replica

SUMMARY_SCHEMA = "aaa.benchmark_summary.v2.1"
MOTION_FAMILY_ORDER = ("constant_velocity", "bouncing", "speed_change", "speed_extrapolation")


class ConfirmationError(RuntimeError):
    """Raised when formal confirmation discipline would be violated."""


@dataclass
class RunOutcome:
    directory: Path
    summary: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.summary["gates"]["all_required_gates_pass"])

    @property
    def unmet(self) -> list[str]:
        return list(self.summary["gates"]["unmet_required_gates"])


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def train_replicas(spec: BenchmarkSpec, *, lineage: str, replicas: int, role: str) -> list[TrainedReplica]:
    return [train_replica(spec, lineage, replica, role=role) for replica in range(replicas)]


def state_fingerprint(state: dict[str, Any]) -> str:
    """Canonical text for a learner state, excluding the mutable role label.

    ``name`` identifies the role a copy plays in an experiment (``frozen``,
    ``online``, ``candidate_frozen``), not the model. Two copies with identical
    numerics must hash identically whatever they are called.
    """

    comparable = {key: value for key, value in state.items() if key != "name"}
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def checkpoint_digest(state: dict[str, Any]) -> str:
    return sha256_text(state_fingerprint(state))


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------


def measure_latency(spec: BenchmarkSpec, model: OnlineRLSPredictor) -> dict[str, Any]:
    """Latency of the *selected* candidate, on the recorded CPU, no I/O."""

    probe = clone_candidate(model, name="latency_probe", update_enabled=True)
    rng = np.random.default_rng(spec.randomness.bootstrap_seed)
    inputs: list[tuple[tuple[float, ...], float]] = []
    for _ in range(spec.latency.samples + spec.latency.warmup_samples):
        position = float(rng.uniform(spec.world.lower_bound + 0.1, spec.world.upper_bound - 0.1))
        velocity = float(rng.uniform(-0.2, 0.2)) * spec.world.dt
        history = tuple(position + velocity * offset for offset in (-3, -2, -1, 0))
        inputs.append((history, position + velocity))
    predict_ns: list[int] = []
    update_ns: list[int] = []
    combined_ns: list[int] = []
    failures = 0
    for index, (history, target) in enumerate(inputs):
        warm = index < spec.latency.warmup_samples
        try:
            start = time.perf_counter_ns()
            probe.predict(history)
            middle = time.perf_counter_ns()
            probe.update(history, target)
            end = time.perf_counter_ns()
        except Exception:
            failures += 1
            continue
        if warm:
            continue
        predict_ns.append(middle - start)
        update_ns.append(end - middle)
        combined_ns.append(end - start)

    def summarize(values: list[int], label: str) -> dict[str, float]:
        array = np.asarray(values, dtype=float) / 1_000_000.0
        return {
            f"{label}_p50_ms": float(np.percentile(array, 50)),
            f"{label}_p95_ms": float(np.percentile(array, 95)),
            f"{label}_p99_ms": float(np.percentile(array, 99)),
        }

    result: dict[str, Any] = {
        "samples": len(combined_ns),
        "warmup_samples": spec.latency.warmup_samples,
        "failures": failures,
        "candidate_forgetting": probe.forgetting,
        "candidate_update_count": probe.update_count,
        "hardware": hardware_metadata(),
    }
    result.update(summarize(predict_ns, "predict"))
    result.update(summarize(update_ns, "update"))
    result.update(summarize(combined_ns, "predict_plus_update"))
    return result


def deterministic_rerun_check(spec: BenchmarkSpec) -> dict[str, Any]:
    """Run a duplicate mini experiment twice from independent objects."""

    def once() -> list[dict[str, Any]]:
        trained = train_replica(spec, "reproducibility_probe", 0, role="development")
        plans = plan_motion_family(spec, "constant_velocity", "reproducibility_probe", replicas=1, episodes=2)
        rows: list[dict[str, Any]] = []
        for plan in plans:
            model = clone_candidate(trained.model, name="candidate_frozen", update_enabled=False)
            identity = TrialIdentity(
                trial_id=plan.trial_id,
                role="development",
                family="constant_velocity",
                scenario=plan.scenario,
                environment_seed=plan.environment_seed,
                replica_id=plan.replica,
                episode=plan.episode,
                stratum=plan.stratum,
                update_mode="frozen",
            )
            records = run_episode(build_environment(plan), [model], identity, learn=False)
            rows.extend(record.to_dict() for record in records)
        rows.append({"final_state": trained.model.state_dict()})
        return rows

    first = once()
    second = once()
    identical = json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    return {"identical": identical, "transitions": len(first) - 1}


def save_resume_check(spec: BenchmarkSpec, tmp_dir: Path) -> dict[str, Any]:
    """Uninterrupted execution versus checkpoint, reload and resume."""

    world = training_world(spec)
    rng = np.random.default_rng(spec.randomness.root_seed + 5)
    stream = []
    position = 0.5
    for _ in range(120):
        velocity = float(rng.uniform(-0.004, 0.004))
        history = (position - 3 * velocity, position - 2 * velocity, position - velocity, position)
        stream.append((history, position + velocity))
        position = min(max(position + velocity, world.lower_bound + 0.05), world.upper_bound - 0.05)

    straight = make_candidate(spec, name="uninterrupted", update_enabled=True)
    for history, target in stream:
        straight.update(history, target)

    interrupted = make_candidate(spec, name="interrupted", update_enabled=True)
    for history, target in stream[:60]:
        interrupted.update(history, target)
    checkpoint = tmp_dir / "resume_probe.json"
    interrupted.save(checkpoint)
    resumed = OnlineRLSPredictor.load(checkpoint, name="interrupted", update_enabled=True)
    for history, target in stream[60:]:
        resumed.update(history, target)

    left = state_fingerprint(straight.state_dict())
    right = state_fingerprint(resumed.state_dict())
    return {
        "identical": left == right,
        "updates": len(stream),
        "max_weight_difference": float(np.max(np.abs(straight.weights - resumed.weights))),
    }


def golden_seed_check(spec: BenchmarkSpec, project_root: Path) -> dict[str, Any]:
    """Compare the seed mapping against a committed fixture."""

    fixture_path = project_root / "benchmarks" / "golden_seeds.json"
    computed = golden_seed_fixture(spec.randomness.root_seed, DEFAULT_GOLDEN_CASES)
    if not fixture_path.exists():
        return {"matches": False, "reason": "benchmarks/golden_seeds.json is missing", "cases": len(computed)}
    stored = json.loads(fixture_path.read_text(encoding="utf-8"))
    if stored.get("root_seed") != spec.randomness.root_seed:
        return {"matches": False, "reason": "fixture root seed differs from the specification"}
    mismatches = {
        key: [stored["seeds"].get(key), value]
        for key, value in computed.items()
        if stored["seeds"].get(key) != value
    }
    return {"matches": not mismatches, "cases": len(computed), "mismatches": mismatches}


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def run_benchmark(
    *,
    role: str = "development",
    batch_id: str | None = None,
    output_root: str | Path = "runs",
    attempt_label: str | None = None,
    replicas: int | None = None,
    episodes: int | None = None,
    spec_path: str | Path | None = None,
    project_root: str | Path | None = None,
    reproduce: bool = False,
    resume: bool = False,
) -> RunOutcome:
    """Execute one benchmark attempt and return its outcome."""

    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    # Validate before defaulting: ``replicas or default`` would silently turn
    # an explicit 0 into the full confirmation budget.
    for name, value in (("replicas", replicas), ("episodes", episodes)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    project = Path(project_root or default_project_root())
    spec = load_spec(spec_path)
    resolved_spec_hash = spec_hash(spec)
    is_confirmation = role in CONFIRMATION_ROLES

    # ---- confirmation invariants, enforced in the core runner -----------
    if is_confirmation:
        if project.resolve() != default_project_root().resolve():
            raise ConfirmationError("confirmation source root must contain the loaded AAA implementation")
        if spec_path is not None and Path(spec_path).resolve() != _canonical_path().resolve():
            raise ConfirmationError(
                "confirmation must use the canonical committed specification; custom specifications are "
                "development experiments only"
            )
        if spec.status != "active":
            raise ConfirmationError(
                f"confirmation requires an active specification, got status {spec.status!r}"
            )
        if not batch_id:
            raise ConfirmationError("confirmation requires a predeclared confirmation batch id")
        if replicas is not None and replicas < spec.confirmation.replicas:
            raise ConfirmationError(f"confirmation requires at least {spec.confirmation.replicas} replicas")
        if episodes is not None and episodes < spec.confirmation.episodes_per_family:
            raise ConfirmationError(
                f"confirmation requires at least {spec.confirmation.episodes_per_family} episodes per family"
            )
        git = git_metadata(project)
        if not git["commit"]:
            raise ConfirmationError("confirmation requires a committed Git source tree")
        if spec.confirmation.require_clean_source_tree and git["dirty"]:
            raise ConfirmationError(
                "confirmation requires a clean source tree; commit or stash before running.\n"
                + "\n".join(f"  {line}" for line in git["status"][:20])
            )

    replica_count = (
        replicas
        if replicas is not None
        else (
            spec.confirmation.high_replication_replicas
            if role == "high_replication"
            else spec.confirmation.replicas
        )
    )
    episode_count = episodes if episodes is not None else spec.confirmation.episodes_per_family

    purpose = purpose_for(role, batch_id)
    lineage = lineage_for(role, spec.confirmation.ab_relationship)
    recovery = recovery_config(spec)

    batch_registry: ConfirmationBatchRegistry | None = None
    confirmation_attempts = 0
    if is_confirmation:
        batch_registry = ConfirmationBatchRegistry.load(project / spec.confirmation.batch_registry)
        batch_registry.claim(batch_id or "", role, resolved_spec_hash, reproduction=reproduce)
        if reproduce:
            confirmation_attempts = _recorded_confirmation_attempts(
                project, str(batch_id), role=role, resolved_spec_hash=resolved_spec_hash
            )
        else:
            confirmation_attempts = 1 + sum(
                1
                for batch in batch_registry.batches()
                if batch.spec_hash == resolved_spec_hash and batch.status in ("consumed", "retired")
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # A confirmation batch id is guaranteed non-empty by the invariants above,
    # and a confirmation attempt directory *is* its batch identity.
    attempt = str(batch_id) if is_confirmation else (attempt_label or f"{timestamp}-{role}")
    directory = Path(output_root) / "benchmark-v2_1" / attempt
    if directory.exists() and not resume:
        raise FileExistsError(
            f"attempt directory already exists: {directory}. Use resume=True to continue it, or choose a "
            "fresh immutable attempt identity."
        )
    directory.mkdir(parents=True, exist_ok=True)

    if is_confirmation and batch_registry is not None and not reproduce:
        batch_registry.reserve(str(batch_id), role, resolved_spec_hash, run_id=attempt, resume=resume)

    registry = ExperimentRegistry.load(directory / "experiment_registry.json")
    started = time.perf_counter()

    json_dump(directory / "benchmark_spec.json", spec.to_dict())
    metadata = {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": attempt,
        "role": role,
        "confirmation_batch": batch_id,
        "purpose": purpose,
        "training_lineage": lineage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "aaa_version": __version__,
        "spec_hash": resolved_spec_hash,
        "git": git_metadata(project),
        "dependency_lock": dependency_lock(project),
        "hardware": hardware_metadata(),
        "parallelism": "none (serial; profiled and not worth parallelizing at this size)",
    }
    json_dump(directory / "metadata.json", metadata)

    # ---- training -------------------------------------------------------
    trained = train_replicas(spec, lineage=lineage, replicas=replica_count, role=role)
    checkpoint_hashes: list[str] = []
    training_seeds_by_replica: dict[int, list[int]] = {}
    for item in trained:
        state = item.model.state_dict()
        digest = checkpoint_digest(state)
        checkpoint_hashes.append(digest)
        training_seeds_by_replica[item.replica] = list(item.seeds)
        json_dump(directory / "checkpoints" / f"replica-{item.replica:02d}.json", state)
        for budget, snapshot in sorted(item.checkpoints.items()):
            json_dump(
                directory
                / "checkpoints"
                / "budgets"
                / f"replica-{item.replica:02d}-budget-{budget:04d}.json",
                snapshot,
            )

    if is_confirmation:
        manifest = load_manifest(project / spec.confirmation.freeze_manifest)
        check_manifest(
            manifest,
            spec,
            project_root=project,
            checkpoint_hashes=checkpoint_hashes,
            batch_id=batch_id or "",
            training_seeds=training_seeds_by_replica,
        )

    models = [item.model for item in trained]
    legacy_states = [item.legacy_state for item in trained]

    # ---- evidence sink --------------------------------------------------
    def start_sink(plan: EpisodePlan) -> None:
        trial_id = plan.trial_id
        registry.plan(
            TrialRecord(
                trial_id=trial_id,
                family=plan.family,
                branch=plan.branch,
                replica=plan.replica,
                episode=plan.episode,
                environment_seed=plan.environment_seed,
                checkpoint_hash=checkpoint_hashes[plan.replica],
            )
        )
        registry.start(trial_id)
        registry.save()

    def sink(plan: EpisodePlan, records) -> None:
        trial_id = records[0].identity.trial_id
        relative = (
            Path("raw")
            / plan.family
            / plan.branch
            / f"replica-{plan.replica:02d}"
            / f"episode-{plan.episode:04d}.jsonl.gz"
        )
        digest = write_jsonl_gz(directory / relative, records)
        registry.complete(trial_id, outputs=[str(relative)], checksums={str(relative): digest})
        registry.save()

    # ---- families -------------------------------------------------------
    collectors: dict[str, Any] = {}
    for family_name in MOTION_FAMILY_ORDER:
        plans = plan_motion_family(spec, family_name, purpose, replicas=replica_count, episodes=episode_count)
        collectors[family_name] = run_motion_family(
            spec,
            family_name,
            plans,
            models,
            role=role,
            batch_id=batch_id,
            training_lineage=training_seeds_by_replica,
            checkpoint_hashes=checkpoint_hashes,
            recovery=recovery,
            legacy=legacy_states,
            start_sink=start_sink,
            sink=sink,
        )
    always_plans = plan_motion_family(
        spec, "always_online", purpose, replicas=replica_count, episodes=episode_count
    )
    collectors["always_online"] = run_always_online_family(
        spec,
        always_plans,
        models,
        role=role,
        batch_id=batch_id,
        training_lineage=training_seeds_by_replica,
        checkpoint_hashes=checkpoint_hashes,
        recovery=recovery,
        legacy=legacy_states,
        start_sink=start_sink,
        sink=sink,
    )

    changed_law = run_changed_law_family(
        spec,
        models,
        purpose=purpose,
        role=role,
        batch_id=batch_id,
        training_lineage=training_seeds_by_replica,
        checkpoint_hashes=checkpoint_hashes,
        recovery=recovery,
        replicas=replica_count,
        episodes=episode_count,
        start_sink=start_sink,
        sink=sink,
    )
    collectors["changed_law:changed"] = changed_law.changed
    collectors["changed_law:unchanged"] = changed_law.unchanged
    json_dump(directory / "metrics" / "changed_law_interventions.json", changed_law.interventions)

    results = {key: collector.finish() for key, collector in collectors.items()}
    for key, value in results.items():
        json_dump(directory / "metrics" / f"{key.replace(':', '_')}.json", value)

    # ---- learning curve -------------------------------------------------
    learning_curve = measure_learning_curve(spec, trained, role=role)
    json_dump(directory / "metrics" / "learning_curve.json", learning_curve)

    # ---- latency --------------------------------------------------------
    latency = measure_latency(spec, models[0])
    json_dump(directory / "metrics" / "latency.json", latency)

    # ---- reproducibility ------------------------------------------------
    probe_dir = directory / "verification"
    probe_dir.mkdir(parents=True, exist_ok=True)
    rerun = deterministic_rerun_check(spec)
    resume_check = save_resume_check(spec, probe_dir)
    golden = golden_seed_check(spec, project)
    reproducibility = {
        "executed": True,
        "checks": {
            "deterministic_rerun_identical": bool(rerun["identical"]),
            "save_resume_identical": bool(resume_check["identical"]),
            "golden_seed_mapping_matches": bool(golden["matches"]),
        },
        "detail": {"deterministic_rerun": rerun, "save_resume": resume_check, "golden_seeds": golden},
        "tolerances": {
            "absolute": spec.tolerances.reproducibility_absolute,
            "relative": spec.tolerances.reproducibility_relative,
        },
    }
    json_dump(directory / "verification" / "reproducibility.json", reproducibility)

    # ---- checksums before correctness so recomputation can verify them --
    registry.mark_interrupted()
    registry.save()
    _write_checksums(directory)

    # ---- correctness ----------------------------------------------------
    correctness = _verify_correctness(
        directory,
        spec,
        results=results,
        expected_trials=_expected_trial_count(replica_count, episode_count),
        registry=registry,
        trained=trained,
        checkpoint_hashes=checkpoint_hashes,
        resolved_spec_hash=resolved_spec_hash,
        metadata=metadata,
        is_confirmation=is_confirmation,
    )
    json_dump(directory / "verification" / "correctness.json", correctness)

    # ---- gates ----------------------------------------------------------
    context = GateContext(
        spec=spec,
        collectors=collectors,
        results=results,
        learning_curve=learning_curve,
        correctness=correctness,
        reproducibility=reproducibility,
        latency=latency,
        confirmation_attempts=confirmation_attempts,
    )
    gates = evaluate_gates(context)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "spec_version": spec.spec_version,
        "spec_hash": resolved_spec_hash,
        "run_id": attempt,
        "role": role,
        "confirmation_batch": batch_id,
        "confirmation_attempts": confirmation_attempts,
        "purpose": purpose,
        "training_lineage": lineage,
        "config": {"replicas": replica_count, "episodes_per_family": episode_count},
        "checkpoints": {
            "relationship": spec.confirmation.ab_relationship,
            "hashes": checkpoint_hashes,
            "training_seeds": {str(k): v for k, v in sorted(training_seeds_by_replica.items())},
            "diagnostics": {str(item.replica): item.diagnostics for item in trained},
        },
        "results": results,
        "learning_curve": learning_curve,
        "latency": latency,
        "correctness": correctness,
        "reproducibility": reproducibility,
        "gates": gates,
        "metadata": metadata,
        "runtime_seconds": time.perf_counter() - started,
        "evidence": {
            "metadata": "metadata.json",
            "benchmark_spec": "benchmark_spec.json",
            "checkpoints": "checkpoints/",
            "raw": "raw/",
            "metrics": "metrics/",
            "verification": "verification/",
            "registry": "experiment_registry.json",
            "checksums": "checksums.json",
            "plots": "plots/",
        },
    }
    # Plots are an auditing aid derived entirely from the summary, so a
    # rendering problem must not destroy an otherwise complete attempt.
    try:
        from ..visualization import write_benchmark_plots

        rendered = write_benchmark_plots(directory / "plots", summary)
        summary["plots"] = {key: str(Path(value).relative_to(directory)) for key, value in rendered.items()}
    except Exception as error:  # pragma: no cover - rendering environment specific
        summary["plots"] = {"error": f"{type(error).__name__}: {error}"}
    json_dump(directory / "summary.json", summary)
    write_report(directory / "report.md", summary)
    _write_checksums(directory)

    if is_confirmation and batch_registry is not None and not reproduce:
        batch_registry.record_outcome(str(batch_id), attempt, passed=bool(gates["all_required_gates_pass"]))
        batch_registry.save()

    return RunOutcome(directory=directory, summary=summary)


def _canonical_path() -> Path:
    from .spec import canonical_spec_path

    return canonical_spec_path()


def _expected_trial_count(replicas: int, episodes: int) -> int:
    motion = len(MOTION_FAMILY_ORDER) + 1  # + always_online
    changed_law = 3  # prefix, changed branch, unchanged branch
    return replicas * episodes * (motion + changed_law)


def _recorded_confirmation_attempts(
    project: Path, batch_id: str, *, role: str, resolved_spec_hash: str
) -> int:
    summary_path = project / "results" / "benchmark_v2_1" / batch_id / "summary.json"
    if not summary_path.is_file():
        raise ConfirmationError(
            "reproduction requires the committed recorded summary so its frozen statistical context is known"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("run_id") != batch_id or summary.get("confirmation_batch") != batch_id:
        raise ConfirmationError("recorded summary identity does not match the requested batch")
    if summary.get("role") != role:
        raise ConfirmationError("recorded summary role does not match the requested reproduction role")
    if summary.get("spec_hash") != resolved_spec_hash:
        raise ConfirmationError("recorded summary specification does not match the active reproduction")
    value = summary.get("confirmation_attempts")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfirmationError("recorded confirmation_attempts is missing or invalid")
    return value


def _write_checksums(directory: Path) -> None:
    json_dump(
        directory / "checksums.json",
        {
            str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "checksums.json" and not path.name.endswith(".tmp")
        },
    )


def _verify_correctness(
    directory: Path,
    spec: BenchmarkSpec,
    *,
    results: dict[str, Any],
    expected_trials: int,
    registry: ExperimentRegistry,
    trained: Sequence[TrainedReplica],
    checkpoint_hashes: Sequence[str],
    resolved_spec_hash: str,
    metadata: dict[str, Any],
    is_confirmation: bool,
) -> dict[str, Any]:
    """Verify the evidence itself, not merely that the process finished."""

    checks: dict[str, Any] = {}
    detail: dict[str, Any] = {}

    counts = registry.counts()
    detail["registry_counts"] = counts
    checks["all_trials_complete"] = counts["COMPLETE"] == expected_trials and not registry.incomplete()
    detail["expected_trials"] = expected_trials

    identifiers = [trial.trial_id for trial in registry.trials.values()]
    checks["no_duplicate_trial_ids"] = len(set(identifiers)) == len(identifiers)

    structural_problems: list[dict[str, Any]] = []
    transitions = 0
    families_seen: set[tuple[str, str]] = set()
    for path, records in _iter_raw(directory):
        if not records:
            structural_problems.append({"file": str(path), "problems": ["empty"]})
            continue
        identity = records[0].identity
        families_seen.add((identity.family, identity.branch))
        if identity.family == "changed_law":
            expected = ["prefix_model"] if identity.branch == "prefix" else list(CHANGED_LAW_PREDICTOR_NAMES)
            if identity.branch == "prefix":
                expected = ["persistence", "constant_motion", "constant_motion_reflected", "prefix_model"]
        elif "candidate_online" in records[0].predictions:
            expected = list(ONLINE_PREDICTOR_NAMES)
        else:
            expected = list(MOTION_PREDICTOR_NAMES)
        report = verify_records(records, expected_predictors=expected)
        transitions += report["transitions"]
        if not report["ok"]:
            structural_problems.append({"file": str(path), "problems": report["problems"]})
    checks["records_structurally_valid"] = not structural_problems
    detail["structural_problems"] = structural_problems[:20]
    detail["scored_transitions"] = transitions
    detail["families_present"] = sorted(f"{family}:{branch}" for family, branch in families_seen)

    checksum_report = verify_checksums(directory)
    checks["checksums_valid"] = bool(checksum_report["ok"])
    detail["checksums"] = checksum_report

    loadable = True
    frozen_intact = True
    for index, item in enumerate(trained):
        path = directory / "checkpoints" / f"replica-{item.replica:02d}.json"
        try:
            reloaded = OnlineRLSPredictor.load(path, name="verify", update_enabled=False)
            reloaded.check_state()
            if checkpoint_digest(reloaded.state_dict()) != checkpoint_hashes[index]:
                frozen_intact = False
        except Exception:
            loadable = False
    checks["checkpoints_load"] = loadable
    checks["checkpoint_hashes_stable"] = frozen_intact

    interventions_path = directory / "metrics" / "changed_law_interventions.json"
    frozen_unchanged = True
    branch_matched = True
    if interventions_path.exists():
        interventions = json.loads(interventions_path.read_text(encoding="utf-8"))
        for row in interventions:
            if row["frozen_weights_before"] != row["frozen_weights_after"]:
                frozen_unchanged = False
            if row["frozen_update_count_before"] != row["frozen_update_count_after"]:
                frozen_unchanged = False
            if row["online_update_count_after"] <= row["frozen_update_count_after"]:
                branch_matched = False
        detail["interventions_checked"] = len(interventions)
    else:
        frozen_unchanged = False
        branch_matched = False
    checks["frozen_branch_state_measured_unchanged"] = frozen_unchanged
    checks["online_branch_actually_updated"] = branch_matched

    checks["spec_hash_matches_canonical_when_confirming"] = (
        not is_confirmation
    ) or resolved_spec_hash == spec_hash(load_spec())
    checks["dependency_lock_recorded"] = bool(metadata["dependency_lock"]["hash"])
    checks["source_tree_clean_when_confirming"] = (not is_confirmation) or not metadata["git"]["dirty"]

    try:
        recomputed = {
            key: collector.finish() for key, collector in rebuild_collectors(directory, spec).items()
        }
        comparison = compare_results(
            _comparable(results), _comparable(recomputed), tolerance=spec.tolerances.recompute_absolute
        )
        checks["summary_recomputes_from_raw_evidence"] = bool(comparison["equivalent"])
        detail["recomputation"] = comparison
    except Exception as error:
        checks["summary_recomputes_from_raw_evidence"] = False
        detail["recomputation"] = {"error": f"{type(error).__name__}: {error}"}

    return {"executed": True, "checks": checks, "detail": detail}


def _comparable(results: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that legitimately differ between run order and rebuild order."""

    trimmed: dict[str, Any] = {}
    for key, value in results.items():
        copy = dict(value)
        copy.pop("stratum_replicas", None)
        trimmed[key] = copy
    return trimmed


def _iter_raw(directory: Path):
    from .evidence import iter_raw_records

    return iter_raw_records(directory)
