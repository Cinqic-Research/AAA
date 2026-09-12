"""Artifacts, provenance, the experiment registry and evidence recomputation."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..experiment import StepRecord, read_step_records

REGISTRY_SCHEMA = "aaa.experiment_registry.v1"
MANIFEST_SCHEMA = "aaa.benchmark_manifest.v1"

TRIAL_STATES = ("PLANNED", "RUNNING", "COMPLETE", "FAILED", "INTERRUPTED", "SUPERSEDED")


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl_gz(path: Path, records: Sequence[StepRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle,
    ):
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return sha256_file(path)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def tree_hash(project_root: Path) -> str | None:
    """Content hash of every tracked file, independent of git object ids."""

    try:
        names = subprocess.run(
            ["git", "ls-files", "-z"], cwd=project_root, check=True, capture_output=True
        ).stdout.split(b"\0")
        digest = hashlib.sha256()
        for raw_name in sorted(name for name in names if name):
            path = project_root / raw_name.decode()
            if not path.is_file():
                continue
            digest.update(raw_name)
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_metadata(project_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=project_root, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain") or ""
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "tree_hash": tree_hash(project_root),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def dependency_lock(project_root: Path) -> dict[str, Any]:
    lock = project_root / "requirements-lock.txt"
    if not lock.exists():
        return {"path": None, "hash": None, "present": False}
    return {"path": "requirements-lock.txt", "hash": sha256_file(lock), "present": True}


def hardware_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
    }
    try:
        model = [
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            if line.startswith("model name")
        ]
        result["cpu_model"] = model[0] if model else None
    except OSError:
        result["cpu_model"] = None
    try:
        import psutil

        result["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        result["memory_bytes"] = None
    try:
        result["blas"] = {
            item.get("name"): item.get("version")
            for item in np.__config__.CONFIG.get("Build Dependencies", {}).values()
            if isinstance(item, dict)
        }
    except Exception:
        result["blas"] = None
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["gpu"] = completed.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        result["gpu"] = None
    try:
        usage = os.statvfs(".")
        result["disk_available_bytes"] = usage.f_bavail * usage.f_frsize
    except OSError:
        result["disk_available_bytes"] = None
    return result


# ---------------------------------------------------------------------------
# experiment registry
# ---------------------------------------------------------------------------


class RegistryError(RuntimeError):
    """Raised when the experiment registry would lose or contradict evidence."""


@dataclass
class TrialRecord:
    trial_id: str
    family: str
    branch: str
    replica: int
    episode: int
    environment_seed: int
    state: str = "PLANNED"
    outputs: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)
    checkpoint_hash: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    failure_stage: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "family": self.family,
            "branch": self.branch,
            "replica": self.replica,
            "episode": self.episode,
            "environment_seed": self.environment_seed,
            "state": self.state,
            "outputs": list(self.outputs),
            "checksums": dict(self.checksums),
            "checkpoint_hash": self.checkpoint_hash,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "failure_stage": self.failure_stage,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrialRecord:
        state = str(value["state"])
        if state not in TRIAL_STATES:
            raise ValueError(f"trial state must be one of {TRIAL_STATES}, got {state!r}")
        return cls(
            trial_id=str(value["trial_id"]),
            family=str(value["family"]),
            branch=str(value["branch"]),
            replica=int(value["replica"]),
            episode=int(value["episode"]),
            environment_seed=int(value["environment_seed"]),
            state=state,
            outputs=[str(item) for item in value.get("outputs", [])],
            checksums={str(k): str(v) for k, v in value.get("checksums", {}).items()},
            checkpoint_hash=None if value.get("checkpoint_hash") is None else str(value["checkpoint_hash"]),
            started_at=value.get("started_at"),
            finished_at=value.get("finished_at"),
            failure_stage=value.get("failure_stage"),
            error=value.get("error"),
        )


class ExperimentRegistry:
    """Durable per-trial state so a run can resume without losing evidence."""

    def __init__(self, path: str | Path, trials: Sequence[TrialRecord] | None = None) -> None:
        self.path = Path(path)
        self.trials: dict[str, TrialRecord] = {trial.trial_id: trial for trial in (trials or ())}

    @classmethod
    def load(cls, path: str | Path) -> ExperimentRegistry:
        source = Path(path)
        if not source.exists():
            return cls(source)
        value = json.loads(source.read_text(encoding="utf-8"))
        if value.get("schema_version") != REGISTRY_SCHEMA:
            raise ValueError(f"unsupported experiment registry schema {value.get('schema_version')!r}")
        return cls(source, [TrialRecord.from_dict(item) for item in value.get("trials", [])])

    def save(self) -> None:
        json_dump(
            self.path,
            {
                "schema_version": REGISTRY_SCHEMA,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "trials": [
                    trial.to_dict() for trial in sorted(self.trials.values(), key=lambda t: t.trial_id)
                ],
                "counts": self.counts(),
            },
        )

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(TRIAL_STATES, 0)
        for trial in self.trials.values():
            counts[trial.state] += 1
        return counts

    def plan(self, trial: TrialRecord) -> TrialRecord:
        """Register a trial, or return the existing record for a resume.

        A resume must reuse the recorded identity exactly. If the immutable
        part of a planned trial has changed, the run is not a resume of this
        experiment and continuing would silently mix two different designs.
        """

        existing = self.trials.get(trial.trial_id)
        if existing is None:
            self.trials[trial.trial_id] = trial
            return trial
        drifted = {
            name: (getattr(existing, name), getattr(trial, name))
            for name in ("family", "branch", "replica", "episode", "environment_seed")
            if getattr(existing, name) != getattr(trial, name)
        }
        if drifted:
            raise RegistryError(
                f"trial {trial.trial_id!r} is already registered with a different identity: {drifted}"
            )
        return existing

    def _trial(self, trial_id: str) -> TrialRecord:
        try:
            return self.trials[trial_id]
        except KeyError:
            raise RegistryError(f"trial {trial_id!r} was never planned") from None

    def start(self, trial_id: str) -> TrialRecord:
        trial = self._trial(trial_id)
        if trial.state == "COMPLETE":
            return trial
        trial.state = "RUNNING"
        trial.started_at = datetime.now(timezone.utc).isoformat()
        trial.failure_stage = None
        trial.error = None
        return trial

    def complete(self, trial_id: str, *, outputs: Iterable[str], checksums: Mapping[str, str]) -> TrialRecord:
        trial = self._trial(trial_id)
        trial.state = "COMPLETE"
        trial.outputs = list(outputs)
        trial.checksums = dict(checksums)
        trial.finished_at = datetime.now(timezone.utc).isoformat()
        return trial

    def fail(self, trial_id: str, *, stage: str, error: str) -> TrialRecord:
        trial = self._trial(trial_id)
        trial.state = "FAILED"
        trial.failure_stage = stage
        trial.error = error
        trial.finished_at = datetime.now(timezone.utc).isoformat()
        return trial

    def mark_interrupted(self) -> int:
        """Any trial still RUNNING when a run ends was interrupted."""

        count = 0
        for trial in self.trials.values():
            if trial.state == "RUNNING":
                trial.state = "INTERRUPTED"
                count += 1
        return count

    def is_complete(self, trial_id: str) -> bool:
        trial = self.trials.get(trial_id)
        return trial is not None and trial.state == "COMPLETE"

    def incomplete(self) -> list[TrialRecord]:
        return [trial for trial in self.trials.values() if trial.state != "COMPLETE"]


# ---------------------------------------------------------------------------
# correctness verification over retained evidence
# ---------------------------------------------------------------------------


@dataclass
class CorrectnessReport:
    executed: bool = False
    checks: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"executed": self.executed, "checks": self.checks, "detail": self.detail}


def verify_records(records: Sequence[StepRecord], *, expected_predictors: Sequence[str]) -> dict[str, Any]:
    """Structural verification of one trial's retained records."""

    problems: list[str] = []
    if not records:
        return {"ok": False, "problems": ["no records"], "transitions": 0}
    steps = [record.step for record in records]
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        problems.append("step indices are not a strictly increasing sequence")
    for record in records:
        if record.target_step != record.step + 1:
            problems.append(f"target_step {record.target_step} does not follow step {record.step}")
            break
    missing = sorted(set(expected_predictors) - set(records[0].predictions))
    if missing:
        problems.append(f"missing predictors {missing}")
    for record in records:
        for name, prediction in record.predictions.items():
            for key in ("raw", "scored", "absolute_error", "normalized_absolute_error"):
                value = prediction.get(key)
                if value is None or not np.isfinite(value):
                    problems.append(f"non-finite {key} for {name} at step {record.step}")
                    break
        if record.interval_width <= 0:
            problems.append(f"non-positive interval width at step {record.step}")
        if record.bounced != bool(record.bounce_walls):
            problems.append(f"bounce flag disagrees with wall metadata at step {record.step}")
        if len(record.history) < 2:
            problems.append(f"history too short at step {record.step}")
        if problems:
            break
    return {"ok": not problems, "problems": problems, "transitions": len(records)}


def verify_checksums(run_dir: Path) -> dict[str, Any]:
    checksum_path = run_dir / "checksums.json"
    if not checksum_path.exists():
        return {"ok": False, "reason": "checksums.json is missing"}
    stored = json.loads(checksum_path.read_text(encoding="utf-8"))
    mismatched: list[str] = []
    missing: list[str] = []
    unsafe: list[str] = []
    for relative, digest in stored.items():
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            unsafe.append(str(relative))
            continue
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
            unsafe.append(relative)
            continue
        target = run_dir / candidate
        try:
            target.resolve(strict=False).relative_to(run_dir.resolve())
        except ValueError:
            unsafe.append(relative)
            continue
        if target.is_symlink():
            unsafe.append(relative)
            continue
        if not target.exists():
            missing.append(relative)
            continue
        if sha256_file(target) != digest:
            mismatched.append(relative)
    return {
        "ok": not mismatched and not missing and not unsafe,
        "files": len(stored),
        "missing": missing,
        "mismatched": mismatched,
        "unsafe": unsafe,
    }


def iter_raw_records(run_dir: Path) -> Iterable[tuple[Path, list[StepRecord]]]:
    for path in sorted((run_dir / "raw").rglob("*.jsonl.gz")):
        yield path, read_step_records(path)
