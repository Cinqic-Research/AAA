"""AAA benchmark v2.1: specification, protocol, statistics, gates and evidence.

The historical ``aaa.benchmark.v2`` protocol is superseded. Its confirmations
are retained as historical/provisional evidence under ``results/benchmark_v2``
and are no longer acceptance evidence; see ``docs/errata.md``.
"""

from __future__ import annotations

from .gates import FAIL, INSUFFICIENT_EVIDENCE, NOT_VERIFIED, PASS, GateResult, evaluate_gates
from .manifest import FreezeMismatch, build_manifest, check_manifest, load_manifest, save_manifest
from .recompute import recompute_run
from .runner import ConfirmationError, RunOutcome, run_benchmark
from .seeds import (
    BatchRegistryError,
    ConfirmationBatch,
    ConfirmationBatchRegistry,
    training_seed,
    trial_seed,
)
from .spec import SPEC_VERSION, BenchmarkSpec, SpecError, canonical_spec_hash, load_spec, spec_hash

__all__ = [
    "FAIL",
    "INSUFFICIENT_EVIDENCE",
    "NOT_VERIFIED",
    "PASS",
    "SPEC_VERSION",
    "BatchRegistryError",
    "BenchmarkSpec",
    "ConfirmationBatch",
    "ConfirmationBatchRegistry",
    "ConfirmationError",
    "FreezeMismatch",
    "GateResult",
    "RunOutcome",
    "SpecError",
    "build_manifest",
    "canonical_spec_hash",
    "check_manifest",
    "evaluate_gates",
    "load_manifest",
    "load_spec",
    "recompute_run",
    "run_benchmark",
    "save_manifest",
    "spec_hash",
    "training_seed",
    "trial_seed",
]
