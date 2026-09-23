"""``python -m research.aaa_python``: the ``aaa.python.v0`` command line.

    spec-hash           the packaged protocol's canonical hash
    fingerprint         the phase's scientific source identity (checkout only)
    subset              the frozen safe subset and sandbox limits
    generate            a reproducibility digest of generated tasks
    golden              check (or, once, write) the cross-version golden answer keys
    safety              validator refusals and sandbox containment
    leakage             causal-boundary probes
    audit               learner parameter and state accounting
    develop             a development run (never confirmation)
    recompute           independent recomputation of a development run
    confirm             refused: v0 admits no confirmation

Exit codes: 0 success, 1 a check failed, 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from . import GENERATOR_VERSION, PROTOCOL_VERSION, checks, generator
from . import spec as spec_module

GOLDEN_RESOURCE = "golden_answers_v0.json"
GOLDEN_SCHEMA = "aaa.python.v0.golden_answers.v1"
GOLDEN_COUNT = 20


def _golden_payload() -> dict[str, Any]:
    spec = spec_module.load()
    keys: dict[str, dict[str, Any]] = {}
    for family in spec["families"]:
        for task in generator.build("development", family, range(GOLDEN_COUNT)):
            keys[task.task_id] = {
                "source_sha256": task.source_sha256,
                "answer": task.answer,
                "attempt": task.attempt,
            }
    return {
        "schema": GOLDEN_SCHEMA,
        "protocol": PROTOCOL_VERSION,
        "generator": GENERATOR_VERSION,
        "keys": keys,
    }


def golden_path() -> Path:
    return Path(str(resources.files("research.aaa_python").joinpath("data", GOLDEN_RESOURCE)))


def golden_differences() -> list[str]:
    expected = json.loads(golden_path().read_text(encoding="utf-8"))
    live = _golden_payload()
    problems = []
    if expected.get("schema") != GOLDEN_SCHEMA or set(expected["keys"]) != set(live["keys"]):
        return ["golden key set or schema differs"]
    for identity, record in expected["keys"].items():
        if live["keys"][identity] != record:
            problems.append(
                f"{identity}: expected {record}, this interpreter produced {live['keys'][identity]}"
            )
    return problems


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _report(results: Sequence[tuple[str, bool, str]], label: str) -> int:
    for name, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not passed else ""))
    failures = checks.failed(list(results))
    print(f"{label}: {len(results) - len(failures)}/{len(results)} passed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m research.aaa_python", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("spec-hash")
    sub.add_parser("fingerprint")
    sub.add_parser("subset")
    generate = sub.add_parser("generate")
    generate.add_argument(
        "--split", default="development", choices=["train", "development", "probe", "attack", "confirmation"]
    )
    generate.add_argument("--family", choices=list(spec_module.FAMILIES), default=None)
    generate.add_argument("--count", type=int, default=20)
    generate.add_argument("--show", action="store_true")
    golden = sub.add_parser("golden")
    golden.add_argument("--write", action="store_true", help="write the golden keys (refuses to overwrite)")
    sub.add_parser("safety")
    sub.add_parser("leakage")
    sub.add_parser("audit")
    develop = sub.add_parser("develop")
    develop.add_argument("--output", type=Path, required=True)
    develop.add_argument("--records", type=Path, required=True, help="gzip JSON-lines per-action records")
    develop.add_argument(
        "--checkpoints", type=Path, default=None, help="directory for resumable trained states"
    )
    develop.add_argument(
        "--quick", action="store_true", help="a reduced smoke plan, labelled quick in the evidence"
    )
    develop.add_argument(
        "--device", default="cpu", help="only 'cpu' is accepted; nothing falls back silently"
    )
    recompute = sub.add_parser("recompute")
    recompute.add_argument("--evidence", type=Path, required=True)
    recompute.add_argument("--records", type=Path, default=None)
    sub.add_parser("confirm")
    args = parser.parse_args(argv)

    if args.command == "spec-hash":
        print(f"protocol: {PROTOCOL_VERSION}")
        print(f"hash: {spec_module.spec_hash()}")
        return 0
    if args.command == "fingerprint":
        from .identity import IdentityError, fingerprint

        try:
            identity = fingerprint()
        except IdentityError as error:
            print(f"refused: {error} (an installed package has no source identity to hash)", file=sys.stderr)
            return 2
        print(identity["sha256"])
        print(f"{identity['file_count']} files, phase {PROTOCOL_VERSION}")
        return 0
    if args.command == "subset":
        print(json.dumps({k: spec_module.load()[k] for k in ("subset", "sandbox")}, indent=1))
        return 0
    if args.command == "generate":
        if args.count < 1:
            print("--count must be positive", file=sys.stderr)
            return 2
        families = [args.family] if args.family else list(spec_module.FAMILIES)
        digest = hashlib.sha256()
        try:
            for family in families:
                for task in generator.build(args.split, family, range(args.count)):
                    digest.update(f"{task.task_id}|{task.source_sha256}|{task.answer!r}\n".encode())
                    if args.show:
                        print(
                            f"# {task.task_id} ({task.slice}, {task.template}) -> {task.answer!r}\n{task.source}"
                        )
        except generator.ConfirmationNotAdmitted as error:
            print(f"refused: {error}", file=sys.stderr)
            return 2
        print(f"{args.split} {families} x {args.count}: {digest.hexdigest()}")
        return 0
    if args.command == "golden":
        if args.write:
            if golden_path().exists():
                print(
                    "refusing: golden keys exist; a change is a recorded generator-version decision",
                    file=sys.stderr,
                )
                return 2
            _write_json(golden_path(), _golden_payload())
            print(f"wrote {golden_path()}")
            return 0
        problems = golden_differences()
        for problem in problems:
            print(f"MISMATCH {problem}", file=sys.stderr)
        print(
            f"golden answer keys on CPython {sys.version.split()[0]}: {'MISMATCH' if problems else 'IDENTICAL'}"
        )
        return 1 if problems else 0
    if args.command == "safety":
        return _report(checks.safety_checks(), "safety")
    if args.command == "leakage":
        return _report(checks.leakage_checks(), "leakage")
    if args.command == "audit":
        from .experiment import make_learner

        spec = spec_module.load()
        for representation in spec["learner"]["representations"]:
            counts = make_learner(spec, 0, representation).parameter_count()
            print(f"{representation}: {json.dumps(counts, sort_keys=True)}")
        return 0
    if args.command == "develop":
        from .experiment import develop as run_develop
        from .experiment import write_records
        from .identity import provenance

        if args.device != "cpu":
            print(f"refused: device {args.device!r}; aaa.python.v0 runs on the CPU only", file=sys.stderr)
            return 2
        captured = provenance(args.device)  # before anything is observed (AAA-179)
        started = time.time()
        evidence, records = run_develop(quick=args.quick, provenance=captured, checkpoints=args.checkpoints)
        evidence["provenance"]["wall_seconds"] = round(time.time() - started, 1)
        write_records(args.records, records)
        _write_json(args.output, evidence)
        print(f"wrote {args.output} ({len(records)} records to {args.records})")
        return 0
    if args.command == "recompute":
        from .experiment import read_records
        from .recompute import load_evidence, verify

        evidence = load_evidence(args.evidence.read_text(encoding="utf-8"))
        loaded = read_records(args.records) if args.records else None
        result = verify(evidence, loaded)
        print(json.dumps(result, indent=1))
        return 0 if result["verdict"] == "PASS" else 1
    if args.command == "confirm":
        print(
            "refused: aaa.python.v0 declares no candidate, promotion criteria, thresholds or confirmation "
            "identities. Formal confirmation is NOT EXECUTED; that is the correct state for this phase.",
            file=sys.stderr,
        )
        return 2
    return 2
