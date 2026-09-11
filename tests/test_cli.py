"""Process-exit contract and confirmation discipline.

The original benchmark command returned 0 whatever the gates said, and the
confirmation minimums lived only in the CLI, so any Python caller could walk
around them. Both are enforced here and in the runner itself.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from aaa import cli
from aaa.benchmark import runner as runner_module
from aaa.benchmark.gates import FAIL, INSUFFICIENT_EVIDENCE, NOT_VERIFIED, PASS
from aaa.benchmark.runner import ConfirmationError, RunOutcome, run_benchmark
from aaa.benchmark.seeds import ConfirmationBatchRegistry
from aaa.benchmark.spec import canonical_spec_hash, load_spec

PROJECT = Path(__file__).resolve().parents[1]


def invoke(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exit_error:  # argparse and explicit SystemExit
            code = int(exit_error.code or 0)
    return code, out.getvalue(), err.getvalue()


def outcome(statuses: dict[str, str], *, required: dict[str, bool] | None = None) -> RunOutcome:
    required = required or dict.fromkeys(statuses, True)
    gates = [
        {
            "name": name,
            "status": status,
            "required": required[name],
            "observed": None,
            "threshold": {},
            "description": "",
            "details": {},
        }
        for name, status in statuses.items()
    ]
    unmet = [gate["name"] for gate in gates if gate["required"] and gate["status"] != PASS]
    return RunOutcome(
        directory=Path("/tmp/aaa-fake-attempt"),
        summary={
            "gates": {"gates": gates, "all_required_gates_pass": not unmet, "unmet_required_gates": unmet}
        },
    )


class ExitCodeTests(unittest.TestCase):
    """Development may record a failure; confirmation may not."""

    def _run(self, role: str, statuses: dict[str, str], extra: list[str] | None = None) -> int:
        with mock.patch.object(cli, "run_benchmark", return_value=outcome(statuses)):
            code, _, _ = invoke(["benchmark", "--role", role, *(extra or [])])
        return code

    def test_development_returns_zero_even_with_a_failed_gate(self):
        self.assertEqual(self._run("development", {"a": PASS, "b": FAIL}), 0)

    def test_confirmation_returns_nonzero_on_a_failed_gate(self):
        self.assertEqual(self._run("confirmation_a", {"a": PASS, "b": FAIL}, ["--batch-id", "x"]), 1)

    def test_confirmation_returns_nonzero_on_not_verified(self):
        self.assertEqual(self._run("confirmation_a", {"a": PASS, "b": NOT_VERIFIED}, ["--batch-id", "x"]), 1)

    def test_confirmation_returns_nonzero_on_insufficient_evidence(self):
        self.assertEqual(
            self._run("confirmation_a", {"a": PASS, "b": INSUFFICIENT_EVIDENCE}, ["--batch-id", "x"]), 1
        )

    def test_confirmation_returns_zero_only_when_every_required_gate_passes(self):
        self.assertEqual(self._run("confirmation_b", {"a": PASS, "b": PASS}, ["--batch-id", "x"]), 0)

    def test_high_replication_also_requires_passing_gates(self):
        self.assertEqual(self._run("high_replication", {"a": FAIL}), 1)

    def test_a_refused_confirmation_returns_a_distinct_code(self):
        with mock.patch.object(cli, "run_benchmark", side_effect=ConfirmationError("dirty tree")):
            code, _, err = invoke(["benchmark", "--role", "confirmation_a", "--batch-id", "x"])
        self.assertEqual(code, 2)
        self.assertIn("dirty tree", err)

    def test_the_unmet_gates_are_named_on_stderr(self):
        with mock.patch.object(cli, "run_benchmark", return_value=outcome({"a": PASS, "b": FAIL})):
            _, _, err = invoke(["benchmark", "--role", "confirmation_a", "--batch-id", "x"])
        self.assertIn("b", err)

    def test_an_optional_gate_failure_does_not_fail_the_process(self):
        statuses = {"a": PASS, "optional": FAIL}
        with mock.patch.object(
            cli, "run_benchmark", return_value=outcome(statuses, required={"a": True, "optional": False})
        ):
            code, _, _ = invoke(["benchmark", "--role", "confirmation_a", "--batch-id", "x"])
        self.assertEqual(code, 0)


class RunnerEnforcementTests(unittest.TestCase):
    """The core runner enforces confirmation invariants, not just the CLI."""

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def _confirm(self, **overrides):
        arguments = dict(
            role="confirmation_a",
            batch_id="any-batch",
            output_root=self.root / "runs",
            project_root=PROJECT,
        )
        arguments.update(overrides)
        return run_benchmark(**arguments)

    def test_a_python_caller_cannot_weaken_the_replica_minimum(self):
        with self.assertRaises(ConfirmationError) as caught:
            self._confirm(replicas=1)
        self.assertIn("replica", str(caught.exception).lower())

    def test_a_python_caller_cannot_weaken_the_episode_minimum(self):
        with self.assertRaises(ConfirmationError) as caught:
            self._confirm(episodes=3)
        self.assertIn("episode", str(caught.exception).lower())

    def test_a_confirmation_without_a_batch_id_is_refused(self):
        with self.assertRaises(ConfirmationError) as caught:
            self._confirm(batch_id=None)
        self.assertIn("batch", str(caught.exception).lower())

    def test_a_confirmation_with_a_custom_specification_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            value = load_spec().to_dict()
            value["confirmation"]["episodes_per_family"] = 1
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ConfirmationError):
                self._confirm(spec_path=path)

    def test_a_confirmation_from_a_dirty_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "tree"
            copy.mkdir()
            (copy / "requirements-lock.txt").write_text("numpy==2.5.3\n", encoding="utf-8")
            with (
                mock.patch.object(
                    runner_module,
                    "git_metadata",
                    return_value={
                        "commit": "abc",
                        "branch": "x",
                        "tree_hash": "t",
                        "dirty": True,
                        "status": ["M x"],
                    },
                ),
                self.assertRaises(ConfirmationError) as caught,
            ):
                self._confirm()
        self.assertIn("clean source tree", str(caught.exception).lower())

    def test_an_unknown_role_is_refused(self):
        with self.assertRaises(ValueError):
            run_benchmark(role="whatever", output_root=self.root / "runs", project_root=PROJECT)

    def test_a_zero_override_is_refused_rather_than_silently_defaulted(self):
        for override in ({"replicas": 0}, {"episodes": 0}):
            with self.assertRaises(ValueError):
                run_benchmark(
                    role="development",
                    output_root=self.root / "runs",
                    project_root=PROJECT,
                    **override,
                )

    def test_a_negative_or_boolean_override_is_refused(self):
        for override in ({"replicas": -1}, {"episodes": True}):
            with self.assertRaises(ValueError):
                run_benchmark(
                    role="development",
                    output_root=self.root / "runs",
                    project_root=PROJECT,
                    **override,
                )

    def test_an_existing_attempt_directory_is_never_overwritten(self):
        run_benchmark(
            role="development",
            output_root=self.root / "runs",
            attempt_label="taken",
            replicas=1,
            episodes=1,
            project_root=PROJECT,
        )
        with self.assertRaises(FileExistsError):
            run_benchmark(
                role="development",
                output_root=self.root / "runs",
                attempt_label="taken",
                replicas=1,
                episodes=1,
                project_root=PROJECT,
            )


class InformationalCommandTests(unittest.TestCase):
    def test_spec_hash_prints_the_canonical_identity(self):
        code, out, _ = invoke(["spec-hash"])
        self.assertEqual(code, 0)
        self.assertIn(canonical_spec_hash(), out)

    def test_batches_lists_the_declared_registry(self):
        code, _, _ = invoke(["batches"])
        self.assertEqual(code, 0)

    def test_declare_batch_writes_a_planned_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            shutil.copytree(PROJECT / "benchmarks", project / "benchmarks")
            code, out, _ = invoke(
                [
                    "declare-batch",
                    "unit-test-batch",
                    "--role",
                    "confirmation_a",
                    "--project-root",
                    str(project),
                ]
            )
            self.assertEqual(code, 0)
            self.assertIn("unit-test-batch", out)
            registry = ConfirmationBatchRegistry.load(project / load_spec().confirmation.batch_registry)
            self.assertEqual(registry.get("unit-test-batch").status, "planned")

    def test_an_unknown_command_is_rejected(self):
        code, _, _ = invoke(["not-a-command"])
        self.assertNotEqual(code, 0)

    def test_no_command_is_rejected(self):
        code, _, _ = invoke([])
        self.assertNotEqual(code, 0)


class RecomputeCommandTests(unittest.TestCase):
    def test_recompute_reports_reproduced_statuses_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_benchmark(
                role="development",
                output_root=Path(directory) / "runs",
                attempt_label="cli-recompute",
                replicas=1,
                episodes=1,
                project_root=PROJECT,
            )
            code, out, _ = invoke(["recompute", str(result.directory)])
        self.assertEqual(code, 0)
        self.assertIn("every stored gate status was reproduced", out)

    def test_recompute_of_a_missing_run_fails(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            invoke(["recompute", str(Path(directory) / "nothing")])


if __name__ == "__main__":
    unittest.main()
