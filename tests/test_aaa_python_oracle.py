"""aaa.python.v0: the frozen subset, the sandboxed CPython oracle, the spec and cross-version keys."""

from __future__ import annotations

import copy
import json
import unittest
from importlib import resources

from research.aaa_python import checks, cli, oracle, rng
from research.aaa_python import spec as spec_module
from research.aaa_python.subset import SubsetError, validate


class StreamTests(unittest.TestCase):
    def test_the_stream_is_a_fixed_function_of_its_seed(self) -> None:
        # Pinned values: the generator's cross-version identity rests on these.
        seed = rng.derive_seed("aaa.python.v0", "pin")
        self.assertEqual(seed, 18182495422584857294)
        stream = rng.Stream(seed)
        self.assertEqual([stream.below(1000) for _ in range(5)], [381, 643, 43, 310, 956])

    def test_bounds_and_arguments_are_checked(self) -> None:
        stream = rng.Stream(1)
        self.assertTrue(all(0 <= stream.below(7) < 7 for _ in range(500)))
        self.assertEqual(sorted(stream.shuffled(range(10))), list(range(10)))
        for bad in (-1, 2**64, True):
            with self.assertRaises(ValueError):
                rng.Stream(bad)
        with self.assertRaises(ValueError):
            stream.below(0)


class SpecTests(unittest.TestCase):
    def test_the_packaged_spec_loads_and_hashes(self) -> None:
        spec = spec_module.load()
        self.assertEqual(spec["protocol"], "aaa.python.v0")
        self.assertRegex(spec_module.spec_hash(), r"^[0-9a-f]{64}$")
        self.assertNotIn("confirmation", spec["splits"]["pool_per_family"])

    def test_malformed_specs_are_refused(self) -> None:
        spec = spec_module.load()
        for mutate in (
            lambda s: s.pop("sandbox"),
            lambda s: s.update(extra=1),
            lambda s: s.update(protocol="aaa.python.v1"),
            lambda s: s["splits"]["pool_per_family"].update(confirmation=10),
            lambda s: s["subset"]["limits"].update(max_lines=0),
            lambda s: s["learner"].update(default_representation="words"),
        ):
            broken = copy.deepcopy(spec)
            mutate(broken)
            with self.assertRaises(spec_module.SpecError):
                spec_module.validate(broken)

    def test_non_standard_json_is_refused(self) -> None:
        with self.assertRaises(spec_module.SpecError):
            json.loads('{"x": NaN}', parse_constant=spec_module._refuse_constant)

    def test_package_resources_ship_with_the_code(self) -> None:
        package = resources.files("research.aaa_python")
        for name in ("data/aaa_python_v0.json", "data/golden_answers_v0.json", "_sandbox_child.py"):
            self.assertTrue(package.joinpath(name).is_file(), name)
        self.assertTrue(oracle.CHILD.is_file())


class SubsetTests(unittest.TestCase):
    def test_every_listed_escape_is_refused_and_every_containment_holds(self) -> None:
        results = checks.safety_checks()
        self.assertEqual(checks.failed(results), [])
        self.assertGreaterEqual(len(results), 60)

    def test_subset_programs_are_accepted(self) -> None:
        for source in (
            "x = 3\nx += 2\nprint(x)",
            "def f(p):\n    return p * 2 + 1\ny = f(4)\nprint(y)",
            "acc = 0\nfor i in range(3):\n    for j in range(2):\n        acc += i - j\nprint(acc)",
            "xs = [1, 2, 3]\nv = xs[1]\nt = [4, 5][v - 2]\nprint(v if v > 1 else -v)",
        ):
            validate(source)

    def test_function_and_parameter_names_follow_the_identifier_rule(self) -> None:
        for source in ("def eval(p):\n    return p", "def f(_p):\n    return 1", "def F(p):\n    return p"):
            with self.assertRaises(SubsetError):
                validate(source)


class OracleTests(unittest.TestCase):
    def test_outcomes_are_exact(self) -> None:
        ok = oracle.execute("x = 3\nx += 2\nprint(x)")
        self.assertEqual((ok.status, ok.stdout), ("ok", "5\n"))
        cases = {
            "x = 1\ny = x // (x - 1)": ("ZeroDivisionError", 2),
            "x = 1\nif x > 0:\n    y = q + 1": ("NameError", 3),
            "x = 1\ny = x + str(x)": ("TypeError", 2),
            "x = [1, 2]\ny = x[5]": ("IndexError", 2),
            'y = int("z")': ("ValueError", 1),
            "def f(p):\n    return p // 0\nprint(f(1))": ("ZeroDivisionError", 2),
        }
        for source, (exception, line) in cases.items():
            outcome = oracle.execute(source)
            self.assertEqual(
                (outcome.status, outcome.exception, outcome.line), ("exception", exception, line), source
            )

    def test_syntax_checks_compile_and_never_execute(self) -> None:
        valid = oracle.check_syntax("print(1)\n")
        self.assertEqual((valid.status, valid.stdout), ("valid", ""))
        self.assertEqual(oracle.check_syntax("if x = 3:\n    pass").status, "syntax_error")
        self.assertEqual(oracle.check_syntax("x = (1\n").status, "syntax_error")

    def test_nothing_unvalidated_is_executed(self) -> None:
        with self.assertRaises(SubsetError):
            oracle.execute("import os")

    def test_the_sandbox_records_its_limits_and_isolation(self) -> None:
        outcome = oracle.execute("print(1)")
        self.assertIn("RLIMIT_CPU", outcome.limits_applied)
        self.assertTrue(outcome.is_program_outcome)
        failure = oracle.Outcome("sandbox_failure", None, None, "", False, {}, 0.0)
        self.assertFalse(failure.is_program_outcome)

    def test_many_runs_preserve_order(self) -> None:
        outcomes = oracle.run_many([("exec", f"print({i})") for i in range(20)])
        self.assertEqual([o.stdout for o in outcomes], [f"{i}\n" for i in range(20)])

    def test_interpreter_provenance_is_recorded(self) -> None:
        record = oracle.interpreter_provenance()
        for key in ("implementation", "version", "version_info", "platform", "compiler"):
            self.assertIn(key, record)


class CrossVersionTests(unittest.TestCase):
    def test_golden_answer_keys_are_identical_on_this_interpreter(self) -> None:
        # CI runs this on CPython 3.10, 3.11, 3.12 and 3.13.
        self.assertEqual(cli.golden_differences(), [])


if __name__ == "__main__":
    unittest.main()
