"""Executable safety and causal-boundary checks, shared by the CLI and the tests.

``safety_checks`` proves that the validator refuses every listed escape and
that the sandbox contains programs that get past it (exercised through the
private child entry point, as if validation had been bypassed).
``leakage_checks`` tries to obtain answer information before a scored action
through every channel the boundary is meant to close.
Each check returns ``(name, passed, detail)``; nothing here observes
confirmation evidence.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from typing import Any

from . import generator
from . import spec as spec_module
from .episode import VIEW_FIELDS, Action, BoundaryError, Environment, TaskView
from .oracle import _request, _run_child, execute
from .subset import SubsetError, validate

ESCAPES = {
    "import": "import os\nprint(1)",
    "from-import": "from os import system\nprint(1)",
    "dunder import": "x = __import__('os')",
    "open": "x = open('secret')",
    "eval": "x = eval('1 + 1')",
    "exec": "exec('x = 1')",
    "compile": "x = compile('1', 'f', 'eval')",
    "getattr": "x = getattr(1, 'real')",
    "globals": "x = globals()",
    "attribute reflection": "x = ().__class__",
    "attribute call": "x = [1]\nx.append(2)",
    "while": "while True:\n    pass",
    "lambda": "f = lambda: 1",
    "comprehension": "x = [i for i in range(3)]",
    "try": "try:\n    x = 1\nexcept Exception:\n    pass",
    "with": "with open('f') as h:\n    pass",
    "class": "class A:\n    pass",
    "power": "x = 9 ** 9",
    "true division": "x = 1 / 2",
    "f-string": "x = f'{1}'",
    "recursion": "def f(p):\n    return f(p)\nx = f(1)",
    "later function": "def f(p):\n    return g(p)\ndef g(p):\n    return p",
    "large literal": "x = 10000000",
    "large range": "for i in range(100000):\n    pass",
    "range over a name": "n = 3\nfor i in range(n):\n    pass",
    "deep nesting": "for i in range(2):\n    for j in range(2):\n        for k in range(2):\n            pass",
    "underscore name": "_x = 1",
    "rebind builtin": "print = 1",
    "keyword argument": "print(1, end='')",
    "slice": "x = [1, 2]\ny = x[0]\nz = x[:1]",
    "global": "def f(p):\n    global x\n    return p",
}

CONTAINMENT: dict[str, tuple[str, set[str]]] = {
    # Programs run through the child directly, as if the validator were bypassed.
    "infinite loop": ("while True:\n    pass", {"cpu_limit", "timeout"}),
    "memory exhaustion": ("x = [0] * (10 ** 10)", {"exception", "sandbox_failure"}),
    "file write (no open builtin)": ("open('f', 'w').write('x')", {"exception"}),
    "import (no __import__ builtin)": ("import os\nos.system('true')", {"exception"}),
    "output flood": ("for i in range(10 ** 7):\n    print('xxxxxxxxxxxxxxxx')", {"output_limit"}),
}


def safety_checks() -> list[tuple[str, bool, str]]:
    spec = spec_module.load()
    results: list[tuple[str, bool, str]] = []
    for name, source in ESCAPES.items():
        try:
            validate(source, spec)
        except SubsetError as error:
            results.append((f"validator refuses {name}", True, str(error)))
        else:
            results.append((f"validator refuses {name}", False, "accepted"))
        try:
            execute(source, spec)
        except SubsetError:
            results.append((f"oracle will not run {name}", True, "refused before execution"))
        else:
            results.append((f"oracle will not run {name}", False, "executed"))
    for name, (source, allowed) in CONTAINMENT.items():
        outcome = _run_child(_request(source, "exec", spec), spec["sandbox"]["wall_timeout_seconds"])
        results.append(
            (
                f"sandbox contains {name}",
                outcome.status in allowed,
                f"{outcome.status} {outcome.exception or ''}",
            )
        )
    probe = execute("x = 1\nprint(x)", spec)
    results.append(
        ("sandbox applies resource limits", bool(probe.limits_applied), json.dumps(probe.limits_applied))
    )
    return results


def _reachable(root: object, limit: int = 2000) -> list[object]:
    seen, frontier, found = {id(root)}, [root], []
    while frontier and len(seen) < limit:
        item = frontier.pop()
        found.append(item)
        for child in gc.get_referents(item):
            if id(child) not in seen and not isinstance(child, type):
                seen.add(id(child))
                frontier.append(child)
    return found


def _expect(error: type[BaseException], call: Callable[[], Any]) -> bool:
    try:
        call()
    except error:
        return True
    return False


def _reveal_refused(env: Environment, view: TaskView) -> bool:
    try:
        env.reveal(view)
    except BoundaryError:
        return True
    return False


def _commit_refused(env: Environment, view: TaskView, label: Any) -> bool:
    try:
        env.commit(view, Action(label, 0.5))
    except BoundaryError:
        return True
    return False


def _out_of_space_refused(task: generator.Task, spec: Any) -> bool:
    env = Environment(spec)
    view = env.present(task)  # a presented task, so only the label-space rule can refuse
    try:
        env.commit(view, Action("not-a-label", 0.5))
    except BoundaryError as error:
        return "label space" in str(error)
    return False


def leakage_checks() -> list[tuple[str, bool, str]]:
    spec = spec_module.load()
    results: list[tuple[str, bool, str]] = []
    for family in spec["families"]:
        task = generator.build("development", family, [3])[0]
        env = Environment(spec)
        view = env.present(task)
        fields = tuple(f for f in TaskView.__dataclass_fields__)
        results.append((f"{family}: view has only the declared fields", fields == VIEW_FIELDS, str(fields)))
        reachable = _reachable(view)
        leaked = [o for o in reachable if isinstance(o, generator.Task)]
        results.append(
            (f"{family}: no evaluator task reachable from the view", not leaked, f"{len(reachable)} objects")
        )
        text = json.dumps({k: getattr(view, k) for k in VIEW_FIELDS}, default=str)
        overlap = set(view.visible_tests) & set(task.hidden_tests)
        results.append((f"{family}: no hidden test case in the view", not overlap, str(sorted(overlap))))
        results.append(
            (
                f"{family}: no oracle observation in the view",
                "oracle" not in text and "stdout" not in text,
                "",
            )
        )
        results.append((f"{family}: reveal before commit is refused", _reveal_refused(env, view), ""))
        label = view.labels[0] if family != "repair" else 0
        env.commit(view, Action(label, 0.5))
        results.append((f"{family}: a second commit is refused", _commit_refused(env, view, label), ""))
        results.append(
            (f"{family}: an out-of-space answer is refused", _out_of_space_refused(task, spec), "")
        )
        _, feedback = env.reveal(view)
        allowed = set(spec["families"][family]["feedback"])
        assert feedback is not None
        results.append(
            (
                f"{family}: feedback holds only declared fields",
                set(feedback.fields) == allowed,
                str(sorted(feedback.fields)),
            )
        )
        if family == "repair":
            revealed = feedback.fields["chosen_candidate_hidden_results"]
            results.append(
                (
                    "repair: feedback covers the chosen candidate only",
                    isinstance(revealed, list) and len(revealed) == len(task.hidden_tests),
                    str(revealed),
                )
            )
        quiet = Environment(spec, feedback_enabled=False)
        v2 = quiet.present(task)
        quiet.commit(v2, Action(label, 0.5))
        results.append((f"{family}: disabled feedback reveals nothing", quiet.reveal(v2)[1] is None, ""))
        order = [event for _, event, _ in env.events]
        results.append(
            (
                f"{family}: event order is present, commit, reveal",
                order == ["present", "commit", "reveal"],
                str(order),
            )
        )
    results.append(
        (
            "confirmation identities cannot be generated",
            _expect(
                generator.ConfirmationNotAdmitted, lambda: generator.build("confirmation", "syntax", [0])
            ),
            "",
        )
    )
    return results


def failed(results: list[tuple[str, bool, str]]) -> list[str]:
    return [f"{name}: {detail}" for name, passed, detail in results if not passed]
