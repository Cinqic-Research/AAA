"""American-English task statements for language-conditioned opaque tasks (``aaa.wm.lang.reports.v1``).

A language-conditioned task is an ``opaque.v0`` task whose visible tests are **not** given as
numbers. The agent receives a short American-English specification or bug report that states
them (and, for bug reports, sometimes what the buggy program returned, which is the kind of
information a real report contains; it is the buggy program's real visible-input result, so it
leaks nothing hidden). Everything else is unchanged: the same program, runs, budgets and
domain-equivalence success.

Phrasing comes from template *families*. Families are split into ``train`` and ``heldout``
before any model sees them. Evaluation uses held-out families only, so success requires
understanding paraphrases rather than matching the training phrasings. Numbers are written as
digits or as English number words (``negative four``, ``twelve``, ``one hundred three``),
chosen per mention.

Every statement carries its gold structured form ``[(x, expected), ...]``, which is used only
for training the language adapter (train split) and for scoring extraction. Agents never see it.
"""

from __future__ import annotations

from typing import Any

from research.aaa_python.rng import Stream, derive_seed

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}


def number_words(n: int) -> str:
    """American English, no 'and': 103 -> 'one hundred three', -4 -> 'negative four'."""

    if n < 0:
        return "negative " + number_words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t * 10] + ("" if o == 0 else "-" + ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        return ONES[h] + " hundred" + ("" if r == 0 else " " + number_words(r))
    th, r = divmod(n, 1000)
    return number_words(th) + " thousand" + ("" if r == 0 else " " + number_words(r))


# {x} input, {e} expected, {g} what the buggy program returned (bug reports only). One clause per test.
CLAUSES: dict[str, list[str]] = {
    "train": [
        "f({x}) should return {e}",
        "calling f with {x} should give {e}",
        "for an input of {x}, the expected output is {e}",
        "when p is {x}, the result must be {e}",
        "f({x}) is supposed to be {e}",
        "given {x}, the function should produce {e}",
        "with {x} as the argument, we expect {e}",
        "the correct value of f({x}) is {e}",
        "passing {x} ought to return {e}",
        "an input of {x} needs to yield {e}",
        "f({x}) returns {g}, but it should return {e}",
        "for {x} I get {g} instead of {e}",
    ],
    "heldout": [
        "if you call f on {x}, you should get back {e}",
        "the answer for {x} has to be {e}",
        "feeding in {x} is expected to produce {e}",
        "{e} is what f should return when it is given {x}",
        "for the argument {x}, f must output {e}",
        "running f on {x} gave {g}; the right answer is {e}",
        "with p set to {x}, the function is supposed to return {e}",
        "we need f({x}) to equal {e}",
    ],
}
# "select": a third disjoint family, used ONLY to choose fine-tuning length (adapter v2); never for evaluation.
CLAUSES["select"] = [
    "when f is handed {x}, it ought to produce {e}",
    "the output for {x} is meant to be {e}",
    "f({x}) was {g} when it needed to be {e}",
    "given the value {x}, the correct return is {e}",
    "calling f on {x} must yield {e}",
    "for input {x}, we expect an answer of {e}",
]
OPENERS = {
    "train": ["Please fix f.", "Bug report:", "The function f is broken.", "Spec for f:", "Task: repair f so that it behaves correctly."],
    "heldout": ["Heads up: f has a bug.", "Here is what f is supposed to do.", "Could you take a look at f?", "Issue:"],
    "select": ["Problem with f:", "Note:", "The tests for f say the following."],
}
JOINERS = {"train": ["; ", ". Also, ", ", and "], "heldout": [". In addition, ", "; moreover, ", ". Finally, "], "select": [". Next, ", "; then ", ". Plus, "]}


def _num(s: Stream, v: int) -> str:
    return number_words(v) if s.chance(1, 2) else str(v)


def statement(task: Any, split_family: str, buggy_results: list[Any] | None = None, *, seed_label: str = "") -> dict[str, Any]:
    """A statement for ``task``'s visible tests using the ``train`` or ``heldout`` template families."""

    s = Stream(derive_seed("aaa.wm.lang.reports.v1", task.task_id, split_family, seed_label))
    clauses = []
    gold = []
    for j, (x, expected) in enumerate(task.visible_tests):
        e = expected[1]
        choices = CLAUSES[split_family]
        has_g = buggy_results is not None and buggy_results[j][0] == "ok" and buggy_results[j][1] != e
        pool = [c for c in choices if ("{g}" in c) == has_g] if has_g and s.chance(1, 2) else [c for c in choices if "{g}" not in c]
        c = s.choice(pool)
        text = c.format(x=_num(s, x), e=_num(s, e), g=_num(s, buggy_results[j][1]) if has_g else "")
        clauses.append(text)
        gold.append((x, e))
    body = clauses[0]
    for c in clauses[1:]:
        body += s.choice(JOINERS[split_family]) + c
    first = body if body.startswith("f(") else body[0].upper() + body[1:]  # never capitalize the name f
    text = f"{s.choice(OPENERS[split_family])} {first}."
    return {"text": text, "gold": gold, "family": split_family}
