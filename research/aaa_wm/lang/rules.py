"""A hand-written language baseline: extract visible tests from a statement with rules.

Written from the **training** phrasings only (``reports.CLAUSES["train"]``), the way an engineer
would write a parser for known report formats. It converts English number words to integers,
splits the statement into clauses on the training joiners and sentence ends, and reads each
clause's first number as the input and its last number as the expected value. Held-out
phrasings are never consulted. The baseline is strong on familiar formats, and it fails exactly
where paraphrase changes the order or splits a clause.
"""

from __future__ import annotations

import re

_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD = r"(?:negative |minus )?(?:(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)(?:[ -](?=zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))?)+"
_NUMBER = re.compile(rf"-?\d+|\b{_WORD}\b", re.IGNORECASE)


def words_to_int(text: str) -> int:
    t = text.lower().replace("-", " ").split()
    sign = 1
    if t and t[0] in ("negative", "minus"):
        sign, t = -1, t[1:]
    total = current = 0
    for w in t:
        if w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current *= 100
        elif w == "thousand":
            total += current * 1000
            current = 0
    return sign * (total + current)


def numbers(text: str) -> list[int]:
    out = []
    for m in _NUMBER.finditer(text):
        s = m.group(0).strip()
        out.append(int(s) if re.fullmatch(r"-?\d+", s) else words_to_int(s))
    return out


def extract(statement: str, n: int = 3) -> list[tuple[int, int]] | None:
    """Up to ``n`` (input, expected) pairs, or ``None`` if the statement does not parse into ``n`` clauses."""

    body = statement.split(":", 1)[1] if statement.split(".")[0].endswith(":") or ":" in statement[:40] else statement
    for opener in ("Please fix f.", "The function f is broken."):
        body = body.replace(opener, " ")
    parts = re.split(r";|\. Also,|, and |\.(?=\s+[A-Z])", body)
    pairs = []
    for p in parts:
        nums = numbers(p)
        if len(nums) >= 2:
            pairs.append((nums[0], nums[-1]))
    return pairs[:n] if len(pairs) >= n else None
