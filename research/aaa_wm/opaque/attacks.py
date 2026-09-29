"""Benchmark attacks for opaque.v0 that need no world model.

``illegal_lines`` / :class:`TellPredictor` were found by the 2026-09-26 red team
(``notes/redteam_opaque_wms.md``, script ``rt2_tells.py``). The recognizer knows only the
generator's grammar: no library, no learning, no execution. A mutation often produces text
the grammar can never emit (a literal outside the drawn ranges, ``p * 4``, a ``//`` divisor of 5,
a ``+= / -=`` pair it never writes), and that flags the faulty line. The same verified planner,
scoring programs only by grammar legality, is therefore a **no-world-model shortcut
baseline**. It is declared as the strongest non-learned baseline, and it documents a
generator weakness to repair in a successor benchmark.
"""

from __future__ import annotations

import re

import numpy as np

LITSET = set(range(2, 10)) | set(range(12, 20))  # in-distribution and novel literal ranges (see note)
N = r"(?:p|q|i)"
API = r"api\d"
L = r"(?P<L>\d+)"


def lit_ok(s):
    return int(s) in LITSET


def expr_ok(e):
    e = e.strip()
    m = re.fullmatch(rf"{N} (\+|-) (\d+)", e)
    if m:
        return lit_ok(m.group(2))
    if re.fullmatch(rf"{N} \* [23]", e) or re.fullmatch(rf"{N} // [234]", e) or re.fullmatch(rf"{N} % [3456]", e):
        return True
    if re.fullmatch(rf"{API}\({N}\)", e):
        return True
    m = re.fullmatch(rf"{API}\({N}\) \+ (\S+)", e)
    if m:
        a = m.group(1)
        return a in ("p", "q", "i") or (a.isdigit() and lit_ok(a))
    return False


def cond_ok(c):
    m = re.fullmatch(rf"(?:{N}|{API}\({N}\)) (?:>|>=|<|<=|==|!=) (\d+)", c.strip())
    return bool(m) and lit_ok(m.group(1))


def illegal_lines(src):
    lines = [l for l in src.split("\n") if l.strip()]
    bad = 0
    body = lines[1:]
    # first statement: q = E
    k = 0
    while k < len(body):
        line = body[k]
        ind = len(line) - len(line.lstrip())
        s = line.strip()
        if s.startswith("for i in range("):
            ok = re.fullmatch(r"for i in range\([234]\):", s) is not None
            bad += not ok
            inner = []
            j = k + 1
            while j < len(body) and len(body[j]) - len(body[j].lstrip()) > ind:
                inner.append(body[j].strip())
                j += 1
            if len(inner) == 1:
                t = inner[0]
                m = re.fullmatch(r"q \+= (.+)", t)
                bad += not (m and (m.group(1) == "i" or expr_ok(m.group(1))))
            elif len(inner) == 2:  # for_if
                bad += not (inner[0].startswith("if ") and inner[0].endswith(":") and cond_ok(inner[0][3:-1]))
                m = re.fullmatch(r"q \+= (.+)", inner[1])
                bad += not (m and expr_ok(m.group(1)))
            else:
                bad += 1
            k = j
            continue
        if s.startswith("if ") and s.endswith(":"):
            bad += not cond_ok(s[3:-1])
            b1 = body[k + 1].strip() if k + 1 < len(body) else ""
            has_else = k + 2 < len(body) and body[k + 2].strip() == "else:"
            b2 = body[k + 3].strip() if has_else and k + 3 < len(body) else None
            m1 = re.fullmatch(r"q (=|\+=|-=) (.+)", b1)
            if not m1:
                bad += 1
            else:
                op1 = m1.group(1)
                bad += not expr_ok(m1.group(2))
                if b2 is not None:
                    m2 = re.fullmatch(r"q (=|\+=|-=) (.+)", b2)
                    if not m2:
                        bad += 1
                    else:
                        bad += not expr_ok(m2.group(2))
                        pair_ok = (op1, m2.group(1)) in (("=", "="), ("+=", "-="))
                        bad += not pair_ok
                else:
                    bad += op1 != "="
            k += 4 if has_else else 2
            continue
        m = re.fullmatch(r"q (=|\+=|-=) (.+)", s)
        if m:
            bad += not expr_ok(m.group(2))
            k += 1
            continue
        m = re.fullmatch(r"return (.+)", s)
        if m:
            bad += m.group(1) not in ("q", "q + p") and re.fullmatch(rf"{API}\(q\)", m.group(1)) is None
            k += 1
            continue
        bad += 1
        k += 1
    return bad


class TellPredictor:
    """Not a world model: a program's score depends only on whether its text is grammar-legal."""

    def __call__(self, programs, view):
        out = np.zeros((len(programs), len(view.visible_tests)))
        for i, s in enumerate(programs):
            out[i, :] = 0.5 * (0.02 ** illegal_lines(s))
        return out
