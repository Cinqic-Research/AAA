"""Cross-check the in-process executor against the v0 sandboxed CPython oracle.

The library functions are all expressible in the safe subset, so a sandbox program is the
opaque program with the library written out as ordinary earlier functions. The results must be
identical (value or exception class) for every sampled (program, input).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .library import ApiSpec
from .program import run


def library_source(lib: Sequence[ApiSpec]) -> str:
    out = []
    for a in lib:
        p = a.params
        if a.family == "affine":
            body = f"    return x * {p[0]} + {p[1]}" if p[1] >= 0 else f"    return x * {p[0]} - {-p[1]}"
        elif a.family == "clip_hi":
            body = f"    return min(x, {p[0]})"
        elif a.family == "clip_lo":
            body = f"    return max(x, {p[0]})"
        elif a.family == "mod":
            body = f"    return x % {p[0]} + {p[1]}"
        elif a.family == "step":
            body = f"    if x > {p[0]}:\n        return {p[1]}\n    return {p[2]}"
        else:
            body = f"    return abs(x - {p[0]})"
        out.append(f"def {a.name}(x):\n{body}\n")
    return "".join(out)


def agreement(samples: Sequence[tuple[str, int]], lib: tuple[ApiSpec, ...]) -> dict[str, Any]:
    from research.aaa_python.oracle import run_many
    from research.aaa_python_v1 import spec as v1_spec

    prefix = library_source(lib)
    jobs = [("exec", prefix + src + f"print(f({x}))\n") for src, x in samples]
    outcomes = run_many(jobs, v1_spec.load())
    bad = []
    for (src, x), o in zip(samples, outcomes, strict=True):
        fast = run(src, lib, x)
        if o.status == "ok":
            slow: tuple[str, Any] = ("ok", int(o.stdout.strip()))
        elif o.status == "runtime_error":
            slow = ("error", o.exception)
        else:
            slow = (o.status, None)
        if fast[0] == "error" and fast[1] == "Overflow":
            continue  # the environment's declared overflow bound has no sandbox analogue
        if fast != slow:
            bad.append({"source": src, "input": x, "fast": fast, "sandbox": slow})
    return {"checked": len(samples), "disagreements": bad}
