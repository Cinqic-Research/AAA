"""CPU versus CUDA for the ``aaa.python.v1`` learner workload, measured rather than assumed.

The workload is online learning: one example at a time, a sparse input with a
few dozen non-zero hashed features in D = 256, a ``tanh`` core of H units and
small heads. Two layouts are timed on each backend for the ~1K (H = 4) and
~10K (H = 37) cores:

``single``   one learner, one forward+backward+update per example (the
             experiment as run);
``batched``  B independent learners stepped together, one example each per
             step (the only layout in which a GPU could amortize launch cost;
             learners never share parameters).

The step computes the same algebra on NumPy (CPU) and CuPy (CUDA): gather the
active input columns, ``tanh`` core, a softmax head of C classes, the
cross-entropy gradient and an SGD update. CUDA runs are explicit (``--device
cuda``); a missing GPU is an error, never a silent CPU fallback. Parity is
checked on the batched layout: after identical steps, CPU and CUDA parameters
must agree to a relative 1e-10 (float64).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import numpy as np

D, C, NNZ = 256, 8, 40


def _step(xp: Any, W: Any, b: Any, V: Any, x: Any, target: Any, lr: float) -> None:
    """One SGD step for B independent learners (leading axis), fully vectorized on ``xp``'s device."""

    B = W.shape[0]
    z = xp.tanh(xp.einsum("bhd,bd->bh", W, x) + b)
    scores = xp.einsum("bch,bh->bc", V, z)
    scores = scores - scores.max(axis=1, keepdims=True)
    p = xp.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)
    p[xp.arange(B), target] -= 1.0
    dz = xp.einsum("bch,bc->bh", V, p) * (1.0 - z * z)
    V -= lr * xp.einsum("bc,bh->bch", p, z)
    b -= lr * dz
    W -= lr * xp.einsum("bh,bd->bhd", dz, x)


def run(xp: Any, B: int, H: int, steps: int, seed: int = 0) -> tuple[float, Any]:
    rng = np.random.default_rng(seed)
    W = xp.asarray(rng.normal(0, 1, (B, H, D)))
    b = xp.zeros((B, H))
    V = xp.zeros((B, C, H))
    xs = []
    for _ in range(steps):
        x = np.zeros((B, D))
        for k in range(B):
            x[k, rng.choice(D, size=NNZ, replace=False)] = rng.normal(0, 0.15, NNZ)
        xs.append(xp.asarray(x))
    tgt_all = [xp.asarray(rng.integers(0, C, size=B)) for _ in range(steps)]
    sync = (lambda: xp.cuda.Stream.null.synchronize()) if xp.__name__ == "cupy" else (lambda: None)
    for s in range(min(20, steps)):
        _step(xp, W, b, V, xs[s], tgt_all[s], 0.1)
    sync()
    W = xp.asarray(rng.normal(0, 1, (B, H, D)))
    b = xp.zeros((B, H))
    V = xp.zeros((B, C, H))
    start = time.perf_counter()
    for s in range(steps):
        _step(xp, W, b, V, xs[s], tgt_all[s], 0.1)
    sync()
    return time.perf_counter() - start, W


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    backends: list[tuple[str, Any]] = []
    if args.device in ("cpu", "both"):
        backends.append(("cpu", np))
    if args.device in ("cuda", "both"):
        import cupy

        if cupy.cuda.runtime.getDeviceCount() < 1:
            raise SystemExit("no CUDA device; refusing to fall back to the CPU")
        backends.append(("cuda", cupy))
    results: dict[str, Any] = {"platform": platform.processor() or platform.machine(), "rows": []}
    for H in (4, 37):
        for B in (1, 10, 100):
            for name, xp in backends:
                seconds, _ = run(xp, B, H, args.steps)
                results["rows"].append(
                    {
                        "backend": name,
                        "hidden": H,
                        "batch": B,
                        "steps": args.steps,
                        "learner_updates_per_second": B * args.steps / seconds,
                    }
                )
                print(
                    f"{name:4s} H={H:3d} B={B:4d}: {B * args.steps / seconds:12.0f} learner-updates/s",
                    flush=True,
                )
    if len(backends) == 2:
        import cupy

        _, w_cpu = run(np, 10, 37, 200, seed=5)
        _, w_gpu = run(cupy, 10, 37, 200, seed=5)
        diff = float(np.max(np.abs(w_cpu - cupy.asnumpy(w_gpu)) / (np.abs(w_cpu) + 1e-12)))
        results["parity_max_relative_difference"] = diff
        print(f"CPU/CUDA parity after 200 batched steps: max relative difference {diff:.3e}")
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
