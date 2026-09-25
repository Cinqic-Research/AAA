# `aaa.python.v1` compute report

**Decision: CPU (NumPy).** Measured on FLOWBOX, not assumed. Raw output:
[`evidence/aaa_python_v1/compute_benchmark.json`](evidence/aaa_python_v1/compute_benchmark.json).

## Hardware and environment

AMD Ryzen 7 5700G (8 cores, 16 threads), 16 GB RAM, NVIDIA GeForce RTX 2060
6 GB (driver 595.91.07), CPython 3.12.3, NumPy 2.5.3, CuPy 14.2 from the
locked CUDA environment (`requirements-cuda-lock.txt`, all 30 pins verified).
All repositories, venvs, pools, resumption caches and outputs were on the
`Cinqic Storage` HDD; the storage preflight passed before every stage. The
HDD was found unmounted at the start of the session and once mid-run; it was
remounted, never replaced by the NVMe (which was 97% full).

## What was measured

`tools/benchmark_aaa_python_v1_compute.py` times the learner's algebra (dense
inputs in D = 256, a `tanh` core, a softmax head, the cross-entropy gradient
and an SGD update) on both backends, for the ~1K core (H = 4) and the ~10K core
(H = 37), with B independent learners stepped together. The machine was idle.

| Core | Learners per step | CPU updates/s | CUDA updates/s | Faster |
|---|---:|---:|---:|---|
| ~1K | 1 | 40,064 | 1,272 | CPU x31 |
| ~1K | 10 | 246,621 | 13,058 | CPU x19 |
| ~1K | 100 | 686,187 | 129,032 | CPU x5 |
| ~10K | 1 | 28,906 | 1,260 | CPU x23 |
| ~10K | 10 | 82,861 | 12,953 | CPU x6 |
| ~10K | 100 | 67,434 | 128,369 | CUDA x1.9 |

AAA's workload is online learning, one task at a time through the causal
boundary. Its throughput is set by per-task Python work (the environment,
encoding, tool calls), about 5,000-8,000 updates per second per process; the
stages ran 14 processes in parallel. A GPU pays off only if about 100 learners
step in lockstep, which would mean redesigning the causal pipeline for at most
a 1.9x gain on a part that is not the bottleneck.

## Parity and fallback

After 200 identical batched steps, CPU and CUDA parameters agree to a relative
1.2e-13 (float64). The benchmark refuses to run CUDA without a device; nothing
falls back silently. No v1 evidence was produced on CUDA, so no v1 result
depends on backend drift.

## Cost of the phase

| Stage | Wall time (14 workers) |
|---|---:|
| encoders (54 tuning runs, 90 evaluations, baselines, v0 anchor) | 1,981 s |
| heads | 584 s |
| capacity | 1,109 s |
| budget extension | about 700 s |
| optimization, tool, adaptation, plasticity, attack | about 3,900 s together |
| post-hoc diagnostic | 658 s |

Stage evidence totals about 7 MB of JSON; the pool cache about 30 MB. A 10K
model's strict-JSON state is about 0.16 MB. None of this approaches FLOWBOX's
limits; the scale question was answered by evidence, not constrained by
hardware.
