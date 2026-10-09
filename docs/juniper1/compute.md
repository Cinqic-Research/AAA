# `aaa.erudition.v0` compute record

All work ran on FLOWBOX, with no cloud or external compute:
- Ryzen 7 5700G (8 cores, 16 threads), 16 GB RAM (15 GiB visible, 2 GiB swap);
- RTX 2060 6 GB;
- Linux Mint (kernel 7.0), Python 3.12.3.

Repositories, virtual environments, data and logs were on the `Cinqic Storage`
HDD (storage preflight passed). The NVMe held only the operating system.

## Environments

| Environment | Lock | Used for |
|---|---|---|
| `venv` | `requirements-lock.txt` (24 distributions, verified) | model-free code, tests, real-model runs, evaluation |
| `venv-torch` | `requirements-torch-lock.txt` (64 distributions, verified; torch 2.14.0+cu126) | Erudition training, simulation with the Erudition continuation, World Model diagnostic |
| llama.cpp | b11270, commit `748d4225`, CUDA sm_75 (Juniper LM 1.1 build) | serving gpt-oss-20b with Juniper-App's qualified profile |

## The resident Language Model

| Item | Measured |
|---|---|
| Artifact | `gpt-oss-20b-6cee5e81ee83-mxfp4-moe.gguf`, SHA-256 verified `9d7364f0…d23d` (2 min 18 s from the HDD) |
| Load | about 3 minutes cold from the HDD |
| Resident | 4.76 GB VRAM; the model file is memory-mapped, about 12 GB of page cache |
| Profile | `-c 16384 -ngl 99 --n-cpu-moe 18 -fa on -t 8 --parallel 1`, temperature 1, top-p 1, reasoning effort low |
| Request | about 330 prompt tokens, 50–135 generated tokens |
| Throughput, idle machine | 27.4–27.9 generated tokens/s, about 4 s per request |
| Throughput, concurrent simulation | about 19 tokens/s, 5–6 s per request; up to 12 s with training as well |

The model's experts run on the CPU. CPU- and memory-bandwidth-heavy work
competes with it regardless of `nice`. Real-model runs were therefore
scheduled away from simulation and training wherever the critical path
allowed. This is the coexistence result for FLOWBOX: inference of all three
components fits together; Erudition training alongside the resident model
works but slows it about threefold.

## Runtime rebuild (2026-10-05)

Between sessions, the qualification's CUDA build of llama.cpp and the shared
conda-forge CUDA 12.9 toolchain were removed from the HDD as part of a
storage clean-up outside this work. The runtime was rebuilt in this workspace:
- the same recipe (micromamba 2.9.0; conda-forge `cuda-version=12.9 cuda-nvcc
  cuda-cudart-dev libcublas-dev cmake ninja`; system GCC 13.3 as host
  compiler; `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`);
- built from the untouched source checkout at commit `748d4225`.

The rebuilt server reports `b11270-748d4225b`. Its binary SHA-256 begins
`a6a5af5c`; the exact package list is retained beside the toolchain.

On 30 recorded development requests re-sent live, generated token sequences
differed (sampling at temperature 1 is not reproducible from a seed across
processes). The parsed action was identical to the recorded runtime's on
30/30, and identical on two identical live requests on 30/30. Every
retained run replays from its call cache, so this affects only new calls.
Each run manifest records the server's own build report.

## Two machine failures, and the scheduling rule they produced

FLOWBOX went down twice on 2026-10-04, at about 07:53 and 12:34. Both times
the GPT-OSS server was resident while heavy work ran beside it: first a
six-worker simulation, then GPU training on about 160k examples. The journal
records no out-of-memory kill, which is consistent with a thrashing freeze:
the model's memory-mapped 12 GB file is evicted and re-read from the HDD.
Interrupted outputs were trimmed to their last complete record and resumed;
the byte-reproducibility of resumed simulation data was verified on sample
streams.

From then on:
- real-model phases ran with only the light experiment runner beside the
  server;
- simulation and training ran with the server stopped.

Training without the server was also about twice as fast (about 45 s per
epoch for the 1.26M model, against about 90 s), and the larger models no
longer ran out of GPU memory.

## Erudition Model

| Item | Value |
|---|---|
| Trainable parameters | 1,259,700 |
| Frozen parameters | 0 |
| Optimizer state while training | AdamW: 2 × 1,259,700 float32 |
| Persistent inference state | 64 × 40 float32 evidence window |
| Weights on disk | about 5 MB |
| Training | RTX 2060, 20 epochs, about 100k examples per data file (see the development report for per-run durations) |
| Inference | CPU, about 1–3 ms per decision |

## World Model and Language Model adapter state

Both are JSON in the content-addressed store. A World Model context holds two
3-parameter Normal-Inverse-Gamma posteriors and at most 48 transitions; a
whole state is a few kilobytes. The adapter holds a handful of notes, under
2 KB. Neither has optimizer state.

## Durations

See the [development report](development_report.md) for each stage's
measured wall time.
