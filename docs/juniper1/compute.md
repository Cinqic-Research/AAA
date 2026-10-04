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
