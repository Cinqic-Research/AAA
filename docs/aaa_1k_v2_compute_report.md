# FLOWBOX compute report (aaa.1k.v2)

Generated from [`docs/evidence/aaa_1k_v2/compute_qualification.json`](evidence/aaa_1k_v2/compute_qualification.json) (sha256 `032273956288cc4a...`) by
`research/aaa_1k_v2/report.py`. Throughput is in cell-steps per second (one cell advancing one
online-learning step: predict, score, update), float64, AAA-1K Champion 1 unless stated. Every point is the
best of 3 timed runs after a warm-up run; CUDA timers synchronize the device before stopping.
The RTX 2060 also drives the desktop, so its numbers include that background load.

| Workload | CPU sequential | CPU parallel (8 workers) | CUDA (RTX 2060) | Winner | Notes |
|---|---:|---:|---:|---|---|
| single model / stream | 2,130 | n/a | 146 | CPU sequential | a lone cell cannot use parallel hardware |
| 32 cells | 36,423 | 36,282 | 4,679 | CPU sequential |  |
| 128 cells | 65,641 | 66,121 | 17,870 | CPU parallel |  |
| 512 cells | 79,200 | 188,071 | 67,261 | CPU parallel |  |
| 1,024 cells | 61,401 | 225,008 | 120,019 | CPU parallel |  |
| 4,096 cells | 55,340 | 264,521 | 273,714 | CUDA |  |
| hyperparameter search (4,096 cells over the lr grid) | 60,762 | 270,373 | 269,598 | CPU parallel |  |
| internal evaluation (online/frozen twins, 1,024 trunks) | 75,658 | 223,360 | 203,539 | CPU parallel |  |
| external benchmark batch (NARMA-10, 1,024 cells) | 80,031 | 361,707 | 188,150 | CPU parallel | 6,000-step sequences |
| prediction only (frozen, 4,096 cells) | 63,799 | 283,637 | 387,968 | CUDA |  |

## Crossover points

* `online`: CUDA beats one CPU process from 1024 cells, and the 8-worker CPU from 4096 cells.
* `prediction`: CUDA beats one CPU process from 1024 cells, and the 8-worker CPU from 4096 cells.
* `sweep`: CUDA beats one CPU process from 1024 cells, and the 8-worker CPU never in the measured range.
* `evaluation`: CUDA beats one CPU process from 512 cells, and the 8-worker CPU from 4096 cells.
* `external`: CUDA beats one CPU process from 1024 cells, and the 8-worker CPU never in the measured range.

## Resources

| Workload | Executor | Cells | Wall s (min / median) | CPU util. | GPU util. | Peak GPU memory MiB |
|---|---|---:|---:|---:|---:|---:|
| online | cpu_seq | 32 | 0.18 / 0.18 | 7% | 16 | 435 |
| online | cpu_seq | 512 | 1.29 / 1.30 | 7% | 16 | 434 |
| online | cpu_seq | 4096 | 14.80 / 14.97 | 7% | 37.2 | 434 |
| online | cpu_par | 32 | 0.18 / 0.18 | 7% | 35 | 422 |
| online | cpu_par | 512 | 0.54 / 0.55 | 49% | 34 | 422 |
| online | cpu_par | 4096 | 3.10 / 3.10 | 50% | 34 | 422 |
| online | cuda | 32 | 1.37 / 1.40 | 7% | 39 | 422 |
| online | cuda | 512 | 1.52 / 1.53 | 7% | 37.3 | 438 |
| online | cuda | 4096 | 2.99 / 3.00 | 7% | 52.9 | 558 |
| online | cuda | 16384 | 11.96 / 12.04 | 7% | 56.3 | 958 |
| external | cpu_seq | 32 | 4.39 / 4.39 | 7% | 39 | 958 |
| external | cpu_seq | 512 | 31.87 / 32.13 | 7% | 38.1 | 958 |
| external | cpu_par | 32 | 4.39 / 4.40 | 7% | 35 | 972 |
| external | cpu_par | 512 | 8.79 / 8.92 | 49% | 34.8 | 987 |
| external | cuda | 32 | 32.10 / 32.10 | 7% | 37.4 | 978 |
| external | cuda | 512 | 32.48 / 33.06 | 7% | 39.4 | 988 |

Peak resident memory: parent process 967 MiB, largest worker 967 MiB.
GPU memory figures are whole-device `memory.used` samples and include the desktop's resident allocation.

## Host-device transfer

| Cells | Stream bytes | Upload s | Download s |
|---:|---:|---:|---:|
| 32 | 51,456 | 0.00012 | 0.00004 |
| 512 | 823,296 | 0.00020 | 0.00015 |
| 4096 | 6,586,368 | 0.00129 | 0.00076 |
| 16384 | 26,345,472 | 0.00379 | 0.00257 |

Transfers are a one-off cost per job and negligible next to the per-step dispatch cost.

## CPU / CUDA parity and determinism

| Arm | Family | Cells | Bitwise CPU=CUDA | Max step diff | Max rel. MAE diff | Failure agreement | CPU repeat bitwise | CUDA repeat bitwise |
|---|---|---:|---:|---:|---:|---|---:|---:|
| c1_champion1 | long_coarse | 16 | 0 | 2.22e-16 | 3.82e-16 | True | 16 | 16 |
| c1_champion1 | v1_coarse_speed | 16 | 2 | 1.11e-16 | 5.30e-16 | True | 16 | 16 |
| c1_champion1 | v1_occlusion | 16 | 4 | 1.11e-16 | 3.75e-16 | True | 16 | 16 |
| c1_champion1 | wall_gaps | 16 | 0 | 1.11e-16 | 1.22e-15 | True | 16 | 16 |
| elman_v2 | long_coarse | 16 | 0 | 3.59e-02 | 3.21e-01 | True | 16 | 16 |
| elman_v2 | v1_coarse_speed | 16 | 0 | 2.61e-15 | 4.55e-14 | True | 16 | 16 |
| elman_v2 | v1_occlusion | 16 | 1 | 1.11e-16 | 6.28e-16 | True | 16 | 16 |
| elman_v2 | wall_gaps | 16 | 0 | 3.26e-05 | 7.48e-06 | True | 16 | 16 |
| gru_v1_keep-2 | long_coarse | 16 | 0 | 1.67e-15 | 1.38e-15 | True | 16 | 16 |
| gru_v1_keep-2 | v1_coarse_speed | 16 | 1 | 1.11e-16 | 3.80e-16 | True | 16 | 16 |
| gru_v1_keep-2 | v1_occlusion | 16 | 4 | 1.11e-16 | 2.72e-16 | True | 16 | 16 |
| gru_v1_keep-2 | wall_gaps | 16 | 0 | 1.11e-16 | 4.82e-15 | True | 16 | 16 |
| gru_v2 | long_coarse | 16 | 0 | 9.63e-14 | 7.50e-14 | True | 16 | 16 |
| gru_v2 | v1_coarse_speed | 16 | 0 | 1.11e-16 | 4.38e-16 | True | 16 | 16 |
| gru_v2 | v1_occlusion | 16 | 4 | 1.11e-16 | 2.89e-16 | True | 16 | 16 |
| gru_v2 | wall_gaps | 16 | 0 | 1.11e-16 | 7.39e-16 | True | 16 | 16 |
| lru_v2 | long_coarse | 16 | 0 | 2.42e-02 | 3.24e-01 | True | 16 | 16 |
| lru_v2 | v1_coarse_speed | 16 | 0 | 1.46e-02 | 1.19e-01 | True | 16 | 16 |
| lru_v2 | v1_occlusion | 16 | 2 | 3.77e-14 | 5.52e-14 | True | 16 | 16 |
| lru_v2 | wall_gaps | 16 | 0 | 5.43e-02 | 1.05e+00 | True | 16 | 16 |
| mgu_v2 | long_coarse | 16 | 0 | 2.00e-15 | 8.43e-16 | True | 16 | 16 |
| mgu_v2 | v1_coarse_speed | 16 | 2 | 1.11e-16 | 7.33e-16 | True | 16 | 16 |
| mgu_v2 | v1_occlusion | 16 | 4 | 1.11e-16 | 1.10e-15 | True | 16 | 16 |
| mgu_v2 | wall_gaps | 16 | 0 | 2.22e-16 | 9.03e-16 | True | 16 | 16 |

Run-to-run determinism: every cell bitwise identical on repeat, on both backends.

Cross-backend agreement: 19 of 24 arm-family groups agree to
better than 1e-9 relative MAE (worst of those 7.5e-14); the remaining groups diverge, the largest by
1.05e+00. Those are the Elman and LRU templates at learning rate 0.03 on long or wall-rich streams:
configurations development found at the edge of instability, where an online learner's trajectory is chaotic
and amplifies last-bit differences between cuBLAS/libdevice and OpenBLAS/glibc arithmetic into different
trajectories (the cross-CPU effect recorded as `AAA-173`). Failure classifications agree in every group.
Equivalence between backends is therefore *numerical* (about 1e-15) for stable configurations and only
*verdict-level* for chaotic ones; paired scientific comparisons never mix backends.

## When to use the Ryzen 7 5700G and when the RTX 2060

For AAA-1K-sized models the deciding quantity is the number of independent cells advancing in lockstep:

* one or a few cells, and any very long sequential stream: one CPU core;
* up to the crossover above: the 8-core CPU in 64-cell chunks;
* beyond it: the RTX 2060, capped by its float64 rate (1/32 of float32).

Larger future models do more arithmetic per step and move the crossover down; measure them when they exist.
