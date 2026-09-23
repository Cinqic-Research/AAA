# AAA compute strategy: CPU and CUDA for new AAA work

This document records how AAA runs on FLOWBOX's CPU and GPU from `aaa.1k.v2`
onward, why the CUDA backend is CuPy, and what the device means scientifically.
The measured throughput, crossover points, parity and determinism results are
in [`aaa_1k_v2_compute_report.md`](aaa_1k_v2_compute_report.md). The machine
is described in [`hardware.md`](hardware.md) and
[`benchmarks/hardware/flowbox.json`](../benchmarks/hardware/flowbox.json).

## Scope

* **Historical work is untouched.** `aaa.1k.v1`, the loop, Champion 0 and
  Champion 1 were CPU/NumPy research and stay exactly that. None of their code
  imports `aaa.compute`, and none of their evidence is relabelled.
* **New work chooses a device explicitly.** `aaa.compute.resolve_backend`
  accepts `cpu`, `cuda`, `cuda:<index>` and `auto`. Requesting CUDA where it is
  unavailable raises; it never falls back silently. `auto` picks by a measured
  workload threshold and is refused for frozen or confirmatory runs, which must
  name the device they resolved to before observing anything.
* **One array program.** The v2 cores and adapters take `xp` (NumPy or CuPy)
  and never branch on the device.

## Backend options evaluated

| Option | Transparency | Dependency cost | CPU/CUDA parity | Batching | Measured on the RTX 2060 (float64 GRU online step) |
|---|---|---|---|---|---|
| **CuPy 14.2** (`cupy-cuda13x`) | same NumPy code path; no second implementation | about 0.7 GB (CuPy plus NVIDIA runtime wheels), MIT | agrees with NumPy to about 1e-15 relative on Champion 1 cells | per-cell batched arrays | saturates near 530k cell-steps/s at 4,096 cells or more; about 6.5 ms per lockstep step below that (dispatch-bound) |
| PyTorch 2.14 eager | a second, differently shaped API | about 3.5 GB with its CUDA libraries, BSD/Apache mix | would need its own implementation and parity tests | yes | saturates near 590k cell-steps/s; 2.6 ms per step floor |
| PyTorch 2.14 + CUDA Graphs | as above | as above | as above | yes | same 590k ceiling; the per-step floor falls to 0.5 ms, which only matters at small batches where the CPU is faster anyway |
| Array API standard (`array-api-compat`) | would let one program target several libraries | small | no parity gain over CuPy's NumPy compatibility | yes | not needed: CuPy already implements the NumPy calls AAA uses |
| Custom fused CUDA kernels | lowest: a second implementation in CUDA C | none | needs its own parity proof | yes | not built; the ceiling analysis below says float64 on this GPU would cap it near a few million cell-steps/s |

**Decision: CuPy.** For AAA-1K the arithmetic is tiny. In float64 the RTX 2060
(FP64 at 1/32 of its FP32 rate, so roughly 0.2 TFLOP/s) caps any framework
near the same throughput, and PyTorch's extra speed is small next to its
dependency weight and a second implementation to keep in parity. CuPy keeps
one transparent code path, the smallest optional dependency, and the closest
CPU/CUDA agreement. PyTorch is not a dependency. It was installed only in a
scratch environment on the HDD to measure it.

FP32 roughly doubled PyTorch's ceiling (about 1.6M cell-steps/s), which shows
the workload is memory- and dispatch-bound rather than compute-bound. All AAA
scientific runs stay float64. FP32 is recorded as an exploratory number only.

## When to use which processor

| Workload shape | Use | Why |
|---|---|---|
| one or a few cells, any length | CPU, single process | about 2,100 cell-steps/s for one cell; CUDA's per-step dispatch floor makes it about 15 times slower (146) |
| 32 to about 2,000 lockstep cells | CPU, 8 single-threaded workers in 64-cell chunks | cache-resident chunks scale across the 8 cores; 188k at 512 cells, 225k at 1,024 |
| 4,096 or more lockstep cells, online learning or frozen prediction | CUDA, narrowly | online 274k vs 265k on the 8-worker CPU; prediction 388k vs 284k |
| hyperparameter sweeps and external benchmarks | CPU, 8 workers | CUDA never beat the 8-worker CPU in the measured range (external 1,024 cells: 362k vs 188k) |
| very long sequential streams (e.g. 230k steps) | CPU | latency-bound: each step's cost is fixed overhead |

The micro-benchmark ceiling of about 530k cell-steps/s quoted above was a bare
learner loop on an idle device. The qualification numbers here include scoring,
failure bookkeeping and the desktop's load on the same GPU, and are the ones to
plan with.

`aaa.compute.device.AUTO_CUDA_MIN_CELLS` encodes the single-process crossover
measured in the compute report.

## Dependency isolation

`requirements-lock.txt` is part of `aaa.1k.v1`'s scientific identity and is
unchanged. `requirements-cuda-lock.txt` repeats every base pin and adds CuPy
and NVIDIA's runtime wheels (CUDA runtime and NVRTC 13.2.86, cuBLAS 13.4.1.3).
`python tools/check_lock.py --lock requirements-cuda-lock.txt` verifies a CUDA
environment. The CPU installation needs nothing new. No system CUDA toolkit is
required; the NVIDIA driver must support CUDA 13.x.

## Provenance

Every v2 artifact records `aaa.compute.backend_provenance`: requested and
resolved device, processor, array library and version, dtype, NumPy and its
BLAS build, CPU model, GPU name, compute capability, memory, NVIDIA driver,
driver-supported CUDA version, the CUDA runtime CuPy uses, NVRTC and cuBLAS
versions, CuPy's reduction accelerators, both lock hashes and the machine
profile hash. The device is part of the scientific environment.

## Numerical policy

* **Paired comparisons use one backend.** No study compares an arm run on the
  CPU with an arm run on the GPU.
* **Historical exactness stays exact.** Batched Champion 1 on the CPU matches
  the historical implementation within 1e-12 per step (tested); the historical
  stage reproductions stay bitwise on Zen 3.
* **Cross-backend equivalence is measured, not assumed.** The compute report
  defines tolerances from measured behaviour and records which cells, if any,
  differ in verdict.
* **Determinism.** Each backend is tested for bitwise run-to-run repeatability
  on the same hardware.

## CI

GitHub-hosted runners have no NVIDIA GPU. CPU CI verifies everything else,
including device parsing, CUDA-absent errors, invalid indices, `auto`
behaviour, provenance and the privacy of the hardware profile, using injected
loaders. CUDA qualification runs on FLOWBOX and is retained as a
hardware-specific artifact. Nothing claims that hosted CI tests CUDA. A future
self-hosted GPU runner could run the `CudaBackendTests` and the parity tests
without making a private machine necessary for ordinary verification.
