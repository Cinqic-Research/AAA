# AAA development hardware and compute strategy

This document records the machine AAA is currently developed on, what AAA can
use of it, what historical work actually used, the revisable long-term
model-size planning goal for this hardware generation (not a ceiling), and
the direction future compute is expected to take.

It is a **project and planning document**. Nothing in it is a scientific
result, a benchmark outcome, or a claim about what any model can do. Where it
describes future hardware, that hardware does not exist yet.

The [research charter](aaa_charter.md) asks how far a persistent artificial
agent can develop autonomous capabilities "using available research,
experimental evidence, and the hardware Cinqic actually has". This document is
where that last phrase is given a concrete current value.

Three things are deliberately kept apart throughout:

| Concern | Where it is recorded |
|---|---|
| the machine development happens on today | this document and [`benchmarks/hardware/flowbox.json`](../benchmarks/hardware/flowbox.json) |
| the platform a specific experiment actually ran on | that experiment's own recorded provenance (`aaa.compute.backend_provenance` from `aaa.1k.v2` on) |
| what the implementation requires of any machine | [`dependencies.md`](dependencies.md) and [`reproduction.md`](reproduction.md) |

## FLOWBOX — the current development workstation

**FLOWBOX** is the primary local AAA/Cinqic development workstation. The table
below was re-audited directly from the running machine on 2026-09-23 with
`python -m aaa.compute probe`; the machine-readable record, with the source of
every field, is [`benchmarks/hardware/flowbox.json`](../benchmarks/hardware/flowbox.json)
(49 machine-verified fields, 3 owner-declared, none unavailable). Serial
numbers, the GPU UUID, network identifiers, the hostname and the username are
deliberately not recorded.

| Component | Specification | Provenance |
|---|---|---|
| CPU | AMD Ryzen 7 5700G (Zen 3, family 25 model 80), 8 cores / 16 threads, 1 socket, 1 NUMA node, boost to 4.67 GHz | machine-verified (`lscpu`) |
| CPU caches | L1d 32 KiB x 8, L1i 32 KiB x 8, L2 512 KiB x 8, L3 16 MiB shared | machine-verified (`lscpu --caches`) |
| CPU vector ISA | SSE4.2, AVX, AVX2, FMA, F16C, BMI2, SHA; **no AVX-512** | machine-verified (`/proc/cpuinfo` flags) |
| GPU | NVIDIA GeForce RTX 2060 (TU106 Rev. A, PCI 10de:1f08), Gigabyte subsystem (1458:3fc1), 6,144 MiB, compute capability 7.5 | machine-verified (`nvidia-smi`, `lspci -nnk`) |
| GPU board model | Gigabyte GeForce RTX 2060 OC | owner-declared (the vendor is verified, the retail model name is not) |
| GPU link and power | PCIe 3.0 x16; power limit 170 W (maximum 200 W) | machine-verified |
| GPU role | also drives the desktop display: Xorg and Cinnamon keep about 0.5 GiB of VRAM resident and take some SM time during measurements | machine-verified (`display_active`) |
| Integrated GPU | AMD Radeon Vega (Cezanne); unused by AAA | machine-verified |
| NVIDIA driver | 595.91.07, supporting CUDA runtimes up to 13.2 | machine-verified |
| CUDA toolkit | **none installed** (no `nvcc`, no `/usr/local/cuda*`) | machine-verified |
| CUDA runtime used by AAA | 13.2 via NVIDIA's PyPI wheels (CUDA runtime and NVRTC 13.2.86, cuBLAS 13.4.1.3) under CuPy 14.2.0, in the optional `.venv-cuda` environment | machine-verified (CuPy runtime queries) |
| System RAM | 15.7 GiB usable (16 GB installed) | machine-verified (`/proc/meminfo`) |
| RAM modules | 16 GB DDR4-3000 CL16 | owner-declared (DIMM tables need root; not read) |
| Swap | 2 GiB swap file | machine-verified |
| Primary storage | Fanxiang S500Pro 256 GB NVMe SSD: EFI partition and the ext4 root filesystem (OS, repository checkout, virtual environments). **About 17 GB free at capture (93% used)** | machine-verified (`lsblk`, `df`) |
| Home directory | ecryptfs-encrypted overlay on the NVMe root; every repository I/O pays the encryption layer | machine-verified (`findmnt`) |
| Cinqic working storage | HGST HTS545050A7E380 500 GB SATA HDD, one ext4 partition labelled "Cinqic Storage", about 419 GB free at capture | machine-verified |
| Motherboard | ASRock B450M/ac R2.0; firmware American Megatrends P3.10 (2022-10-27) | machine-verified (`/sys/class/dmi/id`) |
| Operating system | Linux Mint 22.3 (Zena), kernel 7.0.0-31-generic, glibc 2.39 | machine-verified |
| Python / NumPy | Python 3.12.3; NumPy 2.5.3 on scipy-openblas 0.3.34 (DYNAMIC_ARCH, Haswell kernels, 64-bit integers) | machine-verified |
| Power supply | 650 W | owner-declared (not discoverable from software) |

Every earlier owner-declared value in this document agrees with what the
machine reports. The audit adds what was never recorded before: the exact
caches and vector ISA, the GPU's compute capability, link and power limit, the
driver, the absence of a CUDA toolkit, the encrypted home directory, the
display load on the GPU, and how full the system drive is.

The motherboard and power supply bound future upgrades; no measurement
depends on them. Cooling, case, networking and peripherals are omitted: they
do not affect an AAA compute or reproduction claim.

### Why the CPU in particular is recorded here

FLOWBOX's Zen 3 CPU is not incidental detail. Long online-learning
trajectories in this repository are not bit-stable across all CPU instruction
sets (`AAA-173`), and the post-audit loop evidence was produced on Zen 3. The
repository's reproduction material already depends on that fact; see
[Hardware provenance is per experiment](#hardware-provenance-is-per-experiment)
below. The absence of AVX-512 is part of why: AVX-512 runners (Zen 4) take
different BLAS kernels.

### Storage roles

The two drives have different jobs and different standing:

- the **256 GB NVMe SSD** carries the operating system, the repository and the
  virtual environments. It is nearly full, so large or disposable AAA data does
  not belong on it;
- the **500 GB ext4 HDD** is dedicated to Cinqic work: large experiment data,
  scratch environments (the PyTorch and dysts/River reference environments of
  the v2 backend evaluation live there), and downloaded external datasets. Its
  filesystem behaviour under an AAA workload was measured and retained at
  [`evidence/phase_closure_storage_profile.json`](evidence/phase_closure_storage_profile.json).

No AAA code hard-codes a path to either drive. External datasets resolve from
`$AAA_DATA_ROOT` (default `~/.cache/aaa/external`); on FLOWBOX it points at a
directory on the HDD. Durable scientific evidence is committed to the
repository, and the evidence policy's limitation stands.

The HDD is **verified local working storage and nothing more**. It is not
immutable, not off-site, and not an independent failure domain, so it is not a
publication-grade archive. That distinction is already load-bearing in
[`evidence_policy.md`](evidence_policy.md) and is tracked as `AAA-077`,
`AAA-134` and `AAA-144`; capacity does not change it.

## What AAA uses

### Historical work: CPU only

Every result produced before `aaa.1k.v2` came from CPU execution. That covers
v1, v2.1, observation noise, `aaa.1k.v1` rounds 1-3, loop iterations
0001-0006, Champion 0 and Champion 1. Their code never had a GPU path, and
their evidence is **not** GPU evidence and must never be described as such. A
benchmark attempt's `gpu` metadata field (from `nvidia-smi`) is host
provenance, on the same footing as the CPU model; it records no GPU
computation.

### From `aaa.1k.v2`: CPU or CUDA, chosen explicitly

New work runs one array program on either processor through `aaa.compute`
([`aaa_1k_v2_compute_strategy.md`](aaa_1k_v2_compute_strategy.md)):

- `--device cpu` (NumPy, always available), `--device cuda` or `cuda:<index>`
  (CuPy on the RTX 2060), or `--device auto`, which picks by a measured
  workload threshold. A formal or confirmatory run must name a resolved device
  before it observes anything;
- CPU work runs in single-threaded worker processes (`--workers`, default the 8
  physical cores) with BLAS oversubscription prevented;
- CUDA is optional. `requirements-lock.txt` is unchanged; the CUDA environment
  is `requirements-cuda-lock.txt`;
- CPU and GPU are separate numerical platforms. Paired comparisons use one
  backend, and cross-backend agreement is measured
  ([`aaa_1k_v2_compute_report.md`](aaa_1k_v2_compute_report.md)), never
  assumed.

### When each processor is the right one

For 1K-parameter models the answer is decided by how many independent cells
run in lockstep, not by model size. In brief, from the compute report:

- a single cell or a small batch runs fastest on one CPU core;
- up to about 2,000 cells run fastest on the 8-core CPU in 64-cell chunks
  (225k cell-steps/s at 1,024 cells);
- from 4,096 lockstep cells the RTX 2060 wins, narrowly for online learning
  (274k vs 265k cell-steps/s in float64) and clearly for frozen prediction
  (388k vs 284k). Its FP64 rate, 1/32 of FP32, caps it there;
- hyperparameter sweeps and external benchmarks stay on the CPU: CUDA never
  beat the 8 workers there in the measured range.

Larger future models do more arithmetic per step and shift the balance toward
the GPU; that has to be measured when they exist.

### `aaa.python.v0`: CPU only, by measurement

The Python phase's learner is a 153,600-parameter linear model trained one task
at a time, and its cost is dominated by the CPython oracle: one sandboxed
interpreter process per program, about 20 ms of CPU each, fanned out to 8
processes. A complete development run takes about a minute of wall time and
writes 1.4 MB of evidence (42 MB more with resumable checkpoints). There is
nothing for a GPU to accelerate, so `--device` accepts only `cpu`, and a CUDA
request is refused rather than silently served by the CPU. Development runs
write large or disposable output under `$AAA_DATA_ROOT` on the HDD. The NVMe
had about 15.7 GB free during this transition.

This says nothing about later Python learners. When one exists, measure
whether CUDA helps before offering it, and keep FLOWBOX's 16 GB RAM and 6 GB
VRAM as the constraint until an upgrade actually exists.

## Hardware provenance is per experiment

This document describes the *current* workstation. It does not describe, and
must never be read back onto, the platform any past experiment ran on.

- every benchmark attempt records its own CPU model, core count, memory, OS,
  Python, NumPy, BLAS metadata and available disk space;
- the evidence platform for the post-audit loop stages is AMD Zen 3, and
  `--exact` bitwise reproduction is claimed only there
  ([`reproduction.md`](reproduction.md));
- on other platforms the default standard is verdict-level reproduction:
  identical identities, structure, divergence classifications and adjudicated
  verdicts, with numerical drift reported (`AAA-173`);
- the `Loop full-stage reproduction` workflow replays those stages on hosted
  runners, so FLOWBOX is **not** the only machine on which this repository
  reproduces, and nothing here should be read as requiring the owner's
  hardware.

Historical evidence keeps the machine and platform provenance under which it
was actually produced. Upgrading FLOWBOX does not retroactively change the
environment that produced an earlier result.

## Current AAA 1 model-size goal

The current long-term planning goal for **AAA 1** is approximately **105
million trainable parameters**. This supersedes the earlier owner-selected
125M FLOWBOX planning ceiling as *current guidance*. The older entry in
[`CHANGELOG.md`](../CHANGELOG.md) remains a dated record of that decision.

105M is a revisable planning target, not a hard ceiling, required final count,
immediate next step, promotion criterion, or scientific finding. It does not
show that a particular 105M architecture fits or trains efficiently on
FLOWBOX. Hardware upgrades and a dedicated model server are plans, not present
resources.

The repository's existing discipline is unchanged and remains the stronger
constraint: **complexity must earn its keep** ([charter](aaa_charter.md), claim
10), and capacity increases go through the rules in
[`loop_protocol.md`](loop_protocol.md) — *capacity is an experiment, not a
reward*. A parameter increase requires a persistent measured failure, a
diagnosis that points specifically at capacity, parameter-neutral remedies
having failed, and a gain that transfers to fresh held-out evidence.

AAA may remain far below 105M. The current Champion 1 has 994 trainable
parameters, and the v2 capacity diagnosis on the tested dot/external mixture
was `NOT_CAPACITY_LIMITED` through roughly 4K parameters. A new Python coding
domain may demand a different representation and more capacity, but that must
be measured. Prefer a smaller model whenever it accomplishes the same research
objective. The parameter goal is not enforced by any scientific gate.

### Parameter count is not a compute budget

Trainable parameter count is only one term in whether a model is practical on
given hardware. Feasibility also depends on at least:

- numerical precision;
- model architecture;
- activation and recurrent-state memory;
- optimizer state;
- the training algorithm;
- sequence or trajectory length;
- batch size;
- checkpointing strategy;
- dataset and retained-evidence size;
- CPU versus GPU implementation;
- inference versus training requirements.

For scale only, 105M float32 parameters occupy about 420 MB (401 MiB). A
float32 gradient and two Adam moments raise parameter-related storage to about
1.68 GB (1.56 GiB), before master weights, activations, recurrent state,
sequence length, batch size, temporary kernels, data loading, checkpoints and
display-resident GPU memory. Those omitted terms can dominate 6 GB VRAM.
This arithmetic is an estimate, not a fit or throughput measurement. A
specific architecture and workload need a measured peak-memory and throughput
probe before a feasibility claim.

## Future FLOWBOX upgrades

FLOWBOX is not a permanently fixed platform. The owner plans multiple future
hardware upgrades. No dates, budgets, replacement components or target
specifications are decided, and none are asserted here.

Future evidence or hardware may justify another explicit goal. A change must
be recorded as a new project decision. It cannot retroactively change the
environment that produced historical evidence.

Nothing in this section describes hardware that is purchased, installed or
ordered.

## Planned dedicated model server

**This system does not exist.** It is a future compute and serving target with
no frozen specification.

The owner intends to eventually create a dedicated server or cloud system
specifically for running Juniper models and other appropriate AAA-derived
models. Its expected role:

- running models independently of the owner's primary workstation;
- providing persistent compute availability;
- serving Juniper models to compatible clients such as Juniper App;
- allowing weaker client devices to use models running on stronger dedicated
  hardware;
- eventually providing a deployment environment distinct from FLOWBOX
  development.

The description is deliberately architecture-neutral. CPU, GPU, RAM, storage,
operating system, networking topology, hosting provider, uptime expectations,
remote-access implementation, authentication design, cost and timing are all
undecided and are not stated anywhere in this repository.

This planned server is a **model-serving** target. It is not a solution to the
durable-archival limitations in [`evidence_policy.md`](evidence_policy.md)
(`AAA-077`, `AAA-134`, `AAA-144`); publication-grade evidence archival is a
different problem, and only a later design that explicitly addresses it would
change that.

## How the pieces relate

| Name | What it is |
|---|---|
| AAA | the research programme that develops and validates mechanisms |
| Juniper | the persistent agent the programme is ultimately intended to produce |
| Juniper App | the intended practical harness, runtime and interface for Juniper models |
| FLOWBOX | the current primary local development machine |
| the planned server | a future execution and serving environment; not built |

See the [research charter](aaa_charter.md) for what AAA and Juniper are, and
what this repository does and does not claim about either. Juniper App is
named here only to place the planned server; no Juniper App integration exists
or is in scope.

### Why the charter does not link back here

The link between this document and the charter runs one way on purpose.
`docs/aaa_charter.md` is one of the four protocol documents inside the frozen
`aaa.1k.v1` phase fingerprint (`research/aaa_1k/identity.py`), which Champion 0
and Champion 1 records both cite by hash. Editing the charter — even to add a
cross-reference — changes that fingerprint and invalidates those records. A
planning document is not a reason to disturb a scientific identity, so the
charter is left byte-identical and the reference is made from this side.
