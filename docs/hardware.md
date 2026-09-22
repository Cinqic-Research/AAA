# AAA development hardware and compute strategy

This document records the machine AAA is currently developed on, what the
current implementation actually uses of it, the model-size ceiling the project
owner has adopted for this hardware generation, and the direction future
compute is expected to take.

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
| the machine development happens on today | this document |
| the platform a specific experiment actually ran on | that experiment's own recorded provenance |
| what the implementation requires of any machine | [`dependencies.md`](dependencies.md) and [`reproduction.md`](reproduction.md) |

## FLOWBOX — the current development workstation

**FLOWBOX** is the primary local AAA/Cinqic development workstation. Its
current configuration:

| Component | Current specification |
|---|---|
| CPU | AMD Ryzen 7 5700G — 8 cores, 16 threads, Zen 3 |
| GPU | Gigabyte GeForce RTX 2060 OC — NVIDIA RTX 2060, 6 GB VRAM |
| System RAM | 16 GB DDR4-3000 CL16 |
| Primary storage | 256 GB NVMe SSD — Linux installation and normal development environment |
| Cinqic working storage | 500 GB HDD, ext4, dedicated to Cinqic work |
| Operating system | Linux Mint 22.3 Cinnamon |
| Motherboard | ASRock B450M/ac R2.0 |
| Power supply | 650 W |

The motherboard and power supply are recorded because they bound what future
upgrades are possible, not because any measurement depends on them. Cooling,
case, networking hardware, peripherals and cosmetic components are omitted
deliberately: they do not affect an AAA compute or reproduction claim.

### Why the CPU in particular is recorded here

FLOWBOX's Zen 3 CPU is not incidental detail. Long online-learning
trajectories in this repository are not bit-stable across all CPU instruction
sets (`AAA-173`), and the post-audit loop evidence was produced on Zen 3. The
repository's reproduction material already depends on that fact; see
[Hardware provenance is per experiment](#hardware-provenance-is-per-experiment)
below.

### Storage roles

The two drives have different jobs and different standing:

- the **256 GB NVMe SSD** carries the operating system and the ordinary
  development environment;
- the **500 GB ext4 HDD** is dedicated to Cinqic work and is used as larger
  local working storage for AAA runs whose retained evidence does not fit
  comfortably on the system drive. Its filesystem behaviour under an actual
  AAA workload was measured and retained at
  [`evidence/phase_closure_storage_profile.json`](evidence/phase_closure_storage_profile.json).

The HDD is **verified local working storage and nothing more**. It is not
immutable, not off-site, and not an independent failure domain, so it is not a
publication-grade archive. That distinction is already load-bearing in
[`evidence_policy.md`](evidence_policy.md) and is tracked as `AAA-077`,
`AAA-134` and `AAA-144`; capacity does not change it.

## What the current AAA implementation uses

**The current validated AAA research implementation is CPU-only.** NumPy is
its numerical foundation. All current AAA-1K and improvement-loop evidence was
generated and reproduced under CPU execution. No GPU is required and none is
used; see [`dependencies.md`](dependencies.md).

FLOWBOX containing an RTX 2060 does not change any of that:

- **no AAA model trains on the RTX 2060.** No CUDA, GPU array library, or
  GPU-only dependency is present in `pyproject.toml` or
  `requirements-lock.txt`;
- **no existing AAA evidence was GPU-generated.** The implementation has never
  had a GPU code path, so every retained result in this repository comes from
  CPU execution;
- **no CUDA reproduction standard exists**, because there is nothing to
  reproduce on a GPU;
- **CPU and GPU execution are not interchangeable** for this project's
  purposes. Given that long trajectories already drift between CPU
  microarchitectures (`AAA-173`), a future GPU implementation would be a
  separate numerical platform requiring its own reproduction standard, not a
  faster way to run the same numbers.

A benchmark attempt records a `gpu` field in its environment metadata, from
`nvidia-smi` when it is available. That is provenance about the host, on the
same footing as the CPU model and BLAS build. It is not a record of GPU
computation and the runner never dispatches work to a GPU.

GPU support would be future engineering work: a separate phase that implements
it, demonstrates it, and separates its numerical behaviour from the existing
CPU evidence rather than inheriting it. None of that is designed, and listing
it here is not a commitment to do it.

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

## The current model-size planning ceiling

**For the current version of FLOWBOX, the project owner does not plan to
develop AAA/Juniper models larger than approximately 125 million trainable
parameters.**

This is a self-imposed practical development ceiling for the present hardware
generation. It is stated so that planning has a concrete upper bound. It is
explicitly **not**:

- a scientific finding or a benchmark result;
- a statement that AAA needs 125M parameters;
- a target AAA is expected to reach, or a schedule for reaching it;
- proof that a 125M-parameter model fits on this hardware for any given
  workload;
- proof that such a model could be trained efficiently here;
- a permanent limit on the Juniper architecture.

The repository's existing discipline is unchanged and remains the stronger
constraint: **complexity must earn its keep** ([charter](aaa_charter.md), claim
10), and capacity increases go through the rules in
[`loop_protocol.md`](loop_protocol.md) — *capacity is an experiment, not a
reward*. A parameter increase requires a persistent measured failure, a
diagnosis that points specifically at capacity, parameter-neutral remedies
having failed, and a gain that transfers to fresh held-out evidence.

In practice that means AAA may stay far below the ceiling indefinitely. The
current champion is a 994-parameter recurrent core, roughly five orders of
magnitude below 125M, and nothing about the ceiling suggests that number should
grow faster than the evidence justifies. A smaller model that accomplishes the
same objective remains preferable. **125M is a ceiling, not a destination.**

The ceiling is not a promotion criterion and is not enforced by any test or
gate. Encoding an arbitrary parameter threshold as a scientific check would be
weaker than the discipline above, not stronger.

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

So 125M is an **owner-selected upper planning bound**, not a promise that every
possible 125M-parameter architecture or training regime is practical on present
FLOWBOX hardware. Some much smaller configurations will be infeasible here;
establishing that a specific configuration fits requires measuring it.

## Future FLOWBOX upgrades

FLOWBOX is not a permanently fixed platform. The owner plans multiple future
hardware upgrades. No dates, budgets, replacement components or target
specifications are decided, and none are asserted here.

The consequences for this document:

- the 125M ceiling applies to the **current** FLOWBOX configuration;
- future hardware may justify revisiting that ceiling;
- any change to the ceiling is a new project decision and is documented as one,
  with the configuration it applies to;
- changing hardware does not retroactively change the environment that produced
  historical evidence, which keeps its original provenance.

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
