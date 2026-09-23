# AAA-1K v2 decision log

Every decision of the `aaa.1k.v2` phase, in the order it was taken, with the
evidence available at the time. Entries are appended, never rewritten; a later
entry may supersede an earlier one and says so.

## V2-D1. A new phase, not an edit of `aaa.1k.v1`

`research/aaa_1k/`, the four shared `aaa` modules, `tests/test_aaa_1k.py`,
`requirements-lock.txt` and the four v1 protocol documents are fingerprinted by
`aaa.1k.v1` and cited by hash by Champion 0 and Champion 1. v2 therefore lives
in `research/aaa_1k_v2/` and `aaa/compute/`, imports the historical generators
and agents unchanged, and has its own fingerprint, identity namespace, protocol
and evidence. The v1 fingerprint was `5ce6e019...f771e` before this work and
must still be after it.

## V2-D2. The CUDA backend is CuPy; the CPU backend stays NumPy

Measured, not assumed; see [`aaa_1k_v2_compute_strategy.md`](aaa_1k_v2_compute_strategy.md).
CuPy runs the same code as NumPy and matches it to about 1e-15. On the RTX 2060
in float64, CuPy, PyTorch eager and PyTorch with CUDA Graphs saturate within
about 10% of each other (roughly 530-590k cell-steps/s), so PyTorch's 3.5 GB of
dependencies and second implementation buy almost nothing. CUDA is optional:
`requirements-cuda-lock.txt` is a separate lock.

## V2-D3. Batching across cells is the unit of acceleration

A single 1K-parameter cell cannot use a GPU: one lockstep step costs about
6.5 ms of CuPy dispatch whatever the batch size. The useful parallelism is
across independent cells (initializations x streams x hyperparameters). The
engine is therefore batched from the start, with per-cell hyperparameter
vectors, and the same batching also speeds up the CPU path: a single process
runs 70-95k cell-steps/s against about 3.7k for the historical per-cell code (2,920 cell-steps in 0.79 s in the parity run).

## V2-D4. CPU parallelism uses small chunks

Measured on FLOWBOX: eight workers with 512-cell chunks slowed each worker by a
factor of five, because the working set overflowed the 16 MB L3 and saturated
DRAM. 64-cell chunks gave the best throughput, about 270k cell-steps/s. BLAS
threads are pinned to one per worker, and workers are spawned after the pin.

## V2-D5. Development and confirmation run on the CPU

At development's job shapes (one core type and one horizon per lockstep batch,
about 1-2 thousand cells) the parallel CPU path is faster than the GPU, which
overtakes it only above roughly 2,000-4,000 lockstep cells. Confirmation runs
on the CPU because exact reproduction is established there on the Zen 3
evidence platform. CUDA is exercised for backend parity, the compute
qualification and attack criterion A6. This follows the brief's rule against
forcing CUDA into workloads where the CPU is faster.

## V2-D6. Champion 1 is re-implemented in batched form and proven equal

The reference arm in every v2 comparison is the batched GRU with the v1
feature set and the reach-gated target rule. `tests/test_aaa_1k_v2.py`
requires it to match `research.aaa_1k` + `ReachGatedUnfoldAgent` within 1e-12
per step. The qualification run found most cells bitwise identical and the rest
differing in the last bit (batched products sum in a different order from
OpenBLAS's matrix-vector kernel). Initial parameters are bitwise identical.

## V2-D7. Six candidates, each with a written hypothesis

See [`aaa_1k_v2_research_brief.md`](aaa_1k_v2_research_brief.md). `gru_v1_retuned`
was added so that no architecture comparison is confounded by Champion 1's
v1-selected learning rate: Champion 1 stays frozen as the reference, and its
architecture also competes under the v2 selection rule.

## V2-D8. The explicit missingness flag is paid for inside the cap

Adding an input to the 16-unit GRU gives 1,042 parameters. The `v2` GRU uses 15
units (932). The Elman and MGU candidates are sized to the largest width under
the cap. The flag's contribution is measured by a `no_observed_flag` ablation.

## V2-D9. The diagonal linear candidate learns by exact RTRL and its traces count

The LRU's 240 trace scalars are adaptive state and appear in its footprint
(1,245 total against Champion 1's 1,414 at T=4). The RTRL gradient is verified
against finite differences of the untruncated loss.

## V2-D10. External benchmarks: rule-selected, externally defined, causal

NARMA-10 (Atiya & Parlos) and NARMA-20 (Rodan & Tino, tanh); NARMA-30
excluded because its published constants conflict. Mackey-Glass tau = 17 per
Jaeger's protocol. Twelve dysts systems drawn by a quartile rule with a
registered seed; RK4 integration validated against dysts itself. Four Monash
datasets selected by a written rule from the archive's table, with checksums,
the archive's own MASE semantics (ported from its R code) and its published
baselines. River's drift generators are tabular and multi-feature, so the River
track is rejected, with its inspection retained as evidence. Gymnasium and
MLPerf Tiny are out of scope for a predictor.

## V2-D11. Normalization never sees scored data

Synthetic tasks use calibration realizations from registered development
identities. Monash series use their own warm-up window, and scoring starts after
it. This avoids the full-series normalization leak the brief warns about, even
inside the training split.

## V2-D12. Identities are 64-bit and proven disjoint

Closes the gap `AAA-163` described for new work: no modular reduction, and a
fail-closed set-intersection proof against every AAA-1K seed below index
100,000, AAA-1K and loop evidence, and the loop ledger. The initial registry
(267 blocks) proved disjoint.

## V2-D13. Disclosed pre-freeze looks

Before this protocol was committed, three engineering shakedowns produced
numbers:

1. NARMA-10 and Mackey-Glass runs of Champion 1, `lru_v2` and `elman_v2` on
   ad-hoc unregistered seeds (5, 6, 11, 12), used to check the series pipeline.
2. A scratch-identity run of the complete development pipeline for `gru_v2` and
   `lru_v2` (2 inits x 2 streams per family), used to check selection and the
   screen end to end.
3. Throughput and parity measurements on qualification-style streams.

No threshold, grid value, candidate, family or criterion was chosen or changed
in response to their outcomes. The only protocol edits after them added the
qualification and scratch external blocks and moved development to the CPU
(V2-D5), both on compute grounds. They are disclosed because the protocol
requires every look to be recorded.

## V2-D14. The first development run was stopped; every stage now runs from a pinned worktree

The first development run (launched at `621e2a9`, 2026-09-23 02:04 local)
was stopped after about 15 minutes, before writing any artifact, because two
provenance defects were found in it:

1. `develop` captured the environment (commit, dirty flag, source fingerprint)
   *after* the run, so edits made to the working tree during the run would have
   been recorded as the code that produced the evidence.
2. `run_jobs` spawns fresh worker processes for every call, and each re-imports
   the package from disk. Workers started after stage code was edited during
   the run (an optional optimizer branch, the attack's weight-scale hook) ran
   the edited modules. The SGD path is designed to be numerically unchanged,
   but that is a claim, not evidence, so the run was discarded as a record.

Repair: the CLI captures provenance before anything runs (tracked as
`AAA-179`), and every stage from here on runs from a separate `git worktree`
checked out at a named commit, so no edit can reach a running stage.
Development identities may be observed more than once, so rerunning is
legitimate. The aborted run's log and its one completed raw archive (for
`gru_v1_retuned`) are kept outside the repository. The rerun must reproduce
that archive exactly (a determinism check).

**Disclosed look.** Before it was stopped, the aborted run printed one result:
`gru_v1_retuned` selected `lr=0.01; T=4; live; no clip` with a development
score of 1.27 (27% worse than Champion 1's geometric mean). Nothing was changed
in response. The protocol was already frozen, and the rerun computes the same
thing.

## V2-D15. Development: no challenger, and why

Development ran at `ac905e7` from a pinned worktree (2026-09-23 08:18-08:52,
2,004 s, CPU, 8 workers; `docs/evidence/aaa_1k_v2/development.json`, sha256
`0d5a327d...`). Its raw per-grid-point archives are outside the repository with
recorded hashes. The rerun reproduced the aborted run's `gru_v1_retuned`
archive byte for byte (sha256 `75bd0742...` both times), which checks run-to-run
determinism and the SGD-path equivalence discussed in V2-D14.

Consistency checks: `gru_v1_retuned` at Champion 1's exact configuration
scores exactly 1.000 against Champion 1, with zero unstable cells; Champion 1
has zero failed or diverged development cells.

Result under the preregistered rules: **no candidate passed the screen**, so
there is no challenger. Selected configurations score 1.17-1.30 (development
geometric mean relative to Champion 1; lower is better), and every candidate
fails S2 (v1 non-inferiority) and S3 (stress superiority).

The binding constraint is the stability-margin rule. Each architecture's
*unclipped* reference configuration diverged on some development cells at
lr 0.1 (Elman: once, on NARMA-10, at 0.03). The rule therefore capped eligible
learning rates at 0.01 (Elman 0.003), below Champion 1's frozen 0.03. Several
ineligible configurations, at lr 0.03-0.1 with clipping, score 0.84-0.97,
i.e. better than Champion 1 on development. Under this v2 rule Champion 1's own
configuration would be ineligible too; it is the frozen reference, not a
re-selected candidate.

This is recorded as a finding, not repaired. Changing the margin rule after
seeing which configurations it excludes would be tuning a rule to a result.
What it shows is that the v2 protocol's stability rule is stricter than the v1
rule under which Champion 1 was selected. The v1 rule measured the unclipped
boundary on short streams, where it sat at 0.3; the v2 development streams
include 2,000-step and noisy streams, which move the unclipped boundary to 0.1.
A future phase that wants to test the high-learning-rate, clipped
configurations must predeclare a different stability rule (for example, a
margin measured with the clip in place, or a long-horizon stability gate
instead of a learning-rate margin) and confirm them on fresh identities. They
are *not* evidence of improvement here. They are development observations
excluded by a rule written in advance.

Consequences, as the protocol prescribes: no attack stage runs; the
confirmation is a characterization-only run (no promotion is possible) that
measures Champion 1 and the other candidates on fresh identities, held-out
families and the external suite; Champion 1 remains the phase's final 1K system.
