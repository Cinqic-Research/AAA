# Benchmark v2.1 confirmation evidence

Compact evidence for each formal confirmation attempt under the active
protocol. Raw per-step records are regenerable rather than committed; see
[`docs/evidence_policy.md`](../../docs/evidence_policy.md).

**Failed attempts are kept.** A confirmation that records a required-gate
failure is evidence, its batch is retired permanently, and it stays here. The
batch lifecycle is in
[`benchmarks/confirmation_batches.json`](../../benchmarks/confirmation_batches.json).

Each attempt directory holds its `summary.json`, generated `report.md`,
provenance `metadata.json`, the resolved `benchmark_spec.json`,
`checksums.json` for everything the attempt produced, the five selected
checkpoints plus their budget snapshots, the learning-curve and latency
metrics, and the executed `verification/` results.

## Recorded attempts

### Round 1

| Attempt | Role | Required gates | Outcome |
|---|---|---|---|
| `aaa-v2_1-confirmation-a-0001` | confirmation_a | **1 of 14 failed** | `always_online_stability` FAIL; batch retired |
| `aaa-v2_1-confirmation-b-0001` | confirmation_b | **1 of 14 failed** | `always_online_stability` FAIL; batch retired |

Both attempts failed the same required gate on independent streams, so the
finding replicates. Thirteen gates passed in both, including changed-law
adaptation, recovery and every frozen track.

| | A | B |
|---|---:|---:|
| `candidate_online` normalized MAE | 4.427e-05 | 3.414e-05 |
| `constant_motion_reflected` | 6.621e-06 | 7.012e-06 |
| limit (1.1x baseline + 1e-5) | 1.728e-05 | 1.771e-05 |
| worst replica | 1.519e-04 | 8.784e-05 |

The diagnosis is in [`docs/issue_ledger.md`](../../docs/issue_ledger.md) under
`AAA-120`. No threshold was altered in response.

### Round 2

The candidate was repaired on development evidence after round 1 (`AAA-120`),
which changed the specification hash. A batch declared against one hash is
refused under another, so round 2 required newly declared batches.

| Attempt | Role | Required gates | Outcome |
|---|---|---|---|
| `aaa-v2_1-confirmation-a-0002` | confirmation_a | see `summary.json` | recorded |
| `aaa-v2_1-confirmation-b-0002` | confirmation_b | see `summary.json` | recorded |

See [`docs/handoff_astra.md`](../../docs/handoff_astra.md) for the full
cross-attempt table and the interpretation.
