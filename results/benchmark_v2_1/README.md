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

| Attempt | Role | Required gates | Outcome |
|---|---|---|---|
| `aaa-v2_1-confirmation-a-0001` | confirmation_a | **1 of 14 failed** | `always_online_stability` FAIL; batch retired |

See [`docs/handoff_astra.md`](../../docs/handoff_astra.md) for the full
cross-attempt table and the interpretation.
