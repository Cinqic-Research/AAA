# Evidence and artifact policy

## What lives in git

| Committed | Why |
|---|---|
| compact per-attempt `summary.json` and `report.md` | the result, readable without regenerating anything |
| `metadata.json`, `benchmark_spec.json` | provenance and the exact resolved specification |
| `checksums.json` | integrity of everything the attempt produced |
| `experiment_registry.json` | per-trial state, seeds and completion |
| selected checkpoints (a few kB of JSON) | the model the claim is about |
| `benchmarks/freeze_manifest.json`, `benchmarks/confirmation_batches.json` | what was frozen, and which batches were spent |
| `benchmarks/golden_seeds.json` | a fixture that detects accidental RNG-mapping changes |
| development evidence under `docs/evidence/` | diagnosis and candidate-selection tables, including unsuccessful variants |

## What does not

Compressed per-step record files (`raw/**/*.jsonl.gz`) and the plot images.
A full confirmation attempt writes roughly 100 MB of per-step records. The
previous repository committed tens of megabytes of generated JSON while calling
it compact evidence; this policy is the correction.

## How raw evidence stays recoverable

Raw step records are **regenerable**, not archived, from four committed things:

1. public source at a recorded commit and tree hash;
2. the immutable specification, identified by hash;
3. the immutable seed mapping, keyed by batch identity;
4. the checkpoint lineage, identified by checkpoint hash.

```bash
git checkout <recorded commit>
python -m aaa.cli benchmark --role confirmation_a \
  --batch-id <recorded batch id> --reproduce --output-root runs
python -m aaa.cli recompute runs/benchmark-v2_1/<batch id>
```

The reproduction must yield the recorded checksums. If it does not, that is a
finding, and `recompute` exits non-zero.

CI additionally retains the full attempt directory for 90 days.

## Known limitation

Regenerability is weaker than durable archival. A sufficiently unlucky
combination — the recorded commit lost and the CI artifact expired — would
leave the summaries and checksums without the bytes they describe.

Git LFS, or an external archive with content addresses recorded here, would be
stronger. Neither is in place. This is stated as a recommendation rather than
described as done, and it is tracked as `AAA-077` in the issue ledger.

## Historical evidence

`results/final/` and `results/benchmark_v2/` are preserved unchanged. They are
historical provenance. See [`errata.md`](errata.md) for which of their claims
were affected and why.
