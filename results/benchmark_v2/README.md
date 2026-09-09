# Benchmark v2 evidence

This directory is the tracked home for compact benchmark-v2 evidence summaries. Large per-step artifacts are generated under ignored `runs/benchmark-v2/<attempt-id>/` and must be retained as CI artifacts or another documented durable store when a confirmation attempt is used for a claim. Every attempt contains its own specification copy, source-tree identity, checkpoints, compressed step logs, metrics, checksums, gate decisions, and report.

The required benchmark is frozen in [`benchmarks/benchmark_v2.json`](../../benchmarks/benchmark_v2.json). Run `confirmation_a` and `confirmation_b` from the same committed selected source, with fresh attempt IDs and no intervening tuning. Failed attempts remain evidence and their seeds are retired.

## Recorded confirmation attempts

Both recorded attempts below ran from source commit `82d23b8e0d4ed170a6805ff8ee9bb87641353f34` with `dirty=false` and the same source-tree hash. Each used five independently trained replicas and 100 episodes per family; full raw step evidence remains in the corresponding ignored run directory and is intended to be retained as CI artifacts.

| Attempt | Gate result | Bounce events | Eligible/recovered dynamics events | CPU predict+update p95 |
|---|---|---:|---:|---:|
| `confirmation-a-20260909-clean` | all required gates PASS | 1,681 | 475 / 475 | 0.0234 ms |
| `confirmation-b-20260909-clean` | all required gates PASS | 1,679 | 480 / 480 | 0.0220 ms |

See each attempt's [`summary.json`](confirmation-a-20260909-clean/summary.json) and [`report.md`](confirmation-a-20260909-clean/report.md) for per-family means, p95, signed bias, worst replica, paired hierarchical intervals, coverage, and gate details.

The first clean A/B pair above is retained as development provenance. Its
gate calculations passed, but its straight helper used only a low-speed
training range and therefore did not exercise all declared speed strata. The
corrected confirmation pair below is the acceptance evidence:

| Corrected attempt | Gate result | Straight strata | Bounce events | Eligible/recovered dynamics events | CPU predict+update p95 |
|---|---|---:|---:|---:|---:|
| `confirmation-a-20260909-strata-clean` | all required gates PASS | 32 / 32 | 1,681 | 475 / 475 | 0.0220 ms |
| `confirmation-b-20260909-strata-clean` | all required gates PASS | 32 / 32 | 1,679 | 480 / 480 | 0.0241 ms |

The corrected reports and summaries are in the two corresponding directories.
