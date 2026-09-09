# Benchmark v2 evidence

This directory is the tracked home for compact benchmark-v2 evidence summaries. Large per-step artifacts are generated under ignored `runs/benchmark-v2/<attempt-id>/` and must be retained as CI artifacts or another documented durable store when a confirmation attempt is used for a claim. Every attempt contains its own specification copy, source-tree identity, checkpoints, compressed step logs, metrics, checksums, gate decisions, and report.

The required benchmark is frozen in [`benchmarks/benchmark_v2.json`](../../benchmarks/benchmark_v2.json). Run `confirmation_a` and `confirmation_b` from the same committed selected source, with fresh attempt IDs and no intervening tuning. Failed attempts remain evidence and their seeds are retired.
