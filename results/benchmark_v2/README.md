# Benchmark v2 evidence — SUPERSEDED

> **These attempts are not acceptance evidence.** They are historical and
> provisional evidence produced under benchmark v2, a methodology since found
> to contain defects that changed what several of its gates meant. The active
> protocol is [v2.1](../../docs/benchmark_protocol.md).
>
> In particular, the reported "all required gates PASS" result includes gates
> that could not have failed: a parameterless rule satisfied the learning gate,
> an empty stratum set passed the coverage gate, the reproducibility gate
> performed no computation, the bounce comparison gave the candidate
> information the baseline was denied, and the recovery rate counted only
> events whose 50-transition average stayed elevated.
>
> Every defect is reproduced in
> [`docs/issue_ledger.md`](../../docs/issue_ledger.md) and the affected claims
> are listed in [`docs/errata.md`](../../docs/errata.md).
>
> Nothing here has been deleted. The original text follows, with
> corrections marked inline as **[CORRECTION]** rather than rewritten away.

---

This directory is the tracked home for compact benchmark-v2 evidence summaries. Large per-step artifacts are generated under ignored `runs/benchmark-v2/<attempt-id>/` and must be retained as CI artifacts or another documented durable store when a confirmation attempt is used for a claim. Every attempt contains its own specification copy, source-tree identity, checkpoints, compressed step logs, metrics, checksums, gate decisions, and report.

The required benchmark is frozen in [`benchmarks/benchmark_v2.json`](../../benchmarks/benchmark_v2.json). Run `confirmation_a` and `confirmation_b` from the same committed selected source, with fresh attempt IDs and no intervening tuning. Failed attempts remain evidence and their seeds are retired.

## Recorded confirmation attempts

Both recorded attempts below ran from source commit `82d23b8e0d4ed170a6805ff8ee9bb87641353f34` with `dirty=false` and the same source-tree hash. Each used five independently trained replicas and 100 episodes per family; full raw step evidence remains in the corresponding ignored run directory and is intended to be retained as CI artifacts.

> **[CORRECTION]** "Independently trained" is inaccurate for the A/B
> relationship. Confirmation A and B trained from identical `development`
> streams, so their checkpoints were identical; the `role` parameter threaded
> into training was accepted and never used. The five replicas within an
> attempt were independent of each other, but A and B were not independent of
> one another. Evaluating one frozen selected model set on two independent test
> streams is a defensible design, and v2.1 declares it explicitly rather than
> arriving at it by accident.

| Attempt | Gate result | Bounce events | Eligible/recovered dynamics events | CPU predict+update p95 |
|---|---|---:|---:|---:|
| `confirmation-a-20260909-clean` | all required gates PASS | 1,681 | 475 / 475 | 0.0234 ms |
| `confirmation-b-20260909-clean` | all required gates PASS | 1,679 | 480 / 480 | 0.0220 ms |

See each attempt's [`summary.json`](confirmation-a-20260909-clean/summary.json) and [`report.md`](confirmation-a-20260909-clean/report.md) for per-family means, p95, signed bias, worst replica, paired hierarchical intervals, coverage, and gate details.

The first clean A/B pair above is retained as development provenance. Its
gate calculations passed, but its straight helper used only a low-speed
training range and therefore did not exercise all declared speed strata. The
corrected confirmation pair below is the acceptance evidence:

> **[CORRECTION]** The pair below is **not** acceptance evidence. It was
> produced under the same superseded methodology. The training range was one
> contributing factor, but the underlying defect was that the coverage gate
> could not detect missing strata at all: `all()` over an empty collection is
> `True`, so zero required strata passed. Separately, the v2 training range
> (0.02-0.06) did not match the v2 confirmation range (0.08-0.20), which makes
> those results a speed extrapolation rather than ordinary held-out
> generalization. See [`docs/errata.md`](../../docs/errata.md).

| Corrected attempt | Gate result | Straight strata | Bounce events | Eligible/recovered dynamics events | CPU predict+update p95 |
|---|---|---:|---:|---:|---:|
| `confirmation-a-20260909-strata-clean` | all required gates PASS | 32 / 32 | 1,681 | 475 / 475 | 0.0220 ms |
| `confirmation-b-20260909-strata-clean` | all required gates PASS | 32 / 32 | 1,679 | 480 / 480 | 0.0241 ms |

The corrected reports and summaries are in the two corresponding directories.
