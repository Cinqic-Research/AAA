# The dot-era benchmark and evidence archive

From 2026-09-09 to 2026-09-23, AAA's research centred on one moving dot on a
line. That work is **not obsolete**. It is where AAA built its scientific
machinery: gates that can say no, causal boundaries, crossed designs, frozen
identities, independent recomputation, retained failures and the improvement
loop. Its evidence stays in place, byte for byte, as:

- **historical evidence** of what was and was not established;
- **reproducibility targets**, whose identities are checked on every commit;
- **regression and mechanistic benchmarks** for future adaptive learners;
- **methodological case studies**, including the defects found in them;
- **proof that AAA reports negative and inconclusive results.**

It stopped being the centre of current research on 2026-09-23. AAA's active
specialization is now [Coding, beginning with Python](research_direction.md).
Nothing was moved, renamed or regenerated to make that transition look clean.

## Lineage

| Phase | Where it lives | Scientific status | What it established | What failed or was superseded |
|---|---|---|---|---|
| **v1** online linear SGD (5 parameters) | [`results/final/`](../results/final/), `aaa/experiment.py` (the `legacy_linear_sgd` arm) | historical, preserved unchanged | a learner can lose to an analytic baseline, and that is a result | did not convincingly improve, generalized poorly frozen, lost to constant motion |
| **v2** benchmark introduction | [`results/benchmark_v2/`](../results/benchmark_v2/), [superseded protocol](benchmark_v2_superseded.md) | **superseded**, not acceptance evidence | the first gated protocol | three reviews found gates that could not fail ([pre-repair probes](evidence/pre_repair_probes.json), `AAA-001`-`AAA-100`) |
| **v2.1** repaired benchmark, 3-parameter square-root RLS | `aaa/benchmark/`, [`results/benchmark_v2_1/`](../results/benchmark_v2_1/), [protocol](benchmark_protocol.md) | specification hash `f8e1090b...`; all declared batches spent or retired | a benchmark that says no; online adaptation, numerically soundly | round 1 failed `always_online_stability` on both fresh streams (`AAA-120`) and both batches were retired; `a-0003` was retired under `AAA-127`; historical archives name omitted raw bytes (`AAA-124`) |
| **Observation noise v1 / v1.1** | `aaa/noise/`, [protocol](observation_noise_protocol.md), `benchmarks/observation_noise_*.json` | protocol hash `546e2434...`; full development selection retained; formal A/B **NOT EXECUTED**, declared batches retired unobserved | causal noisy-target updates, matched branches, sharded evidence, an independent verifier | v1 superseded (`AAA-142`); no refinement beat the incumbent; the replication budget had no precision justification |
| **Original neural experiment** (PR #12 TinyMLP, 17 parameters) | recorded in the [charter](aaa_charter.md) and [AAA-1K decisions](aaa_1k_decisions.md) D-1; the branch was closed, not merged | historical | online neural updates beat the same network frozen | still lost badly to simpler methods |
| **AAA-1K** `aaa.1k.v1` (994-parameter GRU, Champion 0) | [`research/aaa_1k/`](../research/aaa_1k/), [report](aaa_1k_report.md), [self-review](aaa_1k_self_review.md), [handoff](aaa_1k_handoff.md) | fingerprint `5ce6e019...`; round 3 is the latest retained v1 round, rounds 1-2 superseded | online learning beats a frozen twin; persistent state helps on memory families | round 1's adaptation estimate did not isolate adaptation (`AAA-153`); rounds 1-2 intervals flattened initializations (`AAA-155`, `AAA-160`); the `coarse_speed_v1` characterization was refuted (`AAA-162`) |
| **Improvement-loop pilot** `aaa.loop.v0-pilot`, iterations 0001-0003 | [`research/aaa_1k_loop/`](../research/aaa_1k_loop/), [pilot report](loop_pilot_report.md), `docs/evidence/aaa1k_loop_0001`-`0003` | retained v0 artifacts under `aaa.loop.v1` governance | the loop can reclassify, diagnose and reject | eight candidates, eight rejections; never reached fresh confirmation |
| **Loop iterations 0004-0006**, **Champion 1** | [report](loop_report_0004_0006.md), [handoff](loop_0004_0006_handoff.md), `docs/evidence/aaa1k_loop_0004`-`0006` | promoted through fresh confirmation and independent recomputation; [PR #20 review](pr20_independent_review.md) established `aaa.loop.v1` | the long-horizon runaway (M2) is a target-unfolding frame lock (`AAA-170`); Champion 1 removes it (21% to 0%) | three candidate fixes rejected; confirmation attempt 1 burned its identities before observing anything (`AAA-171`) |
| **AAA-1K v2** `aaa.1k.v2`, the final 1K pass | [`research/aaa_1k_v2/`](../research/aaa_1k_v2/), [report](aaa_1k_v2_report.md), [self-review](aaa_1k_v2_self_review.md), [handoff](aaa_1k_v2_handoff.md) | fingerprint `e5b11bdb...`; confirmation `NO_CHALLENGER` | Champion 1 remains the 1K reference; capacity verdict `NOT_CAPACITY_LIMITED` to about 4K parameters; retention resolves favourably | round 3's adaptation effect did not replicate on fresh identities; the K5 path is defective (`AAA-180`) and forbidden for promotion |
| **External time-series benchmarks** (NARMA-10/20, Mackey-Glass, 12 dysts systems) | `research/aaa_1k_v2/external/`, [external benchmarks](aaa_1k_v2_external_benchmarks.md) | descriptive | Champion 1 beats persistence on all 15 synthetic tasks and AR-RLS on 11 | not a general forecasting claim |
| **Monash experiments** (4 datasets) | `research/aaa_1k_v2/external/monash.py`, the v2 confirmation evidence | descriptive | online per-series learning measured against published offline methods | not competitive with the best published methods; the `aus_elec_demand` mapping is by name only |

Independent reviews and self-reviews are retained alongside the work they
reviewed: [Sol review](sol_review.md), [self-review](self_review.md),
[AAA-1K Sol review](aaa_1k_sol_review.md),
[independent review 2026-09-22](independent_review_2026-09-22.md) and
[pre-scale review 2026-09-23](pre_scale_review_2026-09-23.md). Every defect
is in the [issue ledger](issue_ledger.md), and every affected earlier claim is
in [errata](errata.md). The two annotated tags `aaa-pre-next-phase-2026-09-19`
and `aaa-pre-next-phase-closure-2026-09-19` mark trees that reproduce the
pre-AAA-1K evidence exactly.

## What is protected, and how

[`benchmarks/protected_identities.json`](../benchmarks/protected_identities.json)
records the v2.1 specification hash, the observation-noise v1.1 protocol hash,
the `aaa.1k.v1` and `aaa.1k.v2` fingerprints, and the SHA-256 of 493 retained
files: all of `docs/evidence/` and `results/`, the frozen registries and
ledgers, and the historical reports, reviews, decision logs and handoffs.
`python tools/check_protected_identities.py` recomputes each one and fails on
any drift; CI runs it on every push. A change to that file is a change to
historical identity and needs a recorded decision in the ledger.

The repository-wide observation-noise source fingerprint is not in that list.
It covers every tracked file and moves whenever any file is added. That is by
design (`AAA-152`), and it is why historical observation-noise work reproduces
from its tagged tree.

## Verifying the archive

```bash
python tools/check_protected_identities.py         # every protected identity and byte
python -m aaa.cli spec-hash                         # v2.1 specification
python -m aaa.cli observation-noise-protocol-hash   # observation-noise v1.1 protocol
python -m research.aaa_1k parameter-audit           # 994 / 982 / 954
python -m research.aaa_1k gradient-check --full
python -m research.aaa_1k fingerprint               # 5ce6e019...
python -m research.aaa_1k recompute --evidence docs/evidence/aaa_1k_evaluation_round3.json
python -m research.aaa_1k_loop champion --verify    # Champion 0
python -m research.aaa_1k_loop validate
python -m research.aaa_1k_loop.champion1 verify     # Champion 1
python -m research.aaa_1k_loop.recompute4 --confirmation docs/evidence/aaa1k_loop_0006/confirmation_2.json \
    --freeze docs/evidence/aaa1k_loop_0006/freeze_2.json
python -m research.aaa_1k_v2 audit
python -m research.aaa_1k_v2 fingerprint            # e5b11bdb...
python -m research.aaa_1k_v2 registry
python -m research.aaa_1k_v2 prove-fresh
python -m research.aaa_1k_v2 recompute --freeze docs/evidence/aaa_1k_v2/freeze.json \
    --confirmation docs/evidence/aaa_1k_v2/confirmation.json
python tools/write_aaa_1k_v2_report.py --check
python tools/check_adaptation_parity.py
python tools/check_promotion_successor.py           # AAA-180, retained and contrasted with the successor
```

All of these run in the required `Locked environment` CI job. Full re-execution
of loop iterations 0004-0006 runs in the weekly and path-triggered
`loop-reproduction.yml` workflow (bitwise on the Zen 3 evidence platform,
verdict-level elsewhere, `AAA-173`). Complete reproduction commands per phase
are in [reproduction](reproduction.md).

## How to use the dot era now

- **As a regression suite.** Any adaptive learner that can read the dot
  interface can be run against the v2.1 families and the 26 v2 stress
  families. Their baselines (reflected constant motion, dead reckoning, RLS,
  Champion 1) stay first-class.
- **As mechanistic benchmarks.** Paired-change adaptation, retention probe
  banks and long-horizon stability are measured designs with known failure
  modes. Their methodology is reused in `aaa.python.v0`'s paired-shift design
  and probe bank, without pretending code and dot dynamics are one domain.
- **As a methodology record.** The ledger is the most useful document here.
  Most of its entries are ways a measurement looked fine and was not.

Frozen phases are never edited to fit new work. A later phase that needs
different code copies it into a new versioned package, as `aaa.1k.v2` did and
as the `aaa.promotion.crossed.v1` successor to the v2 K5 path does.
