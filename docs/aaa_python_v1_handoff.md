# `aaa.python.v1` handoff for independent review

The phase is complete through confirmation. **Nothing is promoted.** The
evidence supports a ~4K member of the v1 family as the next experimental
reference and does not support ~10K; that recommendation needs an independent
reviewer (a separate model session or a human) before anyone calls it
validated.

## Where everything is

| What | Where |
|---|---|
| Question, hypotheses, declared rules, amendments | [research brief](aaa_python_v1_research_brief.md) |
| Mechanism review (written before development) | [literature review](aaa_python_v1_literature_review.md) |
| Exam | [protocol](aaa_python_v1_protocol.md), `research/aaa_python_v1/data/aaa_python_v1.json` |
| Encoders, models, accounting | [architecture](aaa_python_v1_architecture.md) |
| Every number | [development report](aaa_python_v1_development_report.md) (generated) |
| Verdict and evidence chain | [scaling readiness](scaling_readiness.md) |
| Compute decision | [compute report](aaa_python_v1_compute_report.md) |
| Self-review (not independent) | [self-review](aaa_python_v1_self_review.md) |
| Defects | [issue ledger](issue_ledger.md) `AAA-192`..`AAA-204`; [errata](errata.md); [limitations](limitations.md) |
| Evidence | `docs/evidence/aaa_python_v1/`: pre-design diagnostics, the superseded censored run, ten stage documents, `freeze.json`, `confirmation.json`, `compute_benchmark.json` |

## Reproduce

From a clone on the HDD with the locked environment
(`python -m pip install -r requirements-lock.txt && python -m pip install -e . --no-deps`)
and `AAA_DATA_ROOT` on the HDD:

```bash
python -m research.aaa_python_v1 spec-hash        # 992aa2b0...
python -m research.aaa_python_v1 golden           # identical on CPython 3.10-3.13
python -m research.aaa_python_v1 audit            # parameter accounting of every arm
for f in docs/evidence/aaa_python_v1/stage_*.json docs/evidence/aaa_python_v1/confirmation.json; do
  python -m research.aaa_python_v1 summarize --evidence "$f"
  python -m research.aaa_python_v1 recompute --evidence "$f"
done
python -m research.aaa_python_v1 recompute --rerun --evidence docs/evidence/aaa_python_v1/stage_capacity.json
python tools/write_aaa_python_v1_report.py --check
```

To regenerate a stage, check out the commit its provenance records and run
`python -m research.aaa_python_v1 develop --stage <name> --previous <earlier
stage files> --output <path>`. Confirmation cannot be re-observed as fresh:
re-running `confirm` at the freeze commit `a73765b` reproduces the same
identities (a determinism check), which is not new evidence.

## What an independent reviewer should attack first

1. **`e2`'s design.** It was built after reading v0's failures. Look for any
   feature that encodes an answer rather than structure, for example static
   def-use facts that decide `outcome` without values.
2. **The capacity family.** Is a shared `tanh` layer over hashed features a
   fair test of "~10K"? Would a different ~10K architecture change C3?
3. **The one-hot output head.** It dominates every size's parameter count.
4. **Post-hoc material.** The 4K adaptation and plasticity comparison and the
   4K-over-1K contract were added after seeing development data; check that
   they could not have been chosen to produce the verdict.
5. **The plasticity probe** mixes plasticity with interference.
6. **Statistics.** Crossed percentile bootstrap at 10 x 30; Holm per contrast;
   the Jeffreys-smoothed error ratio as the contract metric.
7. **Everything the self-review lists as still weak.**
