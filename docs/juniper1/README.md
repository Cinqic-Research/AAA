# Juniper 1 adaptation research (`aaa.erudition.v0`)

AAA means **Accurate Autonomous Adaptation**. As of 2026-10-04 its active
research question is:

> Can Juniper autonomously detect when its Language Model or World Model is
> inadequate, determine which needs to change, cause it to learn from real
> evidence, verify that the change is better, keep useful improvements
> without destructive interference, roll back harmful changes, and keep doing
> so with minimal human intervention?

Juniper 1 has three model components and no Decision Model:

| Component | What it is here | Adapts through |
|---|---|---|
| **Language Model** | Juniper LM 1.1: OpenAI gpt-oss-20b, immutable and hash-pinned | a separate, versioned adapter of learned notes |
| **World Model** | `aaa.world.cbl.v0`: a context library of Bayesian dynamics models | new, updated or recalled contexts |
| **Erudition Model** | `aaa.erudition.model.v0`: a 1,259,700-parameter transformer, trained from scratch | decides whether, when, where and how the other two adapt |

Host software (validation, execution, the gate, the state store, monitoring
and rollback) is not a model and cannot be changed by the models.

## Result

Confirmed on gpt-oss-20b over 12 fresh streams under the freeze committed at
`a750677` ([development report, section 7](development_report.md#7-confirmation-real-model-frozen-protocol)):

| Frozen contract | Verdict |
|---|---|
| The Erudition Model, adapting both components, beats never adapting without damaging untouched tasks | `PROMOTE`: failure 0.615 → 0.365 |
| Joint adaptation beats World Model-only adaptation | `PROMOTE` |
| Joint adaptation beats Language Model-only adaptation | `INCONCLUSIVE` |
| The learned controller is no worse than the auditable rules on failure, poisoning and false adaptation | `INCONCLUSIVE` |

## Documents

| Document | What it covers |
|---|---|
| [Architecture](architecture.md) | the components, contracts, lifecycle, state and rollback, trust boundary, memory ownership, resource envelope |
| [Decisions](decisions.md) | each major choice: candidates, evidence, falsifier, risks, deferrals |
| [Literature](literature.md) | the primary sources behind the decisions, and what was deliberately not adopted |
| [Data](data.md) | every data source used or inventoried, licences, privacy, contamination controls |
| [Protocol](protocol.md) | the environment, splits, conditions, metrics, hypotheses, the freeze and confirmation rules |
| [Compute](compute.md) | hardware, durations, memory and the CPU/GPU schedule |
| [Development report](development_report.md) | every development stage, including the failures, generated from retained evidence |
| [Handoff](handoff.md) | the state of the work, what is and is not established, where a reviewer should attack first |

## Historical research

Every earlier Erudition line is historical evidence. Each is preserved
exactly as recorded and is not reinterpreted as evidence for this
architecture:
- the moving dot;
- AAA-1K with Champion 0 and Champion 1;
- `aaa.1k.v2`;
- `aaa.python.v0` and `aaa.python.v1`, with their 1K/4K/10K capacity
  conclusions;
- the Python capability ladder.

The same holds for the WM-S world model in `aaa.python.opaque.v0` and the
5.2M-parameter language experiment. See the
[research direction](../research_direction.md) for how they relate to the
current work.

## Reproduce

```bash
python -m unittest tests.test_aaa_erudition tests.test_aaa_erudition_promotion
python -m research.aaa_erudition.recompute docs/evidence/aaa_erudition_v0/development/records/*.json.gz
python tools/check_aaa_erudition_evidence.py --replay              # every stage, confirmation included; needs torch
python -m research.aaa_erudition.wm_diagnostic --trials 20      # needs requirements-torch-lock.txt
python -m unittest tests.test_aaa_erudition_model                 # needs requirements-torch-lock.txt
```

Running the real model needs the qualified GGUF and llama.cpp runtime of
the Juniper LM 1.1 qualification (a private repository). The
exact commands are in the [handoff](handoff.md).
