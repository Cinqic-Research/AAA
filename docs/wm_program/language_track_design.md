# American-English language track: design and pre-registered rules

Written 2026-09-26, **before** any adapter or language-conditioned result existed. At writing
time only the tokenizer experiment and the rule baseline's extraction accuracy were known. Status:
development.

## Scope decision

Markus authorized American English as AAA's first natural-language scope in the 2026-09-26
program instructions. Only American English is in scope, and no multilingual capability is claimed.
Code syntax is not evidence of another natural language. The promotable language model is
trained **from scratch** (random initialization). Historical documents that list language as future
work (`docs/research_direction.md`) are not rewritten. This file and the program index record the
new decision.

## Corpus (`research/aaa_wm/lm/corpus.py`, manifest under `$AAA_DATA_ROOT/lm/corpus/`)

About 1.79 GB of training text from five sources, each with an allocated byte budget:

| Source | License |
|---|---|
| US federal public-domain text (usgpo_filtered) | 17 U.S.C. §105 |
| FineWeb-Edu | ODC-By |
| Cosmopedia stories and wikihow | Apache-2.0 |
| Simple English Wikipedia | CC BY-SA 3.0 / GFDL, with attribution |
| Python 3.12 documentation | PSF |

Processing steps:

- NFC normalization and English-likeness filters.
- A project-written US/UK spelling lexicon, which drops documents whose British-only spellings
  outnumber American ones (at least two hits).
- Exact and MinHash near-duplicate removal across all sources.
- Hash-based train/dev/test splits (98/1/1).
- The download manifest records URL, time, SHA-256 and size for every raw file.
- Excluded sources: TinyStories and OpenWebText (unresolved terms) and the BabyLM corpus (license unknown).

## Tokenizer experiment (decided)

The arms were bytes (vocabulary 257) and byte-level BPE-8K trained on a 100 MB train-split sample.
Both used the same non-embedding shape (d = 256, 4 layers, context 512) and matched FLOPs:

- the BPE head costs about 1.6x more per token;
- bytes ran 3,000 steps and BPE ran 1,900.

The pre-registered rule chose bytes only if they came within 0.03 bits per byte of BPE *and* won on
robustness or a goal probe.

Result, dev bits per byte:

| Seed | Bytes | BPE-8K |
|---:|---:|---:|
| 0 | 1.818 | 1.604 |
| 1 | 1.808 | 1.578 |

**BPE-8K is selected.**

## Size ladder

- **R1:** d256, 4 layers, about 5.2M parameters, 100M tokens.
- **R2:** d384, 6 layers, about 13M parameters, 260M tokens.
- **R3:** d512, 8 layers, about 27M parameters. **Only if** R2 improves the adapter's held-out extraction
  over R1 by more than seed noise *and* by at least 0.05 absolute. A 27M model is permitted, not a goal.

## Language-conditioned task (`research/aaa_wm/lang/`)

An `opaque.v0` task whose three visible tests are stated only in an American-English specification
or bug report. The program, the runs, the budgets and success (domain equivalence) are unchanged.

- **Phrasing families.** 12 training clause templates and 8 disjoint held-out templates, with
  different openers and joiners.
- **Numbers.** Digits or American number words, chosen per mention.
- **Held-out difficulty.** The held-out families include order inversion ("*e* is what f should
  return when it is given *x*") and distractor values ("running f on *x* gave *g*; the right
  answer is *e*").
- **Split rule.** Training statements use train tasks and train families only. Evaluation uses
  development (later, fresh confirmation) tasks and held-out families only.

## Language channels (each supplies the only statement of the tests that an agent sees)

| Channel | What it is |
|---|---|
| `none` | no language understanding: no tests, prior-guided search verified by real runs |
| `rules` | a hand-written parser built from the training phrasings (number words, clause split, first number as input, last as expected). Exact extraction: 1.00 on training phrasings, **0.13** on held-out |
| `adapter:random` | the LM architecture, randomly initialized, fine-tuned on the same training statements for the same steps |
| `adapter:pretrained` | the from-scratch pretrained LM, fine-tuned identically |
| `gold` | perfect extraction (a ceiling, not an arm) |

## Factorial (the decision system crossed with the language channel)

Decision systems: `wms:online` (AAA plus the world model), `policy_aux:plan`, and `policy_aux` (AAA
without the world model). With the `none` channel the neural policies fall back to prior-ordered tool
search, because they cannot act without tests.

Pre-registered comparisons (development first, confirmation later under a separate freeze):

- **L1 (pretraining):** `adapter:pretrained` against `adapter:random`, on held-out extraction and on
  end-to-end success with `wms:online`.
- **L2 (language over rules):** `adapter:pretrained` against `rules`, end to end with `wms:online`.
- **L3 (no regression):** `wms:online` with numeric tests (the non-language benchmark) is unchanged by
  the presence of the language track. The language components never touch the non-language path.
- **Full system:** WM + LM against the pairwise systems (WM + rules, policy + LM, policy + rules),
  under the definitions of the success rules in the program brief.

Language results are **conditional on the world-model result**. The language track is used to
demonstrate integration only after the world-model confirmation.

## Development results, adapter v1 (added 2026-09-26 after the runs; development tune range, one seed each)

| LM | Dev bits per byte | Adapter: held-out exact | Random twin: held-out exact | Training phrasings exact (pretrained / random) |
|---|---:|---:|---:|---|
| R1: d256, 4 layers, 5.2M params, 100M tokens | 1.332 | **0.252** | 0.044 | 0.973 / 0.897 |
| R2: d384, 6 layers, 13.8M params, 262M tokens | 1.137 | 0.102 | 0.114 | 0.980 / 0.923 |
| rules (hand-written, training phrasings) | - | 0.130 | - | 1.000 |

**Reading.**

- At R1, pretraining helps generalization to unseen phrasings (0.252 against 0.044) and beats the rule
  parser.
- At R2 the benefit **does not replicate** (0.102 against 0.114), even though R2 is the better
  language model by bits per byte.
- Every fine-tuned adapter nearly memorizes the training phrasings: the fine-tuning loss falls to
  about 1e-3 in 3,000 fixed steps. The likeliest cause is over-fitting to the 12 training templates,
  which a stronger model does faster. A capacity limit is not the likely cause.

**Pre-registered rule applied.** R3 (about 27M) is **not justified**. R2 does not improve held-out
extraction over R1.

**Next (adapter v2, to be written before it runs).**

- A third, disjoint *selection* family of phrasings, used only to choose the number of fine-tuning
  steps and never for evaluation.
- Three seeds per arm.
- The comparison of pretrained against random twin is repeated at both rungs.

The v1 numbers above are retained as they are.

## Adapter v2: pre-registered protocol (written 2026-09-27, before any v2 run)

v1 fine-tuned for a fixed 3,000 steps and over-fitted the training phrasings. v2 changes only
how the fine-tuning length is chosen. Everything else is as in v1.

- **Selection family.** Six new clause templates, with their own openers and joiners
  (`reports.CLAUSES["select"]`). They are disjoint from the train and held-out families and are
  written in the same register. Selection statements come from the development *tune* tasks (the
  held-out evaluation also uses tune tasks, with held-out phrasings; the two never share phrasing).
- **Length choice.** Checkpoints at 250, 500, 1,000, 2,000 and 3,000 fine-tuning steps. The choice is
  the checkpoint with the best exact extraction on the selection family, with ties going to fewer
  steps. The held-out family is scored only at the chosen checkpoint.
- **Arms.** {R1, R2} x {pretrained, random twin} x seeds {0, 1, 2}: 12 runs, the same optimizer,
  learning rate and batch as v1.
- **Decision rules (development).**
  - Pretraining is *supported* at a rung if the pretrained mean held-out exact extraction exceeds the
    random-twin mean by at least 0.05, with all three paired seed differences positive.
  - R3 stays unjustified unless R2 beats R1 by at least 0.05 (pretrained means, all three seeds).
  - The better supported rung becomes the language channel for the end-to-end factorial.

## Adapter v2 results (development tune tasks, held-out phrasings, exact extraction of all three tests)

| LM | Pretrained (seeds 0, 1, 2) | Mean | Random twin (seeds 0, 1, 2) | Mean | Difference |
|---|---|---:|---|---:|---:|
| R1 (5.2M) | 0.257, 0.313, 0.284 | **0.285** | 0.033, 0.051, 0.016 | 0.033 | +0.251, all seeds positive |
| R2 (13.8M) | 0.296, 0.329, 0.201 | **0.275** | 0.074, 0.048, 0.010 | 0.044 | +0.231, all seeds positive |
| rules (hand-written) | - | 0.130 | - | - | - |

Chosen fine-tuning lengths (selection family only): R1 1000/250/3000 steps; R2 500/500/1000;
random twins mostly 1000-3000.

Pre-registered rules applied:

- **Pretraining is supported at both rungs.** The v1 R2 anomaly (0.102 against 0.114) was an
  over-fitting artifact of the fixed 3,000-step fine-tune, and the selection family removes it.
- **R3 stays unjustified.** R2 does not beat R1 (0.275 against 0.285).
- **The language channel for the factorial is R1** (the better supported rung, and the smaller one).
  The adapter seed *s* pairs with WM-S table seed *s*.
