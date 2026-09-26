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
