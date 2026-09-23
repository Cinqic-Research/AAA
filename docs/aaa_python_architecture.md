# `aaa.python.v0` learner architecture

This is the learner that exercises the [protocol](aaa_python_protocol.md). It is
a **pipeline instrument**, the smallest adaptive system that can validate
representation, causal prediction, online learning from feedback,
persistence, cloning, checkpointing and frozen-versus-adaptive comparison. It
is not an architecture result, and nothing about it is a claim about what
AAA's eventual Python learner should be. It does not reuse AAA-1K's 994-parameter GRU:
that core consumes three real-valued dot features, and forcing program text
through it would test the adapter rather than the question.

## Model

`OnlineLinear` ([`learners.py`](../research/aaa_python/learners.py)):

- **Representation** (an experimental variable, [`representation.py`](../research/aaa_python/representation.py)):
  hashed sparse features in D = 1,024 dimensions (BLAKE2b, so no fitted
  vocabulary and no out-of-vocabulary problem), L2-normalized, plus a constant
  bias feature.
  - `bytes`: UTF-8 byte 1-3 grams;
  - `lexical` (default): token type + lexeme unigrams and bigrams from a
    repository-owned regular-expression lexer. It does not use `tokenize`,
    whose behaviour changed in CPython 3.12;
  - `lexical_types`: token types only, an ablation of `lexical`;
  - `ast_nodes`: parent/child AST node-type pairs from CPython's parser.
    **Refused for `syntax`**, where it falls back to `lexical`: whether
    `ast.parse` succeeds *is* the answer key. The first development smoke
    scored 1.0 through that channel before the rule existed (`AAA-184`).
    Lone repair-candidate lines, which do not parse, also use `lexical`.
- **Heads**: one linear softmax head per family over the valid labels
  (localization masks line numbers beyond the program's length). Repair uses
  one linear scorer over candidate features: half the features of the program
  with that candidate substituted, half the features of the candidate line.
- **Update**: plain SGD on cross-entropy with learning rate 0.2, using the
  target the post-action feedback implies. Repair gets bandit feedback only: a
  logistic update on the *chosen* candidate toward 1 if it passed every hidden
  test, else 0.
- **Initialization**: N(0, 0.01) weights from a seed derived from
  `aaa.python.v0:init:<k>`. Training order is a seeded shuffle of the
  training pool, two epochs, entirely through the causal environment.

## Accounting (`python -m research.aaa_python audit`)

| | count |
|---|---:|
| trainable parameters | **153,600** |
| of which `output` head (101 classes x 1,024) | 103,424 |
| `localize` (40 x 1,024) | 40,960 |
| `outcome` (6 x 1,024) | 6,144 |
| `syntax` (2 x 1,024) | 2,048 |
| `repair` scorer (1 x 1,024) | 1,024 |
| optimizer state | 0 (plain SGD) |
| other persistent state | 1 (the update counter) |
| context memory | none; each task is scored from its own source |

Two thirds of the parameters are the output head's one-hot label space. That
is a representational cost of treating printed integers as classes, not
capacity the task has been shown to need. The feature cache (text to vector)
is a pure-function memo; it holds no learned state and is shared between
clones. Checkpoints are strict JSON (`aaa.python.v0.learner_state.v1`) and
restoration validates schema, configuration, shapes, finiteness and counter
type before replacing any state. `state_hash()` covers configuration, counter
and every weight byte.

## Compute

NumPy on the CPU, single process for learning. The oracle fans out to 8
sandboxed CPython processes. The declared development plan (3 initializations
x 4 representations x 5 families, adaptation and retention) takes about a
minute of wall time on FLOWBOX's Ryzen 7 5700G. Task generation dominates, and
training one learner takes about two seconds. Memory is small: one trained
state is 153,600 float64 weights (1.2 MB); a strict-JSON checkpoint is a few MB. `--device` accepts only
`cpu`, and nothing falls back silently. CUDA would add complexity for no
measured benefit at this size, so it is not offered. Run provenance records
the interpreter, NumPy and BLAS build, CPU, lock hashes, commit, dirty flag
and phase fingerprint, captured before the run (`AAA-179`).

## What would change this document

A new learner is a new versioned candidate with its own identity. It is
measured by the unchanged protocol against these same arms and baselines.
Capacity increases follow [scaling readiness](scaling_readiness.md): define the
deficiency, try parameter-neutral remedies, compare with smaller controls, and
report parameters, optimizer state, memory and compute together.
