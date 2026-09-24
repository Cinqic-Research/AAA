# `aaa.python.v1` learner family and encoders

    TaskView --encoder (fixed, 0 parameters)--> x in R^256 --core--> z --heads--> action

Code: [`encoders.py`](../research/aaa_python_v1/encoders.py),
[`models.py`](../research/aaa_python_v1/models.py). Every component is a
separately versioned experimental variable, and no comparison changes two at
once.

## Encoders (`aaa.python.v1.encoders.v1`)

All encoders read only the issued `TaskView`, emit signed-hashed (BLAKE2b-64,
top bit as sign), L2-normalized sparse vectors in D = 256, and have **zero
trainable parameters and zero adaptive state**. Behind the same core and heads,
therefore, every encoder has exactly the same trainable parameter count, and an
encoder comparison is matched in capacity by construction.

| Encoder | Role | Features |
|---|---|---|
| `e0` | minimal control | token unigrams (type = lexeme) from v0's lexer: order-free |
| `e1` | v0-tied baseline | v0's `lexical` features (unigrams + bigrams), signed hashing, no hashed bias |
| `e2` | structural | `alpha`: e1's n-grams after alpha-renaming identifiers to first-occurrence roles and bucketing literals >= 10; `line`: parser-free line structure (first/last token shape, indentation modulo 4 and its change after a header, running bracket depth, quote parity, final depth); `flow` (never for `syntax`): static def-use (how each used name was last defined: literal, expression, augmented, loop variable, parameter, list, call, undefined; with line distance), operator/operand kinds, subscripts, calls, conditional expressions, comparisons, statement kind and nesting |

`e1`'s relation to v0 is exact except for the sign bit and the bias: v0 hashed
its bias into the shared space and L2-scaled it (`AAA-192`'s sibling finding in
the hashing diagnostic); in v1 every bias belongs to the model.

**What `e2` does not do.** It never evaluates or propagates values: constant
propagation over these programs would compute the answer. It never parses a
`syntax` task (whether the parser succeeds is that family's answer key,
`AAA-184`). For a repair task, if any patched candidate program failed to
parse, *all* candidates of that task lose the `flow` channel, so parse success
cannot single out a candidate. Static facts a linter reports -- a name used
before any definition, a floor division by an expression -- are legitimate
pre-action information; whether they *fire* depends on values.

Per-line vectors (for the pointer head) contain the line's own features plus
two position features (index from the start and from the end). Position
features are identical for every encoder, so position cannot favour one.

## Core

`hidden = 0`: linear (`z = x`). `hidden = H`: `z = tanh(W x + b)` with
`W ~ N(0, 1)` (inputs have unit norm, so pre-activations start near unit
variance) and `b = 0`, shared by every family and by every line and candidate.
`H` is the capacity variable.

## Heads

| Head | Families | Parameters (Z = H, or 256 when linear) |
|---|---|---|
| one-hot softmax (v0's formulation) | syntax (2), outcome (6), output (101), localize (40, masked to the program's lines) | classes x (Z + 1) |
| ordinal discretized Gaussian | output | 2 x (Z + 1): mean (x10) and log-scale over -50..50 |
| pointer (shared line scorer) | localize | Z |
| candidate scorer, bandit logistic update on the chosen candidate | repair | Z + 1 |

Heads start at zero, so an untrained model predicts uniformly (the Gaussian
head starts at mean 0, scale 8).

## Learning

Online SGD on the negative log-likelihood of the target the post-action
feedback implies, one task at a time; for repair, the logistic loss of the
chosen candidate's hidden-test pass. Options, each a separate arm: L2 weight
decay on weights, heavy-ball momentum (velocity counted as adaptive state),
global-norm clipping. A training epoch is one causal pass over the whole
training pool in an initialization-seeded order shared by every arm.

## Accounting

`CoreModel.accounting()` reports trainable parameters per block, optimizer
state, the update counter, and the remembered initial state that the
`reset_each_task` control restores. There is no persistent recurrent state,
replay memory or eligibility trace. The encoders' fixed tables (the lexer's
eleven keywords and ten builtin names) are listed by
`encoders.accounting()`; no learned lookup table exists anywhere. The parameter
counts for the evaluated sizes are recorded in the development evidence and in
the [development report](aaa_python_v1_development_report.md).

## State, cloning and serialization

`clone()` copies every array, including the optimizer velocity. Encodings are
cached as read-only arrays, so a clone cannot alter another's view of an input
(the v0 audit showed writable shared caches could, in principle). State is
strict JSON (`aaa.python.v1.model_state.v1`); `load_state` validates schema,
the complete configuration (including `init_scale`, which v0's did not cover),
block names, shapes, finiteness and counter type before replacing anything;
`state_hash` covers the configuration, counter, parameters and velocity.

## Compute

NumPy on the CPU. The declared development workload trains a model at about
5,000 online updates per second on FLOWBOX (Ryzen 7 5700G). See the
[compute report](aaa_python_v1_compute_report.md) for the measured CPU/CUDA
decision.
