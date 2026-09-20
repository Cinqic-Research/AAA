# AAA-1K architecture specification

> **Independent-review correction (2026-09-20).** The network architecture and
> selected hyperparameters remain unchanged. Evaluation round 3 changes only
> claim identification and uncertainty: Q1 uses a matched frozen-weight copy;
> Q2/Q5 fully cross initializations with environments. Round 2 is retained as
> superseded evidence.

Phase identity: `aaa.1k.v1`. Benchmark family version: `aaa.1k.benchmarks.v1`.
Model format: `aaa.1k.gru.v1`. Architecture id: `AAA1KGRU-3x16x2`.

This is the frozen specification. Anything here that changes requires a new
phase version *before* any fresh evaluation stream is observed.

## Complete persistence boundary

The 994 trainable parameters are not the complete agent state. Exact resume and
branch identity also cover the 16-value hidden state, bounded TBPTT cache,
previous signed error, observation tracker including pending gaps, pending raw
and scored prediction, error estimate, counters, public scales, configuration,
and update diagnostics. Only the branch treatment labels (`name` and
`update_enabled`) are excluded from the equality hash because they intentionally
differ between online and frozen arms. Deserialization rejects unknown fields,
wrong shapes, non-finite cache values, oversized buffers, boolean/negative
counters, impossible gaps, and inconsistent pending predictions before state is
mutated.

## 1. The model

A hand-written NumPy gated recurrent unit. Three inputs, sixteen hidden units,
two outputs.

### Why not a framework

At this size a framework costs transparency and buys nothing. Writing the
forward and backward passes out gives exact parameter accounting, an
inspectable hidden state, gradients checkable against finite differences,
deterministic CPU execution, and no new dependency. AAA already depends on
NumPy.

### Frozen GRU convention

For input `x_t` (3 values) and previous hidden state `h_{t-1}` (16 values):

```text
z_t = sigmoid(W_z x_t + U_z h_{t-1} + b_z)
r_t = sigmoid(W_r x_t + U_r h_{t-1} + b_r)
n_t = tanh(W_n x_t + U_n (r_t * h_{t-1}) + b_n)
h_t = z_t * h_{t-1} + (1 - z_t) * n_t
o_t = W_o h_t + b_o
```

`z_t` is the **keep** gate: it weights the *old* state. This polarity is used
identically in the forward pass, the backward pass, the tests, this document
and the serialized state. Nothing mixes conventions.

One bias vector per gate, not PyTorch's two.

### Parameter count: 994

```text
3 * (I*H + H*H + H) = 3 * (3*16 + 16*16 + 16) = 3 * 320 = 960
H*O + O             = 16*2 + 2               =            34
                                               total     994
```

994 is intentional. Padding the model with six meaningless learned constants to
reach a round 1,000 would be parameter-count theatre.

`python -m research.aaa_1k parameter-audit` recomputes this three ways — from
the array sizes, from the formula, and against the declared constant — and
exits non-zero if they disagree.

### Complete adaptive state footprint

| Category | Scalars |
|---|---|
| trainable parameters | 994 |
| hidden state | 16 |
| **optimizer state** | **0** |
| TBPTT buffer at capacity (`T = 4`) | 404 |
| total | 1414 |

The optimizer is plain SGD and holds nothing. A "1K model" whose optimizer
secretly carried another 100K values would deserve ridicule, so the categories
are counted separately and reported separately.

### Initialization

Deterministic, from a model-local RNG seeded independently of every environment
and bootstrap namespace.

- `W_z, W_r, W_n, U_z, U_r, U_n`: Glorot-uniform, limit `sqrt(6/(fan_in+fan_out))`.
- all gate biases: zero.
- `W_o, b_o`: **exactly zero**.

The zero output head means an untrained model predicts persistence, which is
the same starting behaviour as AAA's existing linear learners and makes "did it
learn anything" a fair question from step one. The cost is that the error head
starts at `softplus(0) = ln 2` in normalized displacement units, which is
large; it is trained down within the first few dozen updates and is reported
rather than hidden.

## 2. The three inputs

All three are derived from public knowledge of the observation format or from
observations the agent has already received.

| # | Value | Normalization |
|---|---|---|
| 1 | centered position | `(x_t - midpoint) / L` |
| 2 | recent displacement | `(x_t - x_prev) / (gap * displacement_scale)` |
| 3 | previous signed prediction error | `(revealed x_t - the prediction made for x_t) / displacement_scale` |

`L = 1.0`, `dt = 0.02`, `displacement_scale = dt * 0.2 = 0.004`. The
displacement constant is taken from the frozen v2.1 specification's
`displacement_scale_speed`, so AAA-1K and the incumbent RLS learner normalize
with the identical public number.

`gap` is the number of simulated steps between the last two *observed*
positions, so input 2 is always a per-step velocity and never a multi-step
jump. This matters: without it, the first revealed step after a gap of four
would hand every rule a displacement four times too large and then blame the
rule for overshooting.

Input 3 closes AAA's predict / reveal / error / adapt loop through the network
itself. It is inspired by prediction-error-driven learning. It is **not** a
claim that this is a biological predictive-coding network.

### Declared deviation from the brief

The AAA-1K brief specified input 3 as `/ interval_width`. That makes the input
numerically inert: a typical realized error on these streams is about 0.003,
against a centered position of order 0.5 and a velocity feature of order 1. A
development probe confirmed the consequence — the `zero_error_input` ablation
was indistinguishable from the full model to five decimal places, so the
mechanism could not have been measured either way. The error is a
displacement-like quantity and is normalized by the same public displacement
constant as input 2, which is what "the same kind of public normalization
discipline already established in AAA" implies in practice. Recorded in
[`aaa_1k_decisions.md`](aaa_1k_decisions.md) as `D-3`.

### Gradient treatment of the inputs

A recorded feature vector is a **constant** as far as gradients are concerned,
even though its third component was built from an earlier prediction.
Differentiating through the model's own history of input construction would
create a gradient shortcut that has nothing to do with predicting the world.

## 3. The two outputs

**Output 1 — next-state prediction.** The predicted next displacement in
normalized units. The evaluator converts it to a position:
`x_pred = reflect(x_t + displacement_scale * o_0)`, where `reflect` is the same
public boundary map applied identically to the reflected constant-motion and
dead-reckoning baselines.

**Output 2 — predicted next prediction-error magnitude.** `softplus(o_1)`,
non-negative by construction. This is a primitive self-estimation signal. It is
a **learned predictive-error-magnitude estimate** — not full Bayesian
uncertainty, not a calibrated predictive distribution — and its calibration is
evaluated separately from the primary prediction score.

## 4. Loss

```text
L_total = L_prediction + lambda_error * L_error_estimate
L_prediction      = (o_0 - d*)^2
L_error_estimate  = (softplus(o_1) - stopgrad(|o_0 - d*|))^2
```

`lambda_error = 0.25`, predeclared. The stop-gradient is real and is tested:
a finite-difference reference that lets the auxiliary target move disagrees
with the analytic gradient completely, and a test asserts that it does.

`d*` is the normalized displacement target. With `unfold_target` enabled
(default) the public reflection map is inverted around the agent's own raw
prediction, so a transition crossing a wall becomes an ordinary sample of the
underlying motion rather than a folded one. This is exactly the mechanism and
the public function the AAA core already uses for `AAA-120`; no evaluator
bounce label is involved.

## 5. Learning

**Truncated backpropagation through time.** On each step the gradient of *that
step's* loss is propagated back through at most `tbptt_steps` stored hidden
transitions. Each step's loss is counted exactly once.

The stored activations are the ones the model actually realized, so the
gradient is exact for the realized trajectory up to the truncation horizon. It
is **not** the exact online gradient: the influence of parameters on hidden
state more than `tbptt_steps` in the past is discarded. Tallec & Ollivier
(2017) show that this bias can cause divergence when a parameter has positive
short-term and negative long-term influence. UORO and RTRL are the unbiased
alternatives and are deliberately *not* implemented here; see
[`aaa_1k_literature_review.md`](aaa_1k_literature_review.md) for why.

**Optimizer:** plain SGD, no momentum, no adaptive rates. Global gradient-norm
clipping is a declared mechanism whose threshold is **selected**, not asserted:
a threshold of 1.0 was originally declared and a development probe found it
firing on 24% of updates and costing 29% of development error, which makes it a
hyperparameter deciding what is learned rather than a guard. Stage 3 of the
development selection chooses it by the same rules as everything else,
preferring the most conservative threshold within the practical margin of the
best. The selected threshold fires on under 1% of updates, and the activation
rate is reported as part of every result.

**A non-finite gradient or parameter raises.** Divergence is evidence. Nothing
is silently reset.

## 6. Required temporal order

Enforced in `research/aaa_1k/runner.py` and nowhere else:

1. the agent holds only causally available observations;
2. every agent produces its prediction;
3. predictions are recorded;
4. the stream advances;
5. the permitted observation — and only that — is revealed;
6. the pre-reveal prediction is scored against latent truth;
7. only then may an enabled learner update;
8. the revealed observation, or a hold, enters the agent's own state.

`accept_observation(value | None)` is the **single** channel into an agent.
There is no parameter through which a regime label, event flag, hidden speed or
future value could reach a learner even by accident.

**Learning only from genuine one-step transitions.** A learner updates only
when both ends of the transition were shown to it. A revealed position that
follows a gap is a multi-step displacement from the held input, not a sample of
the one-step law. This binds the neural arms and the RLS arm equally.

## 7. Hidden state

The 16-value hidden vector persists across normal steps and is **not** reset
when the environment changes — the model is never told that a regime changed.
It is reset only when the protocol begins a new independent episode or when the
`reset_state_every_step` ablation asks for it.

**Frozen does not mean brain-dead.** A frozen arm runs the same forward pass as
its online twin: hidden state keeps evolving, the previous-error input keeps
evolving, and only the weight update is skipped. Freezing the memory too would
answer a different and much less interesting question.

## 8. Serialization

A checkpoint carries the trainable weights, the hidden state, the complete
TBPTT buffer, every counter, the full configuration, the architecture and
format identifiers, and the parent model id. The agent adapter separately
serializes the previous signed prediction error and the observation tracker,
because those describe the interaction history rather than the network.

A test asserts that a run interrupted at step 60 and resumed produces
predictions bitwise identical to the uninterrupted run, and ends on the same
state hash.

## 9. Comparison arms

**Ablations of AAA-1K** — identical initialization, one mechanism removed each:

| Arm | Mechanism removed |
|---|---|
| `aaa1k_state_reset` | hidden state cleared before every step |
| `aaa1k_no_error_input` | input 3 forced to zero |
| `aaa1k_frozen_recurrent` | `U_z, U_r, U_n` left at initialization |

**Matched-capacity neural controls:**

| Control | Parameters | Isolates |
|---|---|---|
| `StatelessMLPControl` 3→28→28→2, tanh | 982 | memory |
| `VanillaRNNControl` 3→28 tanh recurrent→2 | 954 | gating |

**Analytic and incumbent baselines:** persistence, constant motion, reflected
constant motion, dead reckoning, an eight-point windowed linear fit, and the
unmodified v2.1 RLS candidate with its declared mechanism set copied exactly
from the frozen specification.

**Hyperparameters are selected per architecture.** The gated model, the
stateless control and the ungated control each get the learning rate and clip
threshold the same declared rules select for them, on the same development
streams. Imposing one architecture's hyperparameters on another turns a
comparison into a handicap: doing so destabilized the stateless control by two
orders of magnitude on one family and inflated the reported hidden-state
advantage thirty-fold. The three **ablations** do share the gated model's
configuration exactly, because an ablation is the same architecture with one
mechanism removed and must differ in exactly that.

The purpose is not an architecture beauty contest. It is to ask whether memory
matters, whether recurrence matters, whether gates matter, whether explicit
prediction-error feedback matters, and whether training recurrent connections
matters — the mechanism-by-mechanism design Foucault & Meyniel (2021) used.

## 10. Benchmark families

| Family | What it tests |
|---|---|
| `motion_compat` | compatibility: the existing AAA worlds, unchanged and fully observed |
| `occlusion_v1` | holding state through blindness: periodic observation gaps |
| `coarse_speed_v1` | integrating over history: a hidden speed regime behind a coarse quantizer |
| `aba_v1` | continual adaptation: one unlabelled stream, A then B then A |
| `paired_change_v1` | adaptation, isolated: two streams bit-identical until a declared step, after which one changes and one does not |

Under occlusion the evaluator scores against latent truth the agent never saw;
under `coarse_speed_v1` the agent is scored on the quantized observable, which
is what is actually revealed. Both are stated in each family's metadata.

`paired_change_v1` exists because a single online/frozen branch cannot separate
"continued learning helps after a change" from "continued learning helps". A
probe measured that difference at 5%. Running one model through the shared
prefix of a matched pair and differencing the two advantages removes everything
the two worlds have in common; both trunks are asserted to reach an identical
model-state hash at the branch.

Retention is measured against a **fixed frozen probe bank**: eight held-out
regime-A episodes, generated once, never trained on, and evaluated by a frozen
clone with its hidden state reset at the end of each A/B/A segment. Because the
same questions are asked at every checkpoint, accumulated experience cannot
flatter the later ones.

## 11. Lineage

AAA-1K is the **root** of an experimental lineage: `parent_model_id` is `null`.
Checkpoint metadata carries the format version, architecture id, parameter
count, parent id, initialization seed, state hash, benchmark version, source
commit and phase fingerprint.

**No growth is implemented in this phase.** Function-preserving widening
(Net2Net), replay, distillation, adapters and architecture search are future
directions. Growth must eventually be justified by measured inadequacy, not by
boredom.
