# Juniper 1 adaptation architecture (`aaa.erudition.v0`)

Date: 2026-10-04. Code: [`research/aaa_erudition/`](../../research/aaa_erudition/).
This is the active AAA research architecture. It describes what is built; what
it has been shown to do is limited to the [development report](development_report.md)
and, once executed, the confirmation record.

## The three components and the host

```
            user request (text), workspace state
                         |
                         v
   +-------------------------------+      notes (data, versioned)      +---------------------+
   | Language Model  (Juniper LM   | <-------------------------------- | LM adapter state    |
   | 1.1 = gpt-oss-20b, immutable) |                                   | aaa.lm.adapter.v0   |
   +-------------------------------+                                   +---------------------+
          | proposal (tool call)  ^ one revision with the                        ^
          v                       | WM's predicted consequences                  |
   +-------------------------------+                                             |
   | World Model aaa.world.cbl.v0  |  context library, predictive uncertainty   |
   +-------------------------------+                                             |
          | prediction                                                           |
          v                                                                      |
   HOST: validate, execute, record, reveal outcome and feedback, features,       |
         gate, store, canary, rollback  ------------------------------------------+
          | evidence features (40 per step)          ^ adaptation request from a fixed menu
          v                                          |
   +-------------------------------------------------+
   | Erudition Model aaa.erudition.model.v0 (1,259,700 trainable parameters) |
   +-------------------------------------------------+
```

There are three learned components and no Decision Model. Orchestration,
validation, execution, the gate, storage and rollback are host software. They
are not learned and the components cannot change them.

### Language Model: Juniper LM 1.1

- **Base.** OpenAI gpt-oss-20b, MXFP4 GGUF, SHA-256 `9d7364f0…d23d`
  (12,109,566,784 bytes). It is served by the pinned llama.cpp build (b11270,
  `748d4225`) with the qualified FLOWBOX profile copied from Juniper-App
  ([`data/gpt_oss_profile.json`](../../research/aaa_erudition/data/gpt_oss_profile.json),
  source commit and hash recorded). The weights are never written. Every run
  manifest records the artifact identity, and the run that loads the server
  verifies the file hash.
- **Interface.** Tool calls: `act(op, tank, n, expected)` and
  `abstain(reason)`. The host validates every argument. An unknown tank, a
  non-integer `n` or any other tool name becomes `invalid` and executes
  nothing. Model text is never an instruction to the host.
- **Adaptation.** Only `AdapterState`, a versioned, hash-identified list of
  notes:
  - `alias` — "When a user says 'the north tank', they mean the tank cedar";
  - `dynamics` — "fill(tank, n) currently adds about 5*n + 1 units";
  - `observation` — "Observed earlier: fill(cedar, 2) took cedar from 40 to 51".

  Notes are typed records rendered by fixed templates and framed as data. An
  alias key must be a plain phrase that occurred verbatim in an evidence
  request; a tank must exist; coefficients are integers. This is memory-based
  editing ([literature](literature.md) §1). Removing the adapter returns the
  exact base behaviour.
- **Backend boundary.** `LlamaServer` (loopback only), `CachedBackend`
  (records every exchange under the SHA-256 of the exact request; replay
  re-executes a run with no model), `ScriptedLM` (model-free tests only) and
  `SurrogateLM` (simulation only). GPT-OSS specifics (Harmony template,
  `reasoning_effort`, sampler) live only in the profile and `chat_request`.
- **Determinism.** Each request carries a seed derived from its own content,
  so identical inputs in different conditions get identical outputs from the
  cache. This is the Language Model's half of the common-random-numbers design.

### World Model: `aaa.world.cbl.v0`

- **State.** A library of contexts. Each context is one hypothesis about the
  tool semantics: for every operation, a Normal-Inverse-Gamma posterior over
  the change a call produces on the basis `[1, n]`. Results clipped at a
  declared bound are censored. One context is active.
- **Prediction.** A Student-t predictive for the result of a proposed call,
  including its uncertainty. For every `n` it gives the consequence table the
  Language Model can plan with.
- **Evidence for Erudition.** The log-likelihood of each observed transition
  under the active context, under the best other stored context, and under the
  prior.
- **Adaptation mechanisms.**
  - `new_context` fits a fresh context on recent evidence and makes it active;
  - `update_in_place` adds evidence to the active context;
  - `recall` switches to the stored context that best explains recent evidence.

  Other contexts are never modified, so a returning environment is recalled,
  not relearned.
- **Scope, stated plainly.** The basis fits affine tool semantics. Capped
  ("novel") dynamics are outside it on purpose: there the World Model is wrong
  but reports large uncertainty (development diagnostic, D-WM). It is a
  structured-state dynamics model for tool environments, not a general neural
  simulator.

### Erudition Model: `aaa.erudition.model.v0`

- **Input.** The last 64 steps of 40 evidence features, plus the condition's
  allowed-action menu as one token. The features are the final proposal kind,
  revision, Language Model confidence, World Model disagreement and
  uncertainty, outcome and error class, self-consistency of the Language
  Model's expected value, World Model error and surprise, feedback rating and
  structural feedback signals, adaptation history (time since commit, canary,
  last result), state sizes and cooldowns. The full list is `lifecycle.FEATURES`.
- **Network.** A 4-layer pre-LayerNorm transformer encoder (d=160, 4 heads,
  ff=640) with two heads:
  - `q` (12 actions): the expected quality of the next 25 steps if that
    action is requested now;
  - `diagnosis` (8 classes): stable, transient, insufficient evidence,
    unreliable feedback, tool failure, LM deficiency, WM deficiency, joint
    deficiency.
- **Decision.** The best allowed non-wait action if its `q` exceeds `wait`'s
  by the validation-selected margin; otherwise wait.
- **Actions.**
  - `wait`;
  - LM alone: `lm.alias`, `lm.experience`, `lm.retract`, `lm.from_wm`;
  - WM alone: `wm.new`, `wm.update`, `wm.recall`;
  - joint: `joint.new`, `joint.recall`, `joint.new_alias`;
  - `rollback`.

  Joint actions couple a World Model change with distilling it into
  Language Model notes.
- **Training.** From scratch, on training-split simulation only. At sampled
  decision points the simulator copies the whole system once per allowed
  action, forces that action, and measures the following 25 steps (task
  success minus adaptation cost). These counterfactual returns are the `q`
  targets; the hidden regime is the diagnosis target. The Language Model in
  simulation is a surrogate whose rates are fitted to a characterization of
  the real model on training-split situations. No weight comes from any
  earlier AAA model, Juniper LM, Juniper Reference or WM-S.

## The lifecycle (one step)

1. **Observe.** An `Observation`: state, bounds and request.
2. **Propose.** The Language Model proposes with the current adapter.
3. **Predict.** The World Model predicts the proposal's result. If its rounded
   prediction differs from the model's `expected` value and its SD is at most
   1.5, the Language Model gets one revision turn with the predicted
   consequences for every `n`.
4. **Act.** The host executes the final call, or nothing on abstain or invalid.
5. **Record.** An `EpisodeRecord`, in causal order.
6. **Reveal.** The outcome and the user's feedback, which may be false inside
   corruption windows.
7. **Score.** Evaluator side only: the only reader of the hidden label.
8. **Features.** The step's evidence features are computed.
9. **Monitor.** Canaries are checked. If the 8 steps after an accepted change
   are worse than the 8 before by at least 0.25 on both satisfaction and
   self-consistency, the component is restored to its state before that
   change, together with anything accepted on top of it since.
10. **Decide.** The controller chooses from the menu; the host enforces the
    condition's menu and a 4-step cooldown.
11. **Generate candidates.** Each requested mechanism produces a new immutable
    state in isolation.
12. **Gate (`aaa.erudition.gate.v0`).**
    - *World Model:*
      - **Evidence.** The last 16 executed transitions, of which the newest 4
        are held out.
      - **Fit.** A new context is fitted only on the earlier transitions the
        active context found surprising (log-likelihood below about 1%), so a
        window that still holds pre-shift results does not blend two regimes.
      - **Acceptance.** Held-out error must improve by at least 0.25 and end at
        most 0.25, with no regression above 0.125 on any stored context's own
        transitions.
      - **Insufficient evidence.** Fewer than 6 transitions is
        `INSUFFICIENT_EVIDENCE`.
    - *Language Model:* replay up to 6 evidence episodes with parent and
      candidate.
      - **Alias candidates** are scored, in scope, on episodes whose request
        uses a phrase the candidate changes, against the correction extracted
        from that episode. This is *in-sample*. It verifies that the model reads
        the note as intended and that nothing else changes; it cannot verify
        that the correction was true. That rests on the two-agreeing-corrections
        rule, the structural checks, the canary and the measured `poisoned` rate.
      - **Dynamics candidates** are scored on episodes whose executed result
        missed the model's own expected value, against the gate's own estimate
        from recent observations, which is identical in every condition.
        Episodes whose results the candidate shows as observation notes are
        held out.
      - **Acceptance.** The candidate must cut replay error by at least 1/3,
        change at most 25% of up to 4 recently satisfied episodes, and must not
        act on a nonexistent tank where the parent abstained.
13. **Retain, reject or roll back.** Each outcome is committed with
    provenance.
14. **Continue.**

## State, provenance and rollback

The `StateStore` is content-addressed. Each state's identity is the SHA-256 of
its canonical JSON, and every read re-verifies it. Lineage is append-only;
heads move only by commit or rollback, written atomically after the
provenance record. Every `AdaptationRecord` holds:
- parent and result state, component, mechanism;
- the request (with the controller's diagnosis and evidence episode ids);
- the candidate (configuration, evidence);
- the gate evaluation (target parent/candidate, regression, attack, verdict,
  reasons);
- deterministic resource counts.

Tested failure modes:
- tampered objects and corrupted heads are refused;
- an interrupted commit keeps the previous head and appends an explicit
  recovery record;
- a torn lineage tail is dropped and reported;
- a failed write leaves no partial object.

## Trust boundary and adaptation poisoning

Untrusted text includes requests, feedback, model output and notes. The
defences are:
- the host validates every tool call;
- notes come only from structurally verified records;
- alias notes need two agreeing corrections and two-thirds agreement;
- every candidate is tested for collateral change and new fabrication;
- accepted changes are monitored and reversible.

The Erudition Model cannot write state, alter the gate, extend its menu,
touch the Juniper constitution, permissions, credentials or any repository,
or deploy anything. Feedback corruption is in the evaluation (`noise_only`,
`poisoned_shift`). Corrupted feedback that is internally consistent can still
poison an alias note; the metric `poisoned` measures how often that happens
rather than pretending it cannot.

## Memory ownership

| Kind | Owner | Persistence | Contents |
|---|---|---|---|
| LM adapter state | Erudition lifecycle (`StateStore`) | per workspace, versioned | notes estimated from this workspace's evidence |
| World Model state | Erudition lifecycle (`StateStore`) | per workspace, versioned | context library and its bounded transition buffers (48 per context) |
| Erudition working memory | Erudition Model | rolling 64-step window | evidence features only, no text |
| Erudition parameters | AAA research artifact | per training run, hash-pinned | weights trained on synthetic training-split simulation |
| Research experience | AAA evidence | committed or `$AAA_DATA_ROOT` | run records and call caches from synthetic tasks |
| Juniper user memory | Juniper-App | user-controlled | not used, not read, not trained on |

No user conversation, Juniper-App memory or personal data is read, stored or
trained on. Workspace adaptation state would be user data in a product and
would need Juniper-App's deletion and privacy controls before any product
integration. None exists.

## Resource envelope on FLOWBOX

| Item | Size |
|---|---|
| Erudition trainable parameters | 1,259,700 (float32 weights file about 5 MB) |
| Erudition frozen parameters, optimizer state at inference, persistent state | 0; 0; 64 × 40 float32 window (10 KB) |
| World Model state | per context 2 ops × (2 + 4 + 2 + 1) floats + up to 48 transitions; a few KB as JSON |
| LM adapter state | a few notes; under 2 KB |
| GPT-OSS resident | about 4.8 GB VRAM and 9 GB RSS (Juniper LM 1.1 qualification) |
| Erudition inference | CPU, about 1 ms per decision, about 400 MB RSS for the PyTorch runtime |

Training ran separately from inference (see the [compute record](compute.md)).
GPT-OSS's experts run on the CPU (`--n-cpu-moe 18`), so CPU-heavy simulation
during a real-model run roughly halves model throughput. Simulation and
real-model runs are therefore scheduled apart.
