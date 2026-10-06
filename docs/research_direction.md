# Current research direction: Accurate Autonomous Adaptation for Juniper 1

Decision date: 2026-10-04 (owner decision). This document is current planning
guidance. It sits outside the frozen `aaa.1k.v1` charter, whose claim
discipline still applies, and it claims no capability. What has been shown is
limited to the [Juniper 1 development report](juniper1/development_report.md)
and any confirmation recorded there.

## What AAA is for

AAA means **Accurate Autonomous Adaptation**. The Erudition Model exists so
that Juniper can:
- detect deficiencies in its own behaviour or knowledge;
- determine what needs to change;
- learn from real feedback and experience;
- make controlled adaptations and evaluate whether they helped;
- keep improvements, reject or roll back harmful ones, and continue.

Anything unrelated to that goal belongs to another component.

## Juniper 1

Juniper 1 has three model components and **no Decision Model**:

| Component | Role | Current form |
|---|---|---|
| **Language Model** | language, coding, general reasoning interaction, tool calls | Juniper LM 1.1 = OpenAI gpt-oss-20b, selected by the Juniper LM 1.1 qualification; immutable and hash-pinned; adapted only through a separate, versioned adapter |
| **World Model** | environment state, consequences, uncertainty, adaptation to changed dynamics | `aaa.world.cbl.v0`, a context library of Bayesian dynamics models ([architecture](juniper1/architecture.md)) |
| **Erudition Model** | whether, when, where and how the Language Model and World Model adapt; evaluation, retention, rollback | `aaa.erudition.model.v0`, a 1,259,700-parameter transformer trained from scratch |

Host orchestration, validation, permissions, execution, the evaluation gate,
state storage, resource limits and rollback are infrastructure, not a fourth
model. Juniper-App's host authority over tools, memory, secrets and its
constitution is unchanged; no learned component can alter it.

The active research line is [`aaa.erudition.v0`](juniper1/README.md). Its
first confirmation (2026-10-06, live on gpt-oss-20b) promoted the Erudition
Model's joint adaptation over never adapting (failure 0.615 → 0.365) and over
World Model-only adaptation. It left two questions inconclusive: whether joint
adaptation beats Language Model-only adaptation, and whether learned control
is no worse than rules. Its documents are:
- [architecture](juniper1/architecture.md);
- [decisions](juniper1/decisions.md);
- [literature](juniper1/literature.md);
- [protocol](juniper1/protocol.md);
- [data](juniper1/data.md).

## What changed on 2026-10-04

- **The active direction.** Python learning is no longer the active Erudition
  direction. The Language Model owns language and coding; the Erudition Model
  does not learn English or code.
- **The Erudition Model is built from scratch.** No weights, heads or
  task-specific architecture are carried over from Champion 0/1, AAA-1K,
  `aaa.1k.v2`, `aaa.python.v0`/`v1`, WM-S or any language model. Reusable
  engineering was reimplemented in a new package.
- **Scale.** The Erudition Model starts at **at least 1M trainable
  parameters** (owner constraint) and is not padded to reach it. The earlier
  1K/4K/10K conclusions belong to a different task family and architecture
  and do not constrain it. The approximate **105M** AAA 1 planning goal is
  retired as a current target and kept only as dated history.
- **What carries forward is the method:**
  - causal boundaries and post-action feedback;
  - development, attack and confirmation separation;
  - reserved, admission-gated confirmation identities;
  - frozen protocols committed before confirmation;
  - independent recomputation and structurally independent promotion
    implementations;
  - fail-closed evidence and `INSUFFICIENT_EVIDENCE`;
  - retained failures, rollback, provenance and protected historical
    identities.

## Evaluation rules (unchanged in substance)

A scored action may use only information available before it. Evaluator
labels, scheduled regimes and hidden dynamics never reach a component; a test
traps every read of the hidden label outside the scorer. Adaptation is
measured causally, with identical streams across conditions (common random
numbers), against never-adapt, single-component, rule-based and always-adapt
controls. Retention is measured on behaviour that never needed to change.
"Accurate" is measured directly:
- false adaptation;
- missed adaptation;
- misattribution;
- unhelpful adaptation;
- poisoning by corrupted feedback.

A confirmation claim requires the committed freeze and its admission check.

## Self-modification boundary

No learned component may change code, merge, deploy, grant itself tools or
permissions, rewrite the Juniper constitution, or touch canonical Git history.
Runtime adaptation is limited to the versioned adapter and World Model state
held by the host's store, every change of which is reversible. Repository
changes remain externally reviewed.

## Historical direction: Coding, beginning with Python (2026-09-23 to 2026-10-04)

Everything below is the direction that was current until 2026-10-04, kept as
it was last written. Its experiments, results and freezes are unchanged and
remain reproducible. It no longer describes the active work.

Decision date: 2026-09-23. Updated 2026-09-27 for the scoped world-model and
language experiments following `aaa.python.v1`, 2026-09-30 for component
terminology and the language-component direction, and 2026-10-02 for the
Juniper 1 components.
This is current planning guidance. It sits outside the frozen `aaa.1k.v1`
charter, whose principles still apply, and outside the `aaa.1k.v2`
protocol. It claims no capability.

- **AAA** (Accurate Autonomous Adaptation) is the research program.
- **Juniper** is the persistent artificial agent that successful AAA research
  is intended to produce.
- **Autonomy** is the long-term objective.
- **Python coding** is AAA's first deliberate specialization and its current
  primary experimental domain.
- **The moving dot** is a historical and continuing benchmark family
  ([archive](dot_benchmark_archive.md)), not AAA's purpose.

### Components

Current terminology, adopted 2026-09-30 and simplified for Juniper 1 on
2026-10-02. AAA is the program and the system; it is not one model. The components are planning names, not claims that any of
them exist as finished systems.

| Component | Role | Status |
|---|---|---|
| **Erudition Model** | autonomous learning and adaptation | research: Champion 1 and the `aaa.python.v0`/`v1` learners are experiments toward it; a ~4K learner is the evidence-supported next experimental reference in the tested Python family |
| **World Model** | environment state and prediction | scoped result: WM-S in `aaa.python.opaque.v0` only ([handoff](wm_program/handoff.md)) |
| **Language Model** | language and reasoning interaction | OpenAI gpt-oss-20b, selected by the Juniper LM 1.1 qualification; used by the Juniper application, not integrated into AAA |

Juniper 1 has no Decision Model. An earlier plan for a cross-component
Decision Model was dropped on 2026-10-02.

Earlier phase documents call their learners "AAA models" or "the AAA model".
Those names are kept where the evidence was recorded. In current planning,
the learning component is the Erudition Model.

### Scope

Coding is the capability domain. Autonomous adaptation remains the research
problem. The question is whether a system can acquire, retain, apply, test,
revise and adapt programming knowledge through interaction with Python
programs, execution, tests, failures, interpreter feedback, tools,
repositories and unfamiliar tasks, under causal evaluation. It is not whether
it can autocomplete code, and it is not a reason to build a chatbot, chase a
benchmark or scale a Transformer by default.

### Why Python first

Python is already the implementation language of most of AAA. CPython supplies
an exact external ground truth. Execution feedback arrives naturally after an
action. Small programs can be generated rather than scraped, which avoids
licensing ambiguity, contamination and memorized solutions. And Python connects
to the long-term aim that Juniper can eventually understand software. See the
[`aaa.python.v0` brief](aaa_python_research_brief.md).

### Capability ladder

Each rung is measured and reported separately. There is no single Python
score, because one score would hide distinct failures. A rung is earned by
evidence, in roughly this order, and a later rung is not implemented until the
rungs it depends on have measured baselines.

| # | Capability | `aaa.python.v0` |
|---:|---|---|
| 1 | Python syntax validity | family `syntax` |
| 2 | Basic Python semantics | family `output` |
| 3 | State / value tracking | family `output` (assignments, loops, functions) |
| 4 | Output prediction | family `output` |
| 5 | Execution-success prediction | family `outcome` |
| 6 | Exception classification | family `outcome` |
| 7 | Function-level reasoning | partly: `output` function templates, `repair` |
| 8 | Function completion | roadmap |
| 9 | Test-conditioned behavior | partly: `repair` shows two visible tests |
| 10 | Bug localization | family `localize` |
| 11 | Candidate repair selection | family `repair` |
| 12 | Free-form repair, when earned | roadmap |
| 13 | Behaviour-preserving refactoring | roadmap |
| 14 | Tool / execution use (for example, running visible tests before choosing) | measured in `aaa.python.v1`: a logged pre-action tool for `repair`; the tool alone scores 0.95 and a 4K or 10K learner using it matches it |
| 15 | Learning from interpreter feedback | measured: online versus frozen on every family |
| 16 | Learning from test feedback | measured: bandit feedback on the chosen repair |
| 17 | Retention across task sequences | measured: never-trained probe bank |
| 18 | Adaptation to unfamiliar APIs or libraries | roadmap (v0 measures adaptation to unfamiliar *program structure*) |
| 19 | Avoiding regression after adaptation | measured: probe-bank forgetting |
| 20 | Repository navigation | roadmap |
| 21 | Cross-file reasoning | roadmap |
| 22 | Dependency / environment reasoning | roadmap |
| 23 | Long-horizon coding work | roadmap |
| 24 | Calibrated uncertainty and explicit recognition of insufficient evidence | measured as Brier and calibration error; abstention exists in the interface, is not yet rewarded |

`aaa.python.v1` re-measures rungs 1-11 and 14-19 on a repaired exam with
declared generalization slices; its [development report](aaa_python_v1_development_report.md)
has the numbers. "Measured" means an instrument exists. It does not mean the capability exists.
The v0 development result is negative on nearly every rung it measures
([report](aaa_python_development_report.md)).

### Evaluation rules

A scored action may use only information available before it. Hidden tests,
reference patches, answer keys, evaluator labels, future tool output,
held-out repository information and post-hoc success flags must not reach the
system first. Feedback legitimately revealed *after* an action may become
learning data only where the frozen protocol permits it, and exactly what
becomes visible, and when, is recorded.

Training, development, diagnostic, attack and held-out confirmation tasks stay
mechanically separate. Confirmation identities are reserved before use and
never generated early. Contamination risks, including public benchmark
exposure, are recorded. A confirmation claim is refused when separation cannot
be established.

Controls isolate a source of capability: frozen copies of the same model,
exact-match memorization, non-adaptive and parameter-matched systems,
architecture and representation ablations, disabled tools, feedback and
memory ablations, and simple deterministic methods. CPython is an oracle,
never a baseline. Adaptation is measured causally, from identical states with
one property changed, never as "later scores are better". Retention is
measured on fixed probe banks that are never trained on, never as unchanged
training loss.

Representations and architectures are experimental variables. Byte-level,
lexical, AST and execution-trace inputs are all candidates, and standard
next-token prediction is one possible design, not a prerequisite.

### The improvement loop, carried into coding

The dot era's most valuable output is the loop: observe, classify, diagnose,
hypothesize, falsify, intervene minimally, attack, confirm on fresh evidence,
decide, preserve, repeat ([`aaa.loop.v1`](loop_protocol.md)). Its
domain-independent parts carry over directly: the evidence lifecycle,
hypothesis records, candidate identities, development / attack / confirmation
separation, freeze manifests, independent recomputation, promotion decisions,
retained rejections, failure injection, rollback and source identity.

`aaa.python.v0` reuses the *methods* without importing the dot-specific loop
package. It has its own identity separation, freeze-style fingerprint,
independent recomputation, and the shared promotion contract
([`aaa.promotion.crossed.v1`](promotion_contract.md)), which is the one piece
where a real shared abstraction exists. A generic loop framework will be
extracted only when a second domain actually needs the same code.

### World-model and language work now underway

The [world-model program](wm_program/README.md) studies generated Python
debugging with an opaque six-function library. Its structured candidate uses a
hand-written interpreter for visible Python plus a learned table for the
library. Within `aaa.python.opaque.v0`, the retained world-model bits and
declared contracts recompute as reported, with strong seed and transfer
limitations. A separately trained American-English language model was tested
through a structured adapter on templated reports. The stored language
confirmation reports an L2 promotion, but independent review reproduced a
channel-boundary defect: when a proposal had the same number of tests as the
true task, `RUN` returned results and pass labels for the true tests to the
learner and planner. The numerical result remains preserved, but the language
comparison's causal attribution is **not verified**. Pretraining and
full-system success are not established. Neither experiment establishes
general Python or English competence. See the [independent review erratum](wm_program/pr30_independent_review.md).

### Language component direction

The 5.2M-parameter American-English model and its adapter (`aaa.wm.lang.v0`)
are retained research evidence. They are not the intended long-term language
component, and the retained language result stays **not verified**.

Juniper 1's Language Model is OpenAI's unmodified gpt-oss-20b. The Juniper LM
1.1 qualification selected it on evidence, and the Juniper application wraps
it with its own constitution, tool, memory and privacy controls. AAA therefore
does not need to learn English or general coding from scratch; its research
focuses on autonomy. Juniper LM 1, the GPT-2 124M modernization study, was
retired on 2026-10-04 and remains historical research. Its frozen
JuniperBench-Code v1.2 provenance remains valid. No AAA integration of either
model exists. One would need an integration
experiment designed under the same causal rules as the rest of AAA, including
a repaired language-channel boundary. Nothing here claims that integration
exists or will succeed.

The templated extraction study does not establish communication, instruction
following, open-ended conversation, or general understanding.

### Other future directions

- **Other programming languages** can follow when Python work gives a reason.

### Self-inspection boundary

Python competence may eventually support understanding software in Juniper's
own research environment. That is a research question, not authority to change
canonical code. Any self-inspection or self-modification experiment needs:

- an isolated sandbox or worktree;
- immutable external tests and a frozen evaluator;
- a separate proposer and evaluator;
- reproducible patches, retained evidence and rollback;
- no access to external mandatory guardrails;
- external approval before any merge or deployment.

Producing a patch does not establish that a model should merge it, and no
candidate may merge its own code. None of this is implemented.

### Scaling

[Scaling readiness](scaling_readiness.md) sets the rule: define the task and its
measured deficiency, establish baselines, build the minimal trainable system,
diagnose representation, optimization and evaluation problems, test
parameter-neutral remedies, and only then test capacity against smaller
controls. The long-term AAA 1 planning goal of about 105M parameters is
guidance, not a mandate, a minimum, a ceiling to reach, or evidence that Python
needs it.
