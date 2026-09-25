# Current research direction: Coding, beginning with Python

Decision date: 2026-09-23. Updated 2026-09-25 for the `aaa.python.v1` result.
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

Coding is the capability domain. Autonomous adaptation remains the research
problem. The question is whether a system can acquire, retain, apply, test,
revise and adapt programming knowledge through interaction with Python
programs, execution, tests, failures, interpreter feedback, tools,
repositories and unfamiliar tasks, under causal evaluation. It is not whether
it can autocomplete code, and it is not a reason to build a chatbot, chase a
benchmark or scale a Transformer by default.

## Why Python first

Python is already the implementation language of most of AAA. CPython supplies
an exact external ground truth. Execution feedback arrives naturally after an
action. Small programs can be generated rather than scraped, which avoids
licensing ambiguity, contamination and memorized solutions. And Python connects
to the long-term aim that Juniper can eventually understand software. See the
[`aaa.python.v0` brief](aaa_python_research_brief.md).

## Capability ladder

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

## Evaluation rules

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

## The improvement loop, carried into coding

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

## Future directions, recorded and not started

- **English / natural language.** AAA may later learn English as an
  additional capability supporting communication, instruction following,
  explanation, knowledge exchange, reasoning interfaces and interaction with
  people. It is **not** a current objective. No English training, general text
  corpus or conversational behavior is introduced, and the first Python phase
  requires no English competence. Choices that would make a later expansion
  impossible are avoided where that costs nothing; for example, the byte-level
  representation can represent any text. Evidence from the Python work decides
  when expansion is justified.
- **Other programming languages** can follow when Python work gives a reason.

## Self-inspection boundary

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

## Scaling

[Scaling readiness](scaling_readiness.md) sets the rule: define the task and its
measured deficiency, establish baselines, build the minimal trainable system,
diagnose representation, optimization and evaluation problems, test
parameter-neutral remedies, and only then test capacity against smaller
controls. The long-term AAA 1 planning goal of about 105M parameters is
guidance, not a mandate, a minimum, a ceiling to reach, or evidence that Python
needs it.
