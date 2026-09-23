# Current research direction: Coding, beginning with Python

Decision date: 2026-09-23. This is current planning guidance outside the
frozen `aaa.1k.v1` charter and `aaa.1k.v2` protocol. AAA remains the research
programme; Juniper is its intended long-term agent. This decision does not
claim that the present predictor can code, reason generally, or improve itself.

AAA's **first deliberate specialization is Coding, beginning with Python**.
Python is already the implementation language for most of AAA and offers a
coherent environment for studying prediction, memory, tool use, tests and
error-driven adaptation together. Python is the first domain, not AAA's
permanent scope or a requirement to build a generic chatbot. The question is
whether a system can acquire, retain, use, test and adapt coding knowledge
through interaction with Python programs, repositories, execution results,
failures, tools and unfamiliar tasks under causal evaluation.

## Capability ladder

Future protocols should report these separately, with tasks and baselines
appropriate to each rung. A single coding score would hide distinct failures.

1. Python syntax and structural validity.
2. Basic program semantics.
3. Function completion.
4. Test-conditioned code generation.
5. Bug localization.
6. Bug repair.
7. Behaviour-preserving refactoring.
8. Repository navigation and context selection.
9. Understanding interacting files.
10. Dependency and environment reasoning.
11. Execution and tool use.
12. Learning from interpreter and test feedback.
13. Retaining useful information across tasks.
14. Adapting to an unfamiliar library or codebase.
15. Avoiding regressions after adaptation.
16. Long-horizon coding work.
17. Calibrated uncertainty and recognition of insufficient evidence.

## Evaluation rules for a future phase

A scored solution may use only information available before its action. Hidden
tests, reference patches, answer keys, evaluator labels, future tool output,
held-out repository information and post-hoc success flags must not reach the
system before the scored action. Feedback legitimately revealed *after* an
action may become learning data only where the frozen protocol permits it.

Keep training and development tasks, diagnostic tasks, adversarial attacks and
held-out confirmation mechanically separate. Reserve confirmation identities
before use and prefer fresh tasks or repositories. Record contamination risks,
including public benchmark exposure, and refuse a confirmation claim when
separation cannot be established.

Use controls that isolate a source of capability: frozen copies of the same
model, retrieval or memorization, non-adaptive systems, parameter-matched
models, architecture ablations, disabled tools, and context or memory
ablations. Include simple deterministic methods where they are competitive.
Writing code once is not evidence of learning or adaptation.

Choose observations and representations experimentally. Candidates may
include source text, tokens, syntax or AST structure, interpreter errors,
stack traces, tests, repository structure, diffs, traces and tool results.
Standard next-token text prediction is one possible design, not a prerequisite
or a conclusion. Other programming languages can follow when Python work
provides a reason to expand.

## Self-inspection boundary

Python competence may eventually support understanding software used in
Juniper's own research environment. That is a research question, not authority
to change canonical code. A future self-inspection experiment needs an
isolated sandbox or worktree, immutable external tests, separate proposer and
evaluator, frozen evaluation rules and containment controls, retained source
and evidence, reproducible patches, independent recomputation and rollback.
Canonical merge or deployment requires external approval. Producing a patch
does not establish that a model should merge it.

The scaling rule is in [scaling readiness](scaling_readiness.md): define the
task and its measured deficiency first, test parameter-neutral remedies, then
let evidence determine whether and how far to increase capacity. The current
long-term AAA 1 goal of approximately 105M parameters is planning guidance,
not a mandate or a result.
