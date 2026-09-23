# `aaa.python.v0` research brief

Status: **development phase**. Nothing here is a demonstrated Python
capability, and formal confirmation is not admitted in v0.

## Why Python first

AAA studies whether an artificial system can autonomously learn, predict,
reason from feedback, adapt, retain useful knowledge, detect inadequacy and
improve its future behaviour. Its first deliberate specialization is
**Coding, beginning with Python** ([research direction](research_direction.md)).

Python suits that question for concrete reasons:

- **An external, exact ground truth exists.** CPython decides what a program
  does. The learner cannot negotiate with the evaluator, and the evaluator
  needs no human labels.
- **Feedback is naturally post-action.** A program is written or judged
  first, then run; tracebacks, output and test results arrive afterwards. That
  is AAA's foundational loop: observe, predict, act, reveal, score, learn,
  adapt.
- **Tasks can be generated, not scraped.** Small programs from a frozen
  grammar avoid licensing ambiguity, public-benchmark contamination and
  memorized solutions, and they fit on FLOWBOX.
- **Structure can be held out.** Constants, templates and whole program shapes
  can be kept away from training, so memorization, interpolation and
  structural generalization can be told apart.
- **It connects to the long-term goal.** Juniper, the intended persistent
  agent, will eventually need to understand software, including its own
  research environment. That is a research question, not an authority
  (see *Boundaries*).

## The question v0 asks

> Can an adaptive system acquire and retain useful Python knowledge from
> programs and execution feedback, and use it to improve on fresh programs
> without evaluator leakage?

This is deliberately narrower than "can AAA engineer software". v0 exists to
make the narrow question **measurable**: a deterministic, safe, causal
environment, baselines that can win, and statistics that can say no.

## What v0 provides

- a frozen safe subset of Python, enforced by AST validation, executed by
  CPython in an isolated, resource-limited subprocess
  ([protocol](aaa_python_protocol.md));
- five task families: syntax validity, execution outcome and exception class,
  printed output, failing-line localization, and repair selection among four
  candidates;
- train / development / probe / attack identities, mechanically separated by
  identity and by normalized source hash; confirmation identities reserved and
  refused;
- a causal episode boundary that yields feedback only after a committed
  action, restricted to what a real run would reveal;
- one minimal online learner with frozen, feedback-disabled and
  memory-disabled controls, four representations, and chance, majority,
  exact-match lookup and surface-heuristic baselines
  ([architecture](aaa_python_architecture.md));
- a matched paired-shift adaptation design, a never-trained retention probe
  bank, crossed initialization x stream statistics, and independent
  recomputation that re-executes every scored program.

## What v0 does not claim

- It does not claim that the learner knows Python. The retained development
  evidence ([`evidence/aaa_python_v0/`](evidence/aaa_python_v0/)) shows the
  opposite on most families: the minimal learner is near the majority
  baseline and loses clearly to surface heuristics on syntax and localization.
- It does not claim adaptation or retention. Those intervals are unresolved.
- It does not claim generalization beyond the frozen subset, beyond generated
  programs, or to repositories, libraries or real code.
- It does not use English, a text corpus, a pretrained model or a Transformer,
  and it does not assume any of them will be needed.
- It does not scale. Parameter count is a reported cost, not progress.

## Future directions, recorded and not started

- **More of the capability ladder**, earned in order: see
  [research direction](research_direction.md). The next rungs are tool use
  (running visible tests before choosing), free-form repair, and learning
  from interpreter feedback across a longer task sequence.
- **English / natural language** is a possible later capability for
  communication, instruction following, explanation and knowledge exchange.
  It is *not* an objective of this phase, and no English corpus or
  conversational behaviour is introduced. The byte-level representation is
  kept partly so that a later expansion is not made impossible.
- **Other programming languages** may follow once Python work gives a reason.

## Boundaries

Python work is not authority to modify canonical AAA or Juniper code. Any
later self-inspection experiment needs an isolated sandbox or worktree,
external immutable tests, a frozen evaluator, separate proposer and evaluator,
reproducible patches, rollback, retained evidence, no access to mandatory
guardrails, and external approval before any merge. None of that is
implemented here. The learner cannot change its evaluator, hidden tests,
generator, answer keys, promotion rules or containment.

Behaviour will be reported as observed behaviour under stated conditions. It
is not evidence of understanding, consciousness, emotion or general
intelligence.
