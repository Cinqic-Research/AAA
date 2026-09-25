# `aaa.python.v1`: implementer self-review

**This is the implementer's self-review, not an independent review.** One
author (Claude Opus 5.5, working as Markus Gillyard's agent) designed the phase,
wrote the code, ran every stage and wrote this document. Three subagents
(repository claim map, v0 adversarial audit, literature review) supplied
coverage; they share the author's framing and are not independent either.
Every claim below is something a separate reviewer should try to break.

## Starting state

`main` at `cb4170e86e7d15a56dbba319c695b32d01e8206b` (PR #28 merged), clean,
no open PRs, CPU CI green. The canonical repository had moved to
`Cinqic-Research/AAA`; the old `Cinqic/AAA` URL no longer resolves. The
`Cinqic Storage` HDD was unmounted at session start and was mounted (and
remounted once, mid-run) before any work; the NVMe (97% full) was never used
for work. Protected identities were `UNCHANGED` before and after every change.

## The attempt to disprove the conclusions

| Question | What was done | Finding |
|---|---|---|
| Did `e2` leak the answer? | `syntax` never reaches the parser (mock test; mutation caught); no value is ever evaluated; repair parse failure removes `flow` for every candidate; encoders read only the view | no leak found. `e2` does carry designed knowledge of Python structure; that is a limitation, not leakage |
| Did the 10K model get more useful input than 1K? | every capacity size uses the same encoder, D, heads, training pool, order and streams; only the core width and its tuned rate/budget differ | no |
| Did a supposedly fixed component learn? | encoders have no parameters; tests assert zero trainable and adaptive state; encodings are read-only arrays | no |
| Are parameter counts honest? | counted from arrays; heads, core, biases, optimizer velocity and the reset-control copy reported; `hidden_for_budget` tested | yes; the one-hot output head is a third of every size and is reported |
| Did optimizer state become hidden capacity? | momentum arms count velocity (846 and 10,046); all confirmation arms are plain SGD with zero optimizer state | no |
| Are groups complete? | `crossed.v1` refuses missing cells; `summarize` and `recompute` refuse off-design initializations; tamper tests | yes |
| Are held-out identities untouched? | confirmation generated only inside the admitted run; 0 overlap with every earlier v1 pool and v0's observed pools; attack used once | yes |
| Did development choices use attack or confirmation evidence? | selections are code applied to development summaries; the attack stage ran after every selection; the freeze was committed before confirmation existed | no. The 4K-over-1K contract was added after development and before the freeze, and is labelled so |
| Did a baseline get unfairly weakened? | v0's `outcome` heuristic *was* majority (`AAA-193`); v1 adds fitted stupid rules, the medoid attack and the tool; baselines train on the same pool | v1 baselines are stronger than v0's, not weaker |
| Did the benchmark become easier? | repair medoid 0.90 -> 0.32; localization's trivial single-fault half removed; rules and tool reported beside every learner | harder, not easier |
| Is an effect driven by one initialization or template? | crossed intervals over 10 initializations x 30 streams; variance components retained; per-slice tables; fresh initializations in confirmation | no single-initialization dependence found |
| Does a simple heuristic explain it? | fitted rules win on syntax and outcome and are reported as winning; learners beat them on localization and non-tool repair | partly: see limitations |
| Does it survive recomputation? | every stage re-derives from primitives; independent count-weighted recomputation passes; re-adjudication reproduces all four verdicts; post-hoc re-execution reproduced the 1K and 10K adaptation and plasticity rows exactly; v0 `recompute --rerun` reproduces v0's evidence | yes |
| Does 10K still learn late in life? | no late-life decline by the declared ratio; but experienced 10K learns a conflicting mapping 0.155 worse than a fresh one (4K 0.033) | a real weakness of 10K relative to 4K |
| Would 4K produce the same benefit? | tested in development, attack, post-hoc adaptation and confirmation | yes, on every axis measured |
| Does `e2` help when total parameters are matched? | the encoder contrast holds trainable parameters identical (6,662) | yes, confirmed (`PROMOTE`, 0.750) |
| Does the head explain the encoder gain? | the encoder stage used v0's one-hot heads for every encoder; the pointer head was tested after, at a fixed encoder | no |
| Stable across clean clones? | a clean clone built the wheel; the installed package reproduced the spec hash and golden keys outside the checkout; CI runs the golden keys on CPython 3.10-3.13 | yes |

## What broke, and how it was repaired

Pre-observation or tooling defects (all repaired, tested, and in the ledger
under `AAA-204` unless noted): a tool-input model that refused families
without tool evidence; the tool baseline running sandboxes it never used; an
adaptation design larger than its range; Holm pooled across contrasts instead
of per contrast; later stages reading the summary from the wrong level; a
`fingerprint` traceback outside a checkout. A first v1 repair draft reused v0's
asymmetric mutations, which would have left the answer identifiable; caught by
reading generated samples before any evaluation (`AAA-192`).

Design amendments made after observing development data, all recorded in the
brief with their reason: the censored training-budget grid (every arm hit the
top of {1, 3, 8}), the 64-epoch extension rule (declared before capacity), the
post-hoc 4K diagnostic (labelled, outside every verdict) and the 4K-over-1K
contract (added before the freeze).

## Mutation testing

Eight critical invariants were deliberately broken, one at a time, in a clean
tree: the parser on `syntax`, asymmetric repair mutations, single-fault
localization, extra and wrong tool inputs, the tool after commit, state-hash
coverage of optimizer velocity, `recompute` trusting stored means, and a
forged confirmation admission. The v1 suite caught all of them (the first
tool mutation was initially missed; the test was strengthened and then
caught both tool variants).

## Still weak

- The scale question is answered for one architecture family. A different
  ~10K design could behave differently.
- Adaptation and plasticity, the most AAA-specific results, rest on
  development evidence; the 4K comparison is post-hoc.
- Strong rules and the visible-test tool remain better than or equal to the
  learners on three of five families.
- No external benchmark could be attempted meaningfully.
- The same author chose the encoder, the heads and the diagnostics.
