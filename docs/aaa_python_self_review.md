# Python-first transition: implementer self-review

**This is the implementer's self-review, not an independent review.** The
same author built `aaa.python.v0`, the `aaa.promotion.crossed.v1` successor,
the protected-identity guard and the documentation changes, and then tried to
break them. An independent reviewer should treat every conclusion here as a
claim to check, starting with the "still weak" list.

## Scope and starting state

Work began from `main` at `f56caaf0c0c5e38882c1464f24e7a62441df1f32`
(PR #27 merged, no open pull requests, 694 tracked files, verdict
`READY_FOR_NEXT_PHASE_DESIGN`, `AAA-180` open). Before any change, the
branch recorded every protected identity
(`benchmarks/protected_identities.json`), ran the 772-test baseline suite
(OK, 1 skipped) and reproduced `AAA-180` independently from retained
evidence.

## What was attacked, and what broke

| Attack | Found | Disposition |
|---|---|---|
| Reproduce `AAA-180` without the report tool | exactly as recorded, plus the recompute's silent Monash skip | `AAA-182`; prospective successor; frozen v2 unchanged |
| Omit, duplicate, mistype or corrupt successor primitives | unhashable identities escaped as `TypeError` instead of `INVALID_EVIDENCE` | fixed in both implementations independently; tests |
| Inject disagreements between the two successor implementations | none passed | `DISAGREEMENT` on point, bound, status, criterion and verdict |
| Can a representation see the answer? | `ast_nodes` carried CPython's parse verdict into `syntax` (accuracy 1.0) | `AAA-184`; refused for `syntax` and fragments |
| Can a surface rule solve a family? | a fixed fault name and a single risky line let a heuristic localize 92% in the quick smoke | fault names, value-dependent faults and decoys; the heuristic still wins at 0.73, which is reported (`AAA-185`) |
| Can an answer pass the label check without being the label? | `True` and `1.0` passed as label `1` | type-strict commit; malformed confidence refused |
| Do evaluation tasks exist in training? | the runner reached past the declared development pool | pools enforced in `build()`; lookup hit rate 0% measured (`AAA-185`) |
| Does resume work? | it could never match its own checkpoint | JSON-normalized plan comparison; resume reproduces the run (`AAA-185`) |
| Is every declared value read? | four were not, and the first test of it passed vacuously | code reads them; the test fails on an injected unread leaf (`AAA-185`) |
| Does the memory-disabled control forget everything? | it kept its update counter | reset restores all persistent state (`AAA-185`) |
| Do current documents still present the dot as the objective? | README, CONTRIBUTING, CITATION, SECURITY and hardware did | `AAA-183`; protected history untouched, errata instead |

All of these were found and repaired before the committed development
evidence was produced, or they are behaviour-neutral for it. The final HEAD reproduces
the retained records, cells and summary exactly (digest `fe489b73...`) from a
clean worktree.

## Verification performed (FLOWBOX, CPython 3.12.3, Ryzen 7 5700G, CPU only)

- Ruff lint and format, mypy (170 files), the lock check (24 pins,
  `2b5a2309...`), and the exit-code contract all pass.
- The full unit suite passes with branch coverage. The new core modules are
  81-98% covered; `_sandbox_child.py` and some CLI paths run only in
  subprocesses, which coverage does not trace.
- Protected identities are unchanged: v2.1 `f8e1090b...`, observation-noise
  protocol `546e2434...`, `aaa.1k.v1` `5ce6e019...`, `aaa.1k.v2`
  `e5b11bdb...`, and 471 evidence and historical files. The repository-wide
  observation-noise source fingerprint moved (`4d64ceb8...` to
  `a5607423...`), as it must whenever a file is added (`AAA-152`).
- AAA-1K parameter audit and full gradient check (4,918 parameters, 0
  violations); Champion 0 verified; loop records 0001-0006 valid; Champion 1
  valid; loop-0006 recompute `PROMOTE`, agrees; v2 audit, registry (270
  blocks), identity disjointness, recompute `NO_CHALLENGER` agrees, report
  current; the Q2 adaptation-parity diagnostic passes.
- Promotion: `selftest` passes, and the successor agrees with itself on all
  six retained v2 arms, while the frozen code still shows the `AAA-180` split.
- `aaa.python.v0`: 68/68 safety, 52/52 leakage, golden keys identical, the
  generation digest is stable, independent recomputation of the retained
  evidence (1,100 programs re-executed) passes, the report is current, and
  `confirm` exits 2.
- A wheel built with the locked backend and installed into a clean venv ran
  the v2.1 and noise identities, the promotion selftest and the whole v0 smoke
  (golden, safety, leakage, develop, recompute) from an unrelated directory
  whose path contains a space. Its quick run reproduced the checkout's
  records byte for byte. `fingerprint` and `confirm` refuse there with exit 2.

**Not verified locally:** CPython 3.10, 3.11 and 3.13 are not installed on
FLOWBOX, so cross-version answer-key equality rests on the CI matrix. The
first CI run proved the golden keys identical on all four versions. It also
failed on 3.10 and 3.11, because recomputation compared summary floats
exactly and CPython 3.12 changed float `sum()`. That is `AAA-186`, found by
CI rather than by this self-review, and now repaired. No CUDA
evidence exists or is claimed for this work. The full
`loop-reproduction.yml` re-execution was not rerun, because no file it covers
changed.

## Still weak

- **The environment is the implementer's design**, and the learner is too.
  Heuristics still beat the learner on syntax and localization; other
  construction cues may remain undetected.
- **The development evidence is small**: 3 initializations x 4 streams gives
  wide intervals, and most contrasts are `INCONCLUSIVE`, not zero.
- **Summary recomputation reuses `summarize`.** The independent parts of
  `recompute` are the fresh CPython re-execution of every scored task and the
  re-aggregation of counts from records. The step from cells to summary checks
  derivability, not implementation independence.
- **The sandbox is defence in depth for generated code**, not a hardened
  jail: it has POSIX resource limits and no network or filesystem namespace
  isolation.
- **The learner has 153,600 parameters but little structure.** It is a
  pipeline instrument, and its negative result says little about what a
  better-designed learner could do.
- **No confirmation** exists for any Python claim, and none is admitted.
