# Contributing to AAA

AAA is a research programme whose whole point is that its measurements can be
trusted. Its current focus is **Coding, beginning with Python** (`aaa.python.v0`);
the moving-dot work is retained as evidence and benchmark lineage
([archive](docs/dot_benchmark_archive.md)). Contributions are welcome; the
conventions below exist to keep that property.

## Before you change anything

On FLOWBOX, first verify the mounted and writable `Cinqic Storage` HDD with
`lsblk` and `findmnt`, then use an HDD checkout. From that checkout run
`python3 tools/storage_preflight.py --path "$PWD"`; set `$AAA_DATA_ROOT` to
an HDD directory and validate it with a second `--path`. The
[work policy](docs/agent_work_policy.md) is authoritative for engineers and
agents. GitHub `main` is canonical; local and Hugging Face copies follow the
[backup policy](docs/backup_policy.md).

```bash
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -t .
```

## The rules that matter most

**Reproduce before you repair.** If you believe something is wrong, write a
probe that demonstrates it against the current code first, and keep that probe
as a regression test. Every entry in [`docs/issue_ledger.md`](docs/issue_ledger.md)
follows this shape.

**A gate must measure what its name says.** If a criterion can be satisfied by
something that does not have the property being claimed — a parameterless rule
passing a "learning" gate, an empty collection passing a coverage gate — the
gate is broken even when it is green.

**Absence of evidence is not `PASS`.** A missing interval, a missing stratum, an
unrun check: `NOT_VERIFIED` or `INSUFFICIENT_EVIDENCE`. Never silently
satisfied.

**Never tune a threshold to a result you have seen.** Thresholds are frozen
before confirmation batches are generated. If a criterion turns out to measure
the wrong thing, replace the *concept* and document the scientific reason —
and do it on development data.

When a confirmation fails, the failure is the deliverable. Keep it, retire its
batch, diagnose on development data, change the *system* if the diagnosis
warrants it, declare a fresh batch, and let the multiplicity accounting charge
you for the extra attempt. This has happened once already and the whole chain is
in `AAA-120`; follow that shape.

**Improving a model goes through the loop.** A change meant to make a model
better follows [`docs/loop_protocol.md`](docs/loop_protocol.md): use scratch or
diagnostic identities for exploratory work, commit hypotheses and rules before
adjudicated runs, attack on fresh attack identities, and freeze before any
confirmation identity is observed. Spent identities are never reused, even
when an attempt aborts before observation. A promotion requires independent
review and the platform-appropriate reproduction standard. The current
governance version is `aaa.loop.v1`; historical pilot artifacts remain v0.

**Keep confirmation streams clean.** Any confirmation stream that has been
looked at is contaminated for model selection forever. Do selection on
development data.

**Some documents are scientific identity, not prose.** The `aaa.1k.v1` phase
fingerprint (`research/aaa_1k/identity.py`) hashes these files byte for byte:

- `docs/aaa_charter.md`, `docs/aaa_1k_architecture.md`,
  `docs/aaa_1k_literature_review.md` and `docs/aaa_1k_decisions.md`;
- everything under `research/aaa_1k/`;
- `aaa/__init__.py`, `aaa/config.py`, `aaa/environment.py` and
  `aaa/predictors.py`;
- `tests/test_aaa_1k.py` and `requirements-lock.txt`.

The resulting hash, `5ce6e019...`, is recorded in the Champion 0 and Champion 1
records and in the retained AAA-1K evidence. Changing any byte of any file
above, including a typo fix, an added link or reformatting, changes the hash.
`python -m research.aaa_1k_loop champion --verify`,
`python -m research.aaa_1k_loop.champion1 verify` and
`tests.test_aaa_1k_loop` then fail, and so does CI. That is the check working,
not a defect to route around. Do not update the recorded hash, edit the
evidence to match, or remove a file from the fingerprint to make it pass:
historical evidence is never rewritten.

Before touching one of these files, ask whether the change belongs somewhere
else. Documentation that only needs to *refer* to the charter can link to it
from an unfingerprinted document, as [`docs/hardware.md`](docs/hardware.md)
does. `docs/aaa_1k_report.md`, `docs/aaa_1k_handoff.md` and
`docs/aaa_1k_self_review.md` are generated phase outputs and are excluded from
the fingerprint, so editing them does not change the hash. A change
that genuinely has to alter a fingerprinted file is a change to the phase's
scientific identity. It needs its own recorded decision and versioning, like
`AAA-152`, not an ordinary documentation PR. Run
`python -m research.aaa_1k fingerprint` before and after an edit if you are
unsure whether a file is covered.

**`aaa.1k.v2` is frozen too.** Its fingerprint (`research/aaa_1k_v2/identity.py`)
covers `research/aaa_1k_v2/`, `aaa/compute/`, the historical code v2 executes,
`tests/test_aaa_1k_v2.py`, `tests/test_compute.py`, the v2 brief, architecture,
benchmark protocol and external-benchmark documents, and both locks. The
committed freeze (`docs/evidence/aaa_1k_v2/freeze.json`) records it, and
`tests/test_aaa_1k_v2_evidence.py` fails if any covered byte changes. The v2
report, compute report, self-review, handoff, decision log and
`tools/write_aaa_1k_v2_report.py` are outside the fingerprint. A later phase
that needs different v2 code copies it into a new versioned package; it does
not edit `research/aaa_1k_v2/`.

**`aaa.python.v0` has its own identity, and its evidence is protected like the rest.**
`python -m research.aaa_python fingerprint` hashes `research/aaa_python/`,
`aaa/promotion/`, the `tests/test_aaa_python*.py` files, `tests/test_promotion.py`,
the phase brief, protocol, architecture and promotion-contract documents, and the
lock. Changing the exam (the specification, subset, generator, oracle or
answer keys) is a new generator or protocol version: the golden answer keys
refuse to be rewritten (`golden --write` exits 2), and every earlier task
identity must keep meaning the same program. Changing the learner does not
change the exam. `tools/check_protected_identities.py` fails if any retained
dot-era evidence byte or identity moves; adding new evidence is fine.

**Python tasks keep the causal boundary.** A learner sees only a `TaskView`,
never a `Task`. Feedback is post-action and limited to the family's declared
fields. Answer keys come from CPython, never from what a generator meant to
produce. No program runs without passing the subset validator, and none runs
outside the sandbox. A new representation must not smuggle in an interpreter
verdict; `AAA-184` is what that looks like.

**Promotion uses a versioned contract.** A promotion decision declares an
`aaa.promotion` contract before held-out observation. The frozen
`aaa.1k.v2` K5 path is forbidden (`AAA-180`). The primary and independent
implementations must stay structurally independent; do not "fix" a
disagreement by making one call the other.

**Every specification value must be read.** `tests/test_spec.py` fails if a
declared leaf stops being consumed. If you add a value to the specification,
use it.

**Keep the causal boundary.** Predict, record, advance, reveal, score, then
update. No scenario name, event flag, velocity, change schedule, hidden
coefficient or future observation may reach a predictor.

**Failures stay visible.** A numerical problem raises. A failed confirmation is
retained and its batch retired. Nothing is quietly reset, clipped, symmetrized
or dropped.

## Checks that must pass

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m unittest discover -s tests -t .
python tools/check_lock.py
python tools/check_exit_codes.py
python tools/check_protected_identities.py
python -m aaa.promotion selftest
python -m research.aaa_python safety
python -m research.aaa_python leakage
python -m research.aaa_python golden
```

`unittest` is the project's test framework. Please do not introduce a second
one. No blanket `# type: ignore`: if the checker objects, the annotation is
usually the thing that is wrong.

For substantial AI-assisted repository work, report the task start, end, and
elapsed wall-clock time, with phase timings when useful. Keep this lightweight
process record separate from scientific experiment evidence.

## Adding a gate

1. Declare it in `aaa/benchmark/data/benchmark_v2_1.json` with its evaluator,
   `required` flag, description and thresholds.
2. Implement the evaluator in `aaa/benchmark/gates.py`.
3. Add deterministic synthetic cases in `tests/test_gates.py` proving `PASS`,
   `FAIL`, and `NOT_VERIFIED` or `INSUFFICIENT_EVIDENCE` where reachable.
4. Changing the specification changes its hash, which retires every batch
   declared against the old one. That is intentional.

## Scope

The active scope is `aaa.python.v0`: the frozen safe subset, its five task
families, its baselines and its minimal learner. Rungs of the
[capability ladder](docs/research_direction.md) beyond it are added in order,
each with baselines that can beat it. Not in scope now: English or natural-language
training, other programming languages, scraped code corpora, executing
untrusted code, larger models before a measured deficiency justifies them,
and any self-modification of AAA's own code.

The dot-era phases are frozen. Observation noise lives only in the separately
versioned `aaa.observation_noise.v1.1` phase; do not fold it into benchmark
v2.1 or change either protocol's frozen thresholds, samples, endpoints or
historical evidence. See [`docs/limitations.md`](docs/limitations.md) for the
current evidence boundary.
