# `aaa.erudition.v0` handoff

Branch `opus/juniper1-erudition` of Cinqic-Research/AAA, starting from `main`
at `2628a0d`. Freeze `a750677` was pushed before any confirmation identity
existed. This is the implementer's handoff for independent review; nothing is
merged and nothing is promoted by this document.

## What exists

- **Direction.** AAA is Accurate Autonomous Adaptation. Its active work is
  the Juniper 1 Erudition Model. The Python, dot-era and AAA-1K lines are
  historical, unchanged and still reproducible: all protected identities
  and 501 pre-existing protected files are unchanged, and new evidence was
  only added.
- **Code.** [`research/aaa_erudition/`](../../research/aaa_erudition/) contains:
  - contracts and the content-addressed store;
  - the ToolShift environment;
  - the World Model (`aaa.world.cbl.v0`);
  - the Language Model backend, adapter, characterization and surrogate;
  - the lifecycle and gate, and the control policies;
  - the Erudition Model, simulation and training;
  - metrics, recomputation, `aaa.promotion.paired.v1`, evaluation and
    admission.
- **Evidence.** [`docs/evidence/aaa_erudition_v0/`](../evidence/aaa_erudition_v0/)
  holds the characterization, the World Model diagnostic, the sweep, the
  selected model, development, attack, the freeze, and confirmation once
  executed. The report generated from it is the
  [development report](development_report.md).

## Reproduce

From a checkout on a machine with the locked environments:

```bash
python -m unittest tests.test_aaa_erudition tests.test_aaa_erudition_promotion
python -m unittest tests.test_aaa_erudition_model            # torch environment
python tools/check_aaa_erudition_mutations.py                # every injected break must be caught
python tools/check_aaa_erudition_evidence.py                 # re-executes every retained run
python tools/write_aaa_erudition_report.py --check
python tools/check_protected_identities.py
python -m research.aaa_erudition.recompute docs/evidence/aaa_erudition_v0/development/records/*.json.gz
```

Replaying a real-model run needs no model: the exchange record is in each
stage's `calls.jsonl.gz`.

```bash
gunzip -c docs/evidence/aaa_erudition_v0/development/calls.jsonl.gz > /tmp/calls.jsonl
python -m research.aaa_erudition.experiment --split development --indices 2 \
    --conditions joint --controller erudition --model docs/evidence/aaa_erudition_v0/model/erudition.pt \
    --backend replay --cache /tmp/calls.jsonl --out /tmp/replay
```

A live run additionally needs:
- the qualified GGUF (SHA-256 `9d7364f0…d23d`);
- llama.cpp b11270 built for CUDA with the recipe in the
  [compute record](compute.md);
- `--backend llama --key <key file>`.

Simulation and training run with `research.aaa_erudition.simulate` and
`research.aaa_erudition.train`; the commands and data hashes are in each
model's metadata.

## Where to attack first

1. **Sim-to-real.** The Erudition Model was trained against a surrogate.
   Compare the development report's simulated and real tables. One
   correctness leak through the surrogate's confidence was found and
   removed before training (failure record, item 3); check that no other
   surrogate behaviour depends on hidden truth beyond `register`'s tank.
2. **The gate.**
   - The alias gate is in-sample by design: it cannot tell a true correction
     from a consistent lie, and the defences are agreement and the canary.
   - The dynamics gate uses an estimator from recent observations that is
     shared by all arms. Check that it gives no arm information the others
     lack.
3. **Metric definitions.** `metrics.py` (deficiency flags in
   `lifecycle._score`, re-derived independently in `recompute._flags`). Two
   earlier definitions were tautological and were caught; look for more.
4. **Fairness of the factorial.** Only the controller's menu differs between
   arms. The revision and consultation turns are host rules applied to
   every arm. `recompute` refuses a run whose lineage changes a component
   its menu does not allow.
5. **Confirmation discipline.**
   - The freeze commit must precede every confirmation file.
   - The evidence checker verifies this ordering and that every declared run
     exists, and it adjudicates from the freeze's own contracts.
6. **The joint-shift family.** It was built to need both components, yet no
   controller recovers much there. Is the environment too slow, or the
   mechanisms too weak? The report says which evidence exists.
7. **Scale.** The one-million-parameter model is not shown to need its size
   on this environment.

## Self-review (implementer, not independent)

- **Historical identity modified?** No. The protected-identity check passes
  with 501 original files unchanged; new evidence was only added. No
  fingerprinted historical file was touched.
- **Evaluation data contaminated?**
  - Training used training-split identities only.
  - Development tuned the rule-based control and the integration contract,
    as documented.
  - The confirmation contracts were committed before any real-model
    Erudition run.
  - Confirmation streams were generated only under the freeze.
- **Future information across the causal boundary?** A test traps every read
  of the hidden label outside the scorer and the environment itself.
  Features, prompts and gate decisions are built from observed records only.
- **Faked autonomy?**
  - After launch, every adaptation, rejection and rollback was decided by the
    controller and the frozen gate; no human chose examples, candidates or
    components.
  - Humans designed the rules and the environment, and fixed the selection
    rule before the sweep.
- **Faked scale?** No dead parameters (tested). The evidence does not show
  the size is needed, and that is stated.
- **A Decision Model under another name?** No. The gate and canary are fixed
  host code; there is one learned controller.
- **Erudition duplicating the Language Model?** No. It reads 40 numeric
  features and no text.
- **A task-specific World Model presented as general?** It is task-shaped
  (an affine basis), and the architecture, decisions and limitations say so.
- **Rollback.** Every accepted state is reversible; this is tested and
  recomputed.
- **Poisoning.**
  - Notes are typed and template-rendered.
  - In the attack stage, no injected text reached any adapter state.
  - Consistent lies can still plant a wrong alias; this is measured as
    `poisoned`.
- **Base weights untouched?** Yes. The GGUF is read-only, its hash matches,
  and adapters are bound to that hash.
- **LM and WM adaptation proven, and separable?** Yes, by the single-component
  arms against frozen on the real model (development, and confirmation once
  executed).
- **Joint comparison fair?** The same streams and the same host rules; the
  menu is the only factor.
- **Regressions hidden by aggregates?** Retention, per-family failure,
  poisoning, false adaptation and cost are reported separately.
- **Claims supported?** The report's tables are generated from the evidence
  and checked in CI. Its narrative is not machine-checked. A second
  independent review corrected two overclaims: contract timing and
  poisoning.
- **Selection rule followed?** No. By its own tie clause it selects the
  1.02M GRU; the transformer was chosen by mistake (D-4, report §4). The
  rule note predates the sweep only by its file timestamp; it was outside
  Git.
- **Missing checks.**
  - Windows and macOS were not run.
  - The CI neural job runs a 2-trial World Model diagnostic, not the full one.
  - Live real-model runs cannot run in CI.
