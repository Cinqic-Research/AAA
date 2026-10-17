# `aaa.erudition.v0` handoff

Branch `opus/juniper1-erudition` of Cinqic-Research/AAA, starting from `main`
at `2628a0d`. Freeze `a750677` was pushed before any confirmation identity
existed. This handoff records the independent review disposition: **changes
required; not approved or merged**. The fixed-transformer confirmation is
preserved as v0 evidence. The surrogate training path and the runner's
security boundary require successor work before the branch can be treated as
safe, label-blind research code. The architecture-selection deviation also
means the transformer is not the model selected by the written rule.

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
  selected model, development, attack, the freeze and the confirmation. The
  report generated from it is the [development report](development_report.md).

## Confirmation outcome

Attempt 1 ran live on gpt-oss-20b on 2026-10-05/06:
- 12 fresh streams × 5 frozen arms (60 runs, 4,304 model exchanges);
- the code frozen at `a750677`;
- no interruption, and no other attempt.

Two independent implementations adjudicated the four frozen contracts, and
they agree:

| Contract | Verdict | Paired difference [95%] |
|---|---|---|
| `erudition_improves_juniper` | `PROMOTE` | failure −0.250 [−0.369, −0.133]; retention +0.002 [0.000, 0.005], margin 0.05 |
| `joint_over_world_model_only` | `PROMOTE` | failure −0.235 [−0.357, −0.117] |
| `joint_over_language_model_only` | `INCONCLUSIVE` | failure −0.001 [−0.094, 0.096] |
| `learned_control_versus_rules` | `INCONCLUSIVE` | failure +0.017 [−0.052, 0.101], margin 0.03; poisoned +0.019 [−0.062, 0.121], margin 0.02; false adaptation 0 |

The confirmation contradicted two development readings:
- that joint adaptation beats either single component;
- that learned control resists poisoning better than the rules.

The per-stream record shows the causes. On stream 02, the controller treated
a language shift as a World Model problem until step 98. On stream 05, it
accepted a consistent lie. These are recorded in the report (§7) and in
[limitations](../limitations.md). Fixing either needs a successor identity.

## Reproduce

From a checkout on a machine with the locked environments:

```bash
python -m unittest tests.test_aaa_erudition tests.test_aaa_erudition_promotion
python -m unittest tests.test_aaa_erudition_model            # torch environment
python tools/check_aaa_erudition_mutations.py                # every injected break must be caught
python tools/check_aaa_erudition_evidence.py --replay        # re-executes and replays every retained run (exact, except float32 diagnosis probabilities within 1e-4)
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
   Review found that `register_truth()` passes the true entity into the
   surrogate, whose unknown-name branch distinguishes a feasible entity from
   a nonexistent decoy. This changes simulated proposals and Q targets. The
   live confirmation used the real Language Model, so its measured outcomes
   remain results for the frozen transformer; the training trajectories are
   not label-blind. Removing this dependency requires retraining and a
   successor identity.
2. **The gate.**
   - The alias gate is in-sample: its target comes from the same corrections
     that produce the candidate, so consistent false corrections can
     self-certify. The canary misses some wrong-target actions that still
     produce the requested numeric result.
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
6. **The confirmation.**
   - Check that `attempt_log.txt` is consistent with the commit history of
     `confirmation_progress.md`, which was pushed periodically during the run.
   - Check that the 60 runs replay from `calls.jsonl.gz`.
   - Check that the verdicts recompute from the freeze.
   - The checker's file-history rule was changed after the confirmation was
     committed (report §8, item 10). Check that the change does not weaken it.
7. **The joint-shift family.** It was built to need both components, yet no
   controller recovers much there. Is the environment too slow, or the
   mechanisms too weak? The report says which evidence exists.
8. **Scale.** The one-million-parameter model is not shown to need its size
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
  - No scientific code or threshold was tuned after the confirmation. Later
    documentation edits record independent audit findings without changing
    the retained evidence or verdicts; exploratory tables remain labelled
    post hoc.
- **Future information across the causal boundary?** Runtime controller
  features, prompts and gate decisions are built from observed records. The
  training simulator is an exception: the surrogate receives the true entity
  and changes its proposal behavior based on feasibility, which shapes its
  histories and Q targets. The controller input tensor does not include that
  label, but v0 training was not label-blind; a successor is needed to remove
  this dependency.
- **Faked autonomy?**
  - After launch, every adaptation, rejection and rollback was decided by the
    controller and the frozen gate; no human chose examples, candidates or
    components.
  - Humans designed the rules and environment. A selection rule was reported
    in an off-Git working note with a pre-sweep file timestamp, but its
    existence is not Git-backed; the chosen transformer also violated its
    tie clause.
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
  - Consistent lies can self-certify through the alias replay gate. Stream 05
    retained a wrong alias for 57% of post-warm steps; the learned-versus-rules
    poisoning result is inconclusive.
- **Base weights untouched?** Adapters are bound to the expected artifact
  hash, and the runtime manifests record the model path and build. However,
  the live runner does not verify the loaded GGUF or served template
  fail-closed; that runtime identity is a recorded limitation.
- **LM and WM adaptation proven, and separable?** Only partly.
  - Language Model adaptation alone clearly reduces failure on the real model.
  - World Model adaptation alone barely does: the post-hoc confirmation
    interval reaches zero.
  - World Model learning pays off through the joint arm on dynamics shifts,
    but that is exploratory, and the frozen joint-over-Language Model
    contract is `INCONCLUSIVE`.
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

## Independent review disposition

- **BLOCKER — bearer-key disclosure.** `LlamaServer` checks the URL with a
  string prefix. A userinfo URL can pass while parsing to a remote hostname;
  `runtime()` then attaches the bearer key to the request. A no-network probe
  reproduced this with a dummy token. The recorded confirmation omitted
  `--url` and used the default loopback URL, so this does not show that the
  committed confirmation sent a key to a remote host. The runner still needs
  parsed-host validation and a regression test before approval.
- **HIGH — label-aware training simulation.** As described above, hidden
  feasibility changes surrogate proposals and Q targets. The frozen live
  outcomes remain measurements of this transformer; label-blind training
  requires retraining under a successor identity.
- **HIGH — architecture selection.** The declared 0.01 tie rule selects the
  smaller 1.020692M GRU, not the confirmed 1.2597M transformer. The
  transformer's confirmation remains evidence about that artifact, not a
  compliant architecture-selection result. The off-Git note timestamp does
  not establish Git-backed preregistration.
- **MEDIUM — poisoned aliases and canaries.** The alias gate can self-certify
  a consistent false correction; the canary's numeric-consistency condition
  can miss a wrong-target action. The frozen poisoning comparison remains
  `INCONCLUSIVE`.
- **MEDIUM — diagnosis timing.** The diagnosis target is generated after the
  adaptation decision but paired with the just-recorded features; it shares
  the Q head's trunk. It is not an inference input but can shape trained Q
  weights, and does not support a pre-decision diagnosis claim.
- **LOW/MEDIUM — state-store recovery.** Missing `heads.json` with extant
  lineage can discard the active head rather than fail closed. The normal
  interrupted-commit path and the run-store usage did not exercise this case.
- **Identity and decision.** Freeze `a750677` has its declared parent and one
  committed version. All 66 confirmation files were added once afterward;
  the 60 declared runs and 4,304 cache exchanges replay and recompute. The
  four statistical verdicts independently reproduce. No frozen file or
  scientific source was changed by this review. AAA #34 is not approved or
  merged; changing the fingerprinted runner or training path requires a
  successor identity and appropriately new evidence.
