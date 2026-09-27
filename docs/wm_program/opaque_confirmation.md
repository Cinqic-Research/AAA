# `opaque.v0` world-model confirmation

Status: **CONFIRMATION**, fresh identities, observed once. Freeze: `docs/evidence/aaa_wm_opaque_v0/freeze.json`,
committed at `f994d91` before any confirmation identity existed. Evidence:
`docs/evidence/aaa_wm_opaque_v0/confirmation.json`, run from that commit with a clean tree. It holds the
per-(arm, initialization, task) success bits, the task hashes, the freeze and the adjudications.

- **Design.** 4,000 fresh `confirmation` tasks (every pilot, train, development and attack program
  excluded) in 5 slices x 20 slice-pure streams of 40. Initializations 100-104 for every seeded arm. Run
  budget 2, step budget 8.
- **Artifacts.** The WM-S tables and neural checkpoints were produced from the train split before the freeze.
  Their SHA-256 hashes were verified before the run.
- **Runtime.** 7 h 59 min, one capped process, 3.3 GB peak RSS.

## Results (crossed bootstrap over initializations and streams)

| Arm | All [95%] | in-dist | two_fault | novel_literals | novel_grammar | library_B |
|---|---|---:|---:|---:|---:|---:|
| true-library planner (ceiling) | 0.779 [0.763, 0.796] | 0.819 | 0.714 | 0.845 | 0.723 | 0.797 |
| **WM-S online (candidate)** | **0.568 [0.467, 0.658]** | 0.649 | 0.553 | 0.691 | 0.539 | 0.409 |
| WM-S frozen | 0.506 [0.369, 0.631] | 0.596 | 0.481 | 0.629 | 0.480 | 0.347 |
| WM-S online, empty library | 0.197 [0.183, 0.210] | 0.218 | 0.151 | 0.214 | 0.178 | 0.223 |
| tells (grammar shortcut, no WM) | 0.171 [0.157, 0.185] | 0.179 | 0.085 | 0.206 | 0.203 | 0.182 |
| black-box neural WM, depth 1 | 0.148 [0.129, 0.168] | 0.221 | 0.085 | 0.166 | 0.078 | 0.189 |
| policy_aux inside the WM-S planner | 0.111 [0.098, 0.125] | 0.141 | 0.044 | 0.145 | 0.081 | 0.144 |
| policy_aux, tool search | 0.102 [0.088, 0.116] | 0.126 | 0.037 | 0.135 | 0.076 | 0.136 |
| WM-S, empty library (interpreter only) | 0.052 [0.046, 0.059] | 0.065 | 0.019 | 0.059 | 0.058 | 0.062 |
| tool search (gap-reading) | 0.037 [0.031, 0.045] | 0.049 | 0.005 | 0.060 | 0.031 | 0.043 |

`wms:online+empty` has no seed-dependent state, so it scores identically at every initialization, as expected.

## Contracts (`aaa.promotion.crossed.v1`; primary and independent implementations agree on every one)

The metric is the geometric mean over slices of the per-cell Jeffreys-smoothed error ratio, challenger / reference.

| Contract | Reference | Error ratio [95%] | Rule | Verdict |
|---|---|---|---|---|
| C1 WM over the strongest learned non-WM arm | policy_aux:plan | 0.481 [0.377, 0.600] | superior < 0.75 | **PROMOTE** |
| C2a the training-learned library is causal | wms:online+empty | 0.532 [0.416, 0.660] | superior < 0.80 | **PROMOTE** |
| C2b the library model is causal | wms:empty | 0.452 [0.354, 0.560] | superior < 0.75 | **PROMOTE** |
| C3 WM over the best non-learned shortcut | tells | 0.516 [0.404, 0.641] | superior < 0.75 | **PROMOTE** |
| C4 online adaptation | wms (frozen) | 0.873 [0.838, 0.932] | superior < 0.95 | **PROMOTE** |
| C5 structured over black-box WM | wm:d1 | 0.502 [0.392, 0.632] | superior < 0.75 | **PROMOTE** |
| N in_distribution not worse | policy_aux:plan | 0.417 [0.290, 0.564] | noninferior <= 1.0 | **PROMOTE** |
| N two_fault not worse | policy_aux:plan | 0.474 [0.372, 0.591] | noninferior <= 1.0 | **PROMOTE** |
| N novel_literals not worse | policy_aux:plan | 0.370 [0.257, 0.504] | noninferior <= 1.0 | **PROMOTE** |
| N novel_grammar not worse | policy_aux:plan | 0.508 [0.390, 0.634] | noninferior <= 1.0 | **PROMOTE** |
| N library_B not worse | policy_aux:plan | 0.695 [0.632, 0.764] | noninferior <= 1.0 | **PROMOTE** |

The post-run audit (`confirm.audit`) recomputed every stored verdict from the bits and found no disagreements.

## What this licenses, and what it does not

Licensed (subject to independent review), within `aaa.python.opaque.v0`: at a budget of two real
test runs, an AAA agent that plans with **WM-S** makes roughly half the errors of the following:

- the strongest learned model-free agent, even inside the same planner;
- the strongest non-learned shortcut;
- the same agent without its learned library model;
- a black-box neural world model.

WM-S is an exact interpreter of the visible Python plus a tabular model of the opaque library,
learned from real observations and updated online. It is not worse on any slice. Updating the
library model online improves on the frozen model (C4).

Not licensed:

- that a *generic* neural world model helps (the black-box WM loses, and so does JEPA);
- that WM-S learned Python (the interpreter is hand-written knowledge);
- transfer beyond this generated benchmark family;
- freedom from forgetting (adaptation to a changed library costs -0.067 on the old library in development);
- that compute is matched (WM-S uses about 0.3-0.4 CPU seconds per episode against about 0.02-0.3 s for the
  neural arms; the matched resource is real test runs).

The result also reflects a benchmark flaw found by the red team: the grammar shortcut. It is included
as the C3 reference rather than repaired.

Independent adversarial review is pending. Until it has tried to falsify this, no success is declared.
