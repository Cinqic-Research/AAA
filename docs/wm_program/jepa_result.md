# JEPA (J-4) on the opaque.v0 black-box world model: development result

Status: **development**, tune range, one seed. Code: `research/aaa_wm/opaque/jepa.py` and
`experiment.py jepa`. Checkpoints: `wm_s0_n20000_jepa.pt` and `wm_s0_n20000_jepa_novic.pt`.
Train logs, with diagnostics: `$AAA_DATA_ROOT/opaque/train_logs/jepa_s0_n20000*.json`.

The same black-box transformer world model was trained for 20,000 steps in three versions, and the
same depth-1 verified planner was scored on the tune range:

1. **plain:** the consequence objective only (tune-stage result);
2. **J-4:** consequence plus an action-conditioned JEPA term. The predictor maps the pre-edit
   representation and an edit embedding to the representation of the post-edit program, and the
   target is an EMA copy of the encoder under a stop-gradient. The term uses the normalized
   squared error plus a VICReg variance/covariance penalty on both the context and the prediction;
3. **J-4 without VICReg:** the positive control for collapse.

| Variant | Planner success (tune) | Effective rank of z (of 128) | Min per-dim std | Mean pairwise cosine | Action sensitivity |
|---|---:|---:|---:|---:|---:|
| plain | 0.191 | - | - | - | - |
| J-4 + VICReg | 0.175 | 124.6 | 0.66 | 0.64 | 0.139 |
| J-4, no VICReg | 0.167 | 89.9 | 0.31 | 0.63 | 0.102 |

Reading:

- **The anti-collapse machinery works as intended.** With VICReg the representation stays nearly
  full rank. Without it, about 30% of the dimensions collapse (effective rank 90) and action
  sensitivity falls. The JEPA loss reaches 0.004 without VICReg against 0.011 with it: a *lower*
  JEPA loss came with a *more* collapsed representation. That is the failure mode the protocol
  warns about.
- **JEPA did not improve decisions.** 0.175 and 0.167 against 0.191, one seed, all within tune noise.
  The hypothesis that an action-conditioned JEPA auxiliary improves the consequence model's usefulness
  for planning is **not supported**, and it is rejected for this benchmark.
- Why, most likely: the edit transition in text space is exact and cheap, so predicting the post-edit
  representation adds nothing the model lacks. The black-box model's binding limit is execution
  modeling (deficiency A), which a latent consistency objective does not address. That matches the
  literature expectation recorded before the run (`notes/lit_world_models_jepa.md`, sections 9.1 and 9.2).
