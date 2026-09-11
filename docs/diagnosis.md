# Learner diagnosis

```bash
python -m aaa.cli diagnose --output docs/evidence/diagnosis
```

Development data only. This is not confirmation evidence and it does not select
the benchmark candidate; selection is [`candidate_selection.md`](candidate_selection.md).

## What it separates

The original v1 learner performed poorly. The question is *why*, and the honest
answer requires separating factors that a single fit cannot distinguish. The
diagnosis varies each independently — 480 configurations:

| Axis | Values |
|---|---|
| feature basis | legacy absolute positions, centered positions, displacement only, displacement + centered position |
| estimator | batch least squares (stable SVD reference), SGD, normalized LMS, square-root RLS, square-root RLS with the trace bound |
| regime | straight, bouncing, mixed |
| ordering | chronological, shuffled, equal-budget replay |
| hyperparameters | learning rate, forgetting factor, ridge |

and reports, per configuration: train and held-out MAE, weights, divergence,
update-norm mean and maximum, covariance trace, minimum eigenvalue, condition
number, and whether a negative minimum eigenvalue is inside reconstruction
rounding scale.

Per basis and regime it reports conditioning: singular values, effective rank,
condition number, column means and scales, and off-bias correlations.

## What it establishes

- **Conditioning and prediction are different things.** The legacy raw-position
  basis is rank-deficient on straight motion (effective rank 3 of 5, condition
  number ~5e16). Coefficients are then non-identifiable while predictions are
  essentially unchanged, so weight instability is not the same phenomenon as
  predictive error.
- **The trace bound is doing real work.** Unbounded forgetting at lambda=0.90
  on the legacy basis reaches covariance trace ~9e22 and loses positive
  semidefiniteness to reconstruction rounding. The bounded arm stays valid at
  the same held-out error. On the well-conditioned displacement bases the bound
  costs nothing.
- **Batch least squares is the reference.** A large gap between it and an online
  estimator on the same basis is an optimization or conditioning effect, not a
  capacity limit.
- **Ordering matters, and the effect differs by estimator.** Shuffling helps the
  gradient methods substantially on the bouncing regime and is close to neutral
  for RLS.

## What it does not establish

It does not claim a single root cause. Feature scaling, conditioning, optimizer
choice, regime mixing, sequential ordering and the nonlinear reflection rule are
each varied and each contributes; the data does not support collapsing them into
one story, and the report says so.
