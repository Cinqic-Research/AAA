# Candidate selection

```bash
python -m aaa.cli select-candidate --output docs/evidence/candidate_selection.json
```

Development streams only. No confirmation batch was inspected, and every
previously viewed benchmark-v2 confirmation stream is treated as contaminated
for selection purposes. The full machine-readable table, including every
unsuccessful variant, is
[`evidence/candidate_selection.json`](evidence/candidate_selection.json); the
earlier round is retained as
[`evidence/candidate_selection_first_pass.json`](evidence/candidate_selection_first_pass.json).

## Selection rule, declared before the table was read

> among numerically stable variants with development straight-motion normalized MAE <= 1e-5 that also satisfy the declared always-online non-regression requirement on development streams, maximize development changed-law improvement over the identical frozen copy; ties break toward the smaller feature set and then toward less forgetting

Straight-motion admission limit: `1e-05` normalized MAE.
Tie tolerance: `0.02`. Always-online admission is the same
criterion in form as the benchmark's `always_online_stability` gate:
`<= 1.1 x constant_motion_reflected
+ 1e-05`.

Variants evaluated: **36**. Admitted: **20**.
Tied within tolerance: **4**.

## The selected candidate

| Property | Value | Why this and not something else |
|---|---|---|
| feature set | `displacement_position` | `displacement_only` cannot express the boundary-adjacent position term; adding the centered position costs one parameter and is the smallest set that admits the changed-law law |
| forgetting | `0.5` (`exponential`) | the admitted variant with the largest changed-law improvement over its frozen twin; slower forgetting adapts measurably less (see the table below) |
| ridge | `0.0001` | initial covariance is `I / ridge`; alternatives were compared and none was admitted with a better outcome |
| trace bound | `100000` | bounds covariance windup under weak excitation; unbounded arms lose positive semidefiniteness (see the diagnosis) |
| reflection | `True` | programmed public knowledge of the observation format; the like-for-like reflected baseline exists precisely so this is never counted as learning |
| unfold target | `True` | the inverse of the same public map is applied to the update target so the learner regresses in the coordinate its own linear law lives in |
| dead zone | `0` | off; it was not needed once forgetting is trace-bounded and self-triggered |
| detector multiplier | `8` | forgetting is applied only on steps the model's *own* scored error says are surprising; the evaluator never signals an event |
| training budget | `24` episodes | the learning curve is flat well before this point; a larger budget buys nothing measurable |

Measured on development streams:

- straight-motion normalized MAE `3.481e-11`
- bouncing normalized MAE `5.771e-06`
- changed-law improvement over the identical frozen copy `0.6865`
- always-online margin against the reflected baseline `-1.749e-06` (<= 0 satisfies)
- numerically stable under the stress suite: `True`, stress failures `0`

## Every variant considered

Unsuccessful alternatives are retained deliberately. A selection table that
only lists the winner is not evidence.

| Variant | straight MAE | bouncing MAE | changed-law vs frozen | always-online margin | stable |
|---|---:|---:|---:|---:|---|
| `disp+pos|lambda=0.3|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 4.029e-11 | 5.771e-06 | 0.7714 | 0.0005937 | yes |
| `disp+pos|lambda=0.4|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 2.245e-11 | 5.771e-06 | 0.7347 | 4.436e-05 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=2|episodes=24` | 1.741e-11 | 5.771e-06 | 0.709 | 0.0009183 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=4|episodes=24` | 3.481e-11 | 5.771e-06 | 0.7063 | 6.367e-05 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.01|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 8.874e-10 | 5.772e-06 | 0.6922 | -1.747e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=1e+06|deadzone=0|unfold=1|detector=8|episodes=24` | 3.481e-11 | 5.771e-06 | 0.69 | -1.749e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=1e-06|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.343e-12 | 5.771e-06 | 0.6866 | -1.752e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 3.481e-11 | 5.771e-06 | 0.6865 | -1.749e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=8|episodes=24` | 3.481e-11 | 5.771e-06 | 0.6865 | 8.089e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=0|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 3.481e-11 | 1.212e-05 | 0.6865 | 7.489e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=1e-07|unfold=1|detector=8|episodes=24` | 3.944e-08 | 5.811e-06 | 0.6703 | 8.089e-07 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=10000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.343e-10 | 5.771e-06 | 0.6605 | -1.752e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=0|episodes=24` | 1.96e-19 | 5.771e-06 | 0.66 | 0.0009528 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.96e-19 | 5.771e-06 | 0.66 | 8.783e-05 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=16` | 5.062e-11 | 5.771e-06 | 0.6577 | -1.701e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=4` | 3.183e-10 | 5.771e-06 | 0.6553 | -1.513e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=8` | 1.318e-10 | 5.771e-06 | 0.6552 | -1.602e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=16|episodes=24` | 3.481e-11 | 5.771e-06 | 0.6088 | -1.33e-06 | yes |
| `disp+pos|lambda=0.6|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 4.975e-11 | 5.771e-06 | 0.5954 | -3.251e-06 | yes |
| `disp+pos|lambda=0.9|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 4.214e-19 | 5.771e-06 | 0.5715 | 6.352e-05 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=1000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.343e-10 | 5.771e-06 | 0.5288 | -1.541e-06 | yes |
| `disp+pos|lambda=0.7|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 6.722e-11 | 5.771e-06 | 0.5003 | -6.081e-06 | yes |
| `disp+pos|lambda=0.5|dir|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=0|episodes=24` | 1.145e-09 | 5.772e-06 | 0.4868 | 1.284e-05 | yes |
| `disp+pos|lambda=0.5|dir|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.145e-09 | 5.772e-06 | 0.4868 | 2.637e-05 | yes |
| `disp+pos|lambda=0.95|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.176e-19 | 5.771e-06 | 0.3317 | 5.158e-05 | yes |
| `disp+pos|lambda=0.8|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 8.716e-11 | 5.771e-06 | 0.3088 | -6.582e-06 | yes |
| `disp+pos|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=32|episodes=24` | 3.481e-11 | 5.771e-06 | 0.2999 | -5.849e-07 | yes |
| `disp+pos|lambda=0.98|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.96e-19 | 5.771e-06 | 0.0963 | 2.46e-05 | yes |
| `disp+pos|lambda=0.9|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.095e-10 | 5.771e-06 | 0.07344 | -8.877e-06 | yes |
| `disp+pos|lambda=0.9|dir|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 2.113e-11 | 5.771e-06 | 0.06679 | 2.308e-05 | yes |
| `disp+pos|lambda=0.5|dir|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 2.95e-10 | 5.771e-06 | 0.03436 | -4.814e-06 | yes |
| `disp|lambda=0.5|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 3.596e-11 | 5.771e-06 | 0.02587 | -4.7e-07 | yes |
| `disp+pos|lambda=0.99|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.96e-19 | 5.771e-06 | 0.0141 | 1.804e-05 | yes |
| `disp+pos|lambda=0.95|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.216e-10 | 5.771e-06 | 0.005528 | -9.32e-06 | yes |
| `disp+pos|lambda=1.0|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=1|detector=8|episodes=24` | 1.343e-10 | 5.771e-06 | -0.0001668 | -9.646e-06 | yes |
| `disp+pos|lambda=1.0|exp|ridge=0.0001|reflect=1|trace=100000|deadzone=0|unfold=0|detector=0|episodes=24` | 1.343e-10 | 5.771e-06 | -0.0001668 | -4.915e-06 | yes |

## Discipline

- development streams only; no confirmation batch was inspected.
- Thresholds in the benchmark specification were frozen before any
  confirmation batch was generated, and none was chosen by looking at a
  confirmation result.
- The selected candidate is the smallest understandable system that satisfies
  the research objective. Parameter count was not increased because more
  compute was available; the model has three parameters.

