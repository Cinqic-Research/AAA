# Learner diagnosis

Run the reproducible diagnostic with:

```bash
.venv/bin/python -m aaa.cli diagnose --output diagnosis
```

It generates `diagnosis.json` and a short report from newly generated straight-motion rows. The diagnostic fits the exact legacy five-feature displacement family with stable SVD least squares and records rank, singular values, conditioning, target scale, feature correlations, and prediction MAE. Coefficient non-identifiability is kept separate from predictive accuracy.

The legacy `OnlineLinearPredictor` remains available as `linear_online` and is not silently replaced. Benchmark v2 declares `OnlineRLSPredictor` as its smallest candidate: zero-initialized, normalized observation-derived features, full covariance checkpoint state, numerical symmetry/finite checks, and deterministic observation-only reflection. The changed-law family is held out from candidate training; its updating/frozen branches are matched at the intervention boundary.

The diagnosis does not claim that conditioning alone caused the original online result. Mixed-regime fitting, SGD budget, order sensitivity, feature scaling, and the nonlinear reflection rule remain separate causal factors. The benchmark and its recorded ablations are the evidence used for candidate selection.
