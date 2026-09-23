# AAA-1K v2 literature review: mechanisms for the final 1K pass

Written before any v2 development identity was observed. It extends, and does
not replace, the `aaa.1k.v1` review ([`aaa_1k_literature_review.md`](aaa_1k_literature_review.md)).

The rule is unchanged: extract a *mechanism*, tie it to a weakness AAA has
actually measured, count what it costs in parameters and adaptive state, say
how it is expected to fail, and adopt it only if it is testable at 1,000
parameters on the benchmarks this phase has. A result on billion-parameter
models or unrelated tasks is context, not evidence.

The measured weaknesses referred to below are listed in
[`aaa_1k_v2_research_brief.md`](aaa_1k_v2_research_brief.md): M1/Q4 (the gated
core's coarse-family deficit), M2 (repaired in Champion 1, re-tested here at
new horizons and geometries), `AAA-169` (stale-activation TBPTT), R-10
(zeros double as missingness), R-11 (a weak, over-predicting error head) and
R-05 (no independently designed benchmark).

## Summary

| Mechanism | Weakness addressed | Parameter / state cost at 1K | Expected failure | Decision |
|---|---|---|---|---|
| GRU (Cho et al. 2014; Chung et al. 2014) | baseline of Champion 1 | 3 gates: 16 units for 994 | gates idle at the zero-bias point (M1) | retained as reference and as two candidates |
| Minimal gated unit (Zhou et al. 2016) | M1: fewer gates, more width | 2 gate blocks: 19 units for 952 | loses the reset gate's selective forgetting | **adopted** (`mgu_v2`) |
| Ungated tanh recurrence at equal parameters | M1: width over gating | 28 units for 982 | less stable at high learning rates (v1 measured this) | **adopted** (`elman_v2`) |
| Chrono / gate-bias initialization (Tallec & Ollivier 2018) | M1 operating point; long memory | zero parameters | a bias helps one family and hurts another (loop c2) | **adopted** as `gru_v1_keep-2` (the loop's evidence favoured -2, not chrono's positive biases); chrono itself **deferred** |
| Diagonal linear recurrence, stable by construction (Orvieto et al. 2023) | long-horizon stability; long memory | 24 complex modes + readout = 957 | linear state cannot represent sharp nonlinear dynamics without a strong readout | **adopted** (`lru_v2`) |
| Exact RTRL for diagonal recurrence (Zucchet et al. 2023; Williams & Zipser 1989) | `AAA-169` truncation and staleness | eligibility traces: 240 scalars | noisy online gradients; trace state is extra adaptive state | **adopted** with the LRU (traces counted) |
| Truncated BPTT variants (Williams & Peng 1990) | `AAA-169` | cache: horizon x cache size | replay costs a forward window per step | **tested** as grid dimension (`live` / `replay`, horizon 1/4/16) |
| UORO (Tallec & Ollivier 2018b) / SnAp (Menick et al. 2021) / e-prop (Bellec et al. 2020) | unbiased or cheaper online credit | UORO: two rank-1 vectors; SnAp-1: one trace per parameter | high-variance or biased gradients; SnAp-n state grows fast | **deferred**: exact RTRL is affordable for the diagonal core, and the gated cores' horizon effect is measured directly |
| Unitary / orthogonal RNNs (Arjovsky et al. 2016; Lezcano-Casado & Martinez-Rubio 2019) | long-horizon stability | reparameterization overhead | norm preservation hampers forgetting in changing worlds | **rejected** for 1K: the LRU tests stability-by-construction more cheaply |
| IndRNN / antisymmetric RNN (Li et al. 2018; Chang et al. 2019) | stability | IndRNN: diagonal recurrence | neurons interact only through depth | **rejected**: overlaps the LRU hypothesis without its exact online gradient |
| Stable recurrent models (Miller & Hardt 2019) | M2-type runaway | none | a stable recurrence can still diverge through the online target | **context**: M2 was a target-construction lock, not an unstable recurrence (loop `AAA-170`) |
| Missingness indicators (Lipton et al. 2016; Che et al. 2018, GRU-D) | R-10 | one input column: 45-60 parameters at 15-28 units | the flag may be ignored if zeros already suffice | **adopted** (`v2` feature set, with a `no_observed_flag` ablation); GRU-D's decay **deferred** |
| Mean-variance / error-magnitude heads (Nix & Weigend 1994) | R-11 | 1 output: 17 parameters (GRU-16) | weakly informative, biased high (v1 measured) | **tested** by the `no_error_head_loss` ablation |
| Calibrated regression (Kuleshov et al. 2018); proper scoring (Gneiting & Raftery 2007) | R-11 calibration | post-hoc recalibration map | recalibration needs held-out data an online learner lacks | **deferred**; slope, bias and rank correlation are reported, including under shift |
| Deep ensembles (Lakshminarayanan et al. 2017) | uncertainty | k x parameters | exceeds the cap | **rejected** at 1K |
| Echo-state reservoirs (Jaeger 2001; Lukosevicius & Jaeger 2009; Rodan & Tino 2011) | long memory cheaply | fixed reservoir weights are not trainable but are hidden capacity | hides capacity outside the parameter count | **rejected** as a candidate (the capacity would be uncounted); reservoir literature informs the NARMA protocol only |
| Concept-drift adaptation (Gama et al. 2014; Lu et al. 2019) | adaptation after change | detectors, windows or ensembles | extra state and thresholds | **context**: AAA's online SGD adapts implicitly; drift benchmarks are assessed in the external specification |
| Catastrophic forgetting remedies (McCloskey & Cohen 1989; Kirkpatrick et al. 2017; Parisi et al. 2019) | retention | EWC: two parameter-sized vectors | protects the wrong solution when the world changes | **rejected**: no forgetting has been measured to target (v1 D-18); retention stays a measured dimension |
| Adam / momentum (Kingma & Ba 2015) | optimization speed | Adam: 2 x 994 state; momentum: 1 x | hidden state, adaptive rates that fight online change | **diagnostic only** (brief, § Optimizer) |
| Gradient clipping (Pascanu et al. 2013) | exploding gradients | none | a clip that fires often decides what is learned (v1 D-16) | **grid dimension** with a stability-margin rule |
| Function-preserving growth (Chen et al. 2016, Net2Net) | future capacity | growth itself | growth without a diagnosed capacity limit | **deferred** to after the capacity diagnostic |

## Notes on the adopted mechanisms

**Minimal gated unit.** Zhou et al. (2016, *International Journal of
Automation and Computing* 13(3)) merge the GRU's reset and update gates into a
single forget gate that both scales the recurrent input to the candidate state
and interpolates. At 1K it buys three extra hidden units over a 4-input GRU.
The M1 diagnosis blamed the GRU's zero-bias gates for suppressing a
sign-alternating mode; the MGU keeps one gate but at a different operating
point. Expected failure: loss of the reset gate hurts regime switches.

**Ungated recurrence at the widest budget.** The v1 characterization already
measured the 954-parameter ungated control beating the gated core on the
coarse family and being competitive elsewhere, and being less stable. The v2
candidate adds the observed flag and, unlike v1, gets its own learning rate,
horizon and clip from the same grid, so "less stable" is measured under a
fair search.

**Stable-by-construction linear recurrence with exact online gradients.**
Orvieto et al. (2023, *ICML*, "Resurrecting recurrent neural networks for long
sequences") show that a complex-diagonal linear recurrence with a
stable exponential parameterization and a nonlinear readout matches much of
the long-range ability of state-space models. Zucchet et al. (2023, *NeurIPS*,
"Online learning of long-range dependencies") observe that for diagonal linear
recurrences real-time recurrent learning is exact and costs one trace per
recurrent parameter, removing both truncation and the stale-activation
approximation of `AAA-169`. At 1K this is testable: 24 complex modes with a
13-unit tanh readout (957 parameters) and 240 trace scalars, all counted. The
gradient is verified against finite differences of the untruncated loss
(`tests/test_aaa_1k_v2.py`). Expected failures: a linear state may need a
large readout for sharp nonlinear dynamics (walls), and online RTRL gradients
can be noisy.

**Explicit missingness.** Lipton, Kale & Wetzel (2016, *MLHC*) found that
feeding missingness indicators to RNNs helps, and Che et al. (2018, *Scientific
Reports* 8:6085; GRU-D) add learned decays. AAA-1K's v1 held the last position
and zeroed the displacement and error inputs, so a genuinely stationary dot and
a hidden one look identical (R-10). The v2 feature set adds one flag; a
`stationary_mixed` family and a `no_observed_flag` ablation make the mechanism
measurable. GRU-D's learned decay is deferred: it adds per-unit parameters and
the flag must earn its keep first.

**Gate-bias initialization.** Tallec & Ollivier (2018, *ICLR*, "Can recurrent
neural networks warp time?") derive chrono initialization of gate biases from
the time scales a GRU should represent. AAA's loop measured the opposite
direction on the coarse family: a keep-bias of -2 (shorter memory) closed 73%
of the M1 gap and failed only through M2's frame lock, which Champion 1 removes.
The candidate therefore tests -2 on Champion 1's rule; chrono (positive
biases, long memory) is deferred because the evidence points the other way on
the family that matters.

## Deferred and rejected, briefly

UORO, SnAp and e-prop approximate RTRL for general recurrences; the
horizon/semantics grid already measures how much credit assignment matters for
the gated cores, and v1 found it nearly irrelevant at these horizons. Unitary,
orthogonal, antisymmetric and IndRNN families pursue stability through the
recurrence; M2 was not a recurrence instability, and the LRU tests stability by
construction more directly. Reservoirs and ensembles are rejected because the
first hides capacity outside the parameter count and the second exceeds it.
EWC-style consolidation is rejected because no forgetting has been measured.

## References

- Arjovsky, M., Shah, A. & Bengio, Y. (2016). Unitary evolution recurrent neural networks. *ICML*.
- Bellec, G. et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications* 11, 3625.
- Chang, B., Chen, M., Haber, E. & Chi, E. H. (2019). AntisymmetricRNN: a dynamical system view on recurrent neural networks. *ICLR*.
- Che, Z., Purushotham, S., Cho, K., Sontag, D. & Liu, Y. (2018). Recurrent neural networks for multivariate time series with missing values. *Scientific Reports* 8, 6085.
- Chen, T., Goodfellow, I. & Shlens, J. (2016). Net2Net: accelerating learning via knowledge transfer. *ICLR*.
- Cho, K. et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.
- Chung, J., Gulcehre, C., Cho, K. & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv:1412.3555.
- Gama, J., Zliobaite, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys* 46(4).
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA* 102(477).
- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148.
- Kingma, D. P. & Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.
- Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS* 114(13).
- Kuleshov, V., Fenner, N. & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*.
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Lezcano-Casado, M. & Martinez-Rubio, D. (2019). Cheap orthogonal constraints in neural networks. *ICML*.
- Li, S., Li, W., Cook, C., Zhu, C. & Gao, Y. (2018). Independently recurrent neural network (IndRNN). *CVPR*.
- Lipton, Z. C., Kale, D. C. & Wetzel, R. (2016). Directly modeling missing data in sequences with RNNs. *Machine Learning for Healthcare*.
- Lu, J. et al. (2019). Learning under concept drift: a review. *IEEE TKDE* 31(12).
- Lukosevicius, M. & Jaeger, H. (2009). Reservoir computing approaches to recurrent neural network training. *Computer Science Review* 3(3).
- McCloskey, M. & Cohen, N. J. (1989). Catastrophic interference in connectionist networks. *Psychology of Learning and Motivation* 24.
- Menick, J., Elsen, E., Evci, U., Osindero, S., Simonyan, K. & Graves, A. (2021). Practical real time recurrent learning with a sparse approximation. *ICLR*.
- Miller, J. & Hardt, M. (2019). Stable recurrent models. *ICLR*.
- Nix, D. A. & Weigend, A. S. (1994). Estimating the mean and variance of the target probability distribution. *IEEE ICNN*.
- Orvieto, A. et al. (2023). Resurrecting recurrent neural networks for long sequences. *ICML*.
- Parisi, G. I., Kemker, R., Part, J. L., Kanan, C. & Wermter, S. (2019). Continual lifelong learning with neural networks: a review. *Neural Networks* 113.
- Pascanu, R., Mikolov, T. & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
- Rodan, A. & Tino, P. (2011). Minimum complexity echo state network. *IEEE TNN* 22(1).
- Tallec, C. & Ollivier, Y. (2018). Can recurrent neural networks warp time? *ICLR*.
- Tallec, C. & Ollivier, Y. (2018b). Unbiased online recurrent optimization. *ICLR*.
- Williams, R. J. & Peng, J. (1990). An efficient gradient-based algorithm for on-line training of recurrent network trajectories. *Neural Computation* 2(4).
- Williams, R. J. & Zipser, D. (1989). A learning algorithm for continually running fully recurrent neural networks. *Neural Computation* 1(2).
- Zhou, G.-B., Wu, J., Zhang, C.-L. & Zhou, Z.-H. (2016). Minimal gated unit for recurrent neural networks. *International Journal of Automation and Computing* 13(3).
- Zucchet, N., Meier, R., Schug, S., Mujika, A. & Sacramento, J. (2023). Online learning of long-range dependencies. *NeurIPS*.
