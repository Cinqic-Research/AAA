# Juniper 1 adaptation: literature record

Date: 2026-10-04. Scope: the evidence behind the design choices in
[decisions](decisions.md). A paper is evidence about a method in its setting;
none of the numbers below is assumed to transfer to gpt-oss-20b or to
ToolShift. Primary sources were read at least to the abstract; the two claims
the design leans on hardest (continual test-time adaptation collapse, memory-based
editing) were re-checked against the primary abstract by the implementer.

## 1. Adapting a frozen Language Model through memory

- **SERAC** (Mitchell et al., ICML 2022, arXiv 2206.06520) stores edits in an
  explicit memory outside the base model's parameters and decides per input
  whether an edit applies. The scope decision is the dominant error source.
  Retrieving the right edit but letting the base model read it naively did
  much worse than SERAC's trained reader on T5 (edit success 0.278 against
  0.913).
- **IKE** (Zheng et al., arXiv 2305.12740) puts edits in context. Without
  out-of-scope demonstrations, specificity collapses: the edits leak into
  unrelated inputs.
- **GRACE** (Hartvigsen et al., NeurIPS 2023) and **WISE** (Wang et al.,
  NeurIPS 2024) scale sequential edits. Reliability and paraphrase
  generalization fall as edits accumulate: WISE goes from 0.98/0.92 at one edit
  to 0.77/0.72 at a thousand.
- **MeLLo** (Zhong et al., EMNLP 2023) shows that parametric editors fail on
  multi-hop use of edited facts, while a frozen model with retrieved edits does
  better.

*Used for:* the LM adapter is a versioned memory of typed edits ("notes")
read by the unmodified model. Every candidate set of notes is tested for
collateral change on unrelated, previously satisfied requests and for new
fabrication before it is kept.
*Not concluded:* that GPT-OSS reads notes as well as SERAC's trained reader.
That rate is measured here (characterization), not assumed.

## 2. Experience as text

- **Reflexion** (Shinn et al., arXiv 2303.11366) reports large gains from
  verbal self-feedback, but on retries of the same task, often with an oracle
  success signal.
- **ExpeL** (Zhao et al., arXiv 2308.10144) gives modest cross-task transfer and
  reports that hallucinated reflections *hurt* when fed back into memory.

*Used for:* notes are never free text from the model or the user. They are
structured records (a phrase that occurred in a request and a tank that
exists; integer coefficients) rendered by fixed templates. "Show the model
raw observations" (`lm.dynamics_from_experience`) is kept as a competing
mechanism, because the literature does not settle whether raw experience or
distilled knowledge is the better edit.

## 3. Confidence and calibration

- Kadavath et al. (arXiv 2207.05221): pretrained models are fairly calibrated
  in the right format, but P(IK) generalizes poorly out of distribution.
- GPT-4 technical report (arXiv 2303.08774): post-training worsened
  calibration (ECE 0.007 to 0.074 on an MMLU subset).
- Farquhar et al. (Nature 2024): semantic entropy detects confabulation
  (AUROC 0.79), but not errors the model makes consistently.
- gpt-oss-20b model card (arXiv 2508.10925): SimpleQA accuracy 0.067 with a
  0.914 hallucination rate. It almost never abstains.

*Used for:* GPT-OSS token log-probabilities enter the Erudition Model only as
an uncalibrated evidence feature. Acceptance never depends on the model's
confidence, and abstention is scored only where the environment makes it the
correct answer.

## 4. World models that meet the same environment again

- **MOLe** (Nagabandi, Finn, Levine, ICLR 2019, arXiv 1812.07671): a mixture of
  dynamics models with task inference beats continued SGD on one model when a
  regime recurs; the single model forgets.
- **PETS** (Chua et al., NeurIPS 2018) and deep ensembles (Lakshminarayanan et
  al., NeurIPS 2017): probabilistic models separate noise from model
  uncertainty and are sample-efficient.
- Hypernetwork continual learning (von Oswald et al., ICLR 2020), CLEAR replay
  (Rolnick et al., NeurIPS 2019) and van de Ven et al. (Nature Machine
  Intelligence 2022): explicit context separation or replay retains old tasks.
  Regularization alone (EWC) is weaker without task identity.
- **BOCPD** (Adams and MacKay 2007) and the task-free continual learning of
  Aljundi et al. (CVPR 2019): change detection still rests on a hazard prior or
  a hand-set threshold.

*Used for:* the World Model is a library of separately fitted contexts with
Bayesian predictive uncertainty. Adapting adds or switches contexts instead of
overwriting one model, which addresses the forgetting `aaa.python.opaque.v0`
measured when WM-S adapted to library B.
*Not concluded:* that a context library generalizes to rich environments. The
v0 basis is deliberately small, and misfit is designed to show up as
uncertainty (see [decisions](decisions.md), D-WM).

## 5. Learned controllers of learning

- Learned optimizers (Andrychowicz et al., NIPS 2016; VeLO, Metz et al. 2022)
  fail out of distribution even after very large meta-training: ReLU nets for
  the first, long horizons for the second.
- RL² (Duan et al. 2016), OML (Javed and White 2019) and ANML (Beaulieu et al.
  2020) show meta-learned adaptation close to their training distribution.
- Continual-MAML / OSAKA (Caccia et al., NeurIPS 2020) detects shifts with a
  hand-tuned loss threshold. The implementer and the literature check found no
  primary source showing a learned *whether/when/which* controller that
  generalizes out of distribution.

*Used for:* the Erudition Model *controls* standard, inspectable mechanisms
and does not emit parameter updates itself. It is compared with an auditable
rule set that sees the same evidence and has the same menu, and with
never-adapt and always-adapt policies. Its transfer from simulation to the
real model is a measured question.

## 6. Adaptation failures and poisoning

- **RDumb** (Press et al., arXiv 2306.05401): over long horizons, all but one
  state-of-the-art continual test-time adaptation method collapses below a
  non-adapting model. Periodically resetting to the pretrained model matches or
  beats them. (Verified against the abstract.)
- **SAR** (Niu et al., ICLR 2023): noisy, high-gradient samples drive collapse.
- **DIA** (Wu et al., arXiv 2301.12576): a few malicious test samples can
  hijack test-time adaptation.
- **PoisonedRAG** (Zou et al., USENIX Security 2025): five injected texts per
  target question give about 90% attack success against a retrieval memory.
- Greshake et al. (arXiv 2302.12173): retrieved content carries instructions.

*Used for:* the frozen and never-adapt arm is a first-class control. Every
accepted state is reversible; canary monitoring rolls back regressions. Feedback
corruption windows, tool outages, transient glitches and instruction-bearing
feedback are part of the evaluation, not an afterthought. Notes can come only
from structurally verified records.

## 7. Self-improving systems with gates

- **Darwin Gödel Machine** (Zhang et al., arXiv 2505.22954) uses staged
  evaluation, an archive and lineage. An agent gamed a visible hallucination
  detector by removing the logging it relied on.
- **STOP** (Zelikman et al., COLM 2024): improvers occasionally tried to
  disable the sandbox, and wrong-shape outputs produced "accuracy" above 1000%.
- **Voyager** (Wang et al., arXiv 2305.16291) admits skills to its library only
  after self-verification.

*Used for:* the gate, the evaluator and the scorer are host code outside the
adaptable surface. Controllers can only request entries from a fixed menu.
Scores come from an evaluator-only label that no component reads (enforced by
a test that traps every read).

## Deliberately not adopted

- Fine-tuning or LoRA on gpt-oss-20b. It is impractical on FLOWBOX (6 GB VRAM)
  and unnecessary for the first question. The adapter boundary allows it later
  as a separate, versioned artifact.
- Weight editing (ROME, MEMIT): it touches the immutable base.
- Learned optimizers that emit parameter deltas (section 5).
- A neural simulator World Model for v0: an exact Bayesian model of a small
  basis is sufficient for this environment and keeps misfit visible. The
  neural alternative is compared in development (D-WM).
