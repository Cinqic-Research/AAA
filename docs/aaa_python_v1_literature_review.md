# `aaa.python.v1` literature and mechanism review

Written 2026-09-24, **before any v1 learner observed a v1 development
identity.** It records which external mechanisms the pre-scale phase considered,
what AAA weakness each addresses, and the experiment that could show it useless
*for AAA*. External results come almost entirely from models 10^2-10^6 times
larger, pretrained, or trained offline on large corpora. They motivate
experiments. **They are not evidence about AAA; AAA evidence decides AAA.**

## How this review was made

A research subagent searched the literature and verified sources: about 75
arXiv abstracts against their `citation_title`, about 20 DOIs through Crossref,
and the rest on JMLR, OpenReview, AAAI, NeurIPS, Project Euclid or JSTOR pages.
Items it could not confirm are marked UNVERIFIED. The implementer then
independently re-fetched a sample of load-bearing citations: Weinberger et al.
2009 (arXiv 0902.2206), Vasic et al. 2019 (arXiv 1904.01720, ICLR 2019),
Dohare et al. 2024 (Nature 632, doi:10.1038/s41586-024-07711-7; the arXiv
version 2306.13812 carries a different title), and Owen 2007
(doi:10.1214/07-AOAS122, *The pigeonhole bootstrap*). All matched. A subagent
shares the orchestrator's framing, so this is coverage, not independent review.

## Measured AAA weaknesses this review is organized around

These come from v0's retained development evidence and from v1's pre-design
diagnostics on already observed v0 identities
([pre-scale diagnostics](aaa_python_v1_research_brief.md#diagnostics-before-design)):

| # | Weakness | Evidence |
|---|---|---|
| W1 | v0's learner is under-trained | Same features and rate, only more passes: syntax 0.53 -> 0.82 (6 epochs) -> 0.89 (20); outcome 0.64 -> 0.68; output, localize unchanged (`AAA-197`) |
| W2 | `localize` cannot point | A whole-program bag with an absolute 40-way line softmax; accuracy does not move with training budget |
| W3 | `output` needs execution; its head wastes capacity | 52 of 101 classes occur in 600 training programs; the head is 103,424 of 153,600 parameters; accuracy flat in budget |
| W4 | Linear bag-of-n-grams cannot count or relate | Syntax needs bracket/colon/indent structure; localization and outcome need def-use relations |
| W5 | Hashing interference | Signed hashing helps at D=256 (syntax +3.5 pp, 5/5 inits) and is neutral at D >= 1024 |
| W6 | Benchmark shortcuts and weak baselines | Repair medoid rule 0.90-0.93 (`AAA-192`); v0 `outcome` "heuristic" was majority (`AAA-193`); single-fault localization trivial (`AAA-194`) |
| W7 | No resolved adaptation; stationary streams | Adaptation DiD unresolved in every family; a stationary stream gives adaptation little to do |
| W8 | Precision | 3 x 4 cells; stream variance dominates; percentile coverage 0.84-0.90 in null simulation |

## Decisions

Decision codes: `ADOPT` (in v1 now), `TEST` (a v1 development arm),
`DEFER` (recorded, not this phase), `REJECT`, `CONTEXT` (informs design only).
Parameter costs are at D = 256.

### A. Code representation

| Mechanism and source | Weakness | Hypothesis | Params / adaptive state / compute | Complexity; failure mode | ~10K? | Decision | Falsifying AAA experiment |
|---|---|---|---|---|---|---|---|
| Identifier alpha-renaming to first-occurrence roles; name reliance: Rabin et al. 2020 (arXiv 2008.01566, venue UNVERIFIED), Yefet, Alon & Yahav, OOPSLA 2020 (arXiv 1910.07517) | W4, name generalization | Roles generalize to the `novel_names` slice and reduce hash load | 0 / 0 / linear in tokens | Low; loses information if names were informative | yes | **ADOPT** in `e2` (`alpha` channel) | `e2` minus `alpha` vs `e2` on `novel_names` and overall at matched core |
| Line-anchored structure: first/last token shape, indentation change after a header, running bracket depth, quote parity. Algorithmic alignment: Xu et al., ICLR 2020 (arXiv 1905.13211) | W4 (syntax) | Makes the structure the lint reads linearly available; the learner must still learn to use it | 0 / 0 / linear | Low; can make syntax a feature-engineering result, reported as such | yes | **ADOPT** in `e2` (`line` channel), parser-free | `e2` minus `line` on `syntax` |
| Program graphs with def-use (LastUse/LastWrite/ComputedFrom) edges: Allamanis, Brockschmidt & Khademi, ICLR 2018 (arXiv 1711.00740); data flow in GraphCodeBERT: Guo et al., ICLR 2021 (arXiv 2009.08366) | W4 (localize, outcome) | Hashed def-use and operator/operand features help without message passing | features 0; a GGNN would be ~5K | Features low; a GGNN is hard to train online in NumPy | features yes; GGNN partly | **ADOPT** as features (`flow` channel, never for `syntax`); GGNN **DEFER** | `e2` minus `flow` on `localize` and `outcome` |
| AST path-contexts: code2vec, Alon et al., POPL 2019 (arXiv 1803.09473); code2seq, ICLR 2019 (arXiv 1808.01400) | W4 | Longer AST paths beat parent/child pairs | features 0; attention ~4K | Path explosion at small D | partly | **DEFER** (the def-use channel is tested first) | -- |
| Pointer heads for localization: Vasic et al., ICLR 2019 (arXiv 1904.01720); GREAT: Hellendoorn et al., ICLR 2020 (OpenReview B1lnbRNtwr) | W2 | A shared per-line scorer beats an absolute line softmax | Z params / 0 / per line | Low; weak line features give no gain | yes | **TEST** (head ablation, never together with the encoder change) | pointer vs one-hot `localize` at fixed encoder and core |
| Relative position: Shaw, Uszkoreit & Vaswani, NAACL 2018 (arXiv 1803.02155) | W2 | Line index from start and end is enough position | 0 / 0 | Low | yes | **ADOPT** in every encoder's per-line vectors, so position does not favour one encoder | ablation inside the pointer head (deferred) |
| Tree-LSTM (Tai et al., ACL 2015, arXiv 1503.00075), TBCNN (Mou et al., AAAI 2016, arXiv 1409.5718) | W4 | Recursive nets encode scope | 3-6K | High; unstable online | partly | **DEFER** | only if explicit structure is exhausted |
| Byte/char recurrent models: ByT5, Xue et al., TACL 2022 (arXiv 2105.13626) | W4 | A tiny byte GRU learns counting itself | ~3K | Medium; counting not learned online | yes | **DEFER** (next phase: "can a learner learn `line` itself?") | GRU-bytes vs `line` features on syntax |
| Deduplication: Allamanis, Onward! 2019 (arXiv 1812.06469) | W6 | Near-duplicates inflate accuracy | 0 | Low | -- | **ADOPT** as hygiene: exact hashes excluded; near-duplicate rates reported per slice | -- |

### B. Feature hashing

| Mechanism and source | Weakness | Hypothesis | Cost | Failure mode | ~10K? | Decision | Falsifier |
|---|---|---|---|---|---|---|---|
| Signed hashing: Weinberger, Dasgupta, Langford, Smola & Attenberg, ICML 2009 (arXiv 0902.2206) | W5 | Signed collisions cancel in expectation instead of adding | 0 | None expected | yes | **ADOPT** -- AAA evidence: +3.5 pp syntax at D = 256 on v0's own learner, neutral at D >= 1024 | already run (v1 diagnostics) |
| Dedicated, unhashed, unnormalized bias | W5 | The intercept must not collide or scale with 1/norm | +classes | None | yes | **ADOPT** -- the model owns every bias | -- |
| Hash kernels: Shi et al., JMLR 10, 2009; norm preservation: Freksen, Kamma & Larsen, NeurIPS 2018 (arXiv 1805.08539); Moody, NIPS 1988 | W5 | Choose D from the distinct-feature count | -- | -- | -- | **CONTEXT** | -- |
| Multiple hashes / CountSketch: Charikar, Chen & Farach-Colton, ICALP 2002 (doi:10.1007/3-540-45465-9_59) | W5 | k = 2 signed hashes reduce unlucky collisions | 0 | Doubles density | yes | **DEFER** (collision effects were small) | -- |
| Hash/Bloom embeddings: Svenstrup et al., NeurIPS 2017 (arXiv 1709.03933) | W5 | Learned importance weights | small table | Adds input-side parameters | partly | **DEFER** | -- |

### C. Plasticity and continual learning

Forgetting (losing old performance) and loss of plasticity (losing the ability
to learn new things) are different failures. v1 measures both.

| Mechanism and source | Weakness | Hypothesis | Cost | Failure mode | ~10K? | Decision | Falsifier |
|---|---|---|---|---|---|---|---|
| Loss of plasticity and continual backprop: Dohare et al., Nature 632, 2024 (doi:10.1038/s41586-024-07711-7) | plasticity untested | A continuing learner eventually learns new blocks more slowly than a fresh one | utility per unit | Resets destroy features on short streams | partly | **ADOPT the fresh-learner control** as the operational test; continual backprop **TEST only if** degradation appears | late-life learning speed on fresh blocks vs a fresh learner's on the same blocks |
| Causes of plasticity loss: Lyle et al., ICML 2023 (arXiv 2303.01486); Lyle et al. 2024 (arXiv 2402.18762) | diagnostics | Weight/pre-activation growth and saturation precede plasticity loss | 0 | -- | yes | **ADOPT** as diagnostics: weight norms, saturated fraction, dormant fraction, activation effective rank, gradient norms | -- |
| Dormant neurons, ReDo: Sokar et al., ICML 2023 (arXiv 2302.12902) | hidden units only | -- | -- | -- | hidden core only | **ADOPT** dormant fraction as a diagnostic; ReDo **DEFER** | -- |
| Shrink-and-perturb: Ash & Adams, NeurIPS 2020 (arXiv 1910.08475) | warm starts | Restores trainability | 0 | Hurts stationary streams | yes | **TEST only if** plasticity degrades | recovery slope with vs without |
| L2 regularization toward init: Kumar, Marklund & Van Roy 2023 (arXiv 2308.11958); spectral: Lewandowski et al. 2024 (arXiv 2406.06811) | weight growth | Small L2 maintains plasticity | 0 | Underfits | yes | **TEST** as a single, separate optimization arm (plain L2) | development arm vs SGD |
| Implicit under-parameterization / feature rank: Kumar et al., ICLR 2021 (arXiv 2010.14498) | -- | -- | -- | -- | -- | **CONTEXT** (effective rank is reported) | -- |
| Primacy bias: Nikishin et al., ICML 2022 (arXiv 2205.07802); continual RL: Abbas et al., CoLLAs 2023 (arXiv 2303.07507); UPGD: Elsayed & Mahmood, ICLR 2024 (arXiv 2404.00781) | -- | -- | -- | -- | -- | **CONTEXT** / **DEFER** | -- |

### D. Online learning, credit assignment and meta-learning

| Mechanism and source | Weakness | Hypothesis | Cost | Failure mode | ~10K? | Decision | Falsifier |
|---|---|---|---|---|---|---|---|
| Concept drift and change detection: Gama et al., ACM CSUR 46(4), 2014 (doi:10.1145/2523813); ADWIN: Bifet & Gavaldà, SDM 2007 | W7 | Adaptation can only pay when the stream changes; a stationary stream makes the null result expected | 0 | Adaptation appears only under drift, which is correct | yes | **ADOPT** the design: v1's adaptation stage switches the distribution (to held-out program structure) and compares online with frozen twins from one cloned state | online vs frozen after a pre-declared switch |
| Per-coordinate step sizes: AdaGrad, Duchi, Hazan & Singer, JMLR 12, 2011; FTRL-Proximal, McMahan et al., KDD 2013 (doi:10.1145/2487575.2488200) | W1 | Adaptive steps lift rare features | 1 accumulator per weight | Decays under drift | yes (state counted) | **DEFER** -- a momentum arm with counted state is tested instead; AdaGrad is the next optimizer arm if optimization stays limiting | -- |
| IDBD (Sutton, AAAI 1992), Autostep (Mahmood et al., ICASSP 2012) | W7 | Meta-learned per-feature steps help under drift | 2 per weight | Softmax extension needed | yes | **DEFER** | -- |
| RTRL (Williams & Zipser 1989), UORO (Tallec & Ollivier, arXiv 1702.05043), SnAp (Menick et al., ICLR 2021, arXiv 2006.07232), columnar RTRL (Javed et al., arXiv 2302.05326), e-prop (Bellec et al., Nat. Commun. 2020) | -- | Needed only for a *persistent* cross-program recurrent state | O(h^4) exact | No payoff for bounded episodes | no | **DEFER** -- v1 has no cross-task recurrent state; per-program computation is exact backprop | -- |
| Fast weights: Ba et al., NeurIPS 2016 (arXiv 1610.06258); Schlag, Irie & Schmidhuber, ICML 2021 (arXiv 2102.11174) | W3 | Bind variables to values | 1-3K | Slow online | partly | **DEFER** | -- |
| Meta-learned update rules (Metz et al., ICLR 2019; Kirsch & Schmidhuber, NeurIPS 2021); FTML (Finn et al., ICML 2019) | -- | -- | out of budget | -- | no | **REJECT** this phase / **CONTEXT** | -- |
| Contextual bandits: LinUCB, Li et al., WWW 2010 (arXiv 1003.0146) | repair bandit feedback | Exploration improves cumulative repair reward | O(D) | Off-policy evaluation bias | yes | **DEFER** | -- |
| Heuristics as expert inputs: Hedge, Freund & Schapire, JCSS 1997 | W6 | A learner given a heuristic's output must match it | +#rules | -- | yes | **CONTEXT** -- the tool-augmented repair learner is v1's instance of this check | tool learner vs tool baseline |

### E. Execution-grounded learning and tools

| Mechanism and source | Weakness | Hypothesis | Cost | Failure mode | ~10K? | Decision | Falsifier |
|---|---|---|---|---|---|---|---|
| Filter candidates by visible tests: CodeT, Chen et al., ICLR 2023 (arXiv 2207.10397); AlphaCode, Li et al., Science 2022 (arXiv 2203.07814); LEVER, Ni et al., ICML 2023 (arXiv 2302.08468) | W6, tool rung 14 | Running the visible tests decides most repair items | 0 params; 4 x 2 executions | If visible tests decide nearly everything, learning is moot for repair-with-tools | yes | **ADOPT** as a mandatory baseline and a declared, logged tool channel | measured: v1 `visible_tests` 0.936 on development |
| Execution-guided synthesis: Chen, Liu & Song, ICLR 2019 (OpenReview H1gfOiAqYm) | W3 | Executed prefix states make output easy | interpreter | It is the oracle for AAA's families | -- | **CONTEXT** -- running the task's own program is the oracle; v1 refuses such a tool |
| Self-debugging: Chen et al., ICLR 2024 (arXiv 2304.05128); DeepFix (AAAI 2017), DrRepair (ICML 2020, arXiv 2005.10636), Break-It-Fix-It (ICML 2021, arXiv 2106.06600) | W6 | Tracebacks and compiler lines localize for free | 0 | -- | -- | **CONTEXT** -- they define why localization/syntax are *pre-execution* predictions in AAA |
| Execution-trace auxiliary targets: Learning to Execute, Zaremba & Sutskever 2014 (arXiv 1410.4615); TRACED (ICSE 2024, arXiv 2306.07487); NExT (ICML 2024, arXiv 2404.14662) | W3 | Per-line trace targets shape shared features | aux heads | Needs a hidden layer; tracing cost | partly | **DEFER** (requires a protocol version that reveals traces post-action) | -- |

### F. Neural algorithmic reasoning

CLRS (Veličković et al., ICML 2022, arXiv 2205.15659), neural algorithmic
reasoning (Veličković & Blundell, Patterns 2021), neural execution of graph
algorithms (ICLR 2020, arXiv 1910.10593), IPA-GNN (Bieber et al., NeurIPS
2020, arXiv 2010.12621) and Neural Interpreters (NeurIPS 2021) are
**CONTEXT**. The transferable lessons are step-level supervision and
propagation along control flow; the models are out of budget or need offline
training on millions of examples. IPA-GNN is recorded as a possible later
architecture, **DEFER**.

### G. Dynamic capacity

| Mechanism and source | Decision | Why |
|---|---|---|
| Net2Net, Chen, Goodfellow & Shlens, ICLR 2016 (arXiv 1511.05641) | **DEFER** | Function-preserving growth becomes useful once a capacity limit is demonstrated; v1 first asks whether one exists |
| Splitting/Firefly/GradMax/NORTH (NeurIPS 2019, NeurIPS 2020, ICLR 2022 x2) | **DEFER** | At H <= 40 the growth rule is swamped by seed variance |
| RigL (ICML 2020, arXiv 1911.11134), SET (Nat. Commun. 2018) | **DEFER** | Sparse hidden layers are a route to more units per parameter, after a capacity limit is shown |
| Progressive networks (arXiv 1606.04671) | **REJECT** | Parameters grow with tasks |
| Growth triggered by persistent failure (AAA proposal) | **DEFER** | Needs a pre-registered trigger and a shuffled-label control; v1's capacity sweep supplies the evidence such a trigger would need |

### H. External benchmarks

| Benchmark | License (checked) | Decision for this phase |
|---|---|---|
| Learning-to-Execute-style generated programs | regenerate | **ACCEPT** as the style of AAA's own generator |
| CRUXEval, Gu et al., ICML 2024 (arXiv 2401.03065) | MIT | **REJECT** for training (800 items, outside the subset); a filtered output-prediction probe is a later option |
| QuixBugs (SPLASH 2017), ManyBugs (TSE 2015, C) | MIT / not checked | **REJECT** (40 programs; outside the subset; C) |
| VarMisuse/GREAT data | mixed per-file, includes GPL | **REJECT** (license mix, real code far beyond the subset) |
| Project CodeNet (arXiv 2105.12655) | repo Apache-2.0; data terms UNVERIFIED | **DEFER** |
| MBPP, HumanEval, LiveCodeBench | various | **REJECT** (natural-language-to-code generation; not a capability this learner can attempt) |

A ~10K no-pretraining learner cannot meaningfully attempt any of these yet.
Claiming compatibility would be fake evidence. AAA's generated benchmark,
with executable ground truth, declared slices and no contamination, stays
primary.

### I. Statistics

| Method | Decision |
|---|---|
| Pigeonhole (crossed) bootstrap over initializations and streams: Owen, AoAS 1(2), 2007 (doi:10.1214/07-AOAS122); Owen & Eckles, AoAS 2012 (arXiv 1106.2125) | **ADOPT** (already v0's method); v1 raises replication from pilot variance |
| No exact bootstrap for two-way arrays: McCullagh, Bernoulli 6(2), 2000 | **ADOPT** as a caveat; v1 reports variance components as a cross-check |
| Holm, Scand. J. Stat. 1979 | **ADOPT** across declared primary development contrasts |
| Intersection-union tests: Berger 1982 (doi:10.1080/00401706.1982.10487790); Berger & Hsu 1996 (doi:10.1214/ss/1032280304) | **ADOPT** (it is `aaa.promotion.crossed.v1`'s verdict rule) |
| Seed counts and reporting: Colas et al. 2018 (arXiv 1806.08295); Card et al., EMNLP 2020; Bouthillier et al., MLSys 2021; Agarwal et al., NeurIPS 2021 | **CONTEXT** |

## Mechanisms AAA had missed, now in the design

1. Under-training as a confound of v0's negative result (W1): v1 selects each
   arm's training budget on a separate tuning split.
2. A per-line pointer head and an ordinal output head (W2, W3), tested as head
   ablations.
3. Signed hashing and a model-owned bias (W5), adopted from AAA's own evidence.
4. Distribution-switch adaptation design and the fresh-learner plasticity
   control (W7, C1).
5. Execution baselines as ceilings, and the visible-test tool as a declared,
   logged pre-action channel (E).
6. Stupid-but-fitted baselines and permanent shortcut attacks (W6).
