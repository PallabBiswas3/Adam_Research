# Prior-art audit, bounded discovery pass, 2026-10-12

Decision use: the proposed hard-clean/noisy-label/imbalance problem is extensively
studied. The evidence below rules out novelty for classwise filtering, confidence
history, prototype distances, peer agreement, and selection debiasing alone.
It does **not** establish that every possible residual failure has been solved.
No reported paper accuracy is used as an experimental result of this repository.

Scope and queries were registered in [search_protocol.md](search_protocol.md).
This is a targeted 24-paper primary-source audit, not a systematic or exhaustive
review. `M` means full text accessed and the relevant method inspected; it does
not mean all proofs, supplements, experiments or code were verified. `P` means
partial extraction. Unless the source audit below says otherwise, code license,
runtime, exact benchmark settings and theoretical proofs remain unaudited.
Limitations in the last column are applicability cautions inferred from the
method, not newly measured failures or assertions of author error.

| Work / verified source | Date / venue | State; mechanism / equation anchor | Published benchmark family; cost structure | Relevance / applicability caution |
|---|---|---|---|---|
| [Co-teaching](https://arxiv.org/pdf/1804.06872) | 2018, NeurIPS | M, Algorithm 1: two independent networks exchange their smallest-loss subsets; remember rate falls to 1-noise-rate | MNIST, CIFAR-10/100; two forward/backward passes | Filtering foundation; presumed noise rate and memorization schedule matter |
| [DivideMix](https://arxiv.org/pdf/2002.07394) | 2020, ICLR | M, Algorithm 1: per-network loss GMM, peer data division, label refinement/guessing, MixMatch | CIFAR-10/100, Clothing1M, WebVision; two networks, augmentations and SSL | Strong comparator; a GMM alone is not DivideMix |
| [Early-Learning Regularization](https://arxiv.org/pdf/2007.00151) | 2020, NeurIPS | M, regularizer log(1-p dot EMA target), added to CE | CIFAR, Clothing1M, WebVision; per-example class-probability memory; ELR+ adds components | History is established; stale or incorrect early predictions remain an applicability concern |
| [Separating hard clean samples from noisy samples with samples' learning risk](https://link.springer.com/article/10.1007/s40747-024-01360-z) | 2024-03-01, Complex & Intelligent Systems 10:4033-4054 | M, risk estimates effect of learning a sample using difficulty and feature similarity; relabeling and risk-based label augmentation | CIFAR-10/100, Animal-10N, Clothing1M, DocRED; similarity/risk and label-management work beyond CE | Directly covers hard-clean distinction; representation quality affects similarity |
| [Enhanced Sample Selection with Confidence Tracking](https://ojs.aaai.org/index.php/AAAI/article/view/34180/36335) | 2025-04-11, AAAI 39(19):19795-19803 | M, Eqs.9-12: observed-label/competitor probability gaps, minimum Mann-Kendall Z, union with base selector, masked minibatch CE | CIFAR-10/100, CIFAR-N, WebVision, Food-101N; history/trend state plus base learner | Directly covers increasing confidence of hard-clean samples; correlated training histories do not automatically yield calibrated hypothesis tests |
| [Robust long-tailed learning under label noise (RoLT)](https://link.springer.com/content/pdf/10.1007/s11704-025-40860-0.pdf) | Online 2025-10-16; 2026, Frontiers of Computer Science 20:2001321 | M, classwise prototype-distance mixtures and soft labels | Long-tailed noisy CIFAR and real-data settings; feature/prototype and mixture estimation | Direct overlap with minority-clean recovery; journal/preprint implementation equivalence unresolved |
| [Debiased Sample Selection (DSS)](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.pdf) | 2026, CVPR | M, Algorithm 1: marginal distribution adjustment (MDA), candidate class selection (CCS); DSS+ adds co-training | CIFAR noise variants, real noisy data; probability EMA/history and consistency work | Class and instance bias addressed; main text explicitly extends MDA to an estimated/user prior for imbalanced data and points to Appendix 8.1 |
| [PASS](https://arxiv.org/pdf/2303.10802) / [journal record](https://www.sciencedirect.com/science/article/pii/S0262885625004652) | Preprint 2023, inspected v2 2024; online 2025-12-16, journal volume 2026, Image and Vision Computing | M for preprint, P for final version: three-network peer agreement, cosine similarity and Otsu selection, SSL | CIFAR IDN and real noisy images; three learners plus SSL | Agreement is established; final-version changes have not been reconciled |
| [MentorNet](https://proceedings.mlr.press/v80/jiang18c/jiang18c.pdf) | 2018, ICML | M, learned data-driven curriculum/sample weights supplied by mentor | Noisy vision benchmarks; mentor and student training | Difficulty-aware weighting is prior art; supervision used to learn curricula needs matching |
| [Co-teaching+](https://proceedings.mlr.press/v97/yu19b/yu19b.pdf) | 2019, ICML | M, disagreement filtering followed by peer small-loss learning | MNIST/CIFAR/Tiny-ImageNet family; two learners | Preventing peer collapse is established; disagreement population can shrink |
| [JoCoR](https://arxiv.org/pdf/2003.02752) | 2020, CVPR | M, CE plus symmetric prediction KL, select by joint loss, update both | CIFAR-10/100, Clothing1M; two learners | Agreement regularization is prior art; agreement need not imply correctness |
| [Loss correction](https://arxiv.org/pdf/1609.03683) | 2017, CVPR | M, forward T-transformed probabilities; backward inverse-transition loss | MNIST, CIFAR, IMDB and real noisy clothing; transition estimation | Explicit class-conditional assumptions; unknown instance-dependent transitions not covered automatically |
| [Generalized Cross Entropy](https://arxiv.org/pdf/1805.07836) | 2018, NeurIPS | M, Lq=(1-p_y^q)/q and truncated variant | CIFAR, Fashion-MNIST noise; one learner | Robust-loss control; noise-tolerance statements require stated noise/risk assumptions |
| [Symmetric Cross Entropy](https://arxiv.org/pdf/1908.06112) | 2019, ICCV | M, weighted CE plus reverse CE with clipped labels | CIFAR noisy classification; one learner | Hard-class underlearning explicitly motivated; no universal IDN guarantee follows |
| [FINE](https://arxiv.org/pdf/2102.11628) | 2021, NeurIPS | M, classwise leading feature eigenvector alignment, squared scores and mixture selection | CIFAR, Clothing1M and WebVision; feature/eigendecomposition work | Feature geometry is established; unreliable embeddings can impair selection |
| [UniCon](https://arxiv.org/pdf/2203.14542) | 2022, CVPR | M, JSD-based uniform selection, contrastive learning and SSL | Noisy CIFAR and real noisy datasets; multi-view/SSL cost | Class selection bias explicitly addressed; uniform targets must match intended test prior |
| [ITEM](https://arxiv.org/pdf/2401.13360) | First preprint 2024; final venue not verified in this pass | M, multiple expert heads, voting and class-balanced weighted sampling | Noisy CIFAR/real images; expert heads and resampling | Different work from CVPR-2026 DSS despite earlier similar title; class selection bias is already studied |
| [Representation Calibration / RCAL](https://arxiv.org/pdf/2211.10955) | Preprint 2022; final venue not verified | M, distribution and individual representation calibration using self-supervised reference | Long-tailed noisy classification; representation/SSL cost | Exact joint-noise/imbalance overlap; Gaussian assumptions are not guarantees for arbitrary embeddings |
| [Learning From Long-Tailed Data With Noisy Labels](https://arxiv.org/pdf/2108.11096) | Preprint 2021; final venue not verified | M, self-supervised pretraining, logit adjustment and confidence/SuperLoss weighting | Long-tailed noisy vision; two-stage training | Joint problem is established; comparison must account for pretraining cost |
| [Confident Learning](https://jair.org/index.php/jair/article/view/12125) | 2021-04-14, JAIR 70 | P: metadata/abstract verified; PDF viewer did not expose full text; class-conditional confident-joint estimation | CIFAR, Amazon Reviews, ImageNet label audit; out-of-sample probabilities matter | Class-conditional assumptions explicit; no theorem reproduced here |
| [LDAM / DRW](https://arxiv.org/pdf/1906.07413) | 2019, NeurIPS | M, class margins proportional to n_y^(-1/4), deferred reweighting; theorem conditions inspected, proof not reverified | Long-tailed CIFAR/Tiny-ImageNet/iNaturalist; one learner | Imbalance control; observed noisy counts can distort margin assignment |
| [Class-balanced loss](https://arxiv.org/pdf/1901.05555) | 2019, CVPR | M, effective number (1-beta^n)/(1-beta) and reciprocal weights | Long-tailed CIFAR/iNaturalist; one learner | Cheap known remedy; corrupted class counts affect weights |
| [Logit adjustment](https://arxiv.org/pdf/2007.07314) | Preprint 2020; 2021, ICLR | M, training/inference adjustment using log class priors | Long-tailed CIFAR/iNaturalist/ImageNet-LT; one learner | Known calibration remedy; prior estimation under noise is a separate difficulty |
| [Balanced Meta-Softmax](https://arxiv.org/pdf/2007.10740) | 2020, NeurIPS | M, count-weighted softmax n_j exp(z_j)/sum_i n_i exp(z_i), meta sampling | Long-tailed image classification; loss-only variant cheap, meta variant extra training | Prior-shift derivation assumes shared class conditionals; label noise does not satisfy it automatically |

## Closest recent mechanisms and conflicts

SLRLNL studies the exact distinction between hard correct and corrupted samples
using learning risk; CT studies probability trends; RoLT uses classwise geometric
selection; DSS adjusts selection probabilities and masks candidate competitors.
PASS adds peer agreement for instance-dependent corruption. These are distinct
published solutions, not empty comparator slots to fill with an invented method.

DSS uses an EMA probability marginal m and normalizes p/m before selection.
Its imbalanced extension changes the target prior; treating uniform MDA as its
only version would manufacture a gap. CCS is logit-denominator masking rather
than ordinary relabeling. [Supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Pan_Debiased_Sample_Selection_CVPR_2026_supplemental.pdf)
was downloaded with the main paper; exact image benchmark replication was not
run. Official selector stores a `gt` field, but inspected selection methods do
not use it: storage alone is not evidence of oracle leakage.

The journal RoLT authors differ from the 2021 preprint listing, and the original
repository link redirects. Record this version conflict rather than assuming
code reproduces the final article. PASS has preprint/online/volume dates spanning
2023/2025/2026; the inspected preprint is not certified identical to the journal.
ITEM arXiv:2401.13360 has a similar earlier title but different authors/mechanism
from Pan et al.'s CVPR-2026 DSS. Do not count them as the same paper.

## Official implementation and license inspection

Source trees were inspected without executing or copying code into our package.
SHA pins and per-file retrieval hashes are in
[source provenance](../results/reference-audit-20261012-complete/provenance.json)
and [additional assets](../results/reference-details-20261012/provenance.json).
`No LICENSE found` is **not** permission to reuse. Article access/license and
software licensing are different; article reuse permissions were not fully
audited for this pass. All new algorithms are independent equation implementations.

| Paper | Official repository | Inspected commit | Software license / retrieval |
|---|---|---|---|
| Co-teaching | [bhanML/Co-teaching](https://github.com/bhanML/Co-teaching) | 7c7fbe23e15e517db76a0882b6d108e4508e09d6 | No LICENSE found; inspected |
| CT | [Aliinton/ConfidenceTracking](https://github.com/Aliinton/ConfidenceTracking) | 85fa27da92d3171ebb3d919c51166f4396714ce2 | No LICENSE found; inspected |
| DSS | [Aliinton/DSS](https://github.com/Aliinton/DSS) | ebbe5dd705584814d1214fe6cc1f1fb40658cf58 | No LICENSE found; inspected |
| RoLT | [weitongseu/RoLT](https://github.com/weitongseu/RoLT) (redirect from Stomach-ache) | c78db7b14b7cd9bfb1a50928e1aa0e5edde45e49 | No LICENSE found; inspected; version mapping incomplete |
| SLRLNL | [yangbo1973/SLRLNL](https://github.com/yangbo1973/SLRLNL) | Not verified | Source retrieval timed out; license incomplete |
| DivideMix | [LiJunnan1992/DivideMix](https://github.com/LiJunnan1992/DivideMix) | d9d3058fa69a952463b896f84730378cdee6ec39 | MIT text inspected |
| PASS | [arpit2412/PASS-NoisyLabel](https://github.com/arpit2412/PASS-NoisyLabel) | 0573db04aef0ae5c686fa74dee678f481db835f4 | Non-Commercial Research License text inspected |

Author-linked implementations additionally confirmed for
[FINE](https://github.com/Kthyeon/FINE_official),
[ITEM](https://github.com/1998v7/ITEM), and
[LDAM](https://github.com/kaidic/LDAM-DRW); licenses/commits not audited.
Other implementation links/licenses are unverified in this pass. Paper wall-clock
and memory costs were not extracted consistently; the cost column is structural,
not a quantitative cross-paper comparison.

## Audit errors and stopping implication

Initial sandbox source requests failed; a permitted retry encountered an incomplete
HTTP response; a subsequent checkpointed fetch succeeded for six repositories,
with SLRLNL timeout preserved. CVF browser fetch was blocked, so the primary DSS
PDF and supplement were fetched directly and hashed. Local PDF extraction
succeeded; a console Unicode display error did not change the downloaded bytes.
Confident Learning's embedded PDF extraction remains incomplete.

The two faithful strong benchmark reproductions are not feasible within this
short CPU pass using the current inspected environments: official workflows use
image datasets/augmentation, substantially longer training, and CUDA-specific
code paths; several lack verified software reuse licenses. This is a constraint
of this pass, not a claim that CPU implementations are impossible. Stop before
method invention. Finish the bounded controls, quantify their tradeoffs, and
require a reviewed benchmark/fidelity budget before advancing the novelty gate.
