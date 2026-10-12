# DSS fidelity audit: CVPR 2026

Primary specifications: [main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.pdf),
[supplement](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Pan_Debiased_Sample_Selection_CVPR_2026_supplemental.pdf).
Locally preserved PDFs match the prior recorded SHA256. The main paper references
Appendix 8.1; the downloaded supplement labels the long-tail section **3.1** and
contains an unresolved `??` equation reference. Record that numbering defect;
the actual extension equation is present. This is the Pan/Wei/Xie paper, not ITEM.

Official [Aliinton/DSS](https://github.com/Aliinton/DSS), pinned
`ebbe5dd705584814d1214fe6cc1f1fb40658cf58`. Recursive tree contains no LICENSE;
unlicensed code is inspected only, never copied/imported/executed. New MDA, CCS,
trend and update kernels are independently written from equations. The common
backbone is reused separately from MIT-licensed DivideMix, with attribution.
[This sprint's source hashes](../results/strong-baseline-20261012/source_audit.json)
cover loaders, launch scripts and DSS+ trainer beyond the earlier audit.
[Additional hashes](../results/strong-baseline-20261012/additional_source_audit.json)
cover the base trainer/backbone. Its normal PreActResNet-18 topology is consistent
with the licensed common backbone; feature-return/alternative-head utilities
are not invoked here. Inspected true-label fields feed diagnostic reporting;
their storage alone is not evidence of training leakage. Our trainer receives
none of those fields. Official code reports best-test and last-ten accuracy and
uses validation for the saved best checkpoint; this sprint instead preregisters
final-horizon evaluation and forbids test-based model selection.

## Defining algorithm and mathematical checks

Update marginal m <- .99 m + .01 mean_batch(p). Adjust probability by normalizing
p*pi/m, with pi uniform or estimated from a separate clean calibration set.
Select if argmax(adjusted probability) equals observed label. CCS uses increasing
**per-class probabilities**, not CT's observed-label confidence gaps: continuity-
corrected Mann-Kendall Z, variance t(t-1)(2t+5)/18, threshold Z_.9=1.28155.
Exclude candidates from the CE denominator, always retaining the annotated class.
Selected-example CCS losses average over the **whole minibatch**. Excluded logits
have exactly zero gradient; the tests verify the direct reduced-denominator loss.
This is neither relabeling nor a fixed class quota.

The imbalance extension's prior is the class frequency in a small **clean
validation** set. It is not the contaminated observed-label histogram. Our
prior-calibration set has the specified long-tail population distribution and is
separate from balanced model-selection validation. Only its clean labels enter
the prior; true training labels never do. Uniform DSS is retained as a control.
The published long-tail experiment uses CIFAR-10N-Worst and imbalance10, while
this requested benchmark uses controlled synthetic corruption. That dataset
change must be disclosed; no paper accuracy reproduction is claimed.

## Verified schedule and augmentation

| Setting | DSS base | DSS+ CIFAR-10 setting |
|---|---|---|
| Backbone | PreActResNet-18 | PreActResNet-18, two independently initialized learners |
| Horizon / warmup | 150 epochs; inspected default30, launch script accepts warmup argument | 300 epochs /20 warmup |
| SGD | lr .02, momentum .9, wd .001, batch128 | lr .02, momentum .9, wd .0005, batch128 |
| Schedule | lr /10 at80 | cosine to .0004 |
| MDA / CCS | marginal decay .99, significance .10 | same |
| Augmentation | crop32/padding4, horizontal flip; normalized CIFAR statistics | weak crop/flip; strong CIFAR10 AutoAugment; same normalization |
| Additional training | selected CCS supervision | peer masks/candidate sets, weak-to-strong pseudo-target consistency with MixUp beta4, weight2 |

DSS+ uses consistency on all CIFAR minibatch examples, including warmup, and
exchanges frozen selection/candidate policies after epoch updates. Omitting
consistency, sharing initialization or self-selecting would drop essential
mechanisms. Pilot kernels retain peer exchange, two learners, strong augmentation,
MixUp and consistency. Incremental trend scores are tested against direct
pairwise sums rather than replacing them with a different diagnostic.

## Equation/code conflicts and present limitations

Supplement Eq.6 defines pseudo-target exponent **1/T**; its table gives T=2.
Official `mixup_loss` instead raises targets to **T** (default2), sharpening
rather than flattening them. Official consistency targets are also MDA-adjusted;
Eq.6 writes the raw weak prediction. The independent DSS+ kernel follows the
published equation (raw, reciprocal temperature); this is an **explicit mismatch
with official code**, unresolved for a full reproduction. Unit tests demonstrate
both exponent conventions. Do not select a preferred convention after viewing
evaluation results; resolve/specify it before a full benchmark.

Official base symmetric noise excludes the original label; DivideMix's main
protocol allows it. We use a common recorded replacement convention across
comparators, so nominal50% means expected45% actual flips. The warmup/horizon,
dataset noise and calibration-policy departures prevent claiming reproduction
of the paper's headline results.

The current pilot uses128 real images, batch16, four warmup microepochs and
three timed update rounds. Partial late microepochs retain last predictions for
unvisited IDs; early histories cannot assess mature CCS reliability. Full-data
epoch transitions, full schedules, calibration and converged accuracy are not
executed. Base DSS equations and DSS+ operations are verified; **a completed
faithful benchmark result is not yet available**. CPU float32 replaces the
official CUDA autocast path. Memory/time projections are feasibility evidence,
not evidence that DSS fails or solves hard-clean selection.
