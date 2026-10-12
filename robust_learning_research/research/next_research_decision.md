# Next research decision: stop at faithful-benchmark feasibility

**No-go for method invention or a novelty claim.** The requested hard-clean
versus noisy-label failure remains **undetermined on CIFAR-10**. This is neither
evidence that DSS/DivideMix solve it nor evidence that they fail. The hard
stopping rule applies: stop after the bounded pilot, report requirements, and
do not launch the large comparison campaign.

## What this sprint establishes

The earlier decision, method-fidelity, prior-art and failure-study documents
were read; all1,500 pre-existing result artifacts were read and hashed,
including ignored raw results. Prior synthetic controls remain evidence about
their own setup, not realistic published-baseline reproductions.

Official implementations were pinned and loaders, schedules, augmentation and
training operations inspected. DSS source has no verified reuse license, so
its equations were independently implemented. Only DivideMix's MIT backbone
and license are vendored. The imbalance-aware DSS extension uses a separate
clean calibration prior; replacing it with noisy-label frequencies would be
an unfaithful comparator. DSS+ paper/code pseudo-target differences and
DivideMix's paper/code MSE scaling are explicitly recorded.

The one-seed real-image pilot executed six known controls/comparators on full
PreActResNet-18 models, including two-learner peer exchange, DSS+ consistency,
and DivideMix GMM/refinement/co-guessing/MixMatch losses. It took178.81s and
about1.07GiB peak process memory. This verifies operations and feasibility
at batch16, not long training or quality. Fresh development/evaluation seeds
remain unused, and the pilot opened no training-label oracle or test set.

[Measured costs and assumptions](benchmark_feasibility.md) project substantial
CPU requirements; an unmeasured batch128/GPU regime cannot be declared feasible.
Full-data histories, GMM fitting, evaluation and checkpoint costs still need
measurement. No paid GPU, full benchmark or new filtering method was launched.

## Novelty overlap: published mechanisms versus residual questions

| Mechanism/problem | Direct existing overlap | Consequence |
|---|---|---|
| Class/instance selection debiasing and long-tail prior correction | [DSS main paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_Debiased_Sample_Selection_for_Learning_with_Noisy_Labels_CVPR_2026_paper.pdf), [supplement section3.1](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Pan_Debiased_Sample_Selection_CVPR_2026_supplemental.pdf) | Neither MDA nor prior-aware filtering is new |
| History of confidence/class trends | AAAI2025 CT and DSS CCS, documented in the [existing primary-source audit](prior_art_audit.md) | Adding history alone does not establish novelty |
| Peer selection, probabilistic cleanliness and use of rejected labels as unlabeled data | [DivideMix Algorithm1](https://arxiv.org/pdf/2002.07394) | A hard rejection comparison must count its unlabeled training participation |
| Refined/co-guessed targets, sharpening, MixUp and SSL consistency | DivideMix; DSS+ has peer weak/strong consistency | Combining these known components does not establish a contribution |
| Classwise selection, long-tail noisy labels, hard-clean distinction | Existing RoLT/PASS/SLRLNL entries and version/license caveats in [audit](prior_art_audit.md) | Generic quotas here are controls, not faithful replacements for those methods |

Potential residual questions are whether the published prior-aware extension
remains reliable when its *available clean calibration prior* differs from the
training population, and whether quality/retention survives a held-out noise
family under a fixed compute budget. These are **conditional replication
questions**, not established failures or novel contributions. Further direct
prior-art checks would be required even if future experiments reveal a gap.

## Concrete gate to reopen, with no campaign approval implied

1. Resolve/register DSS+ target convention and DivideMix MSE weight convention
   before viewing evaluation results. Validate full-data epoch transitions and
   asymmetric confidence-penalty warmup; the current pilot does not implement
   that unexecuted regime. Preserve essential published schedules/augmentations.
2. On an explicitly provisioned device, measure one full-data warmup and one
   mature-operation epoch at the intended batch size. Measure peak memory,
   growing-history/GMM cost and evaluation overhead. Stop if memory is inadequate.
3. Review a concrete run-count and runtime budget using those measurements.
   The preregistered broad plan is48 development +90 evaluation runs; its
   CPU projection is approximately20,540h in the middle workload scenario.
   Reducing that plan requires a prospective protocol amendment, not hidden
   dropped comparators. GPU speed and cost have not been measured.
4. Only if approved and feasible, select using development clean validation;
   evaluate untouched801-805 seeds and held-out noise configurations. Freeze
   an independent reference difficulty mask before comparator evaluation.
   Report class-wise quality, hard-clean retention, actual rejected-noise
   precision, calibration, macro-F1, balanced accuracy and total time with
   whole-seed comparisons; retain failed and censored runs.
5. If faithful DSS/DivideMix or established class-aware methods close the
   tradeoff, pivot toward replication, calibration-shift characterization or
   compute/fidelity reporting. Do not invent another filtering variation.
   Only a repeated residual after these gates could justify further hypotheses.

Reviewable deliverables: [protocol](strong_baseline_protocol.md),
[DSS fidelity](dss_fidelity.md), [DivideMix fidelity](dividemix_fidelity.md),
[feasibility](benchmark_feasibility.md), [actual results](strong_baseline_results.md).
Earlier work and TrustGeoOpt are preserved; no merge or force push is requested.
