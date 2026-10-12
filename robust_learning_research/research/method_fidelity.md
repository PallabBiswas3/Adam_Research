# Baseline verification and fidelity

The five trained methods are controls or independently implemented **toy
adaptations**. None reproduces a published image benchmark. The workflow's
two-faithful-baseline gate is **incomplete**, despite tests of the equations.
See [audit](prior_art_audit.md) for primary paper links and source/license pins.

| Method | Verified local behavior | Fidelity departures / interpretation |
|---|---|---|
| CE | Observed-label CE, accepts all, no oracle input | Sanity control; post-hoc negative-loss curves are hypothetical filters |
| Global small loss | Frozen epoch-start loss ranks, three warmup epochs, keep .7 | Known control; no published benchmark claim |
| Classwise small loss | Same rule, independent within-observed-class quotas | Known control; noisy labels also determine the quota groups |
| Co-teaching toy | Two independent networks; minibatch peer sets exchanged; forget rate increases to nominal .3; each peer loss is mean CE | Tiny MLP/20 epochs/AdamW replace image network/horizon/optimizer and benchmark schedules. Prediction uses first network; post-hoc curves average two losses. Not official reproduction |
| Confidence tracking toy | Observed-label probability gaps to every competitor; minimum continuity-corrected Mann-Kendall Z; threshold 1.6448536; union with global small loss; masked CE averaged over entire minibatch | Post-epoch full-dataset probability snapshots replace official minibatch pre-update observations; no weak/strong image augmentation. Descriptive final ranking max(sigmoid(Z),global loss percentile) is **not** the union training rule. Not official reproduction |

Co-teaching's inspected official `loss.py` applies another division after mean
cross entropy; our independent implementation follows the paper's selected-set
mean. A test explicitly checks mean normalization and gradients on peer-selected
examples. We disclose the discrepancy rather than silently claiming identical
code. Official `.cuda()` operations were inspected, not executed.

The CT sign-sum and variance T(T-1)(2T+5)/18 are tested against direct equations,
including rising/falling/constant gaps. Ties use the inspected paper/code
convention without a variance tie correction. Correlated epochs and competitor
minimum mean the threshold is a selector setting, not a calibrated statistical
p-value. Full history storage is measured. Masked-full-batch mean differs from
selected-mean control loss scaling; separate learning-rate tuning partly
addresses it but does not establish equivalence.

DSS, SLRLNL, RoLT, DivideMix, ELR and PASS were not implemented or run in this
campaign. Source inspection alone is not baseline reproduction. DSS+ additionally
requires co-training/consistency; DivideMix requires data division and SSL;
RoLT requires prototype-distance mixtures and soft labels. Replacing these with
classwise loss filtering would omit their defining mechanisms. No such shortcut
is used as evidence of their failure. No unlicensed source was vendored or run.

All methods use the same first-network initialization, batch order, data and
clean validation/test split for a given seed/regime. Co-teaching uses a second
independent initialization. Each method gets the same two-rate/four-run-per-rate
development budget. Classwise and CT choose .003; other methods choose .01.
Only final **clean validation** macro-F1 from development seeds selects learning
rates. Pilot/development records have no true-training-label or test metrics.
Evaluation seeds cannot change rates, epochs, thresholds or checkpoints.

The .3 nominal generation setting is known, whereas realized flips are sealed;
instance noise is .3(.25+.75 ambiguity), so nominal .3 is not its actual rate.
The methods' filtering aggressiveness is consequently not matched by default.
Post-hoc precision matching is an oracle-assisted **evaluation** of frozen
scores, never training or a deployable threshold-selection procedure.

`loop_seconds` includes training, selection, clean validation and final scoring.
`total_seconds` additionally includes data generation, storage and evaluation.
Parameter/optimizer/history bytes are exact tensor counts, **not peak RAM**.
Selector timing measures epoch-start work only and excludes peer minibatch
selection; do not compare it alone as total overhead. Same-thread CPU timings
are descriptive, not portable benchmark throughput. No time-to-target claim is
made for this short discovery pass.

The original generator, smoke script, original results, and all TrustGeoOpt files
are preserved. The old seed-zero smoke was rerun with the original settings and
matched the saved counts; it remains a smoke check. The new controlled campaign
is separate and cannot be pooled with that result as independent evidence.
