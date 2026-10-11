# Sprint 3 neural validation

**No structured-optimizer practical advantage is established on these bounded
neural tasks.** AdamW leads on the small correlated-feature MLP, Momentum SGD
helps the deep ReLU MLP, and Fashion-MNIST subset quality is similar across
methods. SOAP costs more and does not improve validation loss.

Evidence: [protocol](sprint3_protocol.md), `processed/neural_selection.json`,
`summary.csv`, `paired_comparisons.json`, `neural_feature_controls.json`, and
`plots/neural_time_quality.png` / `neural_loss.png` under
`results/sprint3/campaign-20261012/`.

## Reproducible scope and fairness

Small MLP: one 32-wide ReLU hidden layer. Deep MLP: six 32-wide ReLU hidden
layers. Correlated synthetic classification has 20 features, 512 training and
256 validation examples; covariance condition is nominally 100. Train-only
feature normalization preserves substantial correlation: measured normalized
training covariance condition is 100.87 for task 700 and 77.44 for held-out
task 701. The latter has a new feature map, teacher and samples, with frozen
optimizer choices from task 700.

Fashion-MNIST: one 32-wide ReLU MLP, 2,048 training and 512 disjoint validation
examples drawn once from the official training split with seed 700. Official
test examples are not used. Download provenance and hashes are in
`research/sprint3_data_provenance.json`; this is a **subset smoke study**, not a
full-dataset benchmark. Pixel normalization uses the fixed torchvision-style
constants .2860/.3530, not statistics estimated from validation data.

CPU, one thread, float32, batch 64, 20 epochs, Kaiming initialization, no decay,
constant rates, identical initialization and minibatch order for paired seeds.
AdamW/Momentum/SOAP each receive four rates x two development seeds per model;
five new evaluation seeds follow. 72 tuning + 45 evaluation + 30 held-out
attempts = 147. Two deep-Momentum tuning attempts at lr=.3 fail numerically and
are retained/disqualified; all evaluations succeed. There is no early stopping,
test-based selection, checkpoint selection or architecture search.

Selected rates: small AdamW .01, Momentum .03, SOAP .03; deep .001/.1/.001;
Fashion .001/.01/.001. Small SOAP hits its upper boundary; deep AdamW/SOAP and
all Fashion choices hit lower boundaries. These are unresolved tuning limits,
especially for claiming that SOAP is intrinsically worse. SOAP's factor limit
128 skips the 784-wide Fashion input-axis factor; it is a disclosed one-sided
structured comparison. Default moments differ across optimizer references.

## Validation quality and total trial wall time

Five paired optimization runs per row; arithmetic mean ± run SD loss:

| Model/task | AdamW loss; seconds/run | Momentum loss; seconds/run | SOAP loss; seconds/run |
| --- | ---: | ---: | ---: |
| Small, task 700 | .0814 ± .0131; .231 | .0988 ± .0197; .166 | .1467 ± .0246; .357 |
| Deep, task 700 | .2606 ± .0423; .327 | .2136 ± .0705; .415 | .3365 ± .0175; .603 |
| Fashion subset | .4861 ± .0259; 1.164 | .4925 ± .0272; .907 | .4899 ± .0227; 1.329 |
| Small, held-out task 701 | .1017 ± .0151; .231 | .1258 ± .0317; .243 | .1283 ± .0199; .359 |
| Deep, held-out task 701 | .3357 ± .0418; .377 | .3073 ± .0708; .381 | .3839 ± .0496; .748 |

Trial time includes data/model/optimizer setup and metrics serialization;
imports and shared process startup are outside trial timers. Very short times
have scheduling/cache noise; this is one machine, not a general throughput
result. Recorded campaign execution includes tuning and failures as well as
evaluations; neural trial timers sum to 81.25 seconds. No GPU was used.

Whole-run paired log10(loss_method/loss_AdamW) intervals:

| Task | Momentum contrast [95% interval] | SOAP contrast [95% interval] |
| --- | ---: | ---: |
| Small task 700 | +.0824 [.0165, .1553] | +.2552 [.2084, .3146] |
| Deep task 700 | -.1020 [-.1819, -.0357] | +.1149 [.0679, .1609] |
| Fashion subset | +.0057 [-.0059, .0182] | +.0034 [-.0180, .0258] |
| Small task 701 | +.0842 [.0400, .1280] | +.1004 [.0386, .1938] |
| Deep task 701 | -.0445 [-.0990, .0100] | +.0577 [.0046, .1181] |

Resampling uses five complete seed pairs, conditional on fixed data and the
two-seed development choice. There is no step-level pseudo-replication,
task-population inference, selection-uncertainty estimate or multiplicity
correction. Deep SGD's original-task quality advantage is less resolved on the
held-out teacher. We do not attribute that advantage causally to Hessian
orientation: neural Hessian/covariance counterfactuals were not constructed.

## Time to target and memory

Targets were predefined: validation loss .25 for synthetic MLPs and .9 for
Fashion. Reached counts AdamW/Momentum/SOAP are 5/5/5 (small), 3/5/0 (deep),
5/5/5 (Fashion), 5/5/5 (held-out small), and 0/5/0 (held-out deep). Unreached
runs are explicitly censored at their full horizon.

On deep task 700, mean **loop** time to first crossing among reached runs is
.161 s for AdamW (three runs) and .051 s for Momentum (five); SOAP has no
crossings. These conditional means cannot be called a paired speedup. On
held-out deep runs Momentum crosses early, yet its final mean loss is .307;
some gains are temporary validation improvements. A sustained-target metric or
development-selected stopping rule is needed before any equal-quality speed
claim. Time-quality plots retain the full curves, including later deterioration.

Mean optimizer tensor storage, in bytes:

| Model | AdamW | Momentum | SOAP |
| --- | ---: | ---: | ---: |
| Small | 5,920 | 2,952 | 25,520 |
| Deep | 48,200 | 24,072 | 149,680 |
| Fashion | 203,616 | 101,800 | 220,784 |

These count tensor state, including basis/factor buffers, and exclude parameters,
gradients, activations and allocator overhead. Sampled process RSS is about
253–254 MB for synthetic tasks and 272–273 MB for Fashion. It is sampled at
initialization/epoch ends within a shared process, not isolated per-trial peak
memory; short temporary allocations can be missed. The tensor-storage counts
are the more interpretable comparison here.

**Decision:** these runs justify no new optimizer, no full Fashion-MNIST claim
and no neural geometric mechanism. More complete baseline schedule/boundary
tuning and sustained-quality timing come before a costly structured proposal.
