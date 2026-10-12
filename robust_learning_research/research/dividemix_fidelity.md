# DivideMix fidelity audit: ICLR 2020

Primary [paper, Algorithm 1 and Eqs.1-12](https://arxiv.org/pdf/2002.07394).
Official [LiJunnan1992/DivideMix](https://github.com/LiJunnan1992/DivideMix), pin
`d9d3058fa69a952463b896f84730378cdee6ec39`. MIT license verified. The exact
`PreResNet.py` and its original LICENSE are vendored with unchanged-byte hashes;
training equations are independently implemented. Source audit records the
official trainer/loader for inspection. There is no reuse of unlicensed DSS code.

## Essential training mechanisms

Two independently initialized networks. At each epoch, scan observed-label CE,
min-max normalize loss, fit two-component GMM; lower-mean posterior is clean
probability. Threshold it and give the resulting partition/weights to the peer.
Keep rejected-label examples as **unlabeled training data**. Two weak views,
peer-weighted labeled refinement, four-prediction co-guessing, sharpening and
MixUp supply supervised soft CE and unlabeled consistency MSE. Uniform-prior
KL regularization is preserved. The peer is in eval mode and predictions detached;
unit tests verify its weights receive no new gradients. No oracle cleanliness
or true training class is used.

## Verified paper and implementation settings

| Setting | Specification / inspected behavior |
|---|---|
| CIFAR backbone | PreActResNet-18; same MIT architecture used for all pilot methods |
| SGD | lr .02, momentum .9, weight decay .0005 |
| Horizon / schedule | paper300 epochs, lr /10 after150; inspected inclusive code loop can execute301 epochs |
| Batch size | paper128; official CLI default64; pilot16 is a disclosed memory-feasibility departure |
| Warmup | CIFAR-10 10epochs, CIFAR-100 30; asymmetric noise adds negative-entropy confidence penalty |
| GMM | max_iter10, tol .01, reg_covar .0005; min-loss component posterior; seeded here for reproducibility |
| Partition | threshold .5 (.6 for90% noise, not this pilot) |
| MixMatch | M=2, temperature .5, beta alpha4 |
| Loss weights | official lambda_u25 with16-epoch linear ramp; prior regularizer1; paper considers0/25/50/150 via validation |
| Augmentation | crop32/padding4, horizontal flip; CIFAR mean(.4914,.4822,.4465), std(.2023,.1994,.2010) |
| Corruption | main symmetric replacement includes original class; official asymmetric map truck->automobile, bird->airplane, deer->horse, cat<->dog |

Noise masks are generated once outside the trainer, with independent RNG and
IDs, rather than reusing official loader-generated files across incompatible
subsets. Our common controlled benchmark reserves clean validation and adds
imbalance10; these are stated departures from the original balanced50K setup.
No published percentage accuracy is treated as an observed result here.

## Explicit equation/code differences

Paper Eq.10 averages the squared probability-vector norm over examples.
Official code averages squared errors over **examples and classes**. At ten
classes this changes the effective MSE weight by a factor10. The pilot follows
the inspected code convention with lambda_u25; the tested function exposes the
paper's class-sum convention explicitly. Do not conflate their numerical scales.
For a future comparison, register one convention and preserve its corresponding
weight before evaluation.

The official trainer hard-codes CUDA operations, array size50000 and older
iterator access. Those are portability assumptions, not algorithm definitions.
The independent CPU kernels use actual subset length and explicit long labels.
GMM degeneracy/empty partitions are censored rather than repaired with an
invented selection rule. The90%-noise five-epoch loss averaging branch is not
used at50% and is outside this pilot's claimed scope.

## Pilot versus faithful training

128 real images, batch16, four compressed warmup microepochs and three main
rounds are sufficient to exercise two learners, peer GMM division, refinement,
co-guessing, all four augmented batches, MixUp and both losses. They are not
10-epoch warmup/300-epoch full-data training. The timed SSL operation uses
fully ramped lambda_u to measure mature-path cost despite nonconverged models;
this is a timing departure, not a claimed early training trajectory.

Full-data GMM scanning, repeated epoch division, exact long training schedule,
final ensemble test predictions and robust accuracy remain unexecuted unless
the feasibility gate is passed in a reviewed campaign. Pilot branch currently
supports the registered symmetric setting; asymmetric confidence penalty and
high-noise history variants must pass preflight before future runs. No claim of
completed DivideMix benchmark reproduction follows from passing kernel tests.
