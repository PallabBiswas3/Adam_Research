# Research foundation and execution plan

2026-10-12. Scope: conventional supervised networks, single workstation/GPU.
Initial deliverable: executable baselines, synthetic failure instrumentation,
independent diagnostics/geometry interfaces, reproducible configuration,
verified primary records, mathematical notes, tests and CPU evidence.

Working question: can causal diagnostics distinguish stale moment estimates
from stochastic noise well enough to selectively buy useful off-diagonal
information? TrustGeoOpt is an internal label, not a novel algorithm claim.

## Sequence and decisions

1. Audit empty workspace and installed environment; preserve preexisting files.
2. Scope prior art; verify metadata, review accessible algorithms, mark gaps.
3. Formulate geometry distinctions, counterexamples and falsifiable hypotheses.
4. Implement trusted PyTorch first-order baselines and a small reference Shampoo.
5. Run paired synthetic studies with exact objective outcomes and causal scores.
6. Evaluate diagnostics on held-out seeds/regimes against gradient norm.
7. Validate diagonal-plus-low-rank solves separately; measure their real overhead.
8. If gates B/C support development, preregister a correction and ablation grid.
9. Tune comparable budgets on validation data; run five held-out seeds on MLP/CNN.
10. Expand to ResNet/CIFAR only after inexpensive evidence merits that cost.

Gate A (novelty) remains open: substantial prior art. Gate B (failure) is a pilot,
not a validated failure reproduction. C (prediction), D (correction), E (compute),
F (robustness) and G (publication) remain unpassed. No prototype is enabled in
the factory. Geometry code is a numerical research primitive, not a contribution.

Budget: small CPU pilots now. No automatic data download, GPU campaign, cloud,
heavy framework or submission. The full prompt describes later research scope;
the first execution is completed against its explicit initial definition.
