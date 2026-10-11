# Targeted Sprint 2 literature audit

Date: 2026-10-12. This is a focused scoping update, not an exhaustive systematic
review. Included primary sources address Adam counterexamples, directional
restart, magnitude safeguards, or the claimed geometry/refresh distinction.
Search strings: `site:proceedings.mlr.press adaptive restart accelerated gradient
schemes ODonoghue Candes 2015`; `"Adaptive Restart for Accelerated Gradient
Schemes" arxiv`; `site:arxiv.org gradient clipping normalized gradient descent
1905.11881 authors`. Existing source URLs were opened directly. Retrieval used
web search plus primary full texts; database hit totals are unavailable.

| Source | Verified content and consequence | Extraction limit |
| --- | --- | --- |
| [Reddi, Kale, Kumar](https://arxiv.org/pdf/1904.09237) | Appendix A assumptions extracted into the counterexample configuration; projected online regret is separate from fixed-rate quadratic progress. | Selected algorithm/footnote/proof, not all theorem audits |
| [AdaBelief](https://arxiv.org/html/2010.07468v5) | Existing residual-moment baseline remains relevant to reliability language. | Selected Algorithm 2; no new proof claim |
| [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html) | Small cumulative-factor reference is an existing comparator. | Original algorithm; no modern implementation efficacy claim |
| [Adaptive Restart](https://arxiv.org/pdf/1204.3982) | Gradient/iterate alignment motivates first-order restarts; any AdamW adaptation needs its own empirical assessment. | Selected observable/restart conditions, no transfer of accelerated-method guarantees |
| [Clipping analysis](https://arxiv.org/abs/1905.11881) | Norm-based magnitude safeguards have prior art. | Metadata/abstract; gradient clipping differs from AdamW step clipping |
| [Clipping bias](https://arxiv.org/abs/2305.01588) | Noise and clipping bias are relevant alternatives to a geometry interpretation. | Metadata/abstract, no imported theorem constants |
| [SOAP](https://arxiv.org/html/2409.11321v2) | Adam in a structured eigenbasis directly overlaps geometric adaptive updates. | Selected mechanism/algorithm; reported benchmark gains not used as our evidence |
| [Purifying Shampoo](https://arxiv.org/html/2506.03595v2) | Section 4 uses approximation-error criteria for basis reuse and refinement. | Selected Section 4; adaptive refresh itself cannot establish novelty |
| [FOAM](https://arxiv.org/abs/2606.02365) | Operator error and adaptive damping address Shampoo staleness. | Preprint metadata/selected HTML; peer-reviewed status not established |

Excluded retrieval: arXiv 1910.03794 is a Gaussian-field paper, not an optimizer
paper; it was opened during discovery and is not retained. Secondary search
results were not used as evidence. Existing 32 records are retained, with the
AMSGrad extraction upgraded and three correction-related primary records added.
The year on the new records is first arXiv posting, explicitly not a verified
journal/proceedings year. No novel mechanism is established by this update.
