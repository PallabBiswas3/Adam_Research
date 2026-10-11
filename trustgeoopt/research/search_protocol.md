# Literature search protocol

Date: 2026-10-12 (Asia/Calcutta). Type: initial scoping review, incomplete for publication.

Question: in supervised MLP/CNN training, can past-gradient diagnostics select low-rank corrections that improve loss progress per wall-clock second over tuned adaptive and structured baselines?

Sources: primary arXiv records and full texts, OpenReview, PMLR, NeurIPS proceedings, JMLR, original lecture notes for RMSProp, and official optimizer documentation. Discovery uses web search; Google Scholar access is not assumed. Include foundational work and explicitly search 2024, 2025, and 2026. English-language primary methods/theory papers; exclude derivative explainers and claims supported only by snippets. LLM-focused work is retained only for mechanistic overlap, not as evidence for our supervised-network scope.

Extraction: exact title, full author list when available, first-posting year, verified venue separately, canonical URL, method, mathematical contribution, assumptions, experiments, complexity, limitations, overlap. Unread full-text details are marked not extracted. Deduplicate by canonical identifier, then title. Preserve discovery queries in search_log.json. This is not an exhaustive systematic review and search absence is not evidence of novelty.

Queries planned: Adam/AMSGrad nonconvergence; adaptive-moment variants; K-FAC/EKFAC/Shampoo/SOAP; streaming PCA/gradient sketches; preconditioner staleness and adaptive refresh 2024/2025/2026; heavy-tail noise and adaptive instability. Screen records first; inspect relevant algorithms/theorems in accessible full texts next. No publication recommendation until this second pass and independent novelty audit are complete.
