# Evaluation Results

## Published baseline

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 starter | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

## Historical locked 100-session holdout

The preceding RRF-60 artifact was trained on the development half defined by `docs/public_split_v1.json` and evaluated once on the disjoint holdout half. Both halves preserve the official 40/40/15/5 scenario mix. This holdout is now consumed and was not used to select or evaluate the active RRF-120 successor.

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Holdout without reranker | 1.000 | 0.683210 | 2.16 | 0.884 | 0.881763 |
| Holdout with development-100 reranker | **1.000** | **0.781397** | **1.91** | **0.909** | **0.916219** |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Buying | 40 | 1.000 | 0.702946 | 1.075 | 0.909384 |
| Browsing | 40 | 1.000 | 0.842212 | 1.95 | 0.933664 |
| Intent Override | 15 | 1.000 | 0.844444 | 3.733333 | 0.898667 |
| Boundary | 5 | 1.000 | 0.733333 | 2.8 | 0.884000 |

The development-only reranker improved holdout TechnicalScore by `0.034456`, MRR by `0.098187`, and MTTC by `0.25` turns while preserving perfect Hit Rate@10. It improved reciprocal rank in 31 sessions, tied in 56, and regressed in 13; it improved conversion turn in 24 and never delayed a conversion. At the per-session TechnicalScore-contribution level, it produced 40 wins, 47 ties, and 13 losses. A deterministic 100,000-resample paired bootstrap (seed `2026`) placed the 95% percentile interval for the gain at `[0.009670, 0.059391]`. Boundary contains only five holdout sessions and must not be treated as a stable standalone estimate.

This split was created after the architecture and route choices had already been developed against the full public set. The holdout therefore tests whether newly fitted 100-session weights transfer to disjoint sessions, but it is not a pristine estimate of the entire architecture-selection process. Only the private 800 can provide that final independent test.

The active development-only RRF-120 artifact SHA-256 is `464d2daa3c0862c049fd497439d3133cc6d1ea18eb7ad6153271bd70e4b0d884`; the split-manifest SHA-256 is `2e92573ca1275da9243135fd1dd0899138f0db971d52c08a3d9810ceb3d0169b`. Its five-fold development OOF TechnicalScore is `0.930420`. Buying, Boundary, and Override routes learn residual weights with scales `1.25`, `0.75`, and `0.75`; ordinary Browsing and fallback routes store explicit zero weights. The RRF-60 holdout artifact remains recoverable at `working-holdout-0.916219`.

## Reproducibility notes

- The official evaluator and public labels are unmodified.
- The runtime agent does not import the evaluator, public session file, or ground truth.
- The deterministic route reports zero prompt and completion tokens.
- Public labels from the development 100 are used only by the trainer; the frozen runtime artifact contains feature weights and never reads labels or evaluator internals.
- The holdout was evaluated once after fitting. It must not be used for subsequent parameter tuning.
- Optional semantic routes are recorded separately and are not claimed unless their assets and configuration are included.

## Diagnostic component ablation

These pre-reranker ablations use a fixed 20-session engineering slice and are included to show component directionality; the locked holdout table above is the current generalization result.

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Full Cartographer | 1.000 | 0.683929 | 2.40 | 0.877179 |
| Without dense route | 1.000 | 0.683929 | 2.40 | 0.877179 |
| Without fingerprints | 1.000 | 0.567560 | 2.65 | 0.837268 |
| Without state | 0.800 | 0.455119 | 4.05 | 0.675536 |
| Without clarification | 0.750 | 0.282202 | 5.60 | 0.567661 |
| BM25-only ablation | 0.500 | 0.154365 | 7.00 | 0.376310 |

The dense row is identical because semantic model assets were intentionally absent from the frozen deterministic configuration. The published organizer BM25 baseline remains the authoritative baseline; the BM25-only diagnostic above differs because it still uses Cartographer's response plumbing and implicit miss handling.

## Performance and model gates

The active RRF-120 candidate's maximum sequential inner-fold p95 was `543.443 ms`; a separate warm run through the promoted default measured mean `144.421 ms`, p95 `380.588 ms`, and maximum `633.244 ms`. Scores from the all-development-trained warm run are not selection evidence. The historical RRF-60 holdout run measured p95 `241.569 ms`. Catalog construction is excluded from turn timings.

The verified local BGE model loaded successfully, but its first 128-document CPU batch required `56.62 s`, projecting roughly six hours for the 391-batch catalog build on the test host. A checksummed GPU handoff now generates this artifact offline; semantic scores remain excluded from every result above until that artifact is imported and re-evaluated. The cross-encoder has not yet been promoted.
