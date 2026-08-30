# Evaluation Results

## Published baseline

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 starter | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

## Cartographer current offline run

Run the full 200-session official evaluator with `python -m evaluator.local_evaluator --output results.json`.

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Cartographer, frozen selective reranker | **1.000** | **0.792323** | **1.89** | **0.911** | **0.919897** |
| Protected checkpoint `working-0.883665` | 1.000 | 0.690885 | 2.18 | 0.882 | 0.883665 |
| Protected checkpoint `checkpoint-0.865767` | 1.000 | 0.636558 | 2.26 | 0.874 | 0.865767 |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 1.000 | 0.719965 | 1.075 |
| Browsing | 80 | 1.000 | 0.851746 | 1.9375 |
| Intent Override | 30 | 1.000 | 0.831667 | 3.60 |
| Boundary | 10 | 1.000 | 0.777778 | 2.90 |

The untouched organizer evaluator reports a `0.036232` absolute improvement over `working-0.883665` and a `0.054130` improvement over `checkpoint-0.865767`, before semantic embeddings are enabled. The current offline configuration improves TechnicalScore by `0.813187` absolute, or approximately `8.62×`, over the published starter.

The frozen route-selective reranker was promoted on a separate five-fold scenario-stratified replay: out-of-fold TechnicalScore increased from `0.883666` to `0.921892` (`+0.038227`), all five folds improved, and no scenario regressed. Buying, Boundary, and Override routes learn residual weights; ordinary Browsing and fallback routes store explicit zero weights. The final artifact was then trained on all public sessions and evaluated twice. Both full result files have SHA-256 `B780D2880479F7ECF64B817FF06614A045FB2641135B45A59E5CF8DAAE45E21D`.

## Reproducibility notes

- The official evaluator and public labels are unmodified.
- The runtime agent does not import the evaluator, public session file, or ground truth.
- The deterministic route reports zero prompt and completion tokens.
- Public labels are used only by the development trainer; the frozen runtime artifact contains feature weights and never reads labels or evaluator internals.
- Optional semantic routes are recorded separately and are not claimed unless their assets and configuration are included.

## Diagnostic component ablation

These pre-reranker ablations use a fixed 20-session engineering slice and are included to show component directionality; only the 200-session row above is a reportable competition result.

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

The frozen configuration's untraced 40-session benchmark initialized in `51.514 s` and evaluated in `24.156 s`. Across 76 agent turns, mean latency was `317.113 ms`, p50 was `278.991 ms`, p95 was `571.942 ms`, and the maximum was `770.988 ms`. A separate allocation-traced run measured `197.577 MiB` of peak Python allocations; the observed development process working set remained below `461 MiB`, including duplicate evaluator structures. Allocation tracing is excluded from the latency gate because it inflated p95 to `1019.025 ms`.

The verified local BGE model loaded successfully, but its first 128-document CPU batch required `56.62 s`, projecting roughly six hours for the 391-batch catalog build on the test host. A checksummed GPU handoff now generates this artifact offline; semantic scores remain excluded from every result above until that artifact is imported and re-evaluated. The cross-encoder has not yet been promoted.
