# Evaluation Results

## Published baseline

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 starter | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

## Cartographer current offline run

Run the full 200-session official evaluator with `python -m evaluator.local_evaluator --output results.json`.

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Cartographer, current offline config | **1.000** | **0.690885** | **2.18** | **0.882** | **0.883665** |
| Protected checkpoint `checkpoint-0.865767` | 1.000 | 0.636558 | 2.26 | 0.874 | 0.865767 |

| Scenario | Sessions | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 1.000 | 0.687991 | 1.6125 |
| Browsing | 80 | 1.000 | 0.741726 | 2.075 |
| Intent Override | 30 | 1.000 | 0.612394 | 3.70 |
| Boundary | 10 | 1.000 | 0.542778 | 3.00 |

The untouched organizer evaluator reports a `0.017898` absolute improvement over the protected checkpoint before semantic embeddings are enabled. The current offline configuration improves TechnicalScore by `0.776955` absolute, or approximately `8.28×`, over the published starter.

## Reproducibility notes

- The official evaluator and public labels are unmodified.
- The runtime agent does not import the evaluator, public session file, or ground truth.
- The deterministic route reports zero prompt and completion tokens.
- Optional semantic routes are recorded separately and are not claimed unless their assets and configuration are included.

## Diagnostic component ablation

These ablations use a fixed 20-session engineering slice and are included to show component directionality; only the 200-session row above is a reportable competition result.

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

The protected checkpoint's fixed 20-session engineering run completed initialization plus evaluation in `47.889 s`. Across 48 agent turns, mean latency was `256.262 ms`, p50 was `254.715 ms`, p95 was `562.459 ms`, and the maximum was `754.542 ms`. The observed full-evaluator process working set remained below `461 MiB`, including the evaluator's own duplicate catalog structures. The current configuration will receive a fresh latency run before final freezing.

The verified local BGE model loaded successfully, but its first 128-document CPU batch required `56.62 s`, projecting roughly six hours for the 391-batch catalog build on the test host. A checksummed GPU handoff now generates this artifact offline; semantic scores remain excluded from every result above until that artifact is imported and re-evaluated. The cross-encoder has not yet been promoted.
