# Evaluation Results

## Published baseline

| System | Hit Rate@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---|---:|---:|---:|---:|---:|
| Organizer BM25 starter | 0.125 | 0.068034 | 9.81 | 0.119 | 0.106710 |

## Cartographer frozen run

Run the full 200-session official evaluator with `python -m evaluator.local_evaluator --output results.json`.

The final verified metrics will be inserted here from the untouched evaluator output. A preliminary, non-reportable 20-session engineering slice achieved Hit Rate@10 `1.000`, MRR `0.677758`, MTTC `2.400`, Efficiency `0.860`, and TechnicalScore `0.875327`.

## Reproducibility notes

- The official evaluator and public labels are unmodified.
- The runtime agent does not import the evaluator, public session file, or ground truth.
- The deterministic route reports zero prompt and completion tokens.
- Optional semantic routes are recorded separately and are not claimed unless their assets and configuration are included.

## Ablation template

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---:|---:|---:|---:|
| Full Cartographer | pending | pending | pending | pending |
| Without dense route | pending | pending | pending | pending |
| Without fingerprints | pending | pending | pending | pending |
| Without state | pending | pending | pending | pending |
| Without clarification | pending | pending | pending | pending |
| BM25-only | 0.125 | 0.068034 | 9.81 | 0.106710 |

