# Learned Reranker Promotion Procedure

Cartographer can fit a small, transparent pairwise residual ranker after the semantic configuration is selected. It learns route-specific weights over runtime-visible scores only; ASINs, labels, and evaluator code are never runtime features.

## Locked public split

`docs/public_split_v1.json` deterministically divides the 200 public sessions into two disjoint, scenario-stratified halves. Each contains 40 Buying, 40 Browsing, 15 Intent Override, and 5 Boundary sessions. The manifest records the source dataset checksum and uses only `sample_id` and `scenario_type` to choose membership; target ASINs are not used.

- Development 100: model fitting, diagnostics, and five-fold model selection.
- Holdout 100: one-time audit after the algorithm is frozen.

Reproduce and verify the manifest:

```powershell
python -m cartographer.data_split
```

The generator refuses to replace an existing manifest with different membership.

## Train and cross-validate on development only

For the current offline route:

```powershell
python -m cartographer.train_ranker --split development --routes buying,boundary,override --cross-validate --output cartographer/ranker_weights.json
```

After semantic screening identifies a finalist:

```powershell
python -m cartographer.train_ranker --split development --with-dense --semantic-config FINALIST --routes buying,boundary,override --cross-validate --output data/cartographer_index/ranker.json
```

Route-selective trials can protect a strong deterministic route while learning only where held-out evidence supports it. For example:

```powershell
python -m cartographer.train_ranker --routes buying,boundary,override --cross-validate --output data/cartographer_index/ranker.json
```

Omitted routes are stored with explicit zero weights, so they retain their exact baseline ordering instead of falling through to another learned route.

The command simulates only the development conversations to collect positive-versus-negative feature pairs. Each of five models is trained without its held-out development fold and evaluated only on that fold. The frozen JSON artifact is fitted on the complete development 100, not the holdout.

## Promotion gate

Before consuming the locked holdout, treat a candidate as promotion-eligible only when `training.cross_validation.promotion.eligible` is `true`. The inner gate requires:

- At least `0.005` out-of-fold TechnicalScore gain over the identical non-learned configuration.
- At least four of five held-out folds match or beat that baseline.
- No scenario Hit Rate regression.
- No scenario MRR regression greater than `0.02`.
- No scenario MTTC regression greater than `0.25` turns.
- Observed held-out p95 latency no greater than `750 ms`.

The fitted model's training score is intentionally not reported as evidence. Inner out-of-fold results guide development; the locked holdout is evaluated only after freezing the selected configuration.

At the user's explicit methodology reset, the current fixed candidate was audited despite its two inner warnings. The disjoint holdout then improved every scenario and measured p95 well below the latency threshold, so the artifact remains enabled on that stronger one-time evidence. This exception cannot be repeated for future tuning because the holdout has now been consumed.

## Development-only and locked-holdout result

The selective `buying,boundary,override` model was trained on the development 100. Inner five-fold TechnicalScore improved from `0.885568` to `0.921516` (`+0.035948`); four of five folds improved. The inner gate warned that Boundary MRR regressed on only five development sessions and one fold measured `765.617 ms` p95.

The model was then evaluated once on the disjoint holdout 100:

| Configuration | Hit Rate@10 | MRR | MTTC | TechnicalScore | p95 latency |
|---|---:|---:|---:|---:|---:|
| Deterministic baseline | 1.000 | 0.683210 | 2.16 | 0.881763 | 256.248 ms |
| Development-100 frozen reranker | 1.000 | 0.781397 | 1.91 | 0.916219 | 241.569 ms |

The holdout gain is `+0.034456`; every scenario's TechnicalScore improved. This one-time result supersedes the earlier `0.919897` in-sample all-200 artifact, which remains recoverable at Git tag `working-0.919897` but is no longer loaded at runtime. Dense features remain zero until a semantic configuration is selected using development-only evidence and receives a new independent evaluation plan.

Reproduce the already-recorded comparison only for audit, not parameter selection:

```powershell
python -m cartographer.evaluate_split --partition holdout --ranker cartographer/ranker_weights.json --output data/cartographer_index/holdout_v1_evaluation.json
```

Because the holdout result has now been inspected, subsequent improvements must be selected using development-only cross-validation. Repeatedly optimizing against this holdout would convert it into another validation set.

## Runtime integrity

The artifact stores only a versioned feature schema and linear weights for Buying, Browsing, Override, and fallback routes. Runtime code loads it with the standard library, fails closed on schema mismatch, and never imports the evaluator or public dataset.
