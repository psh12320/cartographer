# Learned Reranker Promotion Procedure

Cartographer can fit a small, transparent pairwise residual ranker after the semantic configuration is selected. It learns route-specific weights over runtime-visible scores only; ASINs, labels, and evaluator code are never runtime features.

## Train and cross-validate

For the current offline route:

```powershell
python -m cartographer.train_ranker --cross-validate --output data/cartographer_index/ranker.json
```

After semantic screening identifies a finalist:

```powershell
python -m cartographer.train_ranker --with-dense --semantic-config FINALIST --cross-validate --output data/cartographer_index/ranker.json
```

Route-selective trials can protect a strong deterministic route while learning only where held-out evidence supports it. For example:

```powershell
python -m cartographer.train_ranker --routes buying,boundary,override --cross-validate --output data/cartographer_index/ranker.json
```

Omitted routes are stored with explicit zero weights, so they retain their exact baseline ordering instead of falling through to another learned route.

The command simulates public conversations to collect positive-versus-negative feature pairs. Each of five models is trained without its held-out scenario-stratified fold and evaluated only on that fold. The final JSON artifact is then trained on all public sessions for submission use.

## Promotion gate

Enable `AgentConfig.enable_learned_reranker` only when `training.cross_validation.promotion.eligible` is `true`. The gate requires:

- At least `0.005` out-of-fold TechnicalScore gain over the identical non-learned configuration.
- At least four of five held-out folds match or beat that baseline.
- No scenario Hit Rate regression.
- No scenario MRR regression greater than `0.02`.
- No scenario MTTC regression greater than `0.25` turns.
- Observed held-out p95 latency no greater than `750 ms`.

The all-data model's training score is intentionally not reported as evidence. The out-of-fold result is the promotion evidence, and the untouched official evaluator is run again only after freezing the selected configuration.

## Frozen offline result

The selective `buying,boundary,override` model passed: out-of-fold TechnicalScore improved from `0.8836655` to `0.9218924`, all five folds improved, and the maximum held-out fold p95 was `590.589 ms`. Two byte-identical official full-set runs then scored `0.919897`, so the frozen offline configuration enables this artifact. Dense features remain zero until a semantic configuration independently passes its own gate and the ranker is retrained against it.

## Runtime integrity

The artifact stores only a versioned feature schema and linear weights for Buying, Browsing, Override, and fallback routes. Runtime code loads it with the standard library, fails closed on schema mismatch, and never imports the evaluator or public dataset.
