# Semantic Promotion Procedure

Cartographer's official default remains the verified offline configuration until BGE demonstrates a material, stable improvement. Merely placing model files in the index directory does not activate dense inference.

## 1. Import and verify the GPU artifact

Extract these paths under `data/cartographer_index/`:

```text
embeddings.npy
embeddings_manifest.json
bge-small-en-v1.5/
```

Install the optional runtime dependencies, then verify catalog identity, ASIN row ordering, dimensions, dtype, and matrix checksum:

```powershell
python -m pip install -r requirements.txt
python -m cartographer.build_embeddings --verify-only
```

Do not evaluate an artifact unless verification reports `verified: true`, 50,000 rows, and 384 dimensions.

## 2. Screen semantic strategies on 40 sessions

The screening set compares single-query and dual-query formulations, dense similarity weights, dense rank, agreement features, calibration, and diversity. It always includes the offline baseline:

```powershell
python -m cartographer.experiments --mode semantic --limit 40 --include offline_baseline,semantic_default,structured_query,conversation_query,compiled_query,dense_0.5,dense_0.8,dense_1.5,dense_rank_0.5,dense_constraint_1,dense_category_0.5,dense_browsing_heavy,semantic_no_diversity --output semantic_screen_40.json
```

Use this run only to eliminate weak configurations. A 40-session result is not reportable and cannot promote the production default.

## 3. Run finalists on all 200 sessions

Select the best three or four screening configurations and rerun them with `offline_baseline`, omitting `--limit`:

```powershell
python -m cartographer.experiments --mode semantic --include offline_baseline,FINALIST_1,FINALIST_2,FINALIST_3 --output semantic_finalists_200.json
```

The generated `promotion_gate` requires all of the following:

- Dense assets actually loaded.
- TechnicalScore improvement of at least 0.01 over the offline baseline.
- Warm observed p95 turn latency no greater than 750 ms.
- At least four of five stratified folds match or beat the baseline.
- No scenario Hit Rate regression.
- No scenario MRR regression greater than 0.02.
- No scenario MTTC regression greater than 0.25 turns.

The gate reports one `recommended` configuration only when every condition passes. Until then, `AgentConfig.enable_dense` remains `False`.

## 4. Fit the residual reranker

Once a semantic finalist passes, train the route-specific residual ranker against exactly that configuration and require its independent out-of-fold gate to pass:

```powershell
python -m cartographer.train_ranker --with-dense --semantic-config FINALIST --cross-validate --output data/cartographer_index/ranker.json
```

See [LEARNED_RANKER.md](LEARNED_RANKER.md) for its leakage controls and promotion criteria. If it fails, keep the semantic finalist without learned reranking.

## 5. Freeze and independently verify

Copy the recommended configuration's parameters into `AgentConfig`, enable dense retrieval, and run:

```powershell
python -m unittest discover -v
python -m evaluator.local_evaluator --output results.json
python -m cartographer.benchmark --limit 200 --output semantic_benchmark_200.json
```

Repeat the official evaluator once to confirm deterministic product ordering. Record semantic and learned-ranker assets as enabled only after the repeated score matches and the benchmark passes.

## Why the search grid is structured this way

- `structured_query` encodes category and active constraints once per turn.
- `conversation_query` encodes the complete active intent epoch once per turn.
- `compiled_query` creates one compact labeled intent and is the main latency candidate.
- `semantic_default` blends separately normalized structured and conversational vectors.
- Dense similarity scores every catalog row, not only the dense top 300.
- Dense rank and constraint/category agreement are independent features, preventing broad semantic similarity from overpowering exact requirements.
