# Evaluator Observatory Dashboard

The dashboard is a development-only view over the unchanged participant-kit evaluator. It exposes labels and hidden simulator fields for diagnosis, but never adds them to the official `Agent.reset` or `Agent.respond` inputs.

## Install and launch

```powershell
python -m pip install -r requirements-dashboard.txt
python -m cartographer.dashboard --inbrowser
```

The default address is `http://127.0.0.1:7860`. Use another port with `--port 7861`. The dashboard runs locally and does not require network access after installation.

## Session replay

Select any `public_0001` through `public_0200`, choose the runtime features to enable, and replay the exact deterministic evaluator loop. The view shows:

- Every evaluator/customer message and every agent response.
- The structured `ask_attribute` used by the simulator.
- The expected number-one target product, catalog fingerprint, intent card, and override behavior.
- Every recommended product on every turn with title, category, price, rating, and review count.
- Final score, pre-learned score, exact matches, constraint coverage, category agreement, BM25, dense similarity/rank, profile alignment, popularity, learned residual, and cross-encoder score.
- Route, intent epoch, active constraints, compiled query, candidate count, target candidate position, entropy, information gain, cache use, and latency.
- Evidence-based hints that distinguish retrieval, ranking, parsing, and clarification failures.

The replay defaults to the deterministic rule-based ranker. The learned reranker and BGE route are explicit toggles. BGE fails closed when a verified embedding artifact is not installed.

## All 200 expected products

The target browser maps each public development session to its expected ASIN and visible catalog metadata. It makes target distribution and repeated product/category patterns easy to audit.

The repository now records a locked 100/100 split in `docs/public_split_v1.json`. The dashboard can technically display both partitions because all 200 labels are public, but future model selection must use only the development IDs. The holdout was consumed once for the result in `docs/holdout_v1_results.json`; repeatedly tuning against its target browser or live result would invalidate it as unseen evidence.

## Reranker A/B

The batch view runs the unchanged evaluator twice on the same selected prefix:

1. Deterministic baseline with learned reranking disabled.
2. Frozen learned residual reranker enabled.

It reports overall and per-scenario Hit Rate, MRR, MTTC, Efficiency, TechnicalScore, latency, and per-session changes in turn and reciprocal rank. These are public-development diagnostics, not private-test estimates.

## Live latest-code test

Press **Start fresh 200-session test** after saving any agent or Cartographer code changes. The dashboard starts a new Python process, so it imports the current files from disk rather than reusing modules cached by the dashboard server.

After introducing the locked split, this all-200 run is a descriptive regression check, not an independent validation score: half of those sessions trained the current reranker, and the other half has already been used for the recorded one-time audit.

The panel streams one update per completed session and shows:

- Completed count, percentage, elapsed time, and ETA.
- Rolling Hit Rate, MRR, MTTC, Efficiency, and TechnicalScore.
- Rolling per-scenario metrics.
- Hit turn and rank for every completed session.
- Git commit, dirty working-tree paths, and a SHA-256 manifest for every agent/Cartographer Python source file.
- A downloadable final JSON artifact containing all 200 per-session results.

Use **Cancel running test** to terminate the fresh evaluator without stopping the dashboard. Run artifacts are saved under `data/cartographer_index/live_runs/`, which is already excluded from Git.

## Interpreting the result

- Target absent from the candidate pool: prioritize retrieval, embeddings, synonyms, or query parsing.
- Target present but below rank ten: prioritize reranking.
- Target improves after a useful answer: clarification and state tracking are contributing.
- Parsed category/constraints disagree with the customer message: evaluate a compact intent model or LLM.
- Correct behavior with repetitive prose: an LLM may improve presentation, but not necessarily the technical score.
