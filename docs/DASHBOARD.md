# Evaluator Observatory Dashboard

The dashboard is a development-only view over the unchanged participant-kit evaluator. It exposes labels and hidden simulator fields for diagnosis, but never adds them to the official `Agent.reset` or `Agent.respond` inputs.

## Install and launch

```powershell
python -m pip install -r requirements-dashboard.txt
python -m cartographer.dashboard --inbrowser
```

The default address is `http://127.0.0.1:7860`. Use another port with `--port 7861`. The dashboard runs locally and does not require network access after installation.

## Dataset scope

A **Dataset scope** selector sits above the tabs and re-points every panel — session replay, the target browser, the reranker A/B, and the fresh-process live test. Three scopes are offered when the optional held-out file is present:

| Scope | Sessions | Source |
| --- | ---: | --- |
| `public_set` | 200 | `data/public_set.jsonl` |
| `synthetic_800_v1` | 800 | `synthetic_800_v1.jsonl` |
| `All datasets` | 1000 | both, merged and de-duplicated by `sample_id` |

The merged scope is written to the ignored path `data/cartographer_index/dashboard_datasets/combined.jsonl` so the fresh-process evaluator can consume it as a single `--dataset`. Scopes are discovered at startup and missing files are skipped silently, so the dashboard still launches with only the public set present.

Expose different files with a repeatable flag; it replaces the default rather than adding to it:

```bash
python -m cartographer.dashboard --extra-dataset synthetic_800_v1.jsonl --extra-dataset my_probe_set.jsonl
```

Scope selection changes what you *observe*, not what is permitted to drive decisions. Held-out and synthetic scopes are confirmation readouts; configuration choices must still be made on the development partition through the cross-validation procedure, as `docs/NEXT_EXPERIMENTS.md` requires.

## Session replay

Select any session and choose one or more components to remove. The dashboard runs the exact evaluator loop twice: once with the full current agent and once with the selected ablation. Selecting a single component gives the cleanest demonstration of its marginal value; selecting several exposes interactions. The view shows:

- Every evaluator/customer message and every agent response.
- The structured `ask_attribute` used by the simulator.
- The expected number-one target product, catalog fingerprint, intent card, and override behavior.
- Every recommended product on every turn with title, category, price, rating, and review count.
- Final score, pre-learned score, exact matches, constraint coverage, category agreement, BM25, dense similarity/rank, profile alignment, popularity, learned residual, and cross-encoder score.
- Route, intent epoch, active constraints, compiled query, candidate count, target candidate position, entropy, information gain, cache use, and latency.
- Returned recommendation depth, whether the precision gate activated, and why it held or released the full list.
- A paired per-session TechnicalScore contribution, turn, rank, and reciprocal-rank delta.
- Full-agent and ablated conversations, turn state, ranked products, and inference JSON side by side.
- The largest additive score contributions for every recommendation, including fingerprints, constraints, BM25, profile, popularity, learned residual, and optional semantic scores.
- Evidence-based hints that distinguish retrieval, ranking, parsing, and clarification failures.

The component selector exposes individual ablations for lexical FTS5/BM25, category retrieval, intent fingerprints and safe filtering, multi-turn state, entropy clarification, aggregate-profile personalization, popularity, the frozen reranker, the precision recommendation-depth gate, Browsing diversification, and the optional BGE route. BGE fails closed when a verified embedding artifact is not installed.

## All expected products

The target browser maps each session in the selected scope to its expected ASIN and visible catalog metadata. It makes target distribution and repeated product/category patterns easy to audit.

The repository now records a locked 100/100 split in `docs/public_split_v1.json`. The dashboard can technically display both partitions because all 200 labels are public, but future model selection must use only the development IDs. The holdout was consumed once for the result in `docs/holdout_v1_results.json`; repeatedly tuning against its target browser or live result would invalidate it as unseen evidence.

## Component value lab

The batch view runs the unchanged evaluator on the same selected prefix with:

1. The full current agent.
2. One separate run per selected component with only that component removed.

It reports overall and per-scenario Hit Rate, MRR, MTTC, Efficiency, TechnicalScore, latency, and per-session changes in turn, rank, reciprocal rank, and exact TechnicalScore contribution. Positive `full minus ablated` values mean the component helped on that slice. These are public-development diagnostics, not private-test estimates.

The panel also emits report-ready JSON containing the full-agent metrics, component values, model choice, API/network/cost/token disclosure, and limitations. Use development-only scopes for component selection. A held-out or synthetic scope may be displayed for an already frozen confirmation, but must not become a tuning loop.

## Report and video evidence

The dedicated deliverable map identifies where to capture every required proof:

- Architecture and active-component inventory.
- One complete multi-turn evaluator conversation.
- Reranker and precision-gate value with and without each component.
- Transparent recommendation explanations.
- Overall and per-scenario metrics and latency.
- Model, network, cost, and token disclosure.
- Limitations and the remaining team-contribution section.

For the demo video, select one component at a time. Show an Intent Override replay, point out the intent epoch and retained constraints, then compare gate depth and target rank with the gate removed. Repeat with only the reranker removed, and finish on the component-value and operational-disclosure panels.

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
