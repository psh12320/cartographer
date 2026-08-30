# Development-Only Improvement Backlog

The locked public holdout has been consumed. Every experiment below must be selected using only the 100 development IDs in `docs/public_split_v1.json`. Do not rerun the holdout to choose features, weights, routes, thresholds, or models. Use nested development cross-validation when selecting among alternatives.

## What the one-time audit established

- The development-only reranker improved TechnicalScore by `0.034456` on the disjoint weight holdout, with a paired bootstrap interval of `[0.009670, 0.059391]`.
- Hit Rate@10 remained `1.0`; remaining gains are primarily ranking and early-turn ordering.
- Per-session TechnicalScore produced 40 wins, 47 ties, and 13 losses.
- Conversion was earlier in 24 sessions and never later.
- Intent Override was the most consistent route-level gain.
- Buying improved overall but had 11 rank regressions and no increase in rank-one count on the holdout.
- Boundary has only five independent sessions per partition, so its large holdout gain is not stable evidence for a separate high-capacity model.

These are audit observations, not parameters to optimize against. The architecture itself had already been developed on all 200 public sessions before the split, so the private 800 remains the only pristine end-to-end test.

## P0: Make development diagnostics source-aware

Before changing ranking, record the following on development folds only:

- Target rank in lexical, category, exact-fingerprint, and dense routes.
- Candidate-union recall at 10, 40, 300, and the cache limit.
- Target rank before deterministic scoring, after the learned residual, and after any semantic reranker.
- Oracle best rank across retrieval sources.
- Runtime route by public scenario and turn.
- Asked attribute, number of constraints disclosed, target-rank movement, and next-turn conversion.
- Warm latency separated into FTS, embedding, candidate scoring, clarification, and reranking.

Also add a fixed-message replay that measures ranking changes without allowing changed recommendations to alter the simulated dialogue. This separates static rank quality from end-to-end conversation effects.

## P1: BGE as a candidate tie-breaker

Once the verified 50,000-row embedding artifact arrives, first score the existing candidate pool instead of aggressively expanding retrieval. Perfect eventual Hit Rate suggests the initial semantic opportunity is rank ordering.

Compare a deliberately small query set in nested development CV:

1. `compiled`: category plus active requirements, excluding conversational boilerplate.
2. `structured`: category plus normalized constraints.
3. `blend`: structured intent plus active-epoch conversation wording.

Measure raw cosine, dense rank/RRF, dense multiplied by constraint agreement, and dense multiplied by category agreement independently. Retrain the linear reranker whenever dense features change. For Override, embed only the corrected epoch and retained valid constraints. For Boundary, reuse the existing semantic query rather than encoding “no preference.”

## P1: Protect reliable Buying ranks

Buying is the largest remaining score opportunity, but unrestricted residual addition can turn a later deterministic rank-one hit into an earlier low-ranked hit. On development-only nested CV, compare:

- Learned scales `0.25`, `0.5`, `0.75`, and `1.0`.
- Stronger L2 regularization.
- Reciprocal-rank fusion between deterministic and learned orders.
- A guard that preserves a deterministic top product when hard-constraint agreement and score margin are both strong.
- Recommendation depths `1`, `3`, `5`, and `10`, including confidence-gap expansion after clarification.

One earlier turn contributes `0.02` per session to TechnicalScore, while demoting rank one to rank five loses `0.24` through MRR. The experiment objective must therefore combine turn and rank exactly as the evaluator does.

## P1: Fix the Boundary cache transition

`HybridRetriever.search()` currently keys its cache by route, category, query text, and active constraints. A “no preference” answer changes `declined_attributes`, which changes the reranker route to `boundary`, but does not necessarily change that cache signature. The cached ordering can therefore return before Boundary weights are applied.

Add the observable ranker phase or declined-attribute marker to the retrieval signature, then verify on development Boundary folds that the candidate list is rebuilt exactly once after a decline. This is a correctness fix, but its score effect must still be estimated without the consumed holdout.

## P2: Share and shrink route weights

Replace fully independent small-route models with:

```text
phase weights = shared global weights + regularized phase residual
```

Shrink residuals according to independent session count: Boundary strongest, Override next, constraint-rich Buying least. Compare Boundary alternatives using development folds:

- No learned Boundary residual.
- Shared constraint-rich weights.
- Shared default ordering.
- A reduced Boundary feature set.

Correlated product pairs do not increase the number of independent conversations. Report independent sessions, snapshots, and pairs separately.

## P2: Align training with MRR and MTTC

The current uniform pairwise objective optimizes target-over-negative ordering, not the official score. Test:

- Equal total weight per session.
- One snapshot per observable stage instead of every eligible turn.
- Top-rank hard negatives rather than 30 uniformly weighted negatives.
- Position-weighted or listwise loss emphasizing ranks 1–10.
- Higher weight on earlier turns.
- Separate rating quality and log review-count features instead of one popularity signal.
- Within-query score percentile or z-score instead of the saturating `score / (1 + abs(score))` base feature.

Keep ASINs and labels out of runtime features and inspect coefficient stability across folds.

## P2: Clarification and diversification ablations

The evaluator's `other` attribute can reveal up to two remaining constraints, making it unusually strong. Compare only a small development-only policy set:

- Always `other` first.
- Current entropy policy.
- Typed entropy first, `other` after a decline.

Log disclosure count and target-rank movement. Separately test Browsing diversification off versus on; with Hit Rate already saturated, diversity is valuable only if it improves end-to-end reciprocal rank or clarification quality.

## Performance gates

- Repeat warmed development benchmarks and report median and worst p95; the inner-CV maximum of `765.617 ms` conflicts with the much lower one-time audit p95.
- Use `argpartition` plus deterministic top-set sorting for dense top-k rather than sorting all 50,000 scores.
- Benchmark float32 and float16 matrices; the smaller representation is not automatically faster on CPU.
- Measure process RSS in addition to Python allocations.
- Do not evaluate a MiniLM cross-encoder until BGE clears score and latency gates.

## Decision sequence

1. Add diagnostics and nested development CV.
2. Verify the Boundary cache transition.
3. Evaluate BGE candidate scoring.
4. Evaluate Buying shrinkage/rank protection and recommendation depth.
5. Evaluate shared route residuals and the revised training objective.
6. Freeze one final configuration without reopening the consumed holdout.
7. Run the private 800 only through the organizer.
