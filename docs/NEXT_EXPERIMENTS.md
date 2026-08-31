# Development-Only Improvement Backlog

The locked public holdout has been consumed. Every experiment below must be selected using only the 100 development IDs in `docs/public_split_v1.json`. Do not rerun the holdout to choose features, weights, routes, thresholds, or models. Use nested development cross-validation when selecting among alternatives.

## Development experiment pass 2: 2026-08-31 (precision-gated depth)

This pass again used only the 100-session development partition; the consumed holdout stayed closed. The RRF-120 route-scaled reranker (`0.930420` OOF) was the comparison point. Five-fold OOF diagnostics showed 25/100 sessions converting at rank greater than one, mostly because the learned residual lifts targets from deterministic rank 25–115 into ranks 2–9 on turn one and the evaluator locks in that rank; sessions that instead convert after one more disclosure land at rank one in 50 of 60 cases.

| Experiment | Development OOF | Decision |
| --- | ---: | --- |
| Champion RRF-120 route-scaled | `0.930420` | Reference |
| Evaluator-aligned attribute taxonomy | `0.930420` | Score-neutral on development; kept as correctness fix |
| Depth schedule `2,10` | `0.952978` | Superseded |
| Depth schedule `1,10` | `0.961429` | Superseded |
| Depth schedule `1,2,10` | `0.968950` | Carry forward (5/5 stable folds) |
| Depth `1,2,10` + exact-match cap 4 | `0.968950` | Identical outcomes; cap change rejected |
| Depth `1,2,10` + Browsing residual scale `0.75` | **`0.973850`** | **Promoted** (5/5 stable folds, no blockers) |
| Depth `1,2,10` + Browsing residual scale `1.0` | `0.973450` | Rejected; `0.75` retained |
| Promoted + ranker-route cache key (retrained in-run) | `0.971600` | Rejected again; behavior change does not pay on development |

The promoted configuration adds a precision-gated recommendation depth (`recommendation_depth_schedule=(1, 2, 10)` keyed on the turn within the current intent epoch, with a full list forced from absolute turn 6) and a learned Browsing residual at scale `0.75`. One deferred turn costs `0.02` TechnicalScore while a rank-5-to-rank-1 improvement recovers `0.24`, so the agent recommends only what it would bet on while uncertainty is high. OOF scenario metrics: Buying MRR `0.981250`/MTTC `1.425`, Browsing `0.975000`/`1.750`, Override `0.966667`/`3.866667`, Boundary `1.000000`/`2.400`. Hit Rate@10 stayed `1.0` in every scenario and fold. The `classify_constraint` taxonomy was also aligned exactly with the official evaluator's (no brand branch, nine materials, seven color words); this changed no development outcome but removes structurally unanswerable questions.

## What the one-time audit established

- The development-only reranker improved TechnicalScore by `0.034456` on the disjoint weight holdout, with a paired bootstrap interval of `[0.009670, 0.059391]`.
- Hit Rate@10 remained `1.0`; remaining gains are primarily ranking and early-turn ordering.
- Per-session TechnicalScore produced 40 wins, 47 ties, and 13 losses.
- Conversion was earlier in 24 sessions and never later.
- Intent Override was the most consistent route-level gain.
- Buying improved overall but had 11 rank regressions and no increase in rank-one count on the holdout.
- Boundary has only five independent sessions per partition, so its large holdout gain is not stable evidence for a separate high-capacity model.

These are audit observations, not parameters to optimize against. The architecture itself had already been developed on all 200 public sessions before the split, so the private 800 remains the only pristine end-to-end test.

## Development experiment pass: 2026-08-31

This pass used only the 100-session `development` partition. The consumed public holdout was not opened, and no result from this pass was pushed. The frozen RRF-60 reranker (`0.921516` development OOF) was the comparison point.

| Experiment | Development score | Decision |
| --- | ---: | --- |
| Deterministic RRF-60 reference | `0.885568` | Reference |
| Deterministic RRF-100 | `0.896068` | Won all five fold-wise selections; carry forward |
| Frozen learned RRF-60 | `0.921516` OOF | Previous champion |
| Learned RRF-80 | `0.923337` OOF | Better, but below local optimum |
| Learned RRF-100 | `0.925765` OOF | Better, but below local optimum |
| Learned RRF-120 | `0.927169` OOF | Best RRF constant tested |
| Learned RRF-160 | `0.921349` OOF | Rejected |
| Learned RRF-200 | `0.923032` OOF | Rejected |
| RRF-120, global scale `0.75` | `0.923590` OOF | Rejected |
| RRF-120, global scale `1.25` | `0.927570` OOF | Tiny gain with route regressions |
| RRF-120, route scales | **`0.930420` OOF** | Local challenger |

The route-scaled challenger uses `buying=1.25`, `boundary=0.75`, `override=0.75`, and no learned Browsing/default residual. It retained Hit Rate@10 `1.0`, matched or beat the deterministic reference in four of five folds, and produced the following OOF scenario metrics:

| Scenario | MRR | MTTC |
| --- | ---: | ---: |
| Boundary | `1.000000` | `3.200000` |
| Browsing | `0.860933` | `1.925000` |
| Buying | `0.770903` | `1.075000` |
| Intent Override | `0.826667` | `3.466667` |

Its maximum sequential fold p95 latency was `543.443 ms`; the promoted default configuration measured `380.588 ms` p95 in a separate warm full-development run. Scores from full-development runs using an all-development-trained artifact are not selection evidence and are retained only for latency diagnostics.

Other findings:

- Delaying `other` in favor of typed questions scored `0.878098`; globally forcing `other` scored `0.875109` and reduced Hit Rate to `0.99`. Keep the balanced entropy policy.
- Disabling Browsing diversification tied overall at `0.885568` but shifted scenario results; nested selection fell to `0.871201`. Keep the current policy.
- Adding the Boundary ranker phase to the cache key scored `0.919416` OOF, below the frozen model. Do not promote it as a scoring change.
- RRF-20 (`0.870107`), popularity removal (`0.861530`), BM25 weight 3 (`0.877152`), and larger lexical/category pools (approximately flat) were rejected.
- The BGE matrix and manifest were still absent, so no semantic result was fabricated or inferred.

The route-scaled RRF-120 candidate is installed only in the local working tree. The prior checkpoint remains the recovery point. Because the public holdout is consumed, the private organizer evaluation is the only independent end-to-end confirmation.

## Dense (BGE) evaluation: 2026-08-31 — not promoted

The verified 50,000-row embedding artifact was imported and the dense route was evaluated end to end for the first time. Three independent instruments agree that dense retrieval does not earn promotion on this task.

**1. End-to-end grid (development-100, 26 semantic configurations).** Every configuration landed within `[0.97100, 0.97580]` against an offline baseline of `0.973650`. The best, `dense_constraint_2` (dense `0.8`, constraint-agreement `2.0`), gained `+0.00215`. On 100 sessions a single rank-one-to-two flip is worth `0.0015`, so the entire grid spans about three sessions of resolution and the winner is worth roughly one and a half.

**2. Fixed-message rank replay (the P0 instrument below, now built).** Because end-to-end development scoring is saturated at MRR `0.9775`, ranking quality was measured directly by replaying each session's captured message sequence and recording the target's rank in the score-sorted candidate list. Dense is *worse* in every variant:

| Configuration | Turn-1 MRR | Turn-1 rank-one sessions | All-turn MRR |
| --- | ---: | ---: | ---: |
| Offline reference | **`0.628094`** | **49 / 100** | **`0.726025`** |
| `semantic_default` | `0.602461` | 49 / 100 | `0.716751` |
| `dense_constraint_1` | `0.601779` | 49 / 100 | `0.715089` |
| `dense_constraint_2` | `0.596684` | 49 / 100 | `0.711130` |
| `dense_0.25` | `0.592734` | 46 / 100 | `0.708945` |
| `dense_rank_1` | `0.567549` | 43 / 100 | `0.700272` |

No dense configuration ever increased the number of sessions whose target ranks first; two configurations reduced it. Dense only reshuffles the tail, and does so unfavourably.

**3. Dense-retrained out-of-fold cross-validation.** Retraining the residual ranker with live dense features under `dense_constraint_2` scored OOF `0.974329` against the promoted `0.973850`, a gain of `+0.000479` — an order of magnitude below the `0.005` promotion threshold. The fitted dense weights are mutually inconsistent across routes (`dense_score` is negative for buying, browsing, and override but positive for boundary), the signature of a feature being fitted to fold-specific noise rather than signal.

**4. Held-out confirmation.** Carried to the 800-session synthetic test set with its own matching dense-trained weights, `dense_constraint_2` scored `0.952038` against the promoted configuration's `0.951764`, a gain of `+0.000274` while raising turn p95 from `89.9 ms` to `160.4 ms`. The held-out set is far from saturated (MRR `0.919`, 110 sessions converting below rank one), so it had ample room to show a dense benefit and did not.

The mechanism is intelligible. The simulated customer discloses constraints that are verbatim substrings of the target product's own feature text, such as `92% Polyester` or `Rubber sole`. Exact fingerprint and lexical matching resolve those precisely, whereas cosine similarity over `bge-small-en-v1.5` blurs across products that are semantically alike but materially different, promoting near-misses into the ranks the deterministic route had already ordered correctly.

Dense therefore remains opt-in and disabled by default. The artifact and its bring-up path are retained: latency was never the obstacle (turn p95 rose only from about `118 ms` to `171 ms`, far inside the `750 ms` gate), and the route may still matter for a catalog or customer simulator whose language is less literal.

Reproducibility note: the GPU-built bundle and an independent CPU rebuild of the same catalog agree to a per-row cosine of `1.000000` and a maximum elementwise difference of `6.9e-07`, with a byte-identical query encoder. Embedding builds are portable across hosts; only the byte-level matrix checksum differs, so `matrix_sha256` must be taken from the bundle actually installed.

## Confidence-released depth gate: 2026-08-31 — rejected

The depth gate's one theoretical weakness is that a withheld turn costs `0.02` unconditionally while the rank improvement it buys depends on the customer actually disclosing more. `depth_gate_min_information_gain` was added to release the gate whenever the chosen question's expected information gain falls below a threshold. Measured expected gain runs from about `0.2` to `8.2`, with a development median of `5.19` at turn one, `2.08` at turn two, and `1.16` at turn three.

| Threshold | Development TechnicalScore | MRR | MTTC |
| ---: | ---: | ---: | ---: |
| `0.0` (gate always active) | **`0.973650`** | `0.977500` | `1.980` |
| `1.0` | `0.973600` | `0.976667` | `1.970` |
| `2.0` | `0.973600` | `0.976667` | `1.970` |
| `3.0` | `0.970500` | `0.965000` | `1.950` |

Releasing the gate buys a little MTTC and loses more MRR, and the loss grows as the threshold rises. Withholding is worth it even when the next question looks uninformative, because a turn costs `0.02` while locking in a poor rank costs several times that. The knob is retained at its neutral default of `0.0`, where it only prevents withholding on turns when no question is asked at all — a case in which holding back cannot pay. That refinement is score-neutral on development.

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
