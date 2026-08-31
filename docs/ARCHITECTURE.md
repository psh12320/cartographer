# Cartographer Architecture

## Runtime data flow

1. `reset` creates an isolated session with the supplied aggregate profile.
2. The context compiler extracts category and constraints from the current message.
3. Corrections deactivate the superseded initial preference, retain unrelated requirements disclosed through clarification, and start a new text epoch; boundary answers exhaust the requested attribute.
4. The router selects Buying or Browsing weights.
5. Fingerprint, category, FTS5, and optional BGE routes produce a candidate union.
6. Safe hard filters and local reranking produce a candidate posterior.
7. The clarification policy calculates the entropy of possible outcomes for each unasked attribute.
8. The agent returns the current Top 10 and the maximum-information question together.
9. If the evaluator continues the session, returned products become implicit misses and move behind unseen candidates.

## Session state

Each constraint records its allowed attribute, original value, hard or soft strength, source turn, intent epoch, and active/inactive status.

The full natural-language history is not appended to every query. Instead, active state is recompiled into a bounded retrieval context each turn. This keeps behavior explainable and prevents stale intent from contaminating later rankings.

## Ranking features

The default deterministic reranker uses exact intent-fingerprint matches, token and price constraint coverage, coarse-category agreement, normalized FTS5 rank, optional dense similarity, modest profile-attribute agreement, a small rating/popularity tie-breaker, and an optional cross-encoder probability.

Exact fingerprint matches receive the highest weight because they correspond to explicit catalog-derived requirements. Weight presets are evaluated through scenario-stratified five-fold tuning rather than private-data assumptions.

## Clarification policy

For the top candidate pool, scores are converted into a softmax distribution. For each unasked attribute, products are grouped by the answer their fingerprint implies. The entropy of that outcome distribution is the expected information gain of the question. Coverage and aggregate profile tags make small, transparent adjustments.

The catch-all `other` attribute is penalized and becomes available only after a declined answer, late in the session, or when typed attributes have negligible value. This preserves structured, human-readable questions while providing a graceful fallback.

## Offline behavior

The core agent is standard-library only. BGE and the cross-encoder are loaded exclusively from local index assets and are surrounded by a fail-closed optional boundary. Missing ML dependencies, corrupt assets, or a mismatched catalog disable only that route, never the whole agent.
