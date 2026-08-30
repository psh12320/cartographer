# Devpost Project Description: Cartographer

## Inspiration

Most commerce search systems force shoppers to translate an evolving need into rigid keywords and filters. This is especially painful when someone is still exploring or changes their mind halfway through a conversation. We asked a different question: instead of making the shopper do all the search work, can the search engine choose the most useful next question?

## What it does

Cartographer is a privacy-preserving conversational shopping copilot that treats dialogue as active search. It detects whether the customer is buying with firm constraints or browsing with an open-ended goal, constructs a compact session state, retrieves and ranks products through multiple complementary routes, and asks the clarification question expected to shrink the remaining candidate space the most.

Every turn produces both a ranked Top 10 and one structured follow-up question. When the customer changes direction, Cartographer creates a new intent epoch and removes superseded preferences. When the customer has no preference, it marks that attribute exhausted and moves to the next most informative question.

## How we built it

The catalog compiler uses participant-visible product titles, categories, features, descriptions, details, stores, prices, and rating metadata to create structured intent fingerprints. Retrieval combines persistent SQLite FTS5/BM25 lexical search, exact and fuzzy fingerprint lookup, category-aware retrieval, optional local BGE embeddings, and an optional MiniLM cross-encoder feature.

Candidates are fused and reranked with route-specific features. The clarification policy estimates a probability distribution over the current candidates, partitions it by the answer each attribute could produce, and chooses the attribute with the largest expected entropy reduction. Aggregate preference tags influence question priority without inventing values that the user never supplied.

The entire official runtime is local and CPU-only. It requires no external API, account, secret, paid credits, or hosted vector database.

## What makes it different

1. **Active clarification:** questions are selected by measurable information value rather than a static questionnaire.
2. **Dynamic context compilation:** only active constraints enter the next retrieval context; overridden preferences are deactivated explicitly.
3. **Fingerprint retrieval:** product intent representations connect natural customer constraints to catalog metadata without modifying the catalog.
4. **Implicit negative feedback:** a continued session proves that previous recommendations missed, so the agent expands coverage instead of repeating them.
5. **Honest personalization:** profile tags guide what to ask, not unsupported assumptions about what to recommend.

## Tools, libraries, APIs, and data

- Development: Python 3.10+, VS Code/Codex-compatible workflow, Git, and `unittest`.
- Core libraries: Python standard library and SQLite FTS5.
- Optional local ML: NumPy, Sentence Transformers, `BAAI/bge-small-en-v1.5`, and `cross-encoder/ms-marco-MiniLM-L6-v2`.
- External APIs used during inference: none.
- Dataset: the organizer's frozen 50,000-product catalog and 200 public sessions derived from Amazon Reviews 2023 Clothing, Shoes and Jewelry.

## Challenges

The largest challenge was balancing precision and exploration. Hard filtering improves exact buying queries but can silently remove the correct product when metadata is incomplete. Cartographer applies destructive filters only when they retain a safe candidate set; otherwise the constraint remains a strong ranking feature. Another challenge was intent correction: simply appending new text leaves stale preferences in the query, so we introduced explicit intent epochs and active/inactive constraint state.

## Impact

The architecture generalizes beyond retail. Any domain with a finite catalog and evolving user intent—jobs, travel, real estate, support knowledge bases, or enterprise procurement—can use the same combination of hybrid retrieval, active questions, and replaceable context. Local inference also makes the approach practical where privacy, predictable cost, and offline reliability matter.

## Accomplishments

- Preserved the official strict Agent response contract.
- Implemented multi-route retrieval and deterministic offline fallback.
- Added scenario-aware state transitions for Buying, Browsing, Intent Override, and Boundary behavior.
- Added traceable information-gain calculations and per-turn observability.
- Added contract, integrity, state, ranking, determinism, and scenario tests.
- Added reproducible index building, demo, ablation, and five-fold tuning commands.

## Limitations and next steps

Cartographer's current extractors target English clothing metadata. A production version would use configurable domain ontologies, typo-tolerant matching, multilingual local encoders, consented persistent profiles, and online A/B testing. The optional transformer reranker is deliberately gated on measured score and latency improvements rather than enabled for architectural appearance alone.

