from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .catalog import CatalogIndex
from .config import AgentConfig
from .dialog import profile_attributes
from .models import SearchHit, SessionState
from .ranker import LinearReranker
from .semantic import CrossEncoderReranker, SemanticRetriever
from .text import canonical, terms, token_overlap


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    candidate_count: int
    cache_hit: bool


class HybridRetriever:
    """Fuse fingerprints, category routing, FTS5, and optional local transformer scores."""

    def __init__(self, catalog: CatalogIndex, config: AgentConfig) -> None:
        self.catalog = catalog
        self.config = config
        self.semantic = SemanticRetriever(catalog, config)
        self.cross_encoder = CrossEncoderReranker(config.index_dir, config.enable_cross_encoder)
        self.learned_reranker = LinearReranker(
            config.ranker_path or config.index_dir / "ranker.json",
            config.enable_learned_reranker,
            config.learned_reranker_scale,
        )

    def search(self, state: SessionState) -> RetrievalResult:
        signature = (
            state.route,
            canonical(state.category),
            canonical(state.active_query_text),
            tuple((constraint.attribute, canonical(constraint.value), constraint.strength) for constraint in state.active_constraints),
        )
        if signature == state.last_query_signature and state.cached_hits:
            unseen = [hit for hit in state.cached_hits if hit.parent_asin not in state.seen_products]
            seen = [hit for hit in state.cached_hits if hit.parent_asin in state.seen_products]
            state.cached_hits = unseen + seen
            return RetrievalResult(state.cached_hits, len(state.cached_hits), True)

        structured_query = " ".join(
            part
            for part in [state.category, *(constraint.value for constraint in state.active_constraints)]
            if part
        ).strip()
        semantic_query = state.active_query_text or structured_query
        lexical = (
            self.catalog.lexical_search(structured_query, self.config.lexical_limit)
            if self.config.enable_fts
            else []
        )
        lexical_ranks = {product_index: rank for rank, product_index in enumerate(lexical, start=1)}
        semantic_queries = self._semantic_queries(state, structured_query, semantic_query)
        dense_result = self.semantic.search(
            semantic_queries,
            self.config.dense_limit,
        )
        dense = dense_result.hits
        category_all = (
            self.catalog.category_indices(state.category, None)
            if self.config.enable_category and state.category
            else []
        )
        category = category_all[: self.config.category_limit]

        exact_counts: dict[int, int] = {}
        if self.config.enable_fingerprints:
            for constraint in state.active_constraints:
                matches = set(self.catalog.exact_constraint_indices(constraint.value))
                if matches:
                    for product_index in matches:
                        exact_counts[product_index] = exact_counts.get(product_index, 0) + 1

        candidate_indices = set(lexical)
        candidate_indices.update(product_index for product_index, _ in dense)
        candidate_indices.update(category)
        candidate_indices.update(exact_counts)
        candidate_indices.update(
            self.catalog.asin_to_index[asin]
            for asin in state.override_shortlist
            if asin in self.catalog.asin_to_index
        )
        if not candidate_indices:
            candidate_indices.update(range(min(self.config.category_limit, len(self.catalog.products))))

        # A constraint becomes a destructive filter only when metadata coverage is safe.
        hard_sets = [
            set(self.catalog.exact_constraint_indices(constraint.value))
            for constraint in state.active_constraints
            if constraint.strength == "hard" and self.catalog.exact_constraint_indices(constraint.value)
        ]
        if hard_sets:
            eligible = set(category_all) if category_all else set(candidate_indices)
            for matches in hard_sets:
                narrowed = eligible & matches
                if len(narrowed) >= self.config.hard_filter_minimum:
                    eligible = narrowed
            if len(eligible) >= self.config.hard_filter_minimum:
                # Keep the safe intersection and a small escape hatch for imperfect category parsing.
                direct_escape = {
                    index
                    for index, _ in sorted(
                        exact_counts.items(),
                        key=lambda item: (-item[1], self.catalog.products[item[0]].parent_asin),
                    )[:50]
                }
                candidate_indices = (candidate_indices & eligible) | direct_escape

        if len(candidate_indices) > self.config.max_scored_candidates:
            category_members = set(category_all)
            priority = [
                self.catalog.asin_to_index[asin]
                for asin in sorted(state.override_shortlist)
                if asin in self.catalog.asin_to_index
            ]
            priority.extend(
                index
                for index, _ in sorted(
                    exact_counts.items(),
                    key=lambda item: (
                        item[0] not in category_members,
                        -item[1],
                        self.catalog.products[item[0]].parent_asin,
                    ),
                )
            )
            priority.extend(lexical)
            priority.extend(index for index, _ in dense)
            priority.extend(category)
            limited: set[int] = set()
            for index in priority:
                if index in candidate_indices:
                    limited.add(index)
                if len(limited) >= self.config.max_scored_candidates:
                    break
            candidate_indices = limited

        profile_keys = profile_attributes(state.user_profile)
        profile_text = " ".join(
            [
                *(str(value) for value in state.user_profile.get("preference_tags") or []),
                str(state.user_profile.get("summary") or ""),
            ]
        )
        profile_terms = set(terms(profile_text, 40))
        hits: list[SearchHit] = []
        for product_index in candidate_indices:
            product = self.catalog.products[product_index]
            exact_count = exact_counts.get(product_index, 0)
            category_score = self._category_score(state.category, product.category)
            constraint_score = self._constraint_coverage(state, product.search_key, product.price)
            bm25_score = self._rrf_feature(lexical_ranks.get(product_index))
            dense_score = dense_result.score(product_index)
            dense_rank_score = self._rrf_feature(dense_result.rank(product_index))
            product_attributes = set(product.by_attribute)
            attribute_profile_score = (
                len(profile_keys & product_attributes) / len(profile_keys) if profile_keys else 0.0
            )
            padded_product_text = f" {product.search_key} "
            textual_profile_score = (
                sum(f" {term} " in padded_product_text for term in profile_terms) / len(profile_terms)
                if profile_terms
                else 0.0
            )
            if self.config.suppress_textual_profile_after_override and state.intent_epoch > 0:
                profile_score = attribute_profile_score
            else:
                profile_score = 0.75 * textual_profile_score + 0.25 * attribute_profile_score
            review_volume = min(1.0, math.log1p(product.rating_number) / 10.0)
            rating_quality = product.average_rating / 5.0
            popularity = review_volume * (
                (1.0 - self.config.popularity_rating_mix)
                + self.config.popularity_rating_mix * rating_quality
            )
            weights = self.config.weights
            if state.route == "buying":
                exact_weight = weights.exact_fingerprint * 1.15
                coverage_weight = weights.constraint_coverage * 1.20
                category_weight = weights.category
                bm25_weight = weights.bm25
                dense_weight = weights.dense * self.config.dense_buying_multiplier
                dense_rank_weight = weights.dense_rank * self.config.dense_buying_multiplier
                profile_weight = weights.profile * 0.80
                popularity_weight = weights.popularity * self.config.buying_popularity_multiplier
            else:
                exact_weight = weights.exact_fingerprint
                coverage_weight = weights.constraint_coverage
                category_weight = weights.category * 1.15
                bm25_weight = weights.bm25 * 0.95
                dense_weight = weights.dense * self.config.dense_browsing_multiplier
                dense_rank_weight = weights.dense_rank * self.config.dense_browsing_multiplier
                profile_weight = weights.profile * 1.20
                popularity_weight = weights.popularity
            score = (
                exact_weight * exact_count
                + coverage_weight * constraint_score
                + category_weight * category_score
                + bm25_weight * bm25_score
                + dense_weight * dense_score
                + dense_rank_weight * dense_rank_score
                + weights.dense_constraint_agreement * dense_score * constraint_score
                + weights.dense_category_agreement * dense_score * category_score
                + profile_weight * profile_score
                + popularity_weight * popularity
            )
            if product.parent_asin in state.override_shortlist:
                # Reconsider earlier options under the corrected constraint instead of losing them.
                score += 3.5 * constraint_score
            hits.append(
                SearchHit(
                    product_index=product_index,
                    parent_asin=product.parent_asin,
                    score=score,
                    exact_matches=exact_count,
                    category_score=category_score,
                    constraint_score=constraint_score,
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                    dense_rank_score=dense_rank_score,
                    profile_score=profile_score,
                    popularity_score=popularity,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.parent_asin))
        hits = self.learned_reranker.rerank(hits, state)
        if self.cross_encoder.enabled and semantic_query:
            rerank_count = min(40, len(hits))
            documents = [self.catalog.products[hit.product_index].search_text for hit in hits[:rerank_count]]
            scores = self.cross_encoder.score(semantic_query, documents)
            for hit, cross_score in zip(hits[:rerank_count], scores):
                hit.cross_encoder_score = cross_score
                hit.score += self.config.weights.cross_encoder * cross_score
            hits[:rerank_count] = sorted(hits[:rerank_count], key=lambda hit: (-hit.score, hit.parent_asin))

        # A continuing session is implicit negative feedback: previously returned products were misses.
        unseen = [hit for hit in hits if hit.parent_asin not in state.seen_products]
        seen = [hit for hit in hits if hit.parent_asin in state.seen_products]
        hits = unseen + seen
        cache_limit = max(
            self.config.category_limit,
            self.config.clarification_pool,
            self.config.lexical_limit,
            self.config.dense_limit,
        )
        state.cached_hits = hits[:cache_limit]
        state.last_query_signature = signature
        return RetrievalResult(state.cached_hits, len(candidate_indices), False)

    def _semantic_queries(
        self,
        state: SessionState,
        structured_query: str,
        conversational_query: str,
    ) -> list[tuple[str, float]]:
        mode = self.config.dense_query_mode
        if mode == "structured":
            return [(structured_query, 1.0)]
        if mode == "conversation":
            return [(conversational_query, 1.0)]
        if mode == "compiled":
            requirements = "; ".join(
                constraint.value for constraint in state.active_constraints
            )
            compiled = " ".join(
                part
                for part in (
                    f"Category: {state.category}." if state.category else "",
                    f"Requirements: {requirements}." if requirements else "",
                    f"Request: {conversational_query}." if conversational_query else "",
                )
                if part
            )
            return [(compiled or structured_query or conversational_query, 1.0)]
        if mode != "blend":
            raise ValueError(f"Unsupported dense_query_mode: {mode}")
        conversation_weight = (
            self.config.dense_conversation_weight_buying
            if state.route == "buying"
            else self.config.dense_conversation_weight_browsing
        )
        return [
            (structured_query, 1.0 - conversation_weight),
            (conversational_query, conversation_weight),
        ]

    def _rrf_feature(self, rank: int | None) -> float:
        if rank is None:
            return 0.0
        return (self.config.rrf_k + 1.0) / (self.config.rrf_k + float(rank))

    @staticmethod
    def _category_score(query_category: str, product_category: str) -> float:
        if not query_category:
            return 0.0
        if canonical(query_category) == canonical(product_category):
            return 1.0
        return token_overlap(query_category, product_category)

    @staticmethod
    def _constraint_coverage(state: SessionState, search_key: str, price: float | None) -> float:
        constraints = state.active_constraints
        if not constraints:
            return 0.0
        total = 0.0
        padded_search = f" {search_key} "
        for constraint in constraints:
            key = canonical(constraint.value)
            if key and key in search_key:
                match = 1.0
            else:
                query_terms = [value for value in key.split() if len(value) > 1]
                match = (
                    sum(f" {value} " in padded_search for value in query_terms) / len(query_terms)
                    if query_terms
                    else 0.0
                )
            if constraint.attribute == "budget" and price is not None:
                numeric = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", constraint.value)]
                if numeric:
                    target = numeric[-1]
                    match = max(match, max(0.0, 1.0 - abs(price - target) / max(target, 1.0)))
            total += match * (1.25 if constraint.strength == "hard" else 1.0)
        denominator = sum(1.25 if item.strength == "hard" else 1.0 for item in constraints)
        return total / denominator if denominator else 0.0


def diversify_browsing(hits: list[SearchHit], catalog: CatalogIndex, limit: int) -> list[SearchHit]:
    """Keep the top three precise, then add mild intent-fingerprint diversity."""

    if len(hits) <= 3 or limit <= 3:
        return hits[:limit]
    selected = hits[:3]
    remaining = hits[3: min(len(hits), 80)]
    selected_keys = {
        canonical(value)
        for hit in selected
        for value in catalog.products[hit.product_index].constraints
    }
    while remaining and len(selected) < limit:
        best_position = 0
        best_value = float("-inf")
        for position, hit in enumerate(remaining):
            product_keys = {canonical(value) for value in catalog.products[hit.product_index].constraints}
            overlap = len(product_keys & selected_keys) / max(1, len(product_keys))
            value = hit.score - 0.08 * overlap
            if value > best_value:
                best_value = value
                best_position = position
        chosen = remaining.pop(best_position)
        selected.append(chosen)
        selected_keys.update(canonical(value) for value in catalog.products[chosen.product_index].constraints)
    if len(selected) < limit:
        selected.extend(hit for hit in hits if hit not in selected and len(selected) < limit)
    return selected[:limit]
