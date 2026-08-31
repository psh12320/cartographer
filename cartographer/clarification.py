from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .catalog import CatalogIndex
from .dialog import profile_attributes
from .models import SearchHit, SessionState
from .text import canonical


QUESTION_TEMPLATES = {
    "material": "Which material would you prefer?",
    "color": "Do you have a color preference?",
    "size": "What size or fit requirement should I prioritize?",
    "style": "Which style or fit matters most to you?",
    "brand": "Do you have a preferred brand?",
    "budget": "What budget should I stay within?",
    "feature": "Which product feature would most influence your choice?",
    "use_case": "What will you mainly use the product for?",
    "other": "What other requirement would most change your decision?",
}
TYPED_ATTRIBUTES = ("material", "color", "size", "style", "brand", "budget", "feature", "use_case")


@dataclass
class ClarificationDecision:
    attribute: str | None
    message: str
    information_gain: float
    entropy: float


class ClarificationPolicy:
    """Choose the next question by expected reduction in candidate entropy."""

    def __init__(
        self,
        catalog: CatalogIndex,
        pool_size: int = 500,
        other_start_turn: int = 3,
        other_multiplier: float = 0.55,
        other_routes: tuple[str, ...] = ("buying", "browsing"),
        other_max_asks: int = 1,
    ) -> None:
        self.catalog = catalog
        self.pool_size = pool_size
        self.other_start_turn = other_start_turn
        self.other_multiplier = other_multiplier
        self.other_routes = frozenset(other_routes)
        self.other_max_asks = max(1, int(other_max_asks))

    def choose(self, state: SessionState, hits: list[SearchHit], turn: int) -> ClarificationDecision:
        if not hits or turn >= 10:
            return ClarificationDecision(None, "Here are the strongest matches I found.", 0.0, 0.0)
        pool = hits[: self.pool_size]
        probabilities = self._probabilities(pool)
        base_entropy = self._entropy(probabilities)
        preferred = profile_attributes(state.user_profile)
        active_values = {canonical(item.value) for item in state.active_constraints}

        best_attribute: str | None = None
        best_gain = -1.0
        typed_best = -1.0
        for attribute in TYPED_ATTRIBUTES:
            retired = attribute in state.asked_attributes or attribute in state.declined_attributes
            if retired and attribute not in state.deflected_attributes:
                continue
            gain, coverage = self._attribute_gain(attribute, pool, probabilities, active_values)
            gain *= 0.5 + 0.5 * coverage
            if attribute in preferred:
                gain *= 1.08
            if gain > best_gain:
                best_attribute = attribute
                best_gain = gain
            typed_best = max(typed_best, gain)

        allow_other = (
            state.route in self.other_routes
            and (
                bool(state.declined_attributes)
                or turn >= self.other_start_turn
                or typed_best < 0.20
            )
        )
        if allow_other and state.other_ask_count < self.other_max_asks:
            other_gain, coverage = self._attribute_gain("other", pool, probabilities, active_values)
            other_gain *= (0.5 + 0.5 * coverage) * self.other_multiplier
            if other_gain > best_gain:
                best_attribute = "other"
                best_gain = other_gain

        if best_attribute is None:
            return ClarificationDecision(None, "Here are the strongest matches I found.", 0.0, base_entropy)
        return ClarificationDecision(
            best_attribute,
            QUESTION_TEMPLATES[best_attribute],
            max(0.0, best_gain),
            base_entropy,
        )

    def _attribute_gain(
        self,
        attribute: str,
        hits: list[SearchHit],
        probabilities: list[float],
        active_values: set[str],
    ) -> tuple[float, float]:
        outcome_mass: dict[tuple[str, ...], float] = defaultdict(float)
        covered_mass = 0.0
        for hit, probability in zip(hits, probabilities):
            product = self.catalog.products[hit.product_index]
            if attribute == "other":
                values = tuple(
                    canonical(value)
                    for value in product.constraints
                    if canonical(value) not in active_values
                )[:2]
            else:
                values = tuple(
                    canonical(value)
                    for value in product.by_attribute.get(attribute, ())
                    if canonical(value) not in active_values
                )[:2]
            outcome = values or ("__none__",)
            if values:
                covered_mass += probability
            outcome_mass[outcome] += probability
        gain = self._entropy(list(outcome_mass.values()))
        return gain, covered_mass

    @staticmethod
    def _probabilities(hits: list[SearchHit]) -> list[float]:
        scores = [hit.score for hit in hits]
        highest = max(scores)
        spread = max(scores) - min(scores)
        temperature = max(0.75, spread / 8.0)
        weights = [math.exp(max(-40.0, (score - highest) / temperature)) for score in scores]
        total = sum(weights)
        return [weight / total for weight in weights]

    @staticmethod
    def _entropy(probabilities: list[float]) -> float:
        return -sum(value * math.log2(value) for value in probabilities if value > 0.0)
