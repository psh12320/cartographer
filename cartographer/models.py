from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Constraint:
    attribute: str
    value: str
    strength: str
    source_turn: int
    epoch: int
    active: bool = True


@dataclass
class IntentMessage:
    text: str
    source_turn: int
    epoch: int


@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    route: str = "browsing"
    category: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    replaceable_constraint: Constraint | None = None
    intent_messages: list[IntentMessage] = field(default_factory=list)
    intent_epoch: int = 0
    asked_attributes: set[str] = field(default_factory=set)
    declined_attributes: set[str] = field(default_factory=set)
    last_asked: str | None = None
    seen_products: set[str] = field(default_factory=set)
    override_shortlist: set[str] = field(default_factory=set)
    last_query_signature: tuple = field(default_factory=tuple)
    cached_hits: list["SearchHit"] = field(default_factory=list)

    @property
    def active_constraints(self) -> list[Constraint]:
        return [constraint for constraint in self.constraints if constraint.active]

    @property
    def active_query_text(self) -> str:
        return " ".join(
            item.text for item in self.intent_messages if item.epoch == self.intent_epoch
        ).strip()


@dataclass
class ProductRecord:
    index: int
    parent_asin: str
    title: str
    category: str
    category_key: str
    search_text: str
    search_key: str
    price: float | None
    average_rating: float
    rating_number: int
    constraints: tuple[str, ...]
    by_attribute: dict[str, tuple[str, ...]]


@dataclass
class SearchHit:
    product_index: int
    parent_asin: str
    score: float
    exact_matches: int = 0
    category_score: float = 0.0
    constraint_score: float = 0.0
    bm25_score: float = 0.0
    dense_score: float = 0.0
    dense_rank_score: float = 0.0
    profile_score: float = 0.0
    popularity_score: float = 0.0
    cross_encoder_score: float = 0.0
    learned_score: float = 0.0


@dataclass
class TraceEvent:
    turn: int
    route: str
    intent_epoch: int
    category: str
    constraints: list[dict]
    candidate_count: int
    entropy: float
    ask_attribute: str | None
    information_gain: float
    recommendations: list[str]
    latency_ms: float
    cache_hit: bool = False
    dense_enabled: bool = False
    cross_encoder_enabled: bool = False
    learned_reranker_enabled: bool = False
    recommendation_depth: int = 10
    depth_gate_active: bool = False
    depth_gate_reason: str = "disabled"
