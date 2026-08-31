from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


FROZEN_RANKER_PATH = Path(__file__).with_name("ranker_weights.json")


@dataclass(frozen=True)
class SearchWeights:
    """Feature weights used by the deterministic local reranker."""

    exact_fingerprint: float = 7.0
    constraint_coverage: float = 3.0
    category: float = 1.8
    bm25: float = 1.4
    dense: float = 1.1
    dense_rank: float = 0.0
    dense_constraint_agreement: float = 0.0
    dense_category_agreement: float = 0.0
    profile: float = 0.15
    popularity: float = 0.05
    cross_encoder: float = 0.5


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration with safe, offline-first defaults."""

    catalog_path: Path = Path("data/catalog.jsonl")
    index_dir: Path = Path("data/cartographer_index")
    ranker_path: Path | None = FROZEN_RANKER_PATH
    lexical_limit: int = 300
    dense_limit: int = 300
    category_limit: int = 600
    max_scored_candidates: int = 5000
    clarification_pool: int = 500
    max_retained_sessions: int = 64
    hard_filter_minimum: int = 10
    rrf_k: int = 120
    clarification_other_start_turn: int = 1
    clarification_other_multiplier: float = 1.0
    clarification_other_routes: tuple[str, ...] = ("buying", "browsing")
    # The open-ended question draws from every undisclosed requirement rather
    # than one attribute, so it discloses far more per turn than a typed
    # question. Allowing it more than once keeps harvesting the remainder.
    clarification_other_max_asks: int = 2
    # The official customer joins multiple values with "; " only in the
    # "what matters is:" reply. A "key requirement is:" payload is one
    # constraint, so splitting it on ";" shatters constraints that contain one.
    split_hard_requirement_values: bool = False
    # A boundary customer's "please use your judgment" deflection means the
    # requirement still exists, unlike "no additional preference", which means
    # the attribute is exhausted. Retiring it discards the best question.
    boundary_deflection_retires_attribute: bool = True
    # Runtime confidence gate. The normalised rank1-vs-rank2 score margin
    # predicts whether the leader is actually right (40% correct in the lowest
    # quartile, 72% in the highest). Below this threshold the agent declines to
    # widen the list, because converting at rank 2-3 locks in that rank for the
    # rest of the session while deferring a turn costs only 0.02. `0.0`
    # disables the gate and keeps the fixed schedule.
    uncertain_margin_threshold: float = 0.50
    # Over-generality cutoff. When the candidate union is this large the
    # request is too broad to answer with a list, so the agent truncates the
    # recommendation to a probe and spends the turn on a clarification prompt
    # instead. `0` disables the cutoff.
    overgenerality_candidate_threshold: int = 400
    overgenerality_depth: int = 1
    # Long-term personalisation. Preferences observed in earlier sessions are
    # distilled into a durable per-user profile and reloaded on reset, so a
    # returning shopper starts from what they previously cared about.
    enable_profile_memory: bool = False
    profile_memory_path: Path | None = None
    profile_memory_max_tags: int = 8
    buying_popularity_multiplier: float = 3.0
    # Browsing turn one carries almost no constraint information, so the
    # popularity prior is the main discriminative signal available there.
    browsing_popularity_multiplier: float = 1.0
    popularity_rating_mix: float = 1.0
    suppress_textual_profile_after_override: bool = True
    dense_conversation_weight_buying: float = 0.35
    dense_conversation_weight_browsing: float = 0.65
    dense_query_mode: str = "blend"
    dense_buying_multiplier: float = 0.80
    dense_browsing_multiplier: float = 1.20
    dense_calibration_floor_percentile: float = 75.0
    dense_calibration_ceiling_percentile: float = 99.5
    verify_embedding_checksum_on_load: bool = True
    warm_semantic_encoder: bool = True
    learned_reranker_scale: float = 1.0
    learned_reranker_route_scales: tuple[tuple[str, float], ...] = (
        ("boundary", 0.75),
        ("browsing", 0.75),
        ("buying", 1.25),
        ("override", 0.75),
    )
    include_ranker_route_in_cache_key: bool = False
    # Precision-gated recommendation depth: entry i is the depth for the
    # (i+1)-th turn of the current intent epoch; the last entry repeats for
    # later epoch turns. Empty tuple keeps the full top-10 on every turn.
    # From `recommendation_depth_full_turn` (absolute) onward the full list is
    # always returned so Hit Rate cannot be starved by a small depth.
    recommendation_depth_schedule: tuple[int, ...] = (1, 2, 3, 4, 10)
    recommendation_depth_full_turn: int = 4
    # Minimum expected information gain required before the depth gate is
    # allowed to hold products back. `0.0` keeps the gate active whenever a
    # question is asked at all, which is the promoted behavior.
    depth_gate_min_information_gain: float = 0.0
    # Saturation cap for the learned reranker's exact_matches feature.
    exact_match_feature_cap: float = 2.0
    enable_fts: bool = True
    enable_category: bool = True
    enable_fingerprints: bool = True
    # Semantic artifacts are opt-in until they clear the public score and latency gates.
    enable_dense: bool = False
    enable_cross_encoder: bool = False
    enable_learned_reranker: bool = True
    enable_state: bool = True
    enable_clarification: bool = True
    enable_profile: bool = True
    enable_popularity: bool = True
    diversify_browsing: bool = True
    bge_query_instruction: str = "Represent this sentence for searching relevant passages: "
    weights: SearchWeights = field(default_factory=SearchWeights)

    def with_catalog(self, path: str | Path) -> "AgentConfig":
        return replace(self, catalog_path=Path(path))

    def with_overrides(self, **values: object) -> "AgentConfig":
        return replace(self, **values)
