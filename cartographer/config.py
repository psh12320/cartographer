from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class SearchWeights:
    """Feature weights used by the deterministic local reranker."""

    exact_fingerprint: float = 7.0
    constraint_coverage: float = 3.0
    category: float = 1.8
    bm25: float = 1.4
    dense: float = 1.1
    profile: float = 0.15
    popularity: float = 0.05
    cross_encoder: float = 0.5


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration with safe, offline-first defaults."""

    catalog_path: Path = Path("data/catalog.jsonl")
    index_dir: Path = Path("data/cartographer_index")
    lexical_limit: int = 300
    dense_limit: int = 300
    category_limit: int = 600
    max_scored_candidates: int = 5000
    clarification_pool: int = 500
    max_retained_sessions: int = 64
    hard_filter_minimum: int = 10
    rrf_k: int = 60
    clarification_other_start_turn: int = 1
    clarification_other_multiplier: float = 1.0
    clarification_other_routes: tuple[str, ...] = ("buying", "browsing")
    buying_popularity_multiplier: float = 3.0
    popularity_rating_mix: float = 1.0
    suppress_textual_profile_after_override: bool = True
    enable_fts: bool = True
    enable_category: bool = True
    enable_fingerprints: bool = True
    enable_dense: bool = True
    enable_cross_encoder: bool = False
    enable_state: bool = True
    enable_clarification: bool = True
    diversify_browsing: bool = True
    bge_query_instruction: str = "Represent this sentence for searching relevant passages: "
    weights: SearchWeights = field(default_factory=SearchWeights)

    def with_catalog(self, path: str | Path) -> "AgentConfig":
        return replace(self, catalog_path=Path(path))

    def with_overrides(self, **values: object) -> "AgentConfig":
        return replace(self, **values)
