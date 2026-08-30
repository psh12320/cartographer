from __future__ import annotations

import json
from pathlib import Path

from .models import SearchHit, SessionState


FEATURE_NAMES = (
    "base_score",
    "exact_matches",
    "constraint_score",
    "category_score",
    "bm25_score",
    "dense_score",
    "dense_rank_score",
    "dense_constraint_agreement",
    "dense_category_agreement",
    "profile_score",
    "popularity_score",
)


def route_key(state: SessionState) -> str:
    if state.intent_epoch > 0:
        return "override"
    if state.declined_attributes:
        return "boundary"
    return state.route if state.route in {"buying", "browsing"} else "default"


def feature_rows(hits: list[SearchHit]) -> list[dict[str, float]]:
    if not hits:
        return []
    return [
        {
            "base_score": hit.score / (1.0 + abs(hit.score)),
            "exact_matches": min(1.0, hit.exact_matches / 2.0),
            "constraint_score": hit.constraint_score,
            "category_score": hit.category_score,
            "bm25_score": hit.bm25_score,
            "dense_score": hit.dense_score,
            "dense_rank_score": hit.dense_rank_score,
            "dense_constraint_agreement": hit.dense_score * hit.constraint_score,
            "dense_category_agreement": hit.dense_score * hit.category_score,
            "profile_score": hit.profile_score,
            "popularity_score": hit.popularity_score,
        }
        for hit in hits
    ]


class LinearReranker:
    """Optional dependency-free residual ranker loaded from a transparent JSON artifact."""

    def __init__(self, path: Path, enabled: bool, scale: float = 1.0) -> None:
        self.enabled = False
        self.failure_reason: str | None = None
        self.scale = float(scale)
        self.routes: dict[str, dict[str, float]] = {}
        if not enabled:
            return
        if not path.exists():
            self.failure_reason = "Learned ranker artifact is absent."
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format_version") != 1:
                raise ValueError("unsupported ranker format")
            if tuple(payload.get("feature_names") or ()) != FEATURE_NAMES:
                raise ValueError("ranker feature schema does not match runtime")
            routes = payload.get("routes")
            if not isinstance(routes, dict) or not routes:
                raise ValueError("ranker has no route weights")
            self.routes = {
                str(route): {name: float(weights.get(name, 0.0)) for name in FEATURE_NAMES}
                for route, weights in routes.items()
                if isinstance(weights, dict)
            }
            if not self.routes:
                raise ValueError("ranker route weights are invalid")
            self.enabled = True
        except Exception as error:
            self.failure_reason = f"Learned ranker unavailable: {error}"

    def rerank(self, hits: list[SearchHit], state: SessionState) -> list[SearchHit]:
        if not self.enabled or not hits:
            return hits
        weights = self.routes.get(route_key(state)) or self.routes.get("default")
        if not weights:
            return hits
        for hit, features in zip(hits, feature_rows(hits)):
            residual = sum(weights[name] * features[name] for name in FEATURE_NAMES)
            hit.learned_score = residual
            hit.score += self.scale * residual
        return sorted(hits, key=lambda hit: (-hit.score, hit.parent_asin))
