from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from .catalog import CatalogIndex
from .clarification import ClarificationDecision, ClarificationPolicy
from .config import AgentConfig
from .dialog import DialogManager
from .models import SessionState, TraceEvent
from .retrieval import HybridRetriever, diversify_browsing


class CartographerEngine:
    """Offline conversational engine implementing the official Agent behavior."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        config: AgentConfig | None = None,
        catalog_index: CatalogIndex | None = None,
    ) -> None:
        self.config = (config or AgentConfig()).with_catalog(catalog_path)
        self.catalog = catalog_index or CatalogIndex(self.config.catalog_path, self.config.index_dir)
        self.dialog = DialogManager()
        self.retriever = HybridRetriever(self.catalog, self.config)
        self.clarification = ClarificationPolicy(
            self.catalog,
            self.config.clarification_pool,
            other_start_turn=self.config.clarification_other_start_turn,
            other_multiplier=self.config.clarification_other_multiplier,
            other_routes=self.config.clarification_other_routes,
        )
        self.sessions: dict[str, SessionState] = {}
        self.traces: dict[str, list[TraceEvent]] = {}
        self._session_order: list[str] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if session_id in self.sessions:
            self._session_order.remove(session_id)
        self.sessions[session_id] = SessionState(session_id=session_id, user_profile=dict(user_profile or {}))
        self.traces[session_id] = []
        self._session_order.append(session_id)
        while len(self._session_order) > self.config.max_retained_sessions:
            expired = self._session_order.pop(0)
            self.sessions.pop(expired, None)
            self.traces.pop(expired, None)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        started = time.perf_counter()
        if session_id not in self.sessions:
            raise RuntimeError("reset must be called before respond")
        if not isinstance(turn, int) or not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        state = self.sessions[session_id]
        self.dialog.update(state, str(user_message or ""), turn, preserve_state=self.config.enable_state)
        retrieval = self.retriever.search(state)
        if self.config.enable_clarification:
            decision = self.clarification.choose(state, retrieval.hits, turn)
        else:
            decision = ClarificationDecision(None, "Here are the strongest matches I found.", 0.0, 0.0)

        output_limit = min(10, top_k)
        ranked_hits = retrieval.hits
        if state.route == "browsing" and self.config.diversify_browsing:
            ranked_hits = diversify_browsing(ranked_hits, self.catalog, output_limit)
        else:
            ranked_hits = ranked_hits[:output_limit]
        recommendations = [{"parent_asin": hit.parent_asin} for hit in ranked_hits[:output_limit]]

        if decision.attribute is not None:
            state.asked_attributes.add(decision.attribute)
            state.last_asked = decision.attribute
        else:
            state.last_asked = None
        for recommendation in recommendations:
            state.seen_products.add(recommendation["parent_asin"])

        latency_ms = (time.perf_counter() - started) * 1000.0
        trace = TraceEvent(
            turn=turn,
            route=state.route,
            intent_epoch=state.intent_epoch,
            category=state.category,
            constraints=[asdict(constraint) for constraint in state.active_constraints],
            candidate_count=retrieval.candidate_count,
            entropy=round(decision.entropy, 6),
            ask_attribute=decision.attribute,
            information_gain=round(decision.information_gain, 6),
            recommendations=[item["parent_asin"] for item in recommendations],
            latency_ms=round(latency_ms, 3),
            cache_hit=retrieval.cache_hit,
            dense_enabled=self.retriever.semantic.enabled,
            cross_encoder_enabled=self.retriever.cross_encoder.enabled,
        )
        self.traces[session_id].append(trace)
        return {
            "message": decision.message,
            "ask_attribute": decision.attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def get_trace(self, session_id: str) -> list[dict]:
        return [asdict(event) for event in self.traces.get(session_id, [])]
