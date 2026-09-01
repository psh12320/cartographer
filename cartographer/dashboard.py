from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    evaluate,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent

from .catalog import CatalogIndex
from .config import AgentConfig
from .ranker import route_key


DEFAULT_EXTRA_DATASETS = ("synthetic_800_v1.jsonl",)

COMPONENT_SPECS: dict[str, dict[str, str]] = {
    "fts": {
        "label": "Lexical FTS5 / BM25",
        "evidence": "Candidate recall and exact wording match",
    },
    "category": {
        "label": "Category retrieval route",
        "evidence": "Broad catalog coverage when wording is sparse",
    },
    "fingerprints": {
        "label": "Intent fingerprints + safe hard filtering",
        "evidence": "Exact structured requirement matching",
    },
    "state": {
        "label": "Multi-turn intent state",
        "evidence": "Accumulated constraints, boundaries, and overrides",
    },
    "clarification": {
        "label": "Entropy-guided clarification",
        "evidence": "Value of asking the next question",
    },
    "profile": {
        "label": "Safe profile personalization",
        "evidence": "Aggregate-profile ranking and question priority",
    },
    "popularity": {
        "label": "Rating / review popularity prior",
        "evidence": "Catalog-quality tie breaking",
    },
    "reranker": {
        "label": "Frozen learned reranker",
        "evidence": "Target ordering after deterministic retrieval",
    },
    "gate": {
        "label": "Precision recommendation-depth gate",
        "evidence": "Trade one turn for a stronger conversion rank",
    },
    "diversity": {
        "label": "Browsing diversification",
        "evidence": "Coverage across near-duplicate product families",
    },
    "dense": {
        "label": "Optional BGE dense route",
        "evidence": "Semantic candidate recall when verified assets load",
    },
}

COMPONENT_CHOICES = [
    (spec["label"], key)
    for key, spec in COMPONENT_SPECS.items()
]
DEFAULT_SESSION_ABLATIONS = ["reranker", "gate"]

SESSION_HEADERS = [
    "Session",
    "Scenario",
    "Difficulty",
    "Expected #1 ASIN",
    "Expected product",
    "Category",
    "Price",
    "Rating",
    "Reviews",
    "Profile tags",
]

TURN_HEADERS = [
    "Turn",
    "Evaluator / customer says",
    "Agent says",
    "ask_attribute",
    "Route",
    "Epoch",
    "Active constraints",
    "Compiled category",
    "Active query text",
    "Candidates",
    "Target candidate position",
    "Target recommendation rank",
    "Entropy",
    "Information gain",
    "Returned depth",
    "Gate active",
    "Gate reason",
    "Latency ms",
    "Cache hit",
]

PRODUCT_HEADERS = [
    "Turn",
    "Rank",
    "Expected #1?",
    "ASIN",
    "Product",
    "Category",
    "Price",
    "Rating",
    "Reviews",
    "Final score",
    "Pre-learned score",
    "Exact matches",
    "Constraint coverage",
    "Category agreement",
    "BM25 feature",
    "Dense similarity",
    "Dense rank feature",
    "Profile alignment",
    "Popularity",
    "Learned residual",
    "Cross-encoder",
    "Dominant score contributions",
]

COMPARISON_HEADERS = [
    "Variant",
    "Removed component",
    "Scope",
    "Sessions",
    "Hit Rate@10",
    "MRR",
    "MTTC",
    "Efficiency",
    "TechnicalScore",
    "TechnicalScore delta vs full",
    "MRR delta vs full",
    "MTTC delta vs full",
    "Mean latency ms",
    "p95 latency ms",
    "Dense loaded",
    "Reranker loaded",
    "Depth gate configured",
]

SESSION_COMPARISON_HEADERS = [
    "Removed component",
    "Session",
    "Scenario",
    "Full-agent turn",
    "Full-agent rank",
    "Ablated turn",
    "Ablated rank",
    "Full minus ablated contribution",
    "Turn advantage (ablated - full)",
    "RR advantage (full - ablated)",
]

SESSION_OUTCOME_HEADERS = [
    "Metric",
    "Full agent",
    "Selected ablation",
    "Full-agent advantage",
]

COMPONENT_STATUS_HEADERS = [
    "Component",
    "Full agent",
    "Ablated run",
    "Evidence exposed",
]

LIVE_SCENARIO_HEADERS = [
    "Scenario",
    "Sessions",
    "Hit Rate@10",
    "MRR",
    "MTTC",
    "Efficiency",
    "TechnicalScore",
]

LIVE_SESSION_HEADERS = [
    "Completed",
    "Session",
    "Scenario",
    "Hit",
    "First hit turn",
    "Best rank",
    "Reciprocal rank",
    "Running Hit Rate",
    "Running MRR",
    "Running MTTC",
    "Running TechnicalScore",
]

DASHBOARD_CSS = """
.dashboard-shell {max-width: 1680px !important; margin: 0 auto;}
.warning-box {border-left: 4px solid #f59e0b; padding-left: 14px;}
.diagnostic-table {font-size: 12px;}
"""


def _round(value: Any, digits: int = 6) -> Any:
    return round(float(value), digits) if isinstance(value, (int, float)) else value


def _price(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def _technical_score(summary: dict[str, Any]) -> tuple[float, float]:
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    score = (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )
    return efficiency, score


def _session_contribution(session: dict[str, Any]) -> float:
    """The exact per-session contribution implied by the official aggregate formula."""

    turn = int(session.get("first_hit_turn") or 11)
    return (
        0.50 * float(bool(session.get("hit")))
        + 0.30 * float(session.get("reciprocal_rank") or 0.0)
        + 0.02 * (11.0 - float(turn))
    )


def _component_label(key: str) -> str:
    return COMPONENT_SPECS.get(key, {"label": key}).get("label", key)


def _score_contributions(hit: Any, state: Any, config: AgentConfig) -> str:
    """Explain the largest additive ranking signals using the runtime's exact weights."""

    weights = config.weights
    buying = state.route == "buying"
    exact_weight = weights.exact_fingerprint * (1.15 if buying else 1.0)
    coverage_weight = weights.constraint_coverage * (1.20 if buying else 1.0)
    category_weight = weights.category * (1.0 if buying else 1.15)
    bm25_weight = weights.bm25 * (1.0 if buying else 0.95)
    dense_multiplier = config.dense_buying_multiplier if buying else config.dense_browsing_multiplier
    profile_weight = weights.profile * (0.80 if buying else 1.20)
    popularity_weight = (
        weights.popularity * config.buying_popularity_multiplier
        if buying
        else weights.popularity
    )
    learned_scale = dict(config.learned_reranker_route_scales).get(
        route_key(state),
        config.learned_reranker_scale,
    )
    contributions = {
        "exact fingerprints": exact_weight * hit.exact_matches,
        "constraint coverage": coverage_weight * hit.constraint_score,
        "category": category_weight * hit.category_score,
        "BM25": bm25_weight * hit.bm25_score,
        "dense similarity": weights.dense * dense_multiplier * hit.dense_score,
        "dense rank": weights.dense_rank * dense_multiplier * hit.dense_rank_score,
        "dense×constraint": weights.dense_constraint_agreement
        * hit.dense_score
        * hit.constraint_score,
        "dense×category": weights.dense_category_agreement
        * hit.dense_score
        * hit.category_score,
        "profile": profile_weight * hit.profile_score,
        "popularity": popularity_weight * hit.popularity_score,
        "learned residual": learned_scale * hit.learned_score,
        "cross-encoder": weights.cross_encoder * hit.cross_encoder_score,
        "override recovery": 3.5 * hit.constraint_score
        if hit.parent_asin in state.override_shortlist
        else 0.0,
    }
    ranked = sorted(
        ((name, value) for name, value in contributions.items() if abs(value) >= 1e-6),
        key=lambda item: (-abs(item[1]), item[0]),
    )[:4]
    return "; ".join(f"{name} {value:+.3f}" for name, value in ranked) or "deterministic fallback"


def decision_signals(turns: list[dict[str, Any]], hit_turn: int | None) -> list[str]:
    """Translate trace evidence into concrete architecture hypotheses."""

    if not turns:
        return ["No replay evidence was produced."]
    positions = [
        int(turn["target_candidate_position"])
        for turn in turns
        if turn.get("target_candidate_position") is not None
    ]
    signals: list[str] = []
    if not positions:
        signals.append(
            "Retrieval gap: the target never entered the cached candidate pool. "
            "Dense retrieval or better query rewriting is more relevant than a reranker."
        )
    elif min(positions) > 10:
        signals.append(
            f"Ranking gap: the target was retrieved but its best candidate position was {min(positions)}. "
            "A reranker can address this without changing dialogue generation."
        )
    else:
        signals.append(
            f"Retrieval is adequate: the target reached candidate position {min(positions)}."
        )
    if hit_turn is None:
        signals.append(
            "The session missed by turn 10. Inspect parsed constraints and target feature scores before adding an LLM."
        )
    elif hit_turn == 1:
        signals.append(
            "The target converted on turn 1; an LLM is unlikely to improve MTTC for this session."
        )
    else:
        asked = [str(turn.get("ask_attribute")) for turn in turns[: hit_turn - 1]]
        signals.append(
            f"Clarification contributed before the turn-{hit_turn} hit. Asked attributes: {', '.join(asked)}."
        )
    missing_category = any(not str(turn.get("category") or "").strip() for turn in turns)
    if missing_category:
        signals.append(
            "Parsing warning: at least one turn has no compiled category. An LLM or compact intent parser may help."
        )
    else:
        signals.append(
            "The deterministic parser retained a category on every turn; this session alone does not justify an LLM."
        )
    return signals


class DashboardBackend:
    """Development-only replay and comparison layer over the untouched evaluator."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        dataset_path: str | Path = "data/public_set.jsonl",
        index_dir: str | Path = "data/cartographer_index",
        extra_datasets: Sequence[str | Path] = (),
        ranker_path: str | Path | None = None,
        profile_memory_path: str | Path | None = None,
    ) -> None:
        self.catalog_path = str(catalog_path)
        # Long-term personalisation is opt-in. When a store is supplied the
        # replay runs with it enabled so the distilled profile can be shown.
        self.profile_memory_path = Path(profile_memory_path) if profile_memory_path else None
        # An alternate artifact can be shown without replacing the promoted one,
        # which stays bound to the locked development partition.
        self.ranker_path = Path(ranker_path) if ranker_path else None
        self.index_dir = Path(index_dir)
        self.identifiers, self.categories, self.raw_products = catalog_index(self.catalog_path)
        self.catalog = CatalogIndex(self.catalog_path, index_dir)
        self.scopes = self._build_scopes(dataset_path, extra_datasets)
        self.scope_names = list(self.scopes)
        self.select_scope(self.scope_names[0])

    def _build_scopes(
        self,
        primary: str | Path,
        extras: Sequence[str | Path],
    ) -> dict[str, dict[str, Any]]:
        """One scope per readable dataset, plus a merged scope when several exist."""

        ordered: list[str] = []
        for candidate in [primary, *extras]:
            text = str(candidate)
            if text not in ordered and Path(text).exists():
                ordered.append(text)
        if not ordered:
            raise FileNotFoundError(
                f"No readable dataset among {[str(primary), *[str(item) for item in extras]]}"
            )
        scopes: dict[str, dict[str, Any]] = {}
        for text in ordered:
            samples = load_jsonl(text)
            scopes[f"{Path(text).stem} \u00b7 {len(samples)} sessions"] = {
                "paths": [text],
                "samples": samples,
            }
        if len(ordered) > 1:
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for text in ordered:
                for sample in load_jsonl(text):
                    identifier = str(sample["sample_id"])
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    merged.append(sample)
            scopes[f"All datasets \u00b7 {len(merged)} sessions"] = {
                "paths": list(ordered),
                "samples": merged,
            }
        return scopes

    def select_scope(self, name: str | None) -> str:
        """Point every panel at one dataset scope; returns the resolved scope name."""

        resolved = name if name in self.scopes else self.scope_names[0]
        scope = self.scopes[resolved]
        self.scope_name = resolved
        self.samples = scope["samples"]
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in self.samples}
        self.dataset_path = self._materialize(scope)
        return resolved

    def _materialize(self, scope: dict[str, Any]) -> str:
        """A merged scope needs a single file on disk for the fresh-process evaluator."""

        paths = list(scope["paths"])
        if len(paths) == 1:
            return paths[0]
        target = self.index_dir / "dashboard_datasets" / "combined.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for sample in scope["samples"]:
                handle.write(json.dumps(sample) + "\n")
        return str(target)

    def profile_memory_view(self, sample_id: str) -> str:
        """Show what the agent remembers about this shopper from earlier visits."""

        if not self.profile_memory_path:
            return (
                "**Long-term personalization is off.** Start the dashboard with "
                "`--profile-memory <path>` to distil each replayed session into a "
                "durable shopper profile and reload it on the next visit."
            )
        from .profile_memory import ProfileMemory, user_key

        sample = self.samples_by_id.get(str(sample_id))
        if sample is None:
            return "**Long-term personalization is on.** Select a session to see its shopper."
        memory = ProfileMemory(self.profile_memory_path)
        record = memory.records.get(user_key(sample.get("user_profile") or {}))
        if not record:
            return (
                "**Long-term personalization is on.** No prior visits recorded for this "
                "shopper yet — replay a session to distil one."
            )
        attributes = sorted(
            dict(record.get("attribute_counts") or {}).items(), key=lambda item: -item[1]
        )
        categories = sorted(
            dict(record.get("categories") or {}).items(), key=lambda item: -item[1]
        )[:3]
        lines = [
            f"**Long-term personalization is on.** Distilled from "
            f"**{record.get('sessions', 0)} earlier session(s)** by this shopper.",
            "",
            "| Remembered signal | Evidence |",
            "|---|---|",
        ]
        for name, count in attributes[:5]:
            lines.append(f"| specifies `{name}` | seen {count}× |")
        for name, count in categories:
            lines.append(f"| explores *{name}* | seen {count}× |")
        lines.append("")
        lines.append(
            "These are folded into the profile on the next `Agent.reset`, so a returning "
            "shopper starts from what they previously cared about. Only attribute names, "
            "coarse categories and counts are stored — never products, labels or raw text."
        )
        return "\n".join(lines)

    def ranker_banner(self) -> str:
        """State which reranker artifact this dashboard session is showing."""

        import json as _json

        path = self.ranker_path or AgentConfig().ranker_path
        try:
            training = _json.loads(Path(path).read_text(encoding="utf-8"))["training"]
            split = training.get("data_split", {})
            sessions = split.get("partition_sample_count", training.get("sample_count", "?"))
            partition = split.get("partition", "?")
            oof = training.get("cross_validation", {}).get("out_of_fold_technical_score")
            score = f", grouped out-of-fold TechnicalScore `{oof:.6f}`" if oof else ""
            return (
                f"**Reranker artifact:** `{Path(path).name}` — trained on the **{partition}** "
                f"partition ({sessions} sessions){score}."
            )
        except Exception:
            return f"**Reranker artifact:** `{path}`"

    def choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for sample in self.samples:
            sample_id = str(sample["sample_id"])
            target = str(sample["ground_truth"]["parent_asin"])
            product = self.catalog.products[self.catalog.asin_to_index[target]]
            label = f"{sample_id} · {sample['scenario_type']} · {product.title[:70]}"
            choices.append((label, sample_id))
        return choices

    def session_rows(self) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for sample in self.samples:
            target = str(sample["ground_truth"]["parent_asin"])
            product = self.catalog.products[self.catalog.asin_to_index[target]]
            rows.append(
                [
                    sample["sample_id"],
                    sample["scenario_type"],
                    sample.get("difficulty_bucket", ""),
                    target,
                    product.title,
                    product.category,
                    _price(product.price),
                    product.average_rating,
                    product.rating_number,
                    ", ".join(sample.get("user_profile", {}).get("preference_tags") or []),
                ]
            )
        return rows

    def _config(
        self,
        enable_learned_reranker: bool,
        enable_dense: bool,
        enable_clarification: bool,
        diversify_browsing: bool,
    ) -> AgentConfig:
        disabled: list[str] = []
        if not enable_learned_reranker:
            disabled.append("reranker")
        if not enable_clarification:
            disabled.append("clarification")
        if not diversify_browsing:
            disabled.append("diversity")
        return self._component_config(disabled, enable_dense=enable_dense)

    def _component_config(
        self,
        disabled_components: Sequence[str] = (),
        enable_dense: bool = False,
    ) -> AgentConfig:
        """Create the full agent or one controlled ablation from the same defaults."""

        disabled = {str(value) for value in disabled_components}
        return AgentConfig(
            catalog_path=Path(self.catalog_path),
            index_dir=self.catalog.index_dir,
            enable_fts="fts" not in disabled,
            enable_category="category" not in disabled,
            enable_fingerprints="fingerprints" not in disabled,
            enable_state="state" not in disabled,
            enable_clarification="clarification" not in disabled,
            enable_profile="profile" not in disabled,
            enable_popularity="popularity" not in disabled,
            enable_learned_reranker="reranker" not in disabled,
            enable_dense=bool(enable_dense) and "dense" not in disabled,
            diversify_browsing="diversity" not in disabled,
            recommendation_depth_schedule=(
                () if "gate" in disabled else AgentConfig().recommendation_depth_schedule
            ),
            enable_cross_encoder=False,
            **({"ranker_path": self.ranker_path} if self.ranker_path else {}),
            **(
                {"enable_profile_memory": True, "profile_memory_path": self.profile_memory_path}
                if self.profile_memory_path
                else {}
            ),
        )

    def replay(
        self,
        sample_id: str,
        enable_learned_reranker: bool = False,
        enable_dense: bool = False,
        enable_clarification: bool = True,
        diversify_browsing: bool = True,
    ) -> tuple[str, list[dict[str, str]], list[list[Any]], list[list[Any]], dict, dict, dict, str]:
        """Backward-compatible single replay used by tests and external callers."""

        config = self._config(
            enable_learned_reranker,
            enable_dense,
            enable_clarification,
            diversify_browsing,
        )
        return self._replay_with_config(sample_id, config, "Configured replay")

    def _replay_with_config(
        self,
        sample_id: str,
        config: AgentConfig,
        variant_label: str,
    ) -> tuple[str, list[dict[str, str]], list[list[Any]], list[list[Any]], dict, dict, dict, str]:
        if sample_id not in self.samples_by_id:
            raise ValueError(f"Unknown sample id: {sample_id}")
        sample = self.samples_by_id[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        target_record = self.catalog.products[self.catalog.asin_to_index[target]]
        intent_card, behavior = materialize_hidden_fields(sample, self.raw_products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        agent = Agent(self.catalog_path, config=config, catalog_index=self.catalog)
        session_id = f"dashboard_{sample_id}_{uuid.uuid4().hex}"
        agent.reset(session_id, sample.get("user_profile") or {})
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(self.categories.get(target, [])),
            disclosed,
        )
        chat: list[dict[str, str]] = []
        turn_rows: list[list[Any]] = []
        product_rows: list[list[Any]] = []
        inference_turns: list[dict[str, Any]] = []
        first_hit_turn: int | None = None
        first_hit_rank: int | None = None

        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            trace = agent.get_trace(session_id)[-1]
            state = agent.engine.sessions[session_id]
            ranked = [str(item["parent_asin"]) for item in response["recommendations"]]
            target_rank = ranked.index(target) + 1 if target in ranked else None
            hit_by_asin = {hit.parent_asin: hit for hit in state.cached_hits}
            target_candidate_position = next(
                (
                    position
                    for position, hit in enumerate(state.cached_hits, start=1)
                    if hit.parent_asin == target
                ),
                None,
            )
            constraints = [
                {
                    "attribute": constraint.attribute,
                    "value": constraint.value,
                    "strength": constraint.strength,
                    "turn": constraint.source_turn,
                    "epoch": constraint.epoch,
                }
                for constraint in state.active_constraints
            ]
            chat.append({"role": "user", "content": user_message})
            recommendation_text = ", ".join(
                f"#{rank} {asin}{' ← expected target' if asin == target else ''}"
                for rank, asin in enumerate(ranked, start=1)
            )
            chat.append(
                {
                    "role": "assistant",
                    "content": (
                        f"{response['message']}\n\n"
                        f"`ask_attribute`: `{response.get('ask_attribute')}`\n\n"
                        f"{recommendation_text}"
                    ),
                }
            )
            turn_rows.append(
                [
                    turn,
                    user_message,
                    response["message"],
                    response.get("ask_attribute"),
                    trace["route"],
                    trace["intent_epoch"],
                    "; ".join(f"{item['attribute']}: {item['value']} ({item['strength']})" for item in constraints),
                    state.category,
                    state.active_query_text,
                    trace["candidate_count"],
                    target_candidate_position,
                    target_rank,
                    trace["entropy"],
                    trace["information_gain"],
                    trace["recommendation_depth"],
                    trace["depth_gate_active"],
                    trace["depth_gate_reason"],
                    trace["latency_ms"],
                    trace["cache_hit"],
                ]
            )
            recommendation_diagnostics: list[dict[str, Any]] = []
            for rank, asin in enumerate(ranked, start=1):
                product = self.catalog.products[self.catalog.asin_to_index[asin]]
                hit = hit_by_asin.get(asin)
                if hit is None:
                    continue
                learned_scale = dict(config.learned_reranker_route_scales).get(
                    route_key(state),
                    config.learned_reranker_scale,
                )
                pre_learned = (
                    hit.score
                    - learned_scale * hit.learned_score
                    - config.weights.cross_encoder * hit.cross_encoder_score
                )
                row = [
                    turn,
                    rank,
                    "YES" if asin == target else "",
                    asin,
                    product.title,
                    product.category,
                    _price(product.price),
                    product.average_rating,
                    product.rating_number,
                    _round(hit.score),
                    _round(pre_learned),
                    hit.exact_matches,
                    _round(hit.constraint_score),
                    _round(hit.category_score),
                    _round(hit.bm25_score),
                    _round(hit.dense_score),
                    _round(hit.dense_rank_score),
                    _round(hit.profile_score),
                    _round(hit.popularity_score),
                    _round(hit.learned_score),
                    _round(hit.cross_encoder_score),
                    _score_contributions(hit, state, config),
                ]
                product_rows.append(row)
                recommendation_diagnostics.append(dict(zip(PRODUCT_HEADERS[1:], row[1:])))
            inference_turns.append(
                {
                    **trace,
                    "customer_message": user_message,
                    "agent_response": response,
                    "category": state.category,
                    "active_query_text": state.active_query_text,
                    "active_constraints": constraints,
                    "asked_attributes": sorted(state.asked_attributes),
                    "declined_attributes": sorted(state.declined_attributes),
                    "target_candidate_position": target_candidate_position,
                    "target_recommendation_rank": target_rank,
                    "target_is_eligible": override_applied,
                    "recommended_product_features": recommendation_diagnostics,
                }
            )
            if override_applied and target_rank is not None:
                first_hit_turn = turn
                first_hit_rank = target_rank
                break
            if turn == MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(
                    override.get("message", "Actually, please ignore my earlier preference.")
                )
            else:
                user_message, boundary_used = customer_reply(
                    effective,
                    response.get("ask_attribute"),
                    disclosed,
                    boundary_used,
                )

        signals = decision_signals(inference_turns, first_hit_turn)
        status = "HIT" if first_hit_turn is not None else "MISS"
        summary = (
            f"## {variant_label} · {status}: {sample_id}\n\n"
            f"| Expected #1 product | Outcome | Scenario | Turns used | Dense | Learned reranker | Depth gate |\n"
            f"|---|---:|---|---:|---:|---:|---:|\n"
            f"| **{target_record.title}** (`{target}`) | "
            f"{'rank ' + str(first_hit_rank) + ' on turn ' + str(first_hit_turn) if first_hit_turn else 'not found'} | "
            f"{sample['scenario_type']} | {len(inference_turns)} | "
            f"{agent.engine.retriever.semantic.enabled} | "
            f"{agent.engine.retriever.learned_reranker.enabled} | "
            f"{bool(config.recommendation_depth_schedule)} |\n\n"
            "> This is a development replay. The expected product and hidden intent are shown for diagnosis; "
            "they are never sent to `Agent.respond`."
        )
        target_payload = {
            "expected_rank": 1,
            "parent_asin": target,
            "title": target_record.title,
            "category": target_record.category,
            "price": target_record.price,
            "average_rating": target_record.average_rating,
            "rating_number": target_record.rating_number,
            "catalog_fingerprint": {
                "constraints": target_record.constraints,
                "by_attribute": target_record.by_attribute,
            },
            "evaluator_intent_card": intent_card,
            "evaluator_behavior": behavior,
        }
        inference_payload = {
            "sample_id": sample_id,
            "scenario_type": sample["scenario_type"],
            "difficulty_bucket": sample.get("difficulty_bucket"),
            "target": target,
            "result": {
                "hit": first_hit_turn is not None,
                "first_hit_turn": first_hit_turn,
                "best_rank": first_hit_rank,
                "reciprocal_rank": 0.0 if first_hit_rank is None else 1.0 / first_hit_rank,
            },
            "runtime": {
                "variant": variant_label,
                "dense_requested": config.enable_dense,
                "dense_loaded": agent.engine.retriever.semantic.enabled,
                "dense_failure_reason": agent.engine.retriever.semantic.failure_reason,
                "learned_reranker_requested": config.enable_learned_reranker,
                "learned_reranker_loaded": agent.engine.retriever.learned_reranker.enabled,
                "learned_reranker_failure_reason": agent.engine.retriever.learned_reranker.failure_reason,
                "cross_encoder_loaded": agent.engine.retriever.cross_encoder.enabled,
                "component_configuration": {
                    "fts": config.enable_fts,
                    "category": config.enable_category,
                    "fingerprints": config.enable_fingerprints,
                    "state": config.enable_state,
                    "clarification": config.enable_clarification,
                    "profile": config.enable_profile,
                    "popularity": config.enable_popularity,
                    "reranker": config.enable_learned_reranker,
                    "gate": bool(config.recommendation_depth_schedule),
                    "diversity": config.diversify_browsing,
                    "dense": config.enable_dense,
                },
            },
            "decision_signals": signals,
            "turns": inference_turns,
        }
        decision_markdown = "## What this replay suggests\n\n" + "\n".join(
            f"- {signal}" for signal in signals
        )
        # The replay owns the whole conversation, so it can tell the agent the
        # session has ended; distillation is otherwise lazy and the last
        # conversation of a process would never be recorded.
        agent.engine.flush_profile_memory()

        return (
            summary,
            chat,
            turn_rows,
            product_rows,
            target_payload,
            sample.get("user_profile") or {},
            inference_payload,
            decision_markdown,
        )

    def compare_session_components(
        self,
        sample_id: str,
        disabled_components: Sequence[str] = DEFAULT_SESSION_ABLATIONS,
        enable_dense: bool = False,
    ) -> tuple[
        str,
        list[list[Any]],
        list[list[Any]],
        str,
        str,
        list[dict[str, str]],
        list[dict[str, str]],
        list[list[Any]],
        list[list[Any]],
        list[list[Any]],
        list[list[Any]],
        dict,
        dict,
        dict,
        dict,
        str,
    ]:
        """Replay one session with the full agent and one controlled ablation."""

        selected = [
            str(value)
            for value in disabled_components
            if str(value) in COMPONENT_SPECS
        ]
        full_dense = bool(enable_dense) or "dense" in selected
        full = self._replay_with_config(
            sample_id,
            self._component_config((), enable_dense=full_dense),
            "Full agent",
        )
        ablated = self._replay_with_config(
            sample_id,
            self._component_config(selected, enable_dense=full_dense),
            "Without selected components",
        )
        full_inference = full[6]
        ablated_inference = ablated[6]
        full_result = dict(full_inference["result"])
        ablated_result = dict(ablated_inference["result"])
        full_session = {
            **full_result,
            "reciprocal_rank": full_result["reciprocal_rank"],
        }
        ablated_session = {
            **ablated_result,
            "reciprocal_rank": ablated_result["reciprocal_rank"],
        }
        full_value = _session_contribution(full_session)
        ablated_value = _session_contribution(ablated_session)
        full_turn = full_result["first_hit_turn"] or 11
        ablated_turn = ablated_result["first_hit_turn"] or 11
        outcome_rows = [
            ["Hit", int(bool(full_result["hit"])), int(bool(ablated_result["hit"])), int(bool(full_result["hit"])) - int(bool(ablated_result["hit"]))],
            ["First hit turn", full_result["first_hit_turn"], ablated_result["first_hit_turn"], ablated_turn - full_turn],
            ["Best rank", full_result["best_rank"], ablated_result["best_rank"], None if not full_result["best_rank"] or not ablated_result["best_rank"] else ablated_result["best_rank"] - full_result["best_rank"]],
            ["Reciprocal rank", _round(full_result["reciprocal_rank"]), _round(ablated_result["reciprocal_rank"]), _round(float(full_result["reciprocal_rank"]) - float(ablated_result["reciprocal_rank"]))],
            ["TechnicalScore contribution", _round(full_value), _round(ablated_value), _round(full_value - ablated_value)],
        ]
        full_components = dict(full_inference["runtime"]["component_configuration"])
        ablated_components = dict(ablated_inference["runtime"]["component_configuration"])
        status_rows: list[list[Any]] = []
        for key, spec in COMPONENT_SPECS.items():
            full_status: Any = full_components.get(key, False)
            ablated_status: Any = ablated_components.get(key, False)
            if key == "reranker":
                full_status = f"configured={full_status}, loaded={full_inference['runtime']['learned_reranker_loaded']}"
                ablated_status = f"configured={ablated_status}, loaded={ablated_inference['runtime']['learned_reranker_loaded']}"
            elif key == "dense":
                full_status = f"requested={full_status}, loaded={full_inference['runtime']['dense_loaded']}"
                ablated_status = f"requested={ablated_status}, loaded={ablated_inference['runtime']['dense_loaded']}"
            status_rows.append([spec["label"], full_status, ablated_status, spec["evidence"]])

        labels = ", ".join(_component_label(key) for key in selected) or "nothing"
        delta = full_value - ablated_value
        verdict = "helped" if delta > 1e-9 else "hurt" if delta < -1e-9 else "tied"
        summary = (
            f"## Paired session ablation · `{sample_id}`\n\n"
            f"Removed in the second replay: **{labels}**. The full agent **{verdict}** this session by "
            f"`{delta:+.6f}` TechnicalScore contribution (`{full_value:.6f}` vs `{ablated_value:.6f}`).\n\n"
            "> Both variants receive the same evaluator policy and target, but their questions can create different "
            "later customer messages. Treat this as an end-to-end causal replay, and use the turn tables to explain the path."
        )
        signals = (
            "## Report / video interpretation\n\n"
            f"- Selected components: **{labels}**.\n"
            f"- Full-agent outcome: turn `{full_result['first_hit_turn']}`, rank `{full_result['best_rank']}`.\n"
            f"- Ablated outcome: turn `{ablated_result['first_hit_turn']}`, rank `{ablated_result['best_rank']}`.\n"
            f"- Paired contribution delta: **`{delta:+.6f}`**. Positive means the selected components add value on this session.\n"
            "- Use the score-contribution column to explain why products moved; use gate columns to explain why recommendation depth changed."
        )
        return (
            summary,
            outcome_rows,
            status_rows,
            full[0],
            ablated[0],
            full[1],
            ablated[1],
            full[2],
            ablated[2],
            full[3],
            ablated[3],
            full[4],
            full[5],
            full_inference,
            ablated_inference,
            signals,
        )

    def compare_components(
        self,
        sample_count: int = 40,
        components: Sequence[str] = DEFAULT_SESSION_ABLATIONS,
        enable_dense: bool = False,
    ) -> tuple[str, list[list[Any]], list[list[Any]], dict[str, Any]]:
        """Measure the marginal end-to-end value of each selected component."""

        count = max(1, min(len(self.samples), int(sample_count)))
        samples = self.samples[:count]
        selected = [str(value) for value in components if str(value) in COMPONENT_SPECS]
        if not selected:
            selected = list(DEFAULT_SESSION_ABLATIONS)
        full_dense = bool(enable_dense) or "dense" in selected
        configurations: dict[str, tuple[str | None, AgentConfig]] = {
            "Full agent": (None, self._component_config((), enable_dense=full_dense))
        }
        for component in selected:
            configurations[f"Without {_component_label(component)}"] = (
                component,
                self._component_config((component,), enable_dense=full_dense),
            )

        metrics: dict[str, dict[str, Any]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for name, (removed, config) in configurations.items():
            agent = Agent(self.catalog_path, config=config, catalog_index=self.catalog)
            result = evaluate(
                agent,
                samples,
                self.identifiers,
                self.categories,
                self.raw_products,
            )
            latencies = [
                event.latency_ms
                for events in agent.engine.traces.values()
                for event in events
            ]
            metrics[name] = result
            diagnostics[name] = {
                "removed": removed,
                "mean_latency": statistics.fmean(latencies) if latencies else 0.0,
                "p95_latency": _percentile(latencies, 0.95),
                "dense": agent.engine.retriever.semantic.enabled,
                "reranker": agent.engine.retriever.learned_reranker.enabled,
                "gate": bool(config.recommendation_depth_schedule),
            }

        full_metric = metrics["Full agent"]
        full_scopes: dict[str, dict[str, Any]] = {
            "Overall": {
                key: full_metric[key]
                for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")
            },
            **dict(full_metric["scenario_metrics"]),
        }
        comparison_rows: list[list[Any]] = []
        for name, result in metrics.items():
            removed = diagnostics[name]["removed"]
            scopes: dict[str, dict[str, Any]] = {
                "Overall": {
                    key: result[key]
                    for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")
                },
                **dict(result["scenario_metrics"]),
            }
            for scope, summary in scopes.items():
                efficiency, score = _technical_score(summary)
                full_summary = full_scopes[scope]
                _, full_score = _technical_score(full_summary)
                comparison_rows.append(
                    [
                        name,
                        _component_label(removed) if removed else "—",
                        scope,
                        summary["sample_count"],
                        summary["hit_rate_at_10"],
                        summary["mrr"],
                        summary["mttc"],
                        _round(efficiency),
                        _round(score),
                        _round(score - full_score),
                        _round(float(summary["mrr"]) - float(full_summary["mrr"])),
                        _round(float(summary["mttc"]) - float(full_summary["mttc"])),
                        _round(diagnostics[name]["mean_latency"], 3),
                        _round(diagnostics[name]["p95_latency"], 3),
                        diagnostics[name]["dense"],
                        diagnostics[name]["reranker"],
                        diagnostics[name]["gate"],
                    ]
                )

        full_sessions = {
            str(item["sample_id"]): item
            for item in full_metric["sessions"]
        }
        session_rows: list[list[Any]] = []
        for name, result in metrics.items():
            removed = diagnostics[name]["removed"]
            if removed is None:
                continue
            ablated_sessions = {
                str(item["sample_id"]): item
                for item in result["sessions"]
            }
            for sample in samples:
                sample_id = str(sample["sample_id"])
                full_session = full_sessions[sample_id]
                ablated_session = ablated_sessions[sample_id]
                full_turn = full_session["first_hit_turn"] or 11
                ablated_turn = ablated_session["first_hit_turn"] or 11
                session_rows.append(
                    [
                        _component_label(removed),
                        sample_id,
                        sample["scenario_type"],
                        full_session["first_hit_turn"],
                        full_session["best_rank"],
                        ablated_session["first_hit_turn"],
                        ablated_session["best_rank"],
                        _round(
                            _session_contribution(full_session)
                            - _session_contribution(ablated_session)
                        ),
                        ablated_turn - full_turn,
                        _round(
                            float(full_session["reciprocal_rank"])
                            - float(ablated_session["reciprocal_rank"])
                        ),
                    ]
                )

        full_score = float(full_metric["recommended_technical_score"])
        values = []
        for name, result in metrics.items():
            removed = diagnostics[name]["removed"]
            if removed is None:
                continue
            values.append(
                (
                    removed,
                    full_score - float(result["recommended_technical_score"]),
                )
            )
        values.sort(key=lambda item: (-item[1], item[0]))
        ranking = "\n".join(
            f"- **{_component_label(component)}:** `{value:+.6f}` full-minus-ablated TechnicalScore"
            for component, value in values
        )
        summary = (
            f"## Component value lab · first {count} sessions in {self.scope_name}\n\n"
            f"Full-agent TechnicalScore: **`{full_score:.6f}`**. Each row below removes exactly one "
            "component from that same full configuration. Positive full-minus-ablated value means the component helps.\n\n"
            f"{ranking}\n\n"
            "> End-to-end ablations can change the next simulated customer message by changing `ask_attribute`. "
            "Use development-only scopes for model selection; held-out scopes are descriptive readouts."
        )
        performance_path = Path(__file__).resolve().parents[1] / "docs" / "performance_results.json"
        frozen_performance: dict[str, Any] = {}
        if performance_path.exists():
            try:
                frozen_performance = json.loads(performance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                frozen_performance = {"status": "performance artifact could not be read"}
        evidence = {
            "scope": self.scope_name,
            "sample_count": count,
            "full_agent": {
                key: full_metric[key]
                for key in (
                    "hit_rate_at_10",
                    "mrr",
                    "mttc",
                    "efficiency",
                    "recommended_technical_score",
                    "reported_token_usage",
                    "scenario_metrics",
                )
            },
            "component_value": {
                component: _round(value)
                for component, value in values
            },
            "operational_disclosure": {
                "external_api_calls": 0,
                "paid_model_cost_usd": 0,
                "network_required_at_inference": False,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "active_ranker": "dependency-free linear residual ranker",
                "optional_dense_model": "BAAI/bge-small-en-v1.5; only loaded when requested and verified",
            },
            "frozen_performance_evidence": frozen_performance,
            "limitations": [
                "Public labels and hidden intent are exposed only in this development dashboard.",
                "End-to-end dialogue paths may differ after an ablation changes the selected question.",
                "English clothing-specific extraction is not yet a domain-general ontology.",
                "The private organizer evaluation remains the decisive test.",
            ],
            "team_contributions": {
                "status": "TODO before submission",
                "required_action": "Replace the README placeholder with member names and specific contributions.",
            },
        }
        return summary, comparison_rows, session_rows, evidence

    def compare_reranker(
        self,
        sample_count: int = 40,
        enable_dense: bool = False,
        enable_clarification: bool = True,
    ) -> tuple[str, list[list[Any]], list[list[Any]]]:
        """Compatibility wrapper for the original reranker-only dashboard API."""

        summary, comparison, sessions, _ = self.compare_components(
            sample_count,
            ("reranker",),
            enable_dense,
        )
        return summary, comparison, sessions

    @staticmethod
    def _live_scenario_rows(overall: dict[str, Any]) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for scenario, summary in dict(overall.get("scenario_metrics") or {}).items():
            efficiency, score = _technical_score(summary)
            rows.append(
                [
                    scenario,
                    summary["sample_count"],
                    summary["hit_rate_at_10"],
                    summary["mrr"],
                    summary["mttc"],
                    _round(efficiency),
                    _round(score),
                ]
            )
        return rows

    def resolve_scope_path(self, name: str | None) -> str:
        """Dataset path for `name`, independent of any mutated backend state.

        The panels must not depend on a previous change event having fired:
        the dropdown value is authoritative, and it is also per-browser, so two
        viewers cannot clobber each other's selection.
        """

        scope = self.scopes.get(name or "")
        return self._materialize(scope) if scope else self.dataset_path

    def stream_live_evaluation(self, scope_name: str | None = None):
        """Run all 200 sessions in a new interpreter so disk edits are imported."""

        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.catalog.index_dir / "live_runs" / f"evaluation-{timestamp}.json"
        command = [
            sys.executable,
            "-u",
            "-m",
            "cartographer.live_evaluator",
            "--catalog",
            self.catalog_path,
            "--dataset",
            self.resolve_scope_path(scope_name),
            "--output",
            str(output_path),
        ]
        if self.ranker_path:
            command.extend(["--ranker", str(self.ranker_path)])
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        session_rows: list[list[Any]] = []
        metadata: dict[str, Any] = {}
        recent_log: list[str] = []
        completed_normally = False
        try:
            if process.stdout is None:
                raise RuntimeError("Fresh evaluator did not expose a progress stream")
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    recent_log.append(line)
                    recent_log = recent_log[-8:]
                    yield (
                        "## Fresh evaluation is starting\n\n`" + "\n".join(recent_log) + "`",
                        {},
                        [],
                        session_rows,
                        metadata,
                        None,
                    )
                    continue
                kind = event.get("event")
                if kind == "start":
                    metadata = dict(event["metadata"])
                    dirty = metadata.get("git", {}).get("working_tree_changes") or []
                    initialized = (
                        "## Fresh 200-session evaluation initialized\n\n"
                        f"Source digest: `{metadata['source']['combined_sha256'][:16]}…`  \n"
                        f"Git commit: `{metadata.get('git', {}).get('commit') or 'unavailable'}`  \n"
                        f"Working-tree changes: `{len(dirty)}`  \n\n"
                        "The child process imported the current files from disk after you pressed Start."
                    )
                    yield (
                        initialized,
                        {},
                        [],
                        session_rows,
                        metadata,
                        None,
                    )
                elif kind == "progress":
                    completed = int(event["completed"])
                    total = int(event["total"])
                    overall = dict(event["overall"])
                    session = dict(event["session"])
                    score = float(overall["recommended_technical_score"])
                    session_rows.append(
                        [
                            completed,
                            session["sample_id"],
                            session["scenario_type"],
                            session["hit"],
                            session["first_hit_turn"],
                            session["best_rank"],
                            _round(session["reciprocal_rank"]),
                            overall["hit_rate_at_10"],
                            overall["mrr"],
                            overall["mttc"],
                            score,
                        ]
                    )
                    filled = round(30 * completed / total)
                    bar = "█" * filled + "░" * (30 - filled)
                    status = (
                        f"## Running latest code: {completed}/{total} ({100 * completed / total:.1f}%)\n\n"
                        f"`{bar}`\n\n"
                        f"Current session: `{session['sample_id']}` · {session['scenario_type']} · "
                        f"{'hit' if session['hit'] else 'miss'}  \n"
                        f"Running TechnicalScore: **{score:.6f}**  \n"
                        f"Elapsed: `{float(event['elapsed_seconds']):.1f}s` · "
                        f"ETA: `{float(event['eta_seconds']):.1f}s`"
                    )
                    yield (
                        status,
                        {key: value for key, value in overall.items() if key != "scenario_metrics"},
                        self._live_scenario_rows(overall),
                        session_rows,
                        metadata,
                        None,
                    )
                elif kind == "complete":
                    result = dict(event["result"])
                    completed_normally = True
                    status = (
                        "## ✅ Fresh 200-session evaluation complete\n\n"
                        f"TechnicalScore: **{float(result['recommended_technical_score']):.6f}** · "
                        f"Hit Rate: **{float(result['hit_rate_at_10']):.6f}** · "
                        f"MRR: **{float(result['mrr']):.6f}** · MTTC: **{float(result['mttc']):.6f}**\n\n"
                        f"Artifact: `{event['output']}`"
                    )
                    yield (
                        status,
                        {key: value for key, value in result.items() if key not in {"scenario_metrics", "live_evaluation"}},
                        self._live_scenario_rows(result),
                        session_rows,
                        {**metadata, "completed": result.get("live_evaluation")},
                        event["output"],
                    )
            return_code = process.wait()
            if return_code != 0 and not completed_normally:
                raise RuntimeError(
                    f"Fresh evaluator exited with code {return_code}: " + " | ".join(recent_log[-4:])
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()


def build_dashboard(backend: DashboardBackend, presentation: bool = False):
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - depends on optional UI package
        raise RuntimeError(
            "Gradio is not installed. Run: python -m pip install -r requirements-dashboard.txt"
        ) from error

    # Gradio 5 accepts CSS on Blocks; Gradio 6 moves it to launch(). The
    # dashboard dependency is intentionally pinned to the compatible 5.x line.
    with gr.Blocks(title="Cartographer Evaluator Observatory", css=DASHBOARD_CSS) as demo:
        gr.Markdown(
            "# Cartographer Evaluator Observatory\n"
            "Replay the unchanged evaluator, inspect the agent's complete reasoning state, and determine whether a "
            "failure is caused by retrieval, ranking, parsing, or clarification."
        )
        gr.Markdown(
            "**Development-only:** expected targets, intent cards, and evaluator behavior are displayed here for "
            "analysis. The official `Agent.respond` output remains unchanged and never receives these fields.",
            elem_classes="warning-box",
        )
        with gr.Row():
            scope = gr.Dropdown(
                choices=backend.scope_names,
                value=backend.scope_name,
                label="Dataset scope",
                info=(
                    "Every panel below follows this selection. Held-out sets are for confirmation "
                    "readouts only; configuration choices stay on the development partition."
                ),
                scale=3,
            )
            scope_note = gr.Markdown(f"Active scope: **{backend.scope_name}**")
        gr.Markdown(backend.ranker_banner(), elem_classes="warning-box")
        with gr.Tab("Session replay"):
            with gr.Row():
                session = gr.Dropdown(
                    choices=backend.choices(),
                    value=backend.samples[0]["sample_id"],
                    label="Session",
                    scale=4,
                )
                session_dense = gr.Checkbox(
                    value=False,
                    label="Include optional BGE in full agent",
                    info="Selecting the BGE ablation also requests it automatically; assets still fail closed.",
                )
            session_components = gr.CheckboxGroup(
                choices=COMPONENT_CHOICES,
                value=DEFAULT_SESSION_ABLATIONS,
                label="Remove these components in the comparison replay",
                info=(
                    "The first replay always uses the full current agent. Select one component for a clean "
                    "individual ablation, or several to inspect an interaction."
                ),
            )
            run = gr.Button("Replay full agent vs selected ablation", variant="primary")
            ablation_summary = gr.Markdown()
            with gr.Row():
                outcome_comparison = gr.Dataframe(
                    headers=SESSION_OUTCOME_HEADERS,
                    interactive=False,
                    label="Paired session outcome",
                )
                component_status = gr.Dataframe(
                    headers=COMPONENT_STATUS_HEADERS,
                    interactive=False,
                    label="What is on and off in each replay",
                )
            with gr.Row():
                target = gr.JSON(label="Expected #1 product and evaluator intent", visible=not presentation)
                profile = gr.JSON(label="User profile supplied to Agent.reset", visible=not presentation)
            with gr.Row(equal_height=True):
                with gr.Column():
                    full_summary = gr.Markdown()
                    full_conversation = gr.Chatbot(
                        label="Full agent · evaluator ↔ agent conversation",
                        height=620,
                        type="messages",
                        allow_tags=False,
                    )
                with gr.Column():
                    ablated_summary = gr.Markdown()
                    ablated_conversation = gr.Chatbot(
                        label="Without selected components · evaluator ↔ agent conversation",
                        height=620,
                        type="messages",
                        allow_tags=False,
                    )
            replay_interpretation = gr.Markdown()
            gr.Markdown("### Long-term memory of this shopper")
            profile_memory_view = gr.Markdown(backend.profile_memory_view(
                backend.samples[0]["sample_id"]
            ))
            with gr.Tab("Full vs ablated turn state", visible=not presentation):
                with gr.Row():
                    full_turns = gr.Dataframe(
                        headers=TURN_HEADERS,
                        interactive=False,
                        label="Full agent: state, uncertainty, gate, and target position",
                    )
                    ablated_turns = gr.Dataframe(
                        headers=TURN_HEADERS,
                        interactive=False,
                        label="Ablated agent: state, uncertainty, gate, and target position",
                    )
            with gr.Tab("Full vs ablated ranked products", visible=not presentation):
                gr.Markdown(
                    "The final column decomposes the largest additive score signals for a transparent recommendation explanation."
                )
                with gr.Row():
                    full_recommendations = gr.Dataframe(
                        headers=PRODUCT_HEADERS,
                        interactive=False,
                        label="Full agent recommendations",
                        elem_classes="diagnostic-table",
                    )
                    ablated_recommendations = gr.Dataframe(
                        headers=PRODUCT_HEADERS,
                        interactive=False,
                        label="Ablated recommendations",
                        elem_classes="diagnostic-table",
                    )
            with gr.Tab("Complete paired inference JSON", visible=not presentation):
                with gr.Row():
                    full_inference = gr.JSON(label="Full-agent diagnostics")
                    ablated_inference = gr.JSON(label="Ablated diagnostics")
            session.change(
                fn=backend.profile_memory_view,
                inputs=[session],
                outputs=[profile_memory_view],
            )
            run.click(
                fn=backend.profile_memory_view,
                inputs=[session],
                outputs=[profile_memory_view],
            )
            run.click(
                fn=backend.compare_session_components,
                inputs=[session, session_components, session_dense],
                outputs=[
                    ablation_summary,
                    outcome_comparison,
                    component_status,
                    full_summary,
                    ablated_summary,
                    full_conversation,
                    ablated_conversation,
                    full_turns,
                    ablated_turns,
                    full_recommendations,
                    ablated_recommendations,
                    target,
                    profile,
                    full_inference,
                    ablated_inference,
                    replay_interpretation,
                ],
                concurrency_limit=1,
            )
        with gr.Tab("All expected products", visible=not presentation):
            gr.Markdown(
                "Each row is one session in the selected scope. `Expected #1 ASIN` is the evaluator's target."
            )
            all_sessions = gr.Dataframe(
                value=backend.session_rows(),
                headers=SESSION_HEADERS,
                interactive=False,
                label="Sessions and expected number-one products",
            )
        with gr.Tab("Component value lab"):
            gr.Markdown(
                "Run the full agent once, then remove each selected component individually under the unchanged "
                "evaluator. This produces the overall, scenario, latency, and per-session evidence needed for the report."
            )
            with gr.Row():
                batch_count = gr.Slider(1, len(backend.samples), value=40, step=1, label="Session count")
                batch_dense = gr.Checkbox(
                    value=False,
                    label="Include optional BGE in full agent",
                    info="Selecting the BGE ablation requests it automatically.",
                )
            batch_components = gr.CheckboxGroup(
                choices=COMPONENT_CHOICES,
                value=DEFAULT_SESSION_ABLATIONS,
                label="Components to ablate one at a time",
            )
            compare = gr.Button("Measure selected component values", variant="primary")
            comparison_summary = gr.Markdown()
            comparison = gr.Dataframe(
                headers=COMPARISON_HEADERS,
                interactive=False,
                label="Full-agent and per-component overall/scenario metrics",
            )
            session_comparison = gr.Dataframe(
                headers=SESSION_COMPARISON_HEADERS,
                interactive=False,
                label="Per-session component effects",
            )
            report_evidence = gr.JSON(
                label="Report-ready architecture, metric, cost, token, and limitation evidence",
                visible=not presentation,
            )
            compare.click(
                fn=backend.compare_components,
                inputs=[batch_count, batch_components, batch_dense],
                outputs=[comparison_summary, comparison, session_comparison, report_evidence],
                concurrency_limit=1,
            )
        with gr.Tab("Report & video evidence", visible=not presentation):
            gr.Markdown(
                """
## Deliverable map

| Deliverable evidence | Where to capture it |
|---|---|
| Architecture and active components | Component status table plus complete paired inference JSON |
| Multi-turn demonstration | Side-by-side session conversations and turn-state tables |
| Reranker and precision-gate value | Select each separately in Session replay; confirm across scenarios in Component value lab |
| Transparent recommendation explanations | Dominant score contributions in the ranked-products tab |
| Models, cost, tokens, network fallback | Report-ready JSON in Component value lab |
| Overall and per-scenario metrics | Component value lab and Live latest-code test |
| Latency | Replay turn table, component metric table, and live artifact |
| Limitations | Report-ready JSON and the project documentation |
| Team contributions | Complete the named contribution section in the final report before submission |

### Recommended video sequence

1. Show the full-agent component status and the target without revealing it to `Agent.respond`.
2. Replay an Intent Override session and point out route, epoch, retained state, information gain, gate depth, and target rank.
3. Toggle off the precision gate and reranker separately; show the paired contribution delta and why the product order changed.
4. Show overall and per-scenario component value, then the zero-token, zero-API-cost operational disclosure.

The dashboard is diagnostic evidence, not the official response surface. Expected products and evaluator intent remain isolated from agent inference.
"""
            )
        with gr.Tab("Live latest-code test"):
            gr.Markdown(
                "Run the unchanged evaluator across every session in the selected scope in a **new Python "
                "process**. This imports whatever code is currently saved on disk, even if the dashboard was "
                "started before your edits. Complete artifacts are stored under the ignored "
                "`data/cartographer_index/live_runs/` directory."
            )
            with gr.Row():
                start_live = gr.Button("Start fresh full-scope test", variant="primary")
                stop_live = gr.Button("Cancel running test", variant="stop")
            live_status = gr.Markdown("No live evaluation is running.")
            with gr.Row():
                live_overall = gr.JSON(label="Live TechnicalScore, Hit Rate, MRR and MTTC")
                live_metadata = gr.JSON(label="Exact source and Git metadata", visible=not presentation)
            live_scenarios = gr.Dataframe(
                headers=LIVE_SCENARIO_HEADERS,
                interactive=False,
                label="Rolling per-scenario metrics",
            )
            live_sessions = gr.Dataframe(
                headers=LIVE_SESSION_HEADERS,
                interactive=False,
                label="One row appears after every completed session",
            )
            live_artifact = gr.File(label="Download complete evaluation JSON", interactive=False)
            live_event = start_live.click(
                fn=backend.stream_live_evaluation,
                inputs=[scope],
                outputs=[
                    live_status,
                    live_overall,
                    live_scenarios,
                    live_sessions,
                    live_metadata,
                    live_artifact,
                ],
                concurrency_limit=1,
                api_name="run_live_200",
            )
            stop_live.click(fn=None, cancels=[live_event])
        with gr.Tab("How to decide", visible=not presentation):
            gr.Markdown(
                """
## Reading the evidence

- **Target absent from the candidate pool:** improve retrieval—semantic embeddings, synonyms, query rewriting, or category parsing.
- **Target retrieved but below rank 10:** improve ranking—a learned reranker or better deterministic weights.
- **Target rises only after useful questions:** clarification/state tracking is working; an LLM is not automatically needed.
- **Category or constraints are parsed incorrectly:** an LLM or compact intent classifier may help, but test deterministic extraction first.
- **Agent wording feels repetitive but metrics are strong:** an LLM would improve presentation, not retrieval quality.

The dashboard deliberately separates these failure modes so “add an LLM” is a measured decision rather than an architectural assumption.
"""
            )

        def _apply_scope(name: str):
            """Re-point the session picker, the table, and the batch slider."""

            resolved = backend.select_scope(name)
            count = len(backend.samples)
            return (
                f"Active scope: **{resolved}** \u00b7 `{backend.dataset_path}`",
                gr.update(choices=backend.choices(), value=backend.samples[0]["sample_id"]),
                gr.update(value=backend.session_rows()),
                gr.update(maximum=count, value=min(40, count)),
            )

        scope.change(
            fn=_apply_scope,
            inputs=[scope],
            outputs=[scope_note, session, all_sessions, batch_count],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Cartographer evaluator dashboard")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=None,
        help=(
            "Additional labelled dataset to expose as a scope; repeatable. Defaults to "
            "synthetic_800_v1.jsonl when present. Missing files are skipped."
        ),
    )
    parser.add_argument("--index-dir", default="data/cartographer_index")
    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Hide developer-only diagnostic panels and show the three views that "
            "demonstrate the system end to end: session replay, the live test, and "
            "component value."
        ),
    )
    parser.add_argument(
        "--profile-memory",
        default=None,
        help=(
            "Enable long-term personalization and store distilled shopper profiles at "
            "this path, so replays demonstrate Personalized Context Distillation."
        ),
    )
    parser.add_argument(
        "--ranker",
        default=None,
        help=(
            "Alternate reranker artifact to display, e.g. one trained on all 200 public "
            "sessions. The promoted cartographer/ranker_weights.json is left untouched."
        ),
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--auth",
        default=None,
        help=(
            "Require a login as USER:PASSWORD. Strongly recommended whenever --server-name is "
            "not loopback: this dashboard exposes evaluator ground-truth labels and can start "
            "long CPU evaluations."
        ),
    )
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()
    extra = DEFAULT_EXTRA_DATASETS if args.extra_dataset is None else tuple(args.extra_dataset)
    backend = DashboardBackend(
        args.catalog, args.dataset, args.index_dir, extra_datasets=extra, ranker_path=args.ranker,
        profile_memory_path=args.profile_memory,
    )
    demo = build_dashboard(backend, presentation=args.presentation)
    auth = None
    if args.auth:
        user, separator, password = args.auth.partition(":")
        if not separator or not user or not password:
            parser.error("--auth must be USER:PASSWORD")
        auth = (user, password)
    elif args.server_name not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: serving on "
            f"{args.server_name} without --auth exposes ground-truth labels to anyone who can "
            "reach this port.",
            file=sys.stderr,
            flush=True,
        )
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        auth=auth,
    )


if __name__ == "__main__":
    main()
