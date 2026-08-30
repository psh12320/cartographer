from __future__ import annotations

import argparse
import json
import statistics
import uuid
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
]

COMPARISON_HEADERS = [
    "Variant",
    "Scope",
    "Sessions",
    "Hit Rate@10",
    "MRR",
    "MTTC",
    "Efficiency",
    "TechnicalScore",
    "Mean latency ms",
    "p95 latency ms",
    "Dense loaded",
    "Reranker loaded",
]

SESSION_COMPARISON_HEADERS = [
    "Session",
    "Scenario",
    "Baseline turn",
    "Baseline rank",
    "Reranker turn",
    "Reranker rank",
    "Turn improvement",
    "Reciprocal-rank improvement",
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
    ) -> None:
        self.catalog_path = str(catalog_path)
        self.dataset_path = str(dataset_path)
        self.samples = load_jsonl(self.dataset_path)
        self.samples_by_id = {str(sample["sample_id"]): sample for sample in self.samples}
        self.identifiers, self.categories, self.raw_products = catalog_index(self.catalog_path)
        self.catalog = CatalogIndex(self.catalog_path, index_dir)

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
        return AgentConfig(
            catalog_path=Path(self.catalog_path),
            index_dir=self.catalog.index_dir,
            enable_learned_reranker=bool(enable_learned_reranker),
            enable_dense=bool(enable_dense),
            enable_clarification=bool(enable_clarification),
            diversify_browsing=bool(diversify_browsing),
            enable_cross_encoder=False,
        )

    def replay(
        self,
        sample_id: str,
        enable_learned_reranker: bool = False,
        enable_dense: bool = False,
        enable_clarification: bool = True,
        diversify_browsing: bool = True,
    ) -> tuple[str, list[dict[str, str]], list[list[Any]], list[list[Any]], dict, dict, dict, str]:
        if sample_id not in self.samples_by_id:
            raise ValueError(f"Unknown sample id: {sample_id}")
        sample = self.samples_by_id[sample_id]
        target = str(sample["ground_truth"]["parent_asin"])
        target_record = self.catalog.products[self.catalog.asin_to_index[target]]
        intent_card, behavior = materialize_hidden_fields(sample, self.raw_products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        config = self._config(
            enable_learned_reranker,
            enable_dense,
            enable_clarification,
            diversify_browsing,
        )
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
                pre_learned = (
                    hit.score
                    - config.learned_reranker_scale * hit.learned_score
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
            f"## {status}: {sample_id}\n\n"
            f"| Expected #1 product | Outcome | Scenario | Turns used | Dense | Learned reranker |\n"
            f"|---|---:|---|---:|---:|---:|\n"
            f"| **{target_record.title}** (`{target}`) | "
            f"{'rank ' + str(first_hit_rank) + ' on turn ' + str(first_hit_turn) if first_hit_turn else 'not found'} | "
            f"{sample['scenario_type']} | {len(inference_turns)} | "
            f"{agent.engine.retriever.semantic.enabled} | "
            f"{agent.engine.retriever.learned_reranker.enabled} |\n\n"
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
                "dense_requested": enable_dense,
                "dense_loaded": agent.engine.retriever.semantic.enabled,
                "dense_failure_reason": agent.engine.retriever.semantic.failure_reason,
                "learned_reranker_requested": enable_learned_reranker,
                "learned_reranker_loaded": agent.engine.retriever.learned_reranker.enabled,
                "learned_reranker_failure_reason": agent.engine.retriever.learned_reranker.failure_reason,
                "cross_encoder_loaded": agent.engine.retriever.cross_encoder.enabled,
            },
            "decision_signals": signals,
            "turns": inference_turns,
        }
        decision_markdown = "## What this replay suggests\n\n" + "\n".join(
            f"- {signal}" for signal in signals
        )
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

    def compare_reranker(
        self,
        sample_count: int = 40,
        enable_dense: bool = False,
        enable_clarification: bool = True,
    ) -> tuple[str, list[list[Any]], list[list[Any]]]:
        count = max(1, min(len(self.samples), int(sample_count)))
        samples = self.samples[:count]
        configurations = {
            "Deterministic baseline": self._config(
                False, enable_dense, enable_clarification, True
            ),
            "Frozen learned reranker": self._config(
                True, enable_dense, enable_clarification, True
            ),
        }
        metrics: dict[str, dict[str, Any]] = {}
        diagnostic_map: dict[str, dict[str, Any]] = {}
        comparison_rows: list[list[Any]] = []
        for name, config in configurations.items():
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
            diagnostics = {
                "mean_latency": statistics.fmean(latencies) if latencies else 0.0,
                "p95_latency": _percentile(latencies, 0.95),
                "dense": agent.engine.retriever.semantic.enabled,
                "reranker": agent.engine.retriever.learned_reranker.enabled,
            }
            metrics[name] = result
            diagnostic_map[name] = diagnostics
            scopes = {"Overall": {key: result[key] for key in ("sample_count", "hit_rate_at_10", "mrr", "mttc")}}
            scopes.update(result["scenario_metrics"])
            for scope, summary in scopes.items():
                efficiency, score = _technical_score(summary)
                comparison_rows.append(
                    [
                        name,
                        scope,
                        summary["sample_count"],
                        summary["hit_rate_at_10"],
                        summary["mrr"],
                        summary["mttc"],
                        _round(efficiency),
                        _round(score),
                        _round(diagnostics["mean_latency"], 3),
                        _round(diagnostics["p95_latency"], 3),
                        diagnostics["dense"],
                        diagnostics["reranker"],
                    ]
                )
        baseline_sessions = {
            str(item["sample_id"]): item
            for item in metrics["Deterministic baseline"]["sessions"]
        }
        reranked_sessions = {
            str(item["sample_id"]): item
            for item in metrics["Frozen learned reranker"]["sessions"]
        }
        session_rows: list[list[Any]] = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            baseline = baseline_sessions[sample_id]
            reranked = reranked_sessions[sample_id]
            baseline_turn = baseline["first_hit_turn"] or 11
            reranked_turn = reranked["first_hit_turn"] or 11
            session_rows.append(
                [
                    sample_id,
                    sample["scenario_type"],
                    baseline["first_hit_turn"],
                    baseline["best_rank"],
                    reranked["first_hit_turn"],
                    reranked["best_rank"],
                    baseline_turn - reranked_turn,
                    _round(reranked["reciprocal_rank"] - baseline["reciprocal_rank"]),
                ]
            )
        baseline_score = float(metrics["Deterministic baseline"]["recommended_technical_score"])
        reranked_score = float(metrics["Frozen learned reranker"]["recommended_technical_score"])
        gain = reranked_score - baseline_score
        verdict = "improves" if gain > 0 else "does not improve"
        summary = (
            f"## Batch comparison: first {count} public development sessions\n\n"
            f"The frozen reranker **{verdict}** TechnicalScore on this slice: "
            f"`{baseline_score:.6f}` → `{reranked_score:.6f}` "
            f"(`{gain:+.6f}`).\n\n"
            "> This is development-set analysis, not a private-test estimate. Use the per-session table to identify "
            "whether gains come from earlier turns, better rank, or memorized public-set regularities."
        )
        return summary, comparison_rows, session_rows


def build_dashboard(backend: DashboardBackend):
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
        with gr.Tab("Session replay"):
            with gr.Row():
                session = gr.Dropdown(
                    choices=backend.choices(),
                    value=backend.samples[0]["sample_id"],
                    label="Public development session",
                    scale=4,
                )
                learned = gr.Checkbox(
                    value=False,
                    label="Enable frozen learned reranker",
                    info="Off by default so the rule-based algorithm is easy to inspect.",
                )
                dense = gr.Checkbox(
                    value=False,
                    label="Request BGE dense route",
                    info="Fails closed when verified embeddings are absent.",
                )
                clarification = gr.Checkbox(value=True, label="Enable clarification")
                diversify = gr.Checkbox(value=True, label="Diversify Browsing ranks 4–10")
            run = gr.Button("Replay exact evaluator session", variant="primary")
            summary = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=2):
                    conversation = gr.Chatbot(
                        label="Entire evaluator ↔ agent conversation",
                        height=620,
                        type="messages",
                        allow_tags=False,
                    )
                with gr.Column(scale=1):
                    target = gr.JSON(label="Expected #1 product and evaluator intent")
                    profile = gr.JSON(label="User profile supplied to Agent.reset")
            decisions = gr.Markdown()
            with gr.Tab("Turn-by-turn state"):
                turns = gr.Dataframe(
                    headers=TURN_HEADERS,
                    interactive=False,
                    label="Evaluator input, agent output, parsed state, uncertainty, and target position",
                )
            with gr.Tab("Recommended products and ranking signals"):
                recommendations = gr.Dataframe(
                    headers=PRODUCT_HEADERS,
                    interactive=False,
                    label="Every recommendation on every turn with product metadata and score features",
                    elem_classes="diagnostic-table",
                )
            with gr.Tab("Complete inference JSON"):
                inference = gr.JSON(label="All captured non-response diagnostics")
            run.click(
                fn=backend.replay,
                inputs=[session, learned, dense, clarification, diversify],
                outputs=[summary, conversation, turns, recommendations, target, profile, inference, decisions],
                concurrency_limit=1,
            )
        with gr.Tab("All 200 expected products"):
            gr.Markdown(
                "Each row is one public development user/session. `Expected #1 ASIN` is the evaluator's target."
            )
            gr.Dataframe(
                value=backend.session_rows(),
                headers=SESSION_HEADERS,
                interactive=False,
                label="Public sessions and expected number-one products",
            )
        with gr.Tab("Reranker A/B"):
            gr.Markdown(
                "Compare the rule-based deterministic ranker with the frozen public-development-trained residual "
                "reranker under the same unchanged evaluator. Larger slices take longer on CPU."
            )
            with gr.Row():
                batch_count = gr.Slider(1, len(backend.samples), value=40, step=1, label="Session count")
                batch_dense = gr.Checkbox(value=False, label="Request BGE dense route")
                batch_clarification = gr.Checkbox(value=True, label="Enable clarification")
            compare = gr.Button("Run baseline vs reranker", variant="primary")
            comparison_summary = gr.Markdown()
            comparison = gr.Dataframe(
                headers=COMPARISON_HEADERS,
                interactive=False,
                label="Overall and per-scenario metrics",
            )
            session_comparison = gr.Dataframe(
                headers=SESSION_COMPARISON_HEADERS,
                interactive=False,
                label="Per-session turn and rank changes",
            )
            compare.click(
                fn=backend.compare_reranker,
                inputs=[batch_count, batch_dense, batch_clarification],
                outputs=[comparison_summary, comparison, session_comparison],
                concurrency_limit=1,
            )
        with gr.Tab("How to decide"):
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
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Cartographer evaluator dashboard")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--index-dir", default="data/cartographer_index")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    args = parser.parse_args()
    backend = DashboardBackend(args.catalog, args.dataset, args.index_dir)
    demo = build_dashboard(backend)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
