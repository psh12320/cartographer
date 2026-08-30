from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    metric_summary,
    normalize_recommendations,
)
from starter.agent import Agent

from .catalog import CatalogIndex
from .config import AgentConfig
from .data_split import file_sha256, load_manifest, select_split
from .experiments import run_once, semantic_configs, stratified_folds, technical_score
from .ranker import FEATURE_NAMES, feature_rows, route_key


@dataclass
class RankingSnapshot:
    sample_id: str
    route: str
    positive: dict[str, float]
    negatives: list[dict[str, float]]


def collect_snapshots(
    samples: list[dict],
    catalog_path: str,
    config: AgentConfig,
    negative_limit: int = 30,
) -> tuple[list[RankingSnapshot], dict[str, object]]:
    identifiers, categories, products = catalog_index(catalog_path)
    shared_catalog = CatalogIndex(catalog_path, config.index_dir)
    agent = Agent(catalog_path, config=config, catalog_index=shared_catalog)
    snapshots: list[RankingSnapshot] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        session_id = f"ranker_{sample_id}"
        agent.reset(session_id, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        intent_card, behavior = materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": intent_card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective,
            coarse_category(categories.get(target, [])),
            disclosed,
        )
        for turn in range(1, MAX_TURNS + 1):
            response = agent.respond(session_id, user_message, turn, TOP_K)
            state = agent.engine.sessions[session_id]
            hits = state.cached_hits
            rows = feature_rows(hits)
            positions = {
                hit.parent_asin: position for position, hit in enumerate(hits)
            }
            if override_applied and target in positions:
                positive_position = positions[target]
                negatives = [
                    row
                    for hit, row in zip(hits, rows)
                    if hit.parent_asin != target
                ][:negative_limit]
                if negatives:
                    snapshots.append(
                        RankingSnapshot(
                            sample_id=sample_id,
                            route=route_key(state),
                            positive=rows[positive_position],
                            negatives=negatives,
                        )
                    )
            ranked = normalize_recommendations(response.get("recommendations"), identifiers)
            if override_applied and target in ranked:
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
    diagnostics = {
        "sample_count": len(samples),
        "snapshot_count": len(snapshots),
        "route_snapshots": {
            route: sum(snapshot.route == route for snapshot in snapshots)
            for route in ("buying", "browsing", "boundary", "override")
        },
        "dense_enabled": agent.engine.retriever.semantic.enabled,
        "dense_failure_reason": agent.engine.retriever.semantic.failure_reason,
    }
    return snapshots, diagnostics


def fit_pairwise(
    snapshots: list[RankingSnapshot],
    epochs: int = 60,
    learning_rate: float = 0.04,
    l2: float = 0.002,
    seed: int = 2026,
) -> tuple[dict[str, float], int]:
    pairs: list[list[float]] = []
    for snapshot in snapshots:
        for negative in snapshot.negatives:
            pairs.append(
                [snapshot.positive[name] - negative[name] for name in FEATURE_NAMES]
            )
    weights = [0.0] * len(FEATURE_NAMES)
    rng = random.Random(seed)
    for epoch in range(max(0, epochs)):
        rng.shuffle(pairs)
        rate = learning_rate / (1.0 + 0.04 * epoch)
        for difference in pairs:
            margin = max(-30.0, min(30.0, sum(w * x for w, x in zip(weights, difference))))
            correction = 1.0 / (1.0 + math.exp(margin))
            for index, value in enumerate(difference):
                weights[index] += rate * (
                    correction * value - l2 * weights[index]
                )
                weights[index] = max(-12.0, min(12.0, weights[index]))
    return dict(zip(FEATURE_NAMES, weights)), len(pairs)


def train_routes(
    snapshots: list[RankingSnapshot],
    epochs: int,
    learning_rate: float,
    l2: float,
    active_routes: set[str] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    active = active_routes or {"buying", "browsing", "boundary", "override", "default"}
    zero = {name: 0.0 for name in FEATURE_NAMES}
    routes: dict[str, dict[str, float]] = {}
    pair_counts: dict[str, int] = {}
    for route in ("buying", "browsing", "boundary", "override"):
        selected = [snapshot for snapshot in snapshots if snapshot.route == route]
        if route not in active:
            routes[route], pair_counts[route] = dict(zero), 0
            continue
        if not selected:
            continue
        routes[route], pair_counts[route] = fit_pairwise(
            selected,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    if "default" in active:
        routes["default"], pair_counts["default"] = fit_pairwise(
            snapshots,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
    else:
        routes["default"], pair_counts["default"] = dict(zero), 0
    return routes, pair_counts


def model_payload(
    routes: dict[str, dict[str, float]],
    training: dict[str, object],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "routes": routes,
        "training": training,
    }


def write_model(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def cross_validate(
    samples: list[dict],
    snapshots: list[RankingSnapshot],
    catalog_path: str,
    base_config: AgentConfig,
    epochs: int,
    learning_rate: float,
    l2: float,
    active_routes: set[str],
) -> dict[str, object]:
    folds = stratified_folds(samples)
    shared_catalog = CatalogIndex(catalog_path, base_config.index_dir)
    evaluation_data = catalog_index(catalog_path)
    baseline = run_once(
        base_config,
        samples,
        catalog_path,
        shared_catalog,
        evaluation_data,
    )
    fold_ids = [{str(sample["sample_id"]) for sample in fold} for fold in folds]
    baseline_fold_scores = [
        technical_score(
            [
                session
                for session in baseline["sessions"]
                if str(session["sample_id"]) in identifiers
            ]
        )
        for identifiers in fold_ids
    ]
    fold_scores: list[float] = []
    fold_p95_latencies: list[float] = []
    held_out_sessions: list[dict] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for fold_index, fold in enumerate(folds):
            held_out_ids = {str(sample["sample_id"]) for sample in fold}
            training_snapshots = [
                snapshot for snapshot in snapshots if snapshot.sample_id not in held_out_ids
            ]
            routes, pair_counts = train_routes(
                training_snapshots,
                epochs=epochs,
                learning_rate=learning_rate,
                l2=l2,
                active_routes=active_routes,
            )
            ranker_path = root / f"fold-{fold_index}" / "ranker.json"
            write_model(
                ranker_path,
                model_payload(routes, {"pair_counts": pair_counts, "fold": fold_index}),
            )
            config = base_config.with_overrides(
                enable_learned_reranker=True,
                ranker_path=ranker_path,
            )
            metric = run_once(
                config,
                fold,
                catalog_path,
                shared_catalog,
                evaluation_data,
            )
            fold_scores.append(technical_score(list(metric["sessions"])))
            fold_p95_latencies.append(float(metric["diagnostics"]["latency_ms"]["p95"]))
            held_out_sessions.extend(metric["sessions"])
    baseline_score = technical_score(list(baseline["sessions"]))
    candidate_score = technical_score(held_out_sessions)
    gain = candidate_score - baseline_score
    stable_folds = sum(
        candidate + 1e-9 >= reference
        for candidate, reference in zip(fold_scores, baseline_fold_scores)
    )
    scenarios = sorted({str(session["scenario_type"]) for session in baseline["sessions"]})
    baseline_scenarios = {
        scenario: metric_summary(
            [session for session in baseline["sessions"] if session["scenario_type"] == scenario]
        )
        for scenario in scenarios
    }
    candidate_scenarios = {
        scenario: metric_summary(
            [session for session in held_out_sessions if session["scenario_type"] == scenario]
        )
        for scenario in scenarios
    }
    blockers: list[str] = []
    if gain < 0.005:
        blockers.append("out-of-fold TechnicalScore gain is below 0.005")
    if stable_folds < 4:
        blockers.append("fewer than four folds match or beat the baseline")
    if max(fold_p95_latencies, default=0.0) > 750.0:
        blockers.append("observed held-out p95 latency exceeds 750 ms")
    for scenario in scenarios:
        reference = baseline_scenarios[scenario]
        candidate = candidate_scenarios[scenario]
        if float(candidate["hit_rate_at_10"]) < float(reference["hit_rate_at_10"]):
            blockers.append(f"{scenario} Hit Rate regressed")
        if float(candidate["mrr"]) + 0.02 < float(reference["mrr"]):
            blockers.append(f"{scenario} MRR regressed by more than 0.02")
        if float(candidate["mttc"]) > float(reference["mttc"]) + 0.25:
            blockers.append(f"{scenario} MTTC regressed by more than 0.25 turns")
    return {
        "baseline_fold_scores": baseline_fold_scores,
        "fold_scores": fold_scores,
        "mean_fold_score": sum(fold_scores) / len(fold_scores) if fold_scores else 0.0,
        "baseline_technical_score": baseline_score,
        "out_of_fold_technical_score": candidate_score,
        "out_of_fold_gain": round(gain, 6),
        "out_of_fold_session_count": len(held_out_sessions),
        "stable_folds": stable_folds,
        "baseline_scenario_metrics": baseline_scenarios,
        "out_of_fold_scenario_metrics": candidate_scenarios,
        "maximum_fold_p95_latency_ms": max(fold_p95_latencies, default=0.0),
        "promotion": {
            "eligible": not blockers,
            "minimum_gain": 0.005,
            "required_stable_folds": 4,
            "blockers": blockers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Cartographer's dependency-free pairwise residual ranker"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="data/cartographer_index/ranker.json")
    parser.add_argument("--split-manifest", default="docs/public_split_v1.json")
    parser.add_argument(
        "--split",
        choices=("development", "all"),
        default="development",
        help="Train on the locked development split by default; use all only for a final explicit refit",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--negative-limit", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--with-dense", action="store_true")
    parser.add_argument("--semantic-config", default="semantic_default")
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument(
        "--routes",
        default="buying,browsing,boundary,override,default",
        help="Comma-separated routes allowed to learn; omitted routes receive zero weights",
    )
    args = parser.parse_args()

    allowed_routes = {"buying", "browsing", "boundary", "override", "default"}
    active_routes = {value.strip() for value in args.routes.split(",") if value.strip()}
    unknown_routes = active_routes - allowed_routes
    if unknown_routes:
        parser.error(f"unknown route(s): {', '.join(sorted(unknown_routes))}")

    all_samples = load_jsonl(args.dataset)
    split_manifest = load_manifest(args.split_manifest, all_samples, args.dataset)
    samples = select_split(all_samples, split_manifest, args.split)
    if args.limit > 0:
        samples = samples[: args.limit]
    base = AgentConfig(enable_learned_reranker=False)
    if args.with_dense:
        choices = semantic_configs(base)
        if args.semantic_config not in choices or args.semantic_config == "offline_baseline":
            parser.error(f"unknown dense semantic configuration: {args.semantic_config}")
        config = choices[args.semantic_config]
    else:
        config = base.with_overrides(enable_dense=False)
    snapshots, diagnostics = collect_snapshots(
        samples,
        args.catalog,
        config,
        negative_limit=args.negative_limit,
    )
    if args.with_dense and not diagnostics["dense_enabled"]:
        raise RuntimeError(f"Dense training requested but unavailable: {diagnostics['dense_failure_reason']}")
    routes, pair_counts = train_routes(
        snapshots,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        active_routes=active_routes,
    )
    training = {
        **diagnostics,
        "pair_counts": pair_counts,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "semantic_config": args.semantic_config if args.with_dense else None,
        "active_routes": sorted(active_routes),
        "uses_public_labels_only_during_training": True,
        "includes_product_identifiers_as_features": False,
        "data_split": {
            "manifest": args.split_manifest,
            "manifest_sha256": file_sha256(args.split_manifest),
            "name": split_manifest["name"],
            "partition": args.split,
            "partition_sample_count": len(samples),
            "dataset_sha256": split_manifest["source"]["sha256"],
        },
    }
    if args.cross_validate:
        training["cross_validation"] = cross_validate(
            samples,
            snapshots,
            args.catalog,
            config,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            active_routes=active_routes,
        )
    payload = model_payload(routes, training)
    output_path = Path(args.output)
    write_model(output_path, payload)
    print(json.dumps({"output": str(output_path), **payload["training"]}, indent=2))


if __name__ == "__main__":
    main()
