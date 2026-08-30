from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary
from starter.agent import Agent

from .catalog import CatalogIndex
from .config import AgentConfig, SearchWeights


def stratified_folds(samples: list[dict], count: int = 5) -> list[list[dict]]:
    folds: list[list[dict]] = [[] for _ in range(count)]
    groups: dict[tuple[str, str], list[dict]] = {}
    for sample in samples:
        key = (str(sample.get("scenario_type")), str(sample.get("difficulty_bucket")))
        groups.setdefault(key, []).append(sample)
    for key in sorted(groups):
        for position, sample in enumerate(sorted(groups[key], key=lambda item: item["sample_id"])):
            folds[position % count].append(sample)
    return folds


def run_once(
    config: AgentConfig,
    samples: list[dict],
    catalog_path: str,
    shared_catalog: CatalogIndex | None = None,
    evaluation_data: tuple[set[str], dict[str, list[str]], dict[str, dict]] | None = None,
) -> dict:
    identifiers, categories, products = evaluation_data or catalog_index(catalog_path)
    agent = Agent(catalog_path, config=config, catalog_index=shared_catalog)
    result = evaluate(
        agent,
        samples,
        identifiers,
        categories,
        products,
    )
    latencies = [
        event.latency_ms
        for events in agent.engine.traces.values()
        for event in events
    ]
    result["diagnostics"] = {
        "dense_enabled": agent.engine.retriever.semantic.enabled,
        "dense_failure_reason": agent.engine.retriever.semantic.failure_reason,
        "cross_encoder_enabled": agent.engine.retriever.cross_encoder.enabled,
        "learned_reranker_enabled": agent.engine.retriever.learned_reranker.enabled,
        "learned_reranker_failure_reason": agent.engine.retriever.learned_reranker.failure_reason,
        "turn_count": len(latencies),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }
    return result


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def ablation_configs(base: AgentConfig) -> dict[str, AgentConfig]:
    return {
        "full": base,
        "no_dense": replace(base, enable_dense=False, enable_cross_encoder=False),
        "no_fingerprints": replace(base, enable_fingerprints=False),
        "no_state": replace(base, enable_state=False),
        "no_clarification": replace(base, enable_clarification=False),
        "bm25_only": replace(
            base,
            enable_category=False,
            enable_fingerprints=False,
            enable_dense=False,
            enable_cross_encoder=False,
            enable_state=False,
            enable_clarification=False,
            diversify_browsing=False,
        ),
    }


def weight_presets(base: AgentConfig) -> dict[str, AgentConfig]:
    return {
        "balanced": base,
        "precision": replace(
            base,
            weights=SearchWeights(exact_fingerprint=9.0, constraint_coverage=3.5, category=2.0),
        ),
        "semantic": replace(
            base,
            weights=SearchWeights(
                exact_fingerprint=6.0,
                constraint_coverage=2.5,
                category=1.6,
                bm25=1.8,
                dense=1.8,
            ),
        ),
    }


def sweep_configs(base: AgentConfig) -> dict[str, AgentConfig]:
    """Small, interpretable grid focused on breaking exact-match ties."""

    candidates = {"baseline": base}
    for popularity in (0.0, 0.15, 0.35, 0.75, 1.5):
        candidates[f"pop_{popularity:g}"] = replace(
            base,
            weights=replace(base.weights, popularity=popularity),
        )
    for bm25 in (2.0, 3.0, 4.5):
        candidates[f"bm25_{bm25:g}"] = replace(
            base,
            weights=replace(base.weights, bm25=bm25),
        )
    candidates["lexical_tiebreak"] = replace(
        base,
        weights=replace(
            base.weights,
            exact_fingerprint=6.0,
            constraint_coverage=2.8,
            bm25=3.0,
            popularity=0.15,
        ),
    )
    candidates["generic_question"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
    )
    candidates["generic_bm25_2"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        weights=replace(base.weights, bm25=2.0),
    )
    candidates["generic_bm25_3"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        weights=replace(base.weights, bm25=3.0),
    )
    candidates["generic_pop_0.15"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        weights=replace(base.weights, popularity=0.15),
    )
    candidates["generic_lexical_tiebreak"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        weights=replace(
            base.weights,
            exact_fingerprint=6.0,
            constraint_coverage=2.8,
            bm25=3.0,
            popularity=0.15,
        ),
    )
    candidates["generic_buying"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        clarification_other_routes=("buying",),
    )
    candidates["generic_buying_route_pop"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        clarification_other_routes=("buying",),
        buying_popularity_multiplier=3.0,
    )
    candidates["generic_route_pop"] = replace(
        base,
        clarification_other_start_turn=1,
        clarification_other_multiplier=1.0,
        buying_popularity_multiplier=3.0,
    )
    for multiplier in (0.65, 0.75, 0.85):
        candidates[f"route_pop_other_{multiplier:g}"] = replace(
            base,
            clarification_other_start_turn=1,
            clarification_other_multiplier=multiplier,
            buying_popularity_multiplier=3.0,
        )
    candidates["review_volume_only"] = replace(base, popularity_rating_mix=0.0)
    candidates["review_volume_half_rating"] = replace(base, popularity_rating_mix=0.5)
    for profile_weight in (0.05, 0.10, 0.20, 0.30, 0.50):
        candidates[f"profile_{profile_weight:g}"] = replace(
            base,
            weights=replace(base.weights, profile=profile_weight),
        )
    return candidates


def semantic_configs(base: AgentConfig) -> dict[str, AgentConfig]:
    """Two-stage semantic grid; use --limit for screening and rerun finalists on all 200."""

    offline = replace(base, enable_dense=False, enable_cross_encoder=False)
    semantic = replace(base, enable_dense=True, enable_cross_encoder=False)
    candidates: dict[str, AgentConfig] = {
        "offline_baseline": offline,
        "semantic_default": semantic,
    }
    for dense_weight in (0.25, 0.5, 0.8, 1.5, 2.25, 3.0):
        candidates[f"dense_{dense_weight:g}"] = replace(
            semantic,
            weights=replace(semantic.weights, dense=dense_weight, dense_rank=0.0),
        )
    for rank_weight in (0.25, 0.5, 1.0):
        candidates[f"dense_rank_{rank_weight:g}"] = replace(
            semantic,
            weights=replace(semantic.weights, dense=1.1, dense_rank=rank_weight),
        )
    for agreement_weight in (0.5, 1.0, 2.0):
        candidates[f"dense_constraint_{agreement_weight:g}"] = replace(
            semantic,
            weights=replace(
                semantic.weights,
                dense=0.8,
                dense_constraint_agreement=agreement_weight,
            ),
        )
    for agreement_weight in (0.5, 1.0):
        candidates[f"dense_category_{agreement_weight:g}"] = replace(
            semantic,
            weights=replace(
                semantic.weights,
                dense=0.8,
                dense_category_agreement=agreement_weight,
            ),
        )
    candidates["structured_query"] = replace(
        semantic,
        dense_query_mode="structured",
    )
    candidates["balanced_query"] = replace(
        semantic,
        dense_conversation_weight_buying=0.5,
        dense_conversation_weight_browsing=0.5,
    )
    candidates["conversation_query"] = replace(
        semantic,
        dense_query_mode="conversation",
    )
    candidates["compiled_query"] = replace(semantic, dense_query_mode="compiled")
    candidates["dense_route_balanced"] = replace(
        semantic,
        dense_buying_multiplier=1.0,
        dense_browsing_multiplier=1.0,
    )
    candidates["dense_browsing_heavy"] = replace(
        semantic,
        dense_buying_multiplier=0.40,
        dense_browsing_multiplier=1.60,
    )
    candidates["semantic_no_diversity"] = replace(semantic, diversify_browsing=False)
    candidates["semantic_top600"] = replace(semantic, dense_limit=600)
    for floor in (50.0, 65.0, 85.0):
        candidates[f"dense_floor_{int(floor)}"] = replace(
            semantic,
            dense_calibration_floor_percentile=floor,
        )
    return candidates


def technical_score(sessions: list[dict]) -> float:
    if not sessions:
        return 0.0
    summary = metric_summary(sessions)
    mttc = float(summary["mttc"])
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return (
        0.50 * float(summary["hit_rate_at_10"])
        + 0.30 * float(summary["mrr"])
        + 0.20 * efficiency
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Cartographer ablations or weight tuning")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="experiment_results.json")
    parser.add_argument(
        "--mode",
        choices=("ablation", "tune", "sweep", "semantic"),
        default="ablation",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional quick-run sample limit")
    parser.add_argument(
        "--include",
        default="",
        help="Optional comma-separated configuration names for tune/sweep modes",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    base = AgentConfig(catalog_path=Path(args.catalog))
    results: dict[str, object] = {"mode": args.mode, "sample_count": len(samples), "runs": {}}
    shared_catalog = CatalogIndex(args.catalog, base.index_dir)
    evaluation_data = catalog_index(args.catalog)
    if args.mode == "ablation":
        for name, config in ablation_configs(base).items():
            metric = run_once(config, samples, args.catalog, shared_catalog, evaluation_data)
            results["runs"][name] = {key: value for key, value in metric.items() if key != "sessions"}
    else:
        folds = stratified_folds(samples)
        if args.mode == "tune":
            configurations = weight_presets(base)
        elif args.mode == "semantic":
            configurations = semantic_configs(base)
        else:
            configurations = sweep_configs(base)
        if args.include:
            requested = {value.strip() for value in args.include.split(",") if value.strip()}
            unknown = requested - set(configurations)
            if unknown:
                parser.error(f"unknown configuration(s): {', '.join(sorted(unknown))}")
            configurations = {
                name: config for name, config in configurations.items() if name in requested
            }
        session_results: dict[str, list[dict]] = {}
        fold_ids = [{str(sample["sample_id"]) for sample in fold} for fold in folds]
        for name, config in configurations.items():
            metric = run_once(config, samples, args.catalog, shared_catalog, evaluation_data)
            session_results[name] = list(metric["sessions"])
            fold_scores = [
                technical_score(
                    [item for item in metric["sessions"] if str(item["sample_id"]) in identifiers]
                )
                for identifiers in fold_ids
            ]
            results["runs"][name] = {
                "fold_scores": fold_scores,
                "mean_technical_score": statistics.fmean(fold_scores),
                "full_technical_score": metric["recommended_technical_score"],
                "full_hit_rate_at_10": metric["hit_rate_at_10"],
                "full_mrr": metric["mrr"],
                "full_mttc": metric["mttc"],
                "scenario_metrics": metric["scenario_metrics"],
                "diagnostics": metric["diagnostics"],
            }
        if args.mode == "sweep":
            selected: list[str] = []
            held_out_sessions: list[dict] = []
            all_ids = {str(sample["sample_id"]) for sample in samples}
            for identifiers in fold_ids:
                training_ids = all_ids - identifiers
                winner = max(
                    configurations,
                    key=lambda name: (
                        technical_score(
                            [
                                item
                                for item in session_results[name]
                                if str(item["sample_id"]) in training_ids
                            ]
                        ),
                        name,
                    ),
                )
                selected.append(winner)
                held_out_sessions.extend(
                    item
                    for item in session_results[winner]
                    if str(item["sample_id"]) in identifiers
                )
            results["held_out_selection"] = {
                "selected_configurations": selected,
                "technical_score": technical_score(held_out_sessions),
            }
        elif args.mode == "semantic":
            baseline = results["runs"].get("offline_baseline")
            promotion: dict[str, object] = {
                "reportable": args.limit <= 0 and len(samples) == 200,
                "minimum_technical_gain": 0.01,
                "maximum_p95_latency_ms": 750.0,
                "candidates": {},
                "recommended": None,
            }
            if baseline is None:
                promotion["error"] = "offline_baseline must be included for promotion decisions"
            else:
                eligible: list[str] = []
                baseline_score = float(baseline["full_technical_score"])
                baseline_folds = list(baseline["fold_scores"])
                baseline_scenarios = dict(baseline["scenario_metrics"])
                for name, run in results["runs"].items():
                    if name == "offline_baseline":
                        continue
                    blockers: list[str] = []
                    gain = float(run["full_technical_score"]) - baseline_score
                    diagnostics = dict(run["diagnostics"])
                    if not diagnostics.get("dense_enabled"):
                        blockers.append("semantic artifacts were not loaded")
                    if gain < 0.01:
                        blockers.append("TechnicalScore gain is below 0.01")
                    if float(diagnostics["latency_ms"]["p95"]) > 750.0:
                        blockers.append("p95 latency exceeds 750 ms")
                    stable_folds = sum(
                        float(candidate) + 1e-9 >= float(reference)
                        for candidate, reference in zip(run["fold_scores"], baseline_folds)
                    )
                    if stable_folds < 4:
                        blockers.append("fewer than four folds match or beat the offline baseline")
                    for scenario, reference in baseline_scenarios.items():
                        candidate = run["scenario_metrics"][scenario]
                        if float(candidate["hit_rate_at_10"]) < float(reference["hit_rate_at_10"]):
                            blockers.append(f"{scenario} Hit Rate regressed")
                        if float(candidate["mrr"]) + 0.02 < float(reference["mrr"]):
                            blockers.append(f"{scenario} MRR regressed by more than 0.02")
                        if float(candidate["mttc"]) > float(reference["mttc"]) + 0.25:
                            blockers.append(f"{scenario} MTTC regressed by more than 0.25")
                    promotion["candidates"][name] = {
                        "technical_gain": round(gain, 6),
                        "stable_folds": stable_folds,
                        "eligible": not blockers,
                        "blockers": blockers,
                    }
                    if not blockers:
                        eligible.append(name)
                if eligible:
                    promotion["recommended"] = max(
                        eligible,
                        key=lambda name: float(results["runs"][name]["full_technical_score"]),
                    )
            results["promotion_gate"] = promotion
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
