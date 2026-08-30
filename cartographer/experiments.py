from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

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


def run_once(config: AgentConfig, samples: list[dict], catalog_path: str) -> dict:
    identifiers, categories, products = catalog_index(catalog_path)
    return evaluate(Agent(catalog_path, config=config), samples, identifiers, categories, products)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Cartographer ablations or weight tuning")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="experiment_results.json")
    parser.add_argument("--mode", choices=("ablation", "tune"), default="ablation")
    parser.add_argument("--limit", type=int, default=0, help="Optional quick-run sample limit")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    base = AgentConfig(catalog_path=Path(args.catalog))
    results: dict[str, object] = {"mode": args.mode, "sample_count": len(samples), "runs": {}}
    if args.mode == "ablation":
        for name, config in ablation_configs(base).items():
            metric = run_once(config, samples, args.catalog)
            results["runs"][name] = {key: value for key, value in metric.items() if key != "sessions"}
    else:
        folds = stratified_folds(samples)
        for name, config in weight_presets(base).items():
            fold_scores: list[float] = []
            for fold in folds:
                metric = run_once(config, fold, args.catalog)
                fold_scores.append(float(metric["recommended_technical_score"]))
            results["runs"][name] = {
                "fold_scores": fold_scores,
                "mean_technical_score": statistics.fmean(fold_scores),
            }
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

