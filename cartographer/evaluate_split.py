from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, load_jsonl

from .catalog import CatalogIndex
from .config import AgentConfig
from .data_split import file_sha256, load_manifest, select_split
from .experiments import run_once, technical_score


def _without_sessions(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "sessions"}


def _scenario_scores(result: dict[str, Any]) -> dict[str, float]:
    sessions = list(result["sessions"])
    scenarios = sorted({str(item["scenario_type"]) for item in sessions})
    return {
        scenario: technical_score(
            [item for item in sessions if str(item["scenario_type"]) == scenario]
        )
        for scenario in scenarios
    }


def compare(
    catalog_path: str,
    dataset_path: str,
    manifest_path: str,
    ranker_path: str,
    partition: str = "holdout",
) -> dict[str, Any]:
    samples = load_jsonl(dataset_path)
    manifest = load_manifest(manifest_path, samples, dataset_path)
    selected = select_split(samples, manifest, partition)
    shared_catalog = CatalogIndex(catalog_path, AgentConfig().index_dir)
    evaluation_data = catalog_index(catalog_path)
    baseline_config = AgentConfig(
        enable_learned_reranker=False,
        enable_dense=False,
        enable_cross_encoder=False,
    )
    candidate_config = baseline_config.with_overrides(
        enable_learned_reranker=True,
        ranker_path=Path(ranker_path),
    )
    baseline = run_once(
        baseline_config,
        selected,
        catalog_path,
        shared_catalog,
        evaluation_data,
    )
    candidate = run_once(
        candidate_config,
        selected,
        catalog_path,
        shared_catalog,
        evaluation_data,
    )
    if not candidate["diagnostics"]["learned_reranker_enabled"]:
        raise RuntimeError(candidate["diagnostics"]["learned_reranker_failure_reason"])
    baseline_scenarios = _scenario_scores(baseline)
    candidate_scenarios = _scenario_scores(candidate)
    return {
        "evaluation_type": "locked public split comparison",
        "partition": partition,
        "sample_count": len(selected),
        "manifest": {
            "path": manifest_path,
            "sha256": file_sha256(manifest_path),
            "name": manifest["name"],
        },
        "ranker": {"path": ranker_path, "sha256": file_sha256(ranker_path)},
        "baseline": _without_sessions(baseline),
        "candidate": _without_sessions(candidate),
        "delta": {
            "technical_score": round(
                float(candidate["recommended_technical_score"])
                - float(baseline["recommended_technical_score"]),
                6,
            ),
            "hit_rate_at_10": round(
                float(candidate["hit_rate_at_10"]) - float(baseline["hit_rate_at_10"]),
                6,
            ),
            "mrr": round(float(candidate["mrr"]) - float(baseline["mrr"]), 6),
            "mttc": round(float(candidate["mttc"]) - float(baseline["mttc"]), 6),
            "scenario_technical_scores": {
                scenario: round(candidate_scenarios[scenario] - baseline_scenarios[scenario], 6)
                for scenario in baseline_scenarios
            },
        },
        "sessions": {
            "baseline": baseline["sessions"],
            "candidate": candidate["sessions"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ranker on a locked split")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--manifest", default="docs/public_split_v1.json")
    parser.add_argument("--partition", choices=("development", "holdout"), default="holdout")
    parser.add_argument("--ranker", default="cartographer/ranker_weights.json")
    parser.add_argument("--output", default="data/cartographer_index/holdout_v1_evaluation.json")
    args = parser.parse_args()
    result = compare(
        args.catalog,
        args.dataset,
        args.manifest,
        args.ranker,
        args.partition,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
