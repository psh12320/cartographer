from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary
from starter.agent import Agent


def run_shard(args: argparse.Namespace) -> None:
    samples = load_jsonl(args.dataset)
    shard = [sample for position, sample in enumerate(samples) if position % args.shard_count == args.shard_index]
    identifiers, categories, products = catalog_index(args.catalog)
    result = evaluate(Agent(args.catalog), shard, identifiers, categories, products)
    result["shard"] = {
        "index": args.shard_index,
        "count": args.shard_count,
        "source_sample_count": len(samples),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


def merge_shards(args: argparse.Namespace) -> None:
    paths = [Path(value) for value in sorted(glob.glob(args.inputs))]
    if not paths:
        raise SystemExit(f"No shard files matched {args.inputs!r}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    shard_counts = {int(payload["shard"]["count"]) for payload in payloads}
    shard_indices = {int(payload["shard"]["index"]) for payload in payloads}
    if len(shard_counts) != 1 or shard_indices != set(range(shard_counts.pop())):
        raise SystemExit("Shard set is incomplete or inconsistent")
    sessions = [session for payload in payloads for session in payload["sessions"]]
    sessions.sort(key=lambda item: item["sample_id"])
    overall = metric_summary(sessions)
    efficiency = max(0.0, min(1.0, (11.0 - float(overall["mttc"])) / 10.0))
    technical_score = 0.50 * overall["hit_rate_at_10"] + 0.30 * overall["mrr"] + 0.20 * efficiency
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    prompt_tokens = sum(payload["reported_token_usage"]["prompt_tokens"] for payload in payloads)
    completion_tokens = sum(payload["reported_token_usage"]["completion_tokens"] for payload in payloads)
    result = {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {name: metric_summary(grouped[name]) for name in sorted(grouped)},
        "sessions": sessions,
        "evaluation_method": {
            "kind": "merged_official_evaluator_shards",
            "shards": len(paths),
            "formula_matches": "evaluator.local_evaluator.evaluate",
        },
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or merge bounded official-evaluator shards")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--catalog", default="data/catalog.jsonl")
    run.add_argument("--dataset", default="data/public_set.jsonl")
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, default=4)
    run.add_argument("--output", required=True)
    run.set_defaults(handler=run_shard)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--inputs", default="evaluation_shards/shard_*.json")
    merge.add_argument("--output", default="results.json")
    merge.set_defaults(handler=merge_shards)
    args = parser.parse_args()
    if args.command == "run" and not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be between zero and shard-count minus one")
    args.handler(args)


if __name__ == "__main__":
    main()
