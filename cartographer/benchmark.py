from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[position]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Cartographer score, latency, and Python memory")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument(
        "--trace-python-memory",
        action="store_true",
        help="Enable tracemalloc; use a separate untraced run for representative latency",
    )
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    identifiers, categories, products = catalog_index(args.catalog)
    if args.trace_python_memory:
        tracemalloc.start()
    started = time.perf_counter()
    agent = Agent(args.catalog)
    initialization_seconds = time.perf_counter() - started
    evaluation_started = time.perf_counter()
    metrics = evaluate(agent, samples, identifiers, categories, products)
    evaluation_seconds = time.perf_counter() - evaluation_started
    peak_bytes: int | None = None
    if args.trace_python_memory:
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    latencies = [event.latency_ms for events in agent.engine.traces.values() for event in events]
    summary = {
        "sample_count": len(samples),
        "initialization_seconds": round(initialization_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "turn_count": len(latencies),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "allocation_tracing_enabled": args.trace_python_memory,
        "python_peak_memory_mib": (
            round(peak_bytes / (1024 * 1024), 3) if peak_bytes is not None else None
        ),
        "metrics": {key: value for key, value in metrics.items() if key != "sessions"},
    }
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
