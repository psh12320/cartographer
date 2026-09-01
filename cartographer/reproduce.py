"""Reproduce the reported TechnicalScore with the unmodified official evaluator.

Runs the submitted agent over every labelled session we have -- the 200 public
sessions plus the 800-session held-out synthetic set -- and prints the three
splits separately, because they do not mean the same thing:

* ``public-200`` is in-sample. The shipped reranker is fitted on exactly these
  sessions, so this number is optimistic and is reported for completeness only.
* ``synthetic-800`` is out-of-sample. It shares no target product and no sample
  identifier with the public set, so this is the honest generalisation figure.
* ``all-1000`` is the headline reported in the README and project description.

Usage:
    python3 -m cartographer.reproduce
    python3 -m cartographer.reproduce --output reproduction.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent

from .catalog import CatalogIndex
from .config import AgentConfig


def summarise(sessions: list[dict]) -> dict[str, float]:
    """Recompute the official metrics over an arbitrary subset of sessions."""

    if not sessions:
        return {}
    hit_rate = sum(1 for item in sessions if item["hit"]) / len(sessions)
    mrr = statistics.fmean(item["reciprocal_rank"] for item in sessions)
    mttc = statistics.fmean((item["first_hit_turn"] or 11) for item in sessions)
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sessions": len(sessions),
        "technical_score": round(0.50 * hit_rate + 0.30 * mrr + 0.20 * efficiency, 6),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public", default="data/public_set.jsonl")
    parser.add_argument("--synthetic", default="synthetic_800_v1.jsonl")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    public = load_jsonl(args.public)
    synthetic = load_jsonl(args.synthetic) if Path(args.synthetic).exists() else []
    if not synthetic:
        print(f"[reproduce] {args.synthetic} not found; scoring the public set only")
    combined = [*public, *synthetic]

    identifiers, categories, products = catalog_index(args.catalog)
    config = AgentConfig(catalog_path=Path(args.catalog))
    agent = Agent(args.catalog, config=config, catalog_index=CatalogIndex(args.catalog, config.index_dir))
    print(f"[reproduce] scoring {len(combined)} sessions with the unmodified evaluator", flush=True)
    result = evaluate(agent, combined, identifiers, categories, products)

    public_ids = {str(sample["sample_id"]) for sample in public}
    sessions = result["sessions"]
    report = {
        "all_1000": summarise(sessions),
        "public_200_in_sample": summarise(
            [s for s in sessions if str(s["sample_id"]) in public_ids]
        ),
        "synthetic_800_held_out": summarise(
            [s for s in sessions if str(s["sample_id"]) not in public_ids]
        ),
        "scenario_metrics": result["scenario_metrics"],
        "reported_token_usage": result["reported_token_usage"],
    }

    print()
    print(f"{'split':<26}{'sessions':>9}{'score':>11}{'HR@10':>9}{'MRR':>10}{'MTTC':>8}")
    for name in ("all_1000", "public_200_in_sample", "synthetic_800_held_out"):
        m = report[name]
        if m:
            print(f"{name:<26}{m['sessions']:>9}{m['technical_score']:>11.6f}"
                  f"{m['hit_rate_at_10']:>9.4f}{m['mrr']:>10.6f}{m['mttc']:>8.4f}")
    print()
    for scenario, metrics in sorted(report["scenario_metrics"].items()):
        print(f"  {scenario:<18}n={metrics['sample_count']:<5}"
              f"HR={metrics['hit_rate_at_10']:.3f}  MRR={metrics['mrr']:.4f}  "
              f"MTTC={metrics['mttc']:.3f}")
    print()
    print(f"reported tokens: {report['reported_token_usage']['total_tokens']}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
