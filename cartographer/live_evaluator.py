from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl, metric_summary
from starter.agent import Agent


def source_manifest(root: Path) -> dict[str, Any]:
    paths = [
        root / "starter" / "agent.py",
        *sorted((root / "cartographer").glob("*.py")),
        *sorted((root / "cartographer").glob("*.json")),
    ]
    files: dict[str, str] = {}
    combined = hashlib.sha256()
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {"combined_sha256": combined.hexdigest(), "files": files}


def git_metadata(root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={root.as_posix()}", *arguments],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            return completed.stdout.strip() if completed.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("branch", "--show-current") or None,
        "working_tree_changes": run("status", "--short").splitlines(),
    }


def aggregate_result(
    sessions: list[dict[str, Any]],
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    overall = metric_summary(sessions)
    mttc = float(overall["mttc"]) if overall["mttc"] is not None else 11.0
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    technical_score = (
        0.50 * float(overall["hit_rate_at_10"])
        + 0.30 * float(overall["mrr"])
        + 0.20 * efficiency
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for session in sessions:
        grouped[str(session["scenario_type"])].append(session)
    return {
        **overall,
        "efficiency": round(efficiency, 6),
        "recommended_technical_score": round(technical_score, 6),
        "reported_token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "scenario_metrics": {
            name: metric_summary(grouped[name]) for name in sorted(grouped)
        },
        "sessions": sessions,
    }


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    samples = load_jsonl(args.dataset)
    if args.limit > 0:
        samples = samples[: args.limit]
    identifiers, categories, products = catalog_index(args.catalog)
    manifest = source_manifest(root)
    git = git_metadata(root)
    started = time.perf_counter()
    agent = Agent(args.catalog)
    initialization_seconds = time.perf_counter() - started
    metadata = {
        "kind": "fresh_process_official_evaluator_replay",
        "catalog": str(Path(args.catalog).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "sample_count": len(samples),
        "git": git,
        "source": manifest,
        "initialization_seconds": round(initialization_seconds, 3),
    }
    emit({"event": "start", "metadata": metadata})
    sessions: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    turn_latencies: list[float] = []
    evaluation_started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        engine = getattr(agent, "engine", None)
        traces = getattr(engine, "traces", {})
        before_keys = set(traces) if isinstance(traces, dict) else set()
        partial = evaluate(agent, [sample], identifiers, categories, products)
        session = dict(partial["sessions"][0])
        sessions.append(session)
        usage = partial["reported_token_usage"]
        prompt_tokens += int(usage["prompt_tokens"])
        completion_tokens += int(usage["completion_tokens"])
        traces = getattr(getattr(agent, "engine", None), "traces", {})
        new_keys = set(traces) - before_keys if isinstance(traces, dict) else set()
        for session_id in new_keys:
            turn_latencies.extend(
                float(event.latency_ms)
                for event in traces[session_id]
                if hasattr(event, "latency_ms")
            )
        running = aggregate_result(sessions, prompt_tokens, completion_tokens)
        elapsed = time.perf_counter() - evaluation_started
        rate = elapsed / position
        emit(
            {
                "event": "progress",
                "completed": position,
                "total": len(samples),
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": round(rate * (len(samples) - position), 3),
                "session": session,
                "overall": {
                    key: value for key, value in running.items() if key != "sessions"
                },
            }
        )
    result = aggregate_result(sessions, prompt_tokens, completion_tokens)
    result["live_evaluation"] = {
        **metadata,
        "evaluation_seconds": round(time.perf_counter() - evaluation_started, 3),
        "turn_count": len(turn_latencies),
        "latency_ms": {
            "mean": round(statistics.fmean(turn_latencies), 3) if turn_latencies else 0.0,
            "p95": round(
                sorted(turn_latencies)[
                    max(0, min(len(turn_latencies) - 1, round((len(turn_latencies) - 1) * 0.95)))
                ],
                3,
            )
            if turn_latencies
            else 0.0,
            "max": round(max(turn_latencies), 3) if turn_latencies else 0.0,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    emit(
        {
            "event": "complete",
            "output": str(output.resolve()),
            "result": {key: value for key, value in result.items() if key != "sessions"},
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a fresh-process replay of the unchanged official evaluator"
    )
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
