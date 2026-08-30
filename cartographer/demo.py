from __future__ import annotations

import argparse
import json
import uuid

from evaluator.local_evaluator import (
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a traceable Cartographer demo session")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sample-id", default="public_0002")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    sample = next((item for item in samples if item["sample_id"] == args.sample_id), None)
    if sample is None:
        raise SystemExit(f"Unknown sample id: {args.sample_id}")
    _, categories, products = catalog_index(args.catalog)
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(effective, coarse_category(categories[target]), disclosed)

    agent = Agent(args.catalog)
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    print(f"Scenario: {sample['scenario_type']} | Target: {target}")
    for turn in range(1, MAX_TURNS + 1):
        response = agent.respond(session_id, user_message, turn, TOP_K)
        trace = agent.get_trace(session_id)[-1]
        ranked = [item["parent_asin"] for item in response["recommendations"]]
        print(f"\nTurn {turn}")
        print(f"Customer: {user_message}")
        print(
            "State: "
            f"route={trace['route']} epoch={trace['intent_epoch']} "
            f"candidates={trace['candidate_count']} entropy={trace['entropy']:.3f} "
            f"ask={trace['ask_attribute']} gain={trace['information_gain']:.3f} "
            f"latency={trace['latency_ms']:.1f}ms"
        )
        print(f"Cartographer: {response['message']}")
        print("Top 10: " + ", ".join(ranked))
        if override_applied and target in ranked:
            print(f"HIT: target ranked #{ranked.index(target) + 1} on turn {turn}")
            break
        if turn == MAX_TURNS:
            print("MISS: target was not found within ten turns")
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override["message"])
        else:
            user_message, boundary_used = customer_reply(
                effective, response["ask_attribute"], disclosed, boundary_used
            )

    print("\nFinal trace JSON:")
    print(json.dumps(agent.get_trace(session_id), indent=2))


if __name__ == "__main__":
    main()

