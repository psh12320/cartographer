from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cartographer.catalog import CatalogIndex
from cartographer.clarification import ClarificationPolicy
from cartographer.config import AgentConfig
from cartographer.dialog import DialogManager
from cartographer.engine import CartographerEngine
from cartographer.models import SearchHit, SessionState
from cartographer.text import canonical, classify_constraint, intent_fingerprint


ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand", "budget",
    "feature", "use_case", "other", None,
}

def product(
    asin: str,
    title: str,
    feature: str,
    material: str = "cotton",
    category: str = "Shirts",
) -> dict:
    return {
        "parent_asin": asin,
        "title": title,
        "features": [material, feature],
        "description": [feature],
        "price": 25.0,
        "categories": ["Clothing, Shoes & Jewelry", "Men", category],
        "details": {"Department": "mens"},
        "average_rating": 4.5,
        "rating_number": 100,
        "store": "Example",
    }


class CartographerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog_path = self.root / "catalog.jsonl"
        rows = [
            product("A", "Trail shirt", "moisture wicking trail layer"),
            product("B", "Office shirt", "wrinkle resistant office layer", "polyester"),
            product("C", "Winter shirt", "thermal brushed lining", "wool"),
            product("D", "Gym shirt", "quick dry gym top", "nylon"),
            product("E", "Formal shirt", "button down formal style", "silk"),
        ]
        self.catalog_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.config = AgentConfig(
            catalog_path=self.catalog_path,
            index_dir=self.root / "index",
            enable_dense=False,
            enable_cross_encoder=False,
            diversify_browsing=False,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fingerprint_uses_visible_fields_and_classifies_attributes(self) -> None:
        card = intent_fingerprint(product("A", "Trail shirt", "quick dry running layer"))
        self.assertIn("cotton", card["hard_constraints"])
        self.assertEqual(classify_constraint("budget around $25"), "budget")
        self.assertEqual(classify_constraint("quick dry running layer"), "use_case")

    def test_dialog_tracks_boundary_and_replaces_overridden_intent(self) -> None:
        manager = DialogManager()
        state = SessionState("session", {"preference_tags": ["material"]})
        manager.update(
            state,
            "I'm looking for Men Shirts. A key requirement is: cotton.",
            1,
        )
        self.assertEqual(state.route, "buying")
        self.assertEqual(state.category, "Men Shirts")
        state.last_asked = "color"
        manager.update(state, "I don't have a preference for color; please use your judgment.", 2)
        self.assertIn("color", state.declined_attributes)
        manager.update(state, "Actually, ignore my earlier preference. What I need is: wool.", 3)
        self.assertEqual(state.intent_epoch, 1)
        self.assertEqual([canonical(item.value) for item in state.active_constraints], ["wool"])

    def test_entropy_policy_prefers_discriminating_feature(self) -> None:
        catalog = CatalogIndex(self.catalog_path, self.root / "index")
        state = SessionState("session", {"preference_tags": ["feature"]}, category="Men Shirts")
        hits = [
            SearchHit(index, catalog.products[index].parent_asin, 1.0)
            for index in range(len(catalog.products))
        ]
        decision = ClarificationPolicy(catalog).choose(state, hits, 1)
        self.assertIn(decision.attribute, {"material", "feature", "use_case"})
        self.assertGreater(decision.information_gain, 0.0)

    def test_engine_contract_and_exact_fingerprint_ranking(self) -> None:
        engine = CartographerEngine(self.catalog_path, self.config)
        engine.reset("session", {"preference_tags": ["feature"], "summary": "feature focused"})
        response = engine.respond(
            "session",
            "I'm looking for Men Shirts. A key requirement is: wrinkle resistant office layer.",
            1,
            10,
        )
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B")
        self.assertEqual(set(response), {"message", "ask_attribute", "recommendations", "usage"})
        self.assertIn(response["ask_attribute"], ALLOWED_ATTRIBUTES)
        self.assertLessEqual(len(response["recommendations"]), 10)
        self.assertEqual(len({item["parent_asin"] for item in response["recommendations"]}), 5)

    def test_boundary_and_empty_message_have_valid_fallbacks(self) -> None:
        engine = CartographerEngine(self.catalog_path, self.config)
        engine.reset("boundary", {"preference_tags": []})
        first = engine.respond("boundary", "", 1, 10)
        self.assertEqual(len(first["recommendations"]), 5)
        asked = first["ask_attribute"]
        second = engine.respond(
            "boundary",
            f"I don't have a preference for {asked}; please use your judgment.",
            2,
            10,
        )
        self.assertIn(asked, engine.sessions["boundary"].declined_attributes)
        self.assertIn(second["ask_attribute"], ALLOWED_ATTRIBUTES)

    def test_repeated_runs_are_deterministic(self) -> None:
        outputs: list[dict] = []
        for suffix in ("one", "two"):
            engine = CartographerEngine(self.catalog_path, self.config)
            engine.reset(suffix, {"preference_tags": ["material"]})
            outputs.append(engine.respond(suffix, "I'm looking for Men Shirts, but I'm still exploring.", 1, 10))
        self.assertEqual(outputs[0], outputs[1])

    def test_runtime_does_not_import_labels_or_evaluator(self) -> None:
        runtime_files = [
            "catalog.py", "clarification.py", "config.py", "dialog.py", "engine.py",
            "models.py", "retrieval.py", "semantic.py", "text.py",
        ]
        package = Path(__file__).resolve().parents[1] / "cartographer"
        source = "\n".join((package / name).read_text(encoding="utf-8") for name in runtime_files)
        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("evaluator.", source)


if __name__ == "__main__":
    unittest.main()
