from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cartographer.catalog import CatalogIndex
from cartographer.clarification import ClarificationPolicy
from cartographer.config import AgentConfig, SearchWeights
from cartographer.dialog import DialogManager
from cartographer.dashboard import DashboardBackend, decision_signals
from cartographer.data_split import build_manifest, select_split, validate_manifest
from cartographer.engine import CartographerEngine
from cartographer.models import SearchHit, SessionState
from cartographer.live_evaluator import aggregate_result
from cartographer.ranker import FEATURE_NAMES, LinearReranker
from cartographer.train_ranker import RankingSnapshot, fit_pairwise
from cartographer.semantic import (
    SemanticRetriever,
    SemanticSearchResult,
    file_sha256,
    verify_bge_embeddings,
)
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
        state.seen_products.add("A")
        manager.update(state, "Actually, ignore my earlier preference. What I need is: wool.", 3)
        self.assertEqual(state.intent_epoch, 1)
        self.assertEqual([canonical(item.value) for item in state.active_constraints], ["wool"])
        self.assertEqual(state.override_shortlist, {"A"})
        self.assertEqual(state.seen_products, set())
        self.assertIn("color", state.declined_attributes)
        self.assertNotIn("cotton", canonical(state.active_query_text))
        self.assertIn("wool", canonical(state.active_query_text))

    def test_override_replaces_initial_preference_and_retains_later_constraints(self) -> None:
        manager = DialogManager()
        state = SessionState("session", {})
        manager.update(state, "I'm looking for shirts. cotton", 1)
        initial_preference = state.replaceable_constraint
        self.assertIsNotNone(initial_preference)
        self.assertEqual(canonical(initial_preference.value), "cotton")
        manager.update(
            state,
            "For that, what matters is: color: blue; budget around $50.",
            2,
        )
        state.asked_attributes.update({"color", "budget"})

        manager.update(
            state,
            "Actually, ignore my earlier preference. What I need is: wool.",
            3,
        )

        active = {canonical(item.value) for item in state.active_constraints}
        self.assertNotIn("cotton", active)
        self.assertFalse(initial_preference.active)
        self.assertIn("color blue", active)
        self.assertIn("budget around 50", active)
        self.assertIn("wool", active)
        self.assertIsNotNone(state.replaceable_constraint)
        self.assertEqual(canonical(state.replaceable_constraint.value), "wool")
        self.assertEqual(state.replaceable_constraint.strength, "hard")
        self.assertEqual(state.category, "shirts")
        self.assertEqual(state.asked_attributes, {"color", "budget"})

        manager.update(
            state,
            "Actually, linen instead.",
            4,
        )
        active = {canonical(item.value) for item in state.active_constraints}
        self.assertNotIn("wool", active)
        self.assertIn("linen", active)
        self.assertIn("color blue", active)
        self.assertIn("budget around 50", active)
        self.assertEqual(state.replaceable_constraint.strength, "hard")

    def test_full_active_query_accumulates_and_resets_by_epoch(self) -> None:
        manager = DialogManager()
        state = SessionState("session", {})
        manager.update(state, "I need a trail shirt for humid weather.", 1)
        manager.update(state, "Breathability and quick drying matter most.", 2)
        self.assertIn("humid weather", state.active_query_text)
        self.assertIn("quick drying", state.active_query_text)
        manager.update(
            state,
            "Actually, ignore my earlier preference. What I need instead is: a formal office shirt.",
            3,
        )
        self.assertNotIn("humid weather", state.active_query_text)
        self.assertIn("formal office shirt", state.active_query_text)
        self.assertEqual(
            [canonical(item.value) for item in state.active_constraints],
            ["a formal office shirt"],
        )

    def test_dialog_handles_paraphrased_category_override_and_boundary(self) -> None:
        manager = DialogManager()
        state = SessionState("session", {})
        manager.update(state, "I'm shopping for trail shoes, but I am still exploring.", 1)
        self.assertEqual(state.category, "trail shoes")
        state.last_asked = "color"
        manager.update(state, "Any color is fine.", 2)
        self.assertIn("color", state.declined_attributes)
        manager.update(state, "I changed my mind. What I need instead is: waterproof boots.", 3)
        self.assertEqual(state.intent_epoch, 1)
        self.assertEqual(state.route, "buying")
        self.assertEqual(state.category, "trail shoes")
        self.assertEqual(
            [canonical(item.value) for item in state.active_constraints],
            ["waterproof boots"],
        )
        self.assertIn("color", state.declined_attributes)

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
        # The default precision-gated depth schedule recommends a single
        # high-confidence product on the first turn of an intent epoch.
        self.assertEqual(len(response["recommendations"]), 1)

    def test_depth_schedule_expands_across_turns_and_full_turn_override(self) -> None:
        config = self.config.with_overrides(
            recommendation_depth_schedule=(1, 2, 10),
            recommendation_depth_full_turn=4,
            uncertain_margin_threshold=0.0,
        )
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("depth", {"preference_tags": []})
        first = engine.respond("depth", "I'm looking for Men Shirts.", 1, 10)
        self.assertEqual(len(first["recommendations"]), 1)
        first_trace = engine.get_trace("depth")[-1]
        self.assertTrue(first_trace["depth_gate_active"])
        self.assertEqual(first_trace["recommendation_depth"], 1)
        self.assertEqual(first_trace["depth_gate_reason"], "informative question")
        second = engine.respond("depth", "I like a breathable layer.", 2, 10)
        self.assertEqual(len(second["recommendations"]), 2)
        third = engine.respond("depth", "Still browsing.", 3, 10)
        self.assertEqual(len(third["recommendations"]), 5)
        fourth = engine.respond("depth", "Anything else?", 4, 10)
        self.assertEqual(len(fourth["recommendations"]), 5)

    def test_depth_gate_yields_full_list_when_no_question_is_informative(self) -> None:
        config = self.config.with_overrides(
            recommendation_depth_schedule=(1, 2, 10),
            depth_gate_min_information_gain=1e9,
            uncertain_margin_threshold=0.0,
        )
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("gate", {"preference_tags": []})
        response = engine.respond("gate", "I'm looking for Men Shirts.", 1, 10)
        self.assertEqual(len({item["parent_asin"] for item in response["recommendations"]}), 5)
        trace = engine.get_trace("gate")[-1]
        self.assertFalse(trace["depth_gate_active"])
        self.assertEqual(trace["depth_gate_reason"], "no sufficiently informative question")

    def test_empty_depth_schedule_returns_full_lists(self) -> None:
        config = self.config.with_overrides(
            recommendation_depth_schedule=(), uncertain_margin_threshold=0.0
        )
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("full", {"preference_tags": []})
        response = engine.respond("full", "I'm looking for Men Shirts.", 1, 10)
        self.assertEqual(len({item["parent_asin"] for item in response["recommendations"]}), 5)
        self.assertEqual(engine.get_trace("full")[-1]["depth_gate_reason"], "disabled")

    def test_profile_and_popularity_can_be_ablated_without_changing_defaults(self) -> None:
        config = self.config.with_overrides(enable_profile=False, enable_popularity=False)
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("ablated", {"preference_tags": ["material"], "summary": "material"})
        engine.respond("ablated", "I'm looking for Men Shirts.", 1, 10)
        self.assertEqual(engine.sessions["ablated"].user_profile, {})
        self.assertTrue(engine.sessions["ablated"].cached_hits)
        self.assertTrue(
            all(hit.profile_score == 0.0 for hit in engine.sessions["ablated"].cached_hits)
        )
        self.assertTrue(
            all(hit.popularity_score == 0.0 for hit in engine.sessions["ablated"].cached_hits)
        )

    def test_boundary_and_empty_message_have_valid_fallbacks(self) -> None:
        engine = CartographerEngine(self.catalog_path, self.config)
        engine.reset("boundary", {"preference_tags": []})
        first = engine.respond("boundary", "", 1, 10)
        self.assertEqual(len(first["recommendations"]), 1)
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
            "models.py", "ranker.py", "retrieval.py", "semantic.py", "text.py",
        ]
        package = Path(__file__).resolve().parents[1] / "cartographer"
        source = "\n".join((package / name).read_text(encoding="utf-8") for name in runtime_files)
        self.assertNotIn("public_set", source)
        self.assertNotIn("ground_truth", source)
        self.assertNotIn("evaluator.", source)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy is not installed")
    def test_embedding_artifact_verification_rejects_catalog_mismatch(self) -> None:
        import numpy as np

        catalog = CatalogIndex(self.catalog_path, self.root / "semantic-index")
        index_dir = self.root / "semantic-index"
        index_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = index_dir / "embeddings.npy"
        np.save(matrix_path, np.ones((len(catalog.products), 384), dtype=np.float32))
        model_path = index_dir / "bge-small-en-v1.5"
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text("{}\n", encoding="utf-8")
        manifest = {
            "format_version": 1,
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_rows": len(catalog.products),
            "embedding_dimensions": 384,
            "embedding_dtype": "float32",
            "normalized": True,
            "catalog_sha256": catalog.catalog_sha256(),
            "asin_order_sha256": catalog.asin_order_sha256(),
            "matrix_sha256": file_sha256(matrix_path),
        }
        manifest_path = index_dir / "embeddings_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(verify_bge_embeddings(catalog, index_dir)["verified"])
        manifest["asin_order_sha256"] = "wrong"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "asin_order_sha256"):
            verify_bge_embeddings(catalog, index_dir)

    def test_semantic_result_scores_all_candidates_and_deduplicates_queries(self) -> None:
        result = SemanticSearchResult(
            hits=[(1, 0.8)],
            calibrated_scores=[0.1, 0.9, 0.4],
            ranks={1: 1},
            floor=0.2,
            ceiling=0.8,
        )
        self.assertEqual(result.score(1), 0.9)
        self.assertEqual(result.score(2), 0.4)
        self.assertEqual(result.rank(1), 1)
        self.assertIsNone(result.rank(2))
        self.assertEqual(
            SemanticRetriever._prepare_queries(
                [("trail running shoes", 0.25), ("trail running shoes", 0.5), ("", 1.0)]
            ),
            [("trail running shoes", 0.75)],
        )

    def test_semantic_query_modes_preserve_structured_and_active_intent(self) -> None:
        engine = CartographerEngine(self.catalog_path, self.config)
        state = SessionState("semantic", {}, category="Men Shirts")
        manager = DialogManager()
        manager.update(state, "For this, breathability matters during humid trail runs.", 1)
        structured = "Men Shirts breathable"
        active = state.active_query_text
        blend = engine.retriever._semantic_queries(state, structured, active)
        self.assertEqual(len(blend), 2)
        compiled_engine = CartographerEngine(
            self.catalog_path,
            self.config.with_overrides(dense_query_mode="compiled"),
        )
        compiled = compiled_engine.retriever._semantic_queries(state, structured, active)
        self.assertEqual(len(compiled), 1)
        self.assertIn("Category: Men Shirts", compiled[0][0])
        self.assertIn("humid trail runs", compiled[0][0])

    def test_dense_scores_apply_to_candidates_outside_dense_top_k(self) -> None:
        config = self.config.with_overrides(
            weights=SearchWeights(
                exact_fingerprint=0.0,
                constraint_coverage=0.0,
                category=0.0,
                bm25=0.0,
                dense=10.0,
                profile=0.0,
                popularity=0.0,
            )
        )
        engine = CartographerEngine(self.catalog_path, config)

        class FakeSemantic:
            enabled = True
            failure_reason = None

            @staticmethod
            def search(queries: object, limit: int) -> SemanticSearchResult:
                del queries, limit
                # D has the strongest full-matrix score but is deliberately absent
                # from the dense top-hit list to exercise candidate-wide scoring.
                return SemanticSearchResult(
                    hits=[(0, 0.8)],
                    calibrated_scores=[0.2, 0.3, 0.4, 1.0, 0.5],
                    ranks={0: 1},
                )

        engine.retriever.semantic = FakeSemantic()  # type: ignore[assignment]
        state = SessionState("dense", {}, category="Men Shirts")
        result = engine.retriever.search(state)
        self.assertEqual(result.hits[0].parent_asin, "D")

    def test_linear_reranker_loads_transparent_route_weights(self) -> None:
        index_dir = self.root / "ranker-index"
        index_dir.mkdir()
        weights = {name: 0.0 for name in FEATURE_NAMES}
        weights["bm25_score"] = 3.0
        (index_dir / "ranker.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "feature_names": list(FEATURE_NAMES),
                    "routes": {"buying": weights, "default": weights},
                }
            ),
            encoding="utf-8",
        )
        ranker = LinearReranker(index_dir / "ranker.json", enabled=True)
        self.assertTrue(ranker.enabled)
        hits = [
            SearchHit(0, "A", 1.0, bm25_score=0.1),
            SearchHit(1, "B", 1.0, bm25_score=0.9),
        ]
        ranked = ranker.rerank(hits, SessionState("ranker", {}, route="buying"))
        self.assertEqual(ranked[0].parent_asin, "B")

    def test_ranker_separates_boundary_from_browsing(self) -> None:
        browsing = SessionState("browsing", {}, route="browsing")
        boundary = SessionState(
            "boundary",
            {},
            route="browsing",
            declined_attributes={"color"},
        )
        from cartographer.ranker import route_key

        self.assertEqual(route_key(browsing), "browsing")
        self.assertEqual(route_key(boundary), "boundary")

    def test_linear_reranker_applies_route_specific_scales(self) -> None:
        index_dir = self.root / "route-scale-index"
        index_dir.mkdir()
        weights = {name: 0.0 for name in FEATURE_NAMES}
        weights["bm25_score"] = 3.0
        (index_dir / "ranker.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "feature_names": list(FEATURE_NAMES),
                    "routes": {"buying": weights, "boundary": weights},
                }
            ),
            encoding="utf-8",
        )
        ranker = LinearReranker(
            index_dir / "ranker.json",
            enabled=True,
            route_scales={"buying": 0.0, "boundary": 1.0},
        )
        buying_hits = [
            SearchHit(0, "A", 1.0, bm25_score=0.1),
            SearchHit(1, "B", 1.0, bm25_score=0.9),
        ]
        boundary_hits = [
            SearchHit(0, "A", 1.0, bm25_score=0.1),
            SearchHit(1, "B", 1.0, bm25_score=0.9),
        ]
        self.assertEqual(
            ranker.rerank(buying_hits, SessionState("buying", {}, route="buying"))[0].parent_asin,
            "A",
        )
        self.assertEqual(
            ranker.rerank(
                boundary_hits,
                SessionState("boundary", {}, route="browsing", declined_attributes={"color"}),
            )[0].parent_asin,
            "B",
        )

    def test_pairwise_trainer_learns_positive_feature_direction(self) -> None:
        zero = {name: 0.0 for name in FEATURE_NAMES}
        positive = dict(zero)
        positive["dense_score"] = 1.0
        snapshots = [
            RankingSnapshot(
                sample_id="sample",
                route="browsing",
                positive=positive,
                negatives=[zero],
            )
        ]
        weights, pair_count = fit_pairwise(snapshots, epochs=20)
        self.assertEqual(pair_count, 1)
        self.assertGreater(weights["dense_score"], 0.0)

    def test_dashboard_replays_evaluator_and_exposes_target_diagnostics(self) -> None:
        dataset_path = self.root / "public_set.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "sample_id": "public_test",
                    "scenario_type": "buying",
                    "difficulty_bucket": "easy",
                    "user_profile": {"preference_tags": ["material"]},
                    "ground_truth": {"parent_asin": "A"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        backend = DashboardBackend(
            self.catalog_path,
            dataset_path,
            self.root / "dashboard-index",
        )
        summary, chat, turns, products, target, profile, inference, decisions = backend.replay(
            "public_test",
            enable_learned_reranker=False,
            enable_dense=False,
            enable_clarification=True,
            diversify_browsing=False,
        )
        self.assertIn("HIT", summary)
        self.assertEqual(target["parent_asin"], "A")
        self.assertEqual(profile["preference_tags"], ["material"])
        self.assertTrue(chat)
        self.assertTrue(turns)
        self.assertTrue(products)
        self.assertEqual(inference["target"], "A")
        self.assertIn("turns", inference)
        self.assertIn("recommendation_depth", inference["turns"][0])
        self.assertIsInstance(products[0][-1], str)
        self.assertTrue(products[0][-1])
        self.assertIn("What this replay suggests", decisions)

        paired = backend.compare_session_components(
            "public_test",
            ("reranker", "gate"),
            enable_dense=False,
        )
        self.assertEqual(len(paired), 16)
        self.assertIn("Paired session ablation", paired[0])
        self.assertTrue(paired[1])
        self.assertTrue(paired[2])
        self.assertTrue(paired[13]["runtime"]["component_configuration"]["gate"])
        self.assertFalse(paired[14]["runtime"]["component_configuration"]["gate"])

        batch_summary, comparison, session_comparison, evidence = backend.compare_components(
            1,
            ("reranker", "gate"),
            enable_dense=False,
        )
        self.assertIn("Component value lab", batch_summary)
        self.assertTrue(comparison)
        self.assertTrue(session_comparison)
        self.assertEqual(evidence["sample_count"], 1)
        self.assertEqual(set(evidence["component_value"]), {"reranker", "gate"})

    def test_dashboard_decision_signals_distinguish_retrieval_and_ranking(self) -> None:
        retrieval = decision_signals(
            [{"target_candidate_position": None, "ask_attribute": "material", "category": "Shirts"}],
            None,
        )
        ranking = decision_signals(
            [{"target_candidate_position": 24, "ask_attribute": "material", "category": "Shirts"}],
            None,
        )
        self.assertIn("Retrieval gap", retrieval[0])
        self.assertIn("Ranking gap", ranking[0])

    def test_live_evaluator_aggregation_matches_official_formula(self) -> None:
        sessions = [
            {
                "sample_id": "hit",
                "scenario_type": "buying",
                "hit": True,
                "first_hit_turn": 1,
                "best_rank": 1,
                "reciprocal_rank": 1.0,
            },
            {
                "sample_id": "miss",
                "scenario_type": "browsing",
                "hit": False,
                "first_hit_turn": None,
                "best_rank": None,
                "reciprocal_rank": 0.0,
            },
        ]
        result = aggregate_result(sessions)
        self.assertEqual(result["hit_rate_at_10"], 0.5)
        self.assertEqual(result["mrr"], 0.5)
        self.assertEqual(result["mttc"], 6.0)
        self.assertEqual(result["efficiency"], 0.5)
        self.assertEqual(result["recommended_technical_score"], 0.5)

    def test_public_split_is_deterministic_stratified_disjoint_and_exhaustive(self) -> None:
        dataset_path = self.root / "split.jsonl"
        samples = [
            {
                "sample_id": f"{scenario}_{index}",
                "scenario_type": scenario,
                "ground_truth": {"parent_asin": f"{scenario}_{index}"},
            }
            for scenario in ("buying", "browsing", "intent_override", "boundary")
            for index in range(4)
        ]
        dataset_path.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        first = build_manifest(samples, dataset_path, seed="test-seed")
        second = build_manifest(samples, dataset_path, seed="test-seed")
        self.assertEqual(first, second)
        validate_manifest(samples, first, dataset_path)
        development = select_split(samples, first, "development")
        holdout = select_split(samples, first, "holdout")
        self.assertEqual(len(development), 8)
        self.assertEqual(len(holdout), 8)
        self.assertEqual(
            {sample["sample_id"] for sample in development}
            & {sample["sample_id"] for sample in holdout},
            set(),
        )
        self.assertEqual(
            Counter(sample["scenario_type"] for sample in development),
            Counter({"buying": 2, "browsing": 2, "intent_override": 2, "boundary": 2}),
        )

    def test_public_split_accepts_only_line_ending_normalization(self) -> None:
        dataset = self.root / "split-newlines.jsonl"
        lf_content = (
            b'{"sample_id":"one","scenario_type":"buying"}\n'
            b'{"sample_id":"two","scenario_type":"buying"}\n'
        )
        dataset.write_bytes(lf_content)
        samples = [
            {"sample_id": "one", "scenario_type": "buying"},
            {"sample_id": "two", "scenario_type": "buying"},
        ]
        manifest = build_manifest(samples, dataset, seed="newline-test")

        dataset.write_bytes(lf_content.replace(b"\n", b"\r\n"))
        validate_manifest(samples, manifest, dataset)

        dataset.write_bytes(lf_content.replace(b'"two"', b'"changed"').replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_manifest(samples, manifest, dataset)

    def test_over_generality_cutoff_truncates_a_broad_request(self) -> None:
        config = self.config.with_overrides(
            recommendation_depth_schedule=(),
            overgenerality_candidate_threshold=2,
            overgenerality_depth=1,
        )
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("broad", {"preference_tags": []})
        response = engine.respond("broad", "I'm looking for Men Shirts.", 1, 10)
        self.assertEqual(len(response["recommendations"]), 1)
        self.assertIn("over-generality", engine.get_trace("broad")[-1]["depth_gate_reason"])

    def test_confidence_gate_narrows_when_the_top_margin_is_thin(self) -> None:
        wide = self.config.with_overrides(
            recommendation_depth_schedule=(), uncertain_margin_threshold=0.0
        )
        narrow = self.config.with_overrides(
            recommendation_depth_schedule=(), uncertain_margin_threshold=1.01
        )
        for config, expected in ((wide, 5), (narrow, 1)):
            engine = CartographerEngine(self.catalog_path, config)
            engine.reset("margin", {"preference_tags": []})
            response = engine.respond("margin", "I'm looking for Men Shirts.", 1, 10)
            self.assertEqual(len(response["recommendations"]), expected)

    def test_profile_memory_distils_and_recalls_across_sessions(self) -> None:
        store = self.root / "profiles.json"
        config = self.config.with_overrides(
            enable_profile_memory=True, profile_memory_path=store
        )
        engine = CartographerEngine(self.catalog_path, config)
        profile = {"preference_tags": ["fit"], "purchase_frequency": "3-4 prior purchases"}
        engine.reset("visit-1", profile)
        engine.respond("visit-1", "I'm looking for Men Shirts. A key requirement is: wool.", 1, 10)

        # A second visit by the same shopper reloads what the first one revealed.
        engine.reset("visit-1", profile)
        remembered = engine.sessions["visit-1"].user_profile
        self.assertEqual(remembered["remembered_sessions"], 1)
        self.assertIn("fit", remembered["preference_tags"])
        self.assertIn("material", remembered["preference_tags"])
        self.assertTrue(store.exists())

        # The record survives a fresh process and never stores product identity.
        payload = json.loads(store.read_text(encoding="utf-8"))
        self.assertEqual(payload["format_version"], 1)
        serialized = json.dumps(payload)
        for asin in ("A", "B", "C", "D", "E"):
            self.assertNotIn(f'"{asin}"', serialized)

    def test_profile_memory_distils_across_distinct_session_identifiers(self) -> None:
        """The official evaluator allocates a fresh identifier per conversation.

        Keying distillation on a repeated `session_id` silently never fires
        under that caller, so this mirrors the evaluator's naming exactly.
        """

        store = self.root / "uuid_profiles.json"
        config = self.config.with_overrides(
            enable_profile_memory=True, profile_memory_path=store
        )
        engine = CartographerEngine(self.catalog_path, config)
        profile = {"preference_tags": ["fit"], "purchase_frequency": "3-4 prior purchases"}
        for index in range(3):
            engine.reset(f"public_{index:032x}", profile)
            engine.respond(
                f"public_{index:032x}",
                "I'm looking for Men Shirts. A key requirement is: wool.",
                1,
                10,
            )
        engine.reset("public_final", profile)
        remembered = engine.sessions["public_final"].user_profile
        self.assertGreaterEqual(remembered["remembered_sessions"], 2)
        self.assertIn("material", remembered["preference_tags"])
        self.assertTrue(store.exists())

    def test_profile_memory_is_off_by_default_and_survives_a_broken_store(self) -> None:
        self.assertFalse(AgentConfig().enable_profile_memory)
        broken = self.root / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        config = self.config.with_overrides(
            enable_profile_memory=True, profile_memory_path=broken
        )
        engine = CartographerEngine(self.catalog_path, config)
        engine.reset("session", {"preference_tags": ["fit"]})
        response = engine.respond("session", "I'm looking for Men Shirts.", 1, 10)
        self.assertTrue(response["recommendations"])
        self.assertIsNotNone(engine.profile_memory.failure_reason)

    def test_frozen_ranker_is_fitted_on_the_declared_public_partition(self) -> None:
        """The shipped artifact must declare exactly the data it was fitted on.

        Selection originally used only the locked 100-session development half so
        the holdout stayed meaningful. Both public halves have since been
        consumed, so the final model is fitted on all 200 labelled sessions and
        generalisation is judged on the disjoint held-out synthetic set instead.
        This test pins that declaration so the provenance can never drift
        silently from what the documentation claims.
        """

        repository = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (repository / "cartographer" / "ranker_weights.json").read_text(encoding="utf-8")
        )
        training = payload["training"]
        split = training["data_split"]
        self.assertEqual(training["sample_count"], 200)
        self.assertEqual(split["partition"], "all")
        self.assertEqual(split["partition_sample_count"], 200)
        self.assertEqual(split["name"], "public-stratified-100-100-v1")
        # Provenance must be self-describing: no labels, no product identity.
        self.assertNotIn("parent_asin", json.dumps(payload["routes"]))


if __name__ == "__main__":
    unittest.main()
