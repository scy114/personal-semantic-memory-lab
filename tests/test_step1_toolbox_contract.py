import json
import unittest
from pathlib import Path

from tools.step1.step1_toolbox import LocalStep1Toolbox


FIXTURE = Path(__file__).parent / "fixtures" / "step1_toolbox"


class Step1ToolboxContractTests(unittest.TestCase):
    def setUp(self):
        self.toolbox = LocalStep1Toolbox(FIXTURE)

    def test_fixture_json_parses(self):
        for path in FIXTURE.rglob("*.jsonl"):
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    obj = json.loads(line)
                    self.assertIn("schema_version", obj, f"{path}:{line_no}")
        for path in FIXTURE.rglob("*.json"):
            obj = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("schema_version", obj, str(path))

    def test_store_loads_s0b_minimal_manifest(self):
        self.assertTrue(self.toolbox.store.raw_sources)
        self.assertEqual(self.toolbox.store.raw_bundle["schema_version"], "s0b.bundle.v0.1")
        raw_source = self.toolbox.store.resolve_raw_source("raw:fixture:conv_001")
        self.assertIsNotNone(raw_source)
        self.assertEqual(raw_source["organization_degree"], "high")
        self.assertEqual(raw_source["adapter_recommendation"], "fixture_conversation_adapter")

    def test_retrieve_evidence_defaults_to_raw_evidence(self):
        results = self.toolbox.retrieve_evidence(
            "What is Jon currently working on?",
            filters={"subject_ids": ["Jon"]},
            top_k=5,
            route_policy={"route": "mixed"},
        )
        self.assertTrue(results)
        self.assertTrue(all(result["item_layer"] == "raw_evidence" for result in results))
        self.assertTrue(all("warnings" in result for result in results))
        self.assertTrue(all("missing_raw_source_id" not in result["warnings"] for result in results))
        self.assertTrue(all(result["raw_source_id"] == "raw:fixture:conv_001" for result in results))
        self.assertTrue(all(result["adapter_run_id"] == "adapter_run:fixture:conv_001:v1" for result in results))
        self.assertTrue(all(result["modality"] == "text" for result in results))
        self.assertTrue(all(result["locator"]["kind"] == "conversation_turn" for result in results))

    def test_retrieve_evidence_memory_requires_opt_in(self):
        default_results = self.toolbox.retrieve_evidence(
            "grant proposal",
            filters={},
            top_k=5,
            route_policy={"route": "mixed"},
        )
        self.assertFalse(any(result["item_layer"] == "memory_unit" for result in default_results))

        opt_in_results = self.toolbox.retrieve_evidence(
            "grant proposal",
            filters={"include_memory_units": True},
            top_k=5,
            route_policy={"route": "mixed"},
        )
        memory_results = [result for result in opt_in_results if result["item_layer"] == "memory_unit"]
        self.assertTrue(memory_results)
        self.assertTrue(all("memory_unit_not_raw_evidence" in result["warnings"] for result in memory_results))

    def test_retrieve_memory_units_default_status_filter(self):
        results = self.toolbox.retrieve_memory_units("poster grant Jon", filters={}, top_k=10)
        statuses = {result["status"] for result in results}
        self.assertNotIn("candidate", statuses)
        self.assertIn("accepted_for_experiment", statuses)

        with_candidates = self.toolbox.retrieve_memory_units(
            "poster grant Jon",
            filters={"memory_statuses": ["accepted_for_experiment", "candidate"]},
            top_k=10,
        )
        self.assertIn("candidate", {result["status"] for result in with_candidates})

    def test_resolve_ref_valid_and_unresolved(self):
        resolved = self.toolbox.resolve_ref("evidence:fixture:conv_001:turn_001", "evidence_ref")
        self.assertTrue(resolved["resolved"])
        self.assertEqual(resolved["ref_type"], "evidence_ref")
        self.assertEqual(resolved["metadata"]["raw_source_id"], "raw:fixture:conv_001")
        self.assertEqual(resolved["metadata"]["adapter_run_id"], "adapter_run:fixture:conv_001:v1")
        self.assertEqual(resolved["metadata"]["source_specific_ref"], "turn_001")
        self.assertEqual(resolved["metadata"]["display_ref"], "Fixture conversation turn 001")
        self.assertEqual(resolved["metadata"]["raw_source_organization_degree"], "high")
        self.assertEqual(resolved["resolution_warnings"], [])

        unresolved = self.toolbox.resolve_ref("evidence:fixture:missing", "evidence_ref")
        self.assertFalse(unresolved["resolved"])
        self.assertEqual(unresolved["failure_reason"], "ref_not_found")
        self.assertIn("unresolved_ref", unresolved["resolution_warnings"])

    def test_extension_ref_routes_are_not_core_by_default(self):
        report = self.toolbox.resolve_ref("report:fixture:review_001", "report_ref")
        self.assertFalse(report["resolved"])
        self.assertIn("extension_resolution_route", report["resolution_warnings"])

    def test_check_claim_support_trusts_raw_evidence_by_default(self):
        direct = self.toolbox.check_claim_support(
            "Jon is currently focused on a grant proposal.",
            ["evidence:fixture:conv_001:turn_001"],
        )
        self.assertEqual(direct["support_strength"], "direct")

        unknown = self.toolbox.check_claim_support(
            "Jon is preparing a conference poster.",
            ["memory:fixture:jon_poster_001"],
        )
        self.assertEqual(unknown["support_strength"], "unknown")
        self.assertIn("raw_evidence_ref_required", unknown["missing_refs"])

        contradicts = self.toolbox.check_claim_support(
            "Jon is currently preparing a conference poster.",
            ["evidence:fixture:conv_001:turn_002"],
        )
        self.assertEqual(contradicts["support_strength"], "contradicts")

    def test_grounding_and_handoff_packet(self):
        retrieval_results = self.toolbox.retrieve_evidence("grant proposal Jon", filters={}, top_k=3)
        resolved_refs = [
            self.toolbox.resolve_ref("evidence:fixture:conv_001:turn_001", "evidence_ref")
        ]
        support_checks = [
            self.toolbox.check_claim_support(
                "Jon is currently focused on a grant proposal.",
                ["evidence:fixture:conv_001:turn_001"],
            )
        ]
        grounding_reports = [
            self.toolbox.evaluate_grounding(
                {
                    "target_type": "packet",
                    "target_ref": "test_packet",
                    "claims": ["Jon is currently focused on a grant proposal."],
                },
                ["evidence:fixture:conv_001:turn_001"],
                options={"target_type": "packet"},
            )
        ]

        packet = self.toolbox.adapt_step1_results_for_step2(
            {
                "query_id": "q_fixture_jon_current_work",
                "query_text": "What is Jon currently working on?",
                "route_hint": "mixed",
            },
            {
                "retrieval_results": retrieval_results,
                "resolved_refs": resolved_refs,
                "support_checks": support_checks,
                "grounding_reports": grounding_reports,
            },
            options={"packet_mode": "by_ref"},
        )
        self.assertEqual(packet["packet_mode"], "by_ref")
        self.assertIn("use_policy", packet["retrieval_results"][0])
        self.assertNotIn("text", packet["retrieval_results"][0])

        packet_by_value = self.toolbox.adapt_step1_results_for_step2(
            {
                "query_id": "q_fixture_jon_current_work",
                "query_text": "What is Jon currently working on?",
                "route_hint": "mixed",
            },
            {
                "retrieval_results": retrieval_results,
                "resolved_refs": resolved_refs,
                "support_checks": support_checks,
                "grounding_reports": grounding_reports,
            },
            options={"packet_mode": "by_value"},
        )
        self.assertEqual(packet_by_value["packet_mode"], "by_value")
        self.assertIn("text", packet_by_value["retrieval_results"][0])


if __name__ == "__main__":
    unittest.main()
