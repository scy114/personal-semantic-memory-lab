import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.step1.step1_build_runner import (
    build_s1_proposal_index,
    build_s1_route_skip_rows,
    build_s1_reject_index,
    run_build,
    select_s1_processed_proposal,
    select_s1_reject_rows,
)
from tools.step2.step2_build_runner import run_build as run_step2_build
from tools.step0b.s0b_section_policy_runner import run_section_policy


ROOT = Path(__file__).resolve().parents[1]


def deterministic_prebuild_args() -> dict:
    return {
        "pre_build_route_mode": "none",
        "route_policy": None,
        "proposal_provider": "mock",
        "proposal_profile": None,
        "api_mode": "responses",
        "allow_live_api": False,
        "external_model_outputs": None,
        "max_items": None,
        "item_offset": 0,
        "sample_stride": 1,
    }


class Step1BuildRunnerTests(unittest.TestCase):
    def test_s1_sentence_evidence_does_not_match_sibling_proposal_by_raw_span(self):
        proposal_index = build_s1_proposal_index(
            [
                {
                    "output_kind": "memory_candidate",
                    "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
                    "raw_span_id": "raw:generic:notes:text_span:0001",
                    "original_text": "First sentence.",
                    "processed_text": "Processed first sentence.",
                }
            ]
        )
        first = {
            "evidence_ref": "evidence:test:notes:unit-0001",
            "canonical_evidence_ref": "evidence:test:notes:unit-0001",
            "source_specific_ref": "raw:generic:notes:text_span:0001:P0001.S0001",
            "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
            "raw_span_id": "raw:generic:notes:text_span:0001",
            "text": "First sentence.",
            "locator": {"kind": "text_unit_sentence"},
        }
        second = {
            "evidence_ref": "evidence:test:notes:unit-0002",
            "canonical_evidence_ref": "evidence:test:notes:unit-0002",
            "source_specific_ref": "raw:generic:notes:text_span:0001:P0001.S0002",
            "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0002",
            "raw_span_id": "raw:generic:notes:text_span:0001",
            "text": "Second sentence.",
            "locator": {"kind": "text_unit_sentence"},
        }

        self.assertEqual(select_s1_processed_proposal(first, proposal_index)["processed_text"], "Processed first sentence.")
        self.assertIsNone(select_s1_processed_proposal(second, proposal_index))

    def test_s1_skipped_prebuild_proposal_vetoes_script_fallback(self):
        reject_index = build_s1_reject_index(
            [
                {
                    "output_kind": "skipped",
                    "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
                    "raw_span_id": "raw:generic:notes:text_span:0001",
                    "original_text": "2nd.",
                    "warnings": ["s1_low_value_not_materialized"],
                }
            ]
        )
        evidence_item = {
            "evidence_ref": "evidence:test:notes:unit-0001",
            "canonical_evidence_ref": "evidence:test:notes:unit-0001",
            "source_specific_ref": "raw:generic:notes:text_span:0001:P0001.S0001",
            "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
            "raw_span_id": "raw:generic:notes:text_span:0001",
            "text": "2nd.",
            "locator": {"kind": "text_unit_sentence"},
        }

        rows = select_s1_reject_rows(evidence_item, reject_index)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["output_kind"], "skipped")

    def test_s1_route_skip_rows_veto_script_fallback(self):
        route_skip_rows = build_s1_route_skip_rows(
            [
                {
                    "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
                    "span_id": "raw:generic:notes:text_span:0001:P0001.S0001",
                    "raw_span_id": "raw:generic:notes:text_span:0001",
                    "text": "2nd.",
                    "task_routes": [
                        {
                            "target_task": "s1_memory_candidate",
                            "recommended_route": "skip_or_background_only",
                            "route_decision_id": "route:test-skip",
                            "warnings": ["s1_low_value_not_materialized"],
                        }
                    ],
                }
            ]
        )
        reject_index = build_s1_reject_index(route_skip_rows)
        evidence_item = {
            "evidence_ref": "evidence:test:notes:unit-0001",
            "canonical_evidence_ref": "evidence:test:notes:unit-0001",
            "source_specific_ref": "raw:generic:notes:text_span:0001:P0001.S0001",
            "text_unit_id": "raw:generic:notes:text_span:0001:P0001.S0001",
            "raw_span_id": "raw:generic:notes:text_span:0001",
            "text": "2nd.",
            "locator": {"kind": "text_unit_sentence"},
        }

        rows = select_s1_reject_rows(evidence_item, reject_index)

        self.assertEqual(len(route_skip_rows), 1)
        self.assertEqual(rows[0]["output_kind"], "skipped")
        self.assertIn("prebuild_route_skip_not_materialized", rows[0]["warnings"])

    def test_s0b_section_policy_filters_generic_text_before_s1_and_s2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "generic_text_section_fixture"
            raw_dir = fixture / "raw"
            org_dir = raw_dir / "organization"
            org_dir.mkdir(parents=True)
            source_path = raw_dir / "mixed_text.txt"
            source_path.write_text(
                "PROJECT GUTENBERG LICENSE AND FRONT MATTER\n"
                "This license text should not become ordinary evidence.\n\n"
                "2026-04-01. Samuel Pepys wrote that dated body spans should not be treated as front matter.\n\n"
                "Jan. 1, 1660. This morning I went to the office and worked on Navy accounts.\n\n"
                "[1] Editorial footnote: this note explains an editor's convention.",
                encoding="utf-8",
            )
            raw_source = {
                "schema_version": "s0b.raw_source.v0.1",
                "raw_source_id": "raw:generic:mixed_text",
                "workspace_id": "generic_text_section_fixture",
                "bundle_id": "bundle:generic:mixed_text",
                "source_type": "public_domain_text",
                "modality": "text",
                "local_path": str(source_path),
                "original_uri_or_path": str(source_path),
                "original_format": "txt",
                "content_hash": "test-hash-section-policy",
                "privacy_class": "public_dataset",
                "organization_degree": "medium",
                "processing_status": "ready_for_s1_intake",
                "inclusion_decision": "include",
                "adapter_recommendation": "generic_text_adapter",
                "coverage_notes": "Mixed front matter, body, and footnote fixture.",
                "perspective_notes": "Author body plus editor/front matter.",
                "quality_notes": "Plain text with deliberate non-body sections.",
                "ingested_at": "2026-05-20T00:00:00+08:00",
                "time_source": "user_provided",
                "time_confidence": "medium",
                "source_specific_metadata": {
                    "record_id": "mixed_text",
                    "title": "Mixed text",
                    "author_subject_id": "Samuel Pepys",
                    "modeled_subject_is_author": True,
                    "max_segment_chars": 80,
                },
            }
            (org_dir / "raw_sources.jsonl").write_text(json.dumps(raw_source, ensure_ascii=False) + "\n", encoding="utf-8")
            (org_dir / "bundle.json").write_text(
                json.dumps(
                    {
                        "schema_version": "s0b.bundle.v0.1",
                        "bundle_id": "bundle:generic:mixed_text",
                        "workspace_id": "generic_text_section_fixture",
                        "modeled_subject_id": "Samuel Pepys",
                        "raw_root": str(raw_dir),
                        "source_count": 1,
                        "organization_degree": "medium",
                        "processing_status": "ready_for_s1_intake",
                        "default_adapter_recommendation": "generic_text_adapter",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            section_result = run_section_policy(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(fixture),
                    run_id="s0b-section:test:mixed-text",
                    max_segment_chars=80,
                    duplicate_policy="fail",
                )
            )
            self.assertEqual(section_result["counts"]["sections"], 4)

            section_rows = [
                json.loads(line)
                for line in (org_dir / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            range_rows = [
                json.loads(line)
                for line in (org_dir / "source_range_map.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            queue_rows = [
                json.loads(line)
                for line in (org_dir / "llm_assist_queue.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            section_types = {row["final_section_type"] for row in section_rows}
            self.assertIn("license", section_types)
            self.assertIn("body", section_types)
            self.assertIn("footnote", section_types)
            self.assertTrue(all("raw_text_hash" in row for row in section_rows))
            self.assertTrue(all(row["raw_backpointer"]["locator"]["kind"] == "text_span" for row in section_rows))
            self.assertTrue(range_rows[0]["selected_ranges"])
            self.assertTrue(range_rows[0]["excluded_ranges"])
            self.assertTrue(queue_rows)
            self.assertTrue(all(row["status"] == "queued" for row in queue_rows))

            output_workspace = Path(temp_dir) / "generic_text_section_s1"
            build_result = run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(fixture),
                    output_workspace=str(output_workspace),
                    workspace_id="generic_text_section_s1",
                    modeled_subject_id="Samuel Pepys",
                    run_id="s1-build:test:section-policy",
                    run_scope="evidence_plus_memory",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            evidence_rows = [
                json.loads(line)
                for line in Path(build_result["evidence"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            units = [
                json.loads(line)
                for line in (output_workspace / "memory" / "memory_units.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(evidence_rows), 2)
            self.assertEqual(evidence_rows[0]["section_type"], "body")
            self.assertEqual(evidence_rows[1]["section_type"], "body")
            self.assertEqual(evidence_rows[0]["s1_storage_policy"], "ordinary_evidence")
            self.assertNotIn("PROJECT GUTENBERG", evidence_rows[0]["text"])
            self.assertNotIn("Editorial footnote", evidence_rows[0]["text"])
            self.assertEqual(len(units), 2)
            self.assertIn("wrote in", units[0]["content"])
            self.assertEqual(units[0]["s2_policy"], "candidate_allowed")

            s2_result = run_step2_build(
                Namespace(
                    workspace=str(output_workspace),
                    workspace_id="generic_text_section_s1",
                    modeled_user_id="Samuel Pepys",
                    target_participant="Samuel Pepys",
                    run_id="s2-build:test:section-policy",
                    max_units=10,
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )
            reviewed = [
                json.loads(line)
                for line in (output_workspace / "portrait" / "reviewed_units.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(s2_result["counts"]["reviewed_units"], 2)
            self.assertEqual(len(reviewed), 2)
            self.assertNotIn("PROJECT GUTENBERG", reviewed[0]["content"])
            self.assertNotIn("Editorial footnote", reviewed[0]["content"])

    def test_generic_text_adapter_builds_text_span_evidence_and_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "generic_text_fixture"
            raw_dir = fixture / "raw"
            org_dir = raw_dir / "organization"
            org_dir.mkdir(parents=True)
            source_path = raw_dir / "pepys_excerpt.txt"
            source_path.write_text(
                "This morning I went to the office and then discussed Navy business.\n\n"
                "In the afternoon I considered accounts and noted that careful records are necessary.\n\n"
                "At night I returned home, tired but resolved to continue the work.",
                encoding="utf-8",
            )
            raw_source = {
                "schema_version": "s0b.raw_source.v0.1",
                "raw_source_id": "raw:generic:pepys_excerpt",
                "workspace_id": "generic_text_fixture",
                "bundle_id": "bundle:generic:pepys_excerpt",
                "source_type": "public_domain_text",
                "modality": "text",
                "local_path": str(source_path),
                "original_uri_or_path": str(source_path),
                "original_format": "txt",
                "content_hash": "test-hash-generic-text",
                "privacy_class": "public_dataset",
                "organization_degree": "medium",
                "processing_status": "ready_for_s1_intake",
                "inclusion_decision": "include",
                "adapter_recommendation": "generic_text_adapter",
                "coverage_notes": "Tiny plain-text fixture excerpt.",
                "perspective_notes": "First-person diary-like public-domain text.",
                "quality_notes": "Plain text, not conversation-shaped.",
                "source_specific_metadata": {
                    "record_id": "pepys_excerpt",
                    "title": "Pepys excerpt",
                    "author_subject_id": "Samuel Pepys",
                    "modeled_subject_is_author": True,
                    "max_segment_chars": 80,
                },
            }
            (org_dir / "raw_sources.jsonl").write_text(json.dumps(raw_source, ensure_ascii=False) + "\n", encoding="utf-8")
            (org_dir / "bundle.json").write_text(
                json.dumps(
                    {
                        "schema_version": "s0b.bundle.v0.1",
                        "bundle_id": "bundle:generic:pepys_excerpt",
                        "workspace_id": "generic_text_fixture",
                        "modeled_subject_id": "Samuel Pepys",
                        "raw_root": str(raw_dir),
                        "source_count": 1,
                        "organization_degree": "medium",
                        "processing_status": "ready_for_s1_intake",
                        "default_adapter_recommendation": "generic_text_adapter",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            output_workspace = Path(temp_dir) / "generic_text_s1_build"
            result = run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(fixture),
                    output_workspace=str(output_workspace),
                    workspace_id="generic_text_s1_build",
                    modeled_subject_id="Samuel Pepys",
                    run_id="s1-build:test:generic-text:evidence-plus-memory",
                    run_scope="evidence_plus_memory",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            evidence_rows = [
                json.loads(line)
                for line in Path(result["evidence"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            units = [
                json.loads(line)
                for line in (output_workspace / "memory" / "memory_units.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertGreaterEqual(len(evidence_rows), 2)
            self.assertEqual(len(units), len(evidence_rows))
            self.assertTrue(all(row["locator"]["kind"] == "text_span" for row in evidence_rows))
            self.assertTrue(all("D1:" not in row["evidence_ref"] for row in evidence_rows))
            self.assertTrue(all(row["extraction_method"] == "generic_text_adapter" for row in evidence_rows))
            self.assertTrue(all(row["subject_role"] == "target" for row in evidence_rows))
            self.assertTrue(all(unit["backpointer_refs"] == unit["evidence_refs"] for unit in units))

    def test_fixture_evidence_only_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_workspace = Path(temp_dir) / "fixture_s1_build"
            result = run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(output_workspace),
                    workspace_id="fixture_s1_build",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:fixture:evidence-only",
                    run_scope="evidence_only",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            evidence_path = Path(result["evidence"])
            manifest_path = Path(result["build_manifest"])
            source_manifest_path = Path(result["source_manifest"])

            evidence_rows = [
                json.loads(line)
                for line in evidence_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source_manifest_rows = [
                json.loads(line)
                for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(evidence_rows), 2)
            self.assertEqual(len(source_manifest_rows), 1)
            self.assertEqual(manifest["run_scope"], "evidence_only")
            self.assertTrue(manifest["validation_summary"]["valid"])
            self.assertTrue(all(row["raw_source_id"] == "raw:fixture:conv_001" for row in evidence_rows))
            self.assertTrue(all(row["adapter_run_id"] for row in evidence_rows))
            self.assertTrue(all(row.get("locator") for row in evidence_rows))
            self.assertFalse((output_workspace / "memory").exists())
            self.assertFalse((output_workspace / "portrait").exists())

    def test_fixture_evidence_plus_memory_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_workspace = Path(temp_dir) / "fixture_s1_build_memory"
            result = run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(output_workspace),
                    workspace_id="fixture_s1_build_memory",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:fixture:evidence-plus-memory",
                    run_scope="evidence_plus_memory",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            manifest = json.loads(Path(result["build_manifest"]).read_text(encoding="utf-8"))
            decisions_path = output_workspace / "memory" / "preprocessing_decisions.jsonl"
            candidates_path = output_workspace / "memory" / "memory_candidates.jsonl"
            units_path = output_workspace / "memory" / "memory_units.jsonl"
            memory_manifest_path = output_workspace / "memory" / "memory_build_manifest.json"

            decisions = [
                json.loads(line)
                for line in decisions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            candidates = [
                json.loads(line)
                for line in candidates_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            units = [
                json.loads(line)
                for line in units_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            memory_manifest = json.loads(memory_manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["run_scope"], "evidence_plus_memory")
            self.assertEqual(len(decisions), 2)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(len(units), 2)
            self.assertTrue(memory_manifest["validation_summary"]["valid"])
            self.assertTrue(all(unit["status"] == "accepted_for_experiment" for unit in units))
            self.assertTrue(all(unit["generation_method"] == "script_generated" for unit in units))
            self.assertTrue(all(unit["backpointer_refs"] == unit["evidence_refs"] for unit in units))
            self.assertFalse((output_workspace / "portrait").exists())

    def test_fixture_full_s1_build(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_workspace = Path(temp_dir) / "fixture_s1_full_build"
            result = run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(output_workspace),
                    workspace_id="fixture_s1_full_build",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:fixture:full",
                    run_scope="full_s1_build",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            manifest = json.loads(Path(result["build_manifest"]).read_text(encoding="utf-8"))
            summaries_path = output_workspace / "memory" / "summaries.jsonl"
            summaries = [
                json.loads(line)
                for line in summaries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(manifest["run_scope"], "full_s1_build")
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["item_layer"], "doc_level_summary")
            self.assertEqual(summaries[0]["summary_type"], "session_summary")
            self.assertTrue(summaries[0]["backpointer_refs"])
            self.assertEqual(
                summaries[0]["generated_from_evidence_count"],
                len(summaries[0]["evidence_refs"]),
            )
            self.assertTrue(manifest["validation_summary"]["summary_validation"]["valid"])
            self.assertFalse((output_workspace / "portrait").exists())

    def test_memory_ids_are_based_on_canonical_evidence_ref_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_workspace = Path(temp_dir) / "fixture_s1_build_hashed_ids"
            run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(output_workspace),
                    workspace_id="fixture_s1_build_hashed_ids",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:fixture:hashed-ids",
                    run_scope="evidence_plus_memory",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            decisions = [
                json.loads(line)
                for line in (output_workspace / "memory" / "preprocessing_decisions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            candidates = [
                json.loads(line)
                for line in (output_workspace / "memory" / "memory_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            units = [
                json.loads(line)
                for line in (output_workspace / "memory" / "memory_units.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

            self.assertTrue(all(row["decision_id"].startswith("prep:") for row in decisions))
            self.assertTrue(all(row["candidate_id"].startswith("candidate:") for row in candidates))
            self.assertTrue(all(row["memory_id"].startswith("memory:") for row in units))
            combined_ids = " ".join(
                [*(row["decision_id"] for row in decisions), *(row["candidate_id"] for row in candidates), *(row["memory_id"] for row in units)]
            )
            self.assertNotIn("turn_001", combined_ids)
            self.assertNotIn("turn_002", combined_ids)

    def test_run_scope_downgrade_fails_when_higher_scope_assets_are_stale(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_workspace = Path(temp_dir) / "fixture_s1_scope_downgrade"
            run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(output_workspace),
                    workspace_id="fixture_s1_scope_downgrade",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:fixture:full-before-downgrade",
                    run_scope="full_s1_build",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            with self.assertRaisesRegex(FileExistsError, "Existing S1 build assets"):
                run_build(
                    Namespace(
                        project_root=str(ROOT),
                        workspace="tests/fixtures/step1_toolbox",
                        output_workspace=str(output_workspace),
                        workspace_id="fixture_s1_scope_downgrade",
                        modeled_subject_id="Jon",
                        run_id="s1-build:test:fixture:evidence-only-downgrade",
                        run_scope="evidence_only",
                        duplicate_policy="fail",
                        **deterministic_prebuild_args(),
                    )
                )


if __name__ == "__main__":
    unittest.main()
