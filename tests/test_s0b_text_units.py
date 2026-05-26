import json
import shutil
import unittest
from argparse import Namespace
from pathlib import Path

from tools.memory_proposal_router import run_router
from tools.step0b.s0b_section_policy_runner import run_section_policy
from tools.step0b.s0b_text_units_from_evidence import build_text_units


ROOT = Path(__file__).resolve().parents[1]


class S0BTextUnitTests(unittest.TestCase):
    def make_workspace(self, name: str, text: str, source_specific_metadata: dict | None = None) -> Path:
        workspace = ROOT / "users" / name
        if workspace.exists():
            shutil.rmtree(workspace)
        (workspace / "raw" / "organization").mkdir(parents=True)
        raw_path = workspace / "raw" / "sample.txt"
        raw_path.write_text(text, encoding="utf-8")
        raw_source = {
            "schema_version": "s0b.raw_source.v0.1",
            "raw_source_id": f"raw:{name}:sample",
            "workspace_id": name,
            "bundle_id": f"bundle:{name}",
            "source_type": "plain_text",
            "modality": "text",
            "local_path": str(raw_path.relative_to(ROOT)),
            "original_format": "txt",
            "organization_degree": "medium",
            "processing_status": "ready_for_s1_intake",
            "inclusion_decision": "include",
            "adapter_recommendation": "generic_text_adapter",
            "source_specific_metadata": source_specific_metadata or {"max_segment_chars": 1200},
        }
        (workspace / "raw" / "organization" / "raw_sources.jsonl").write_text(
            json.dumps(raw_source, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return workspace

    def tearDown(self) -> None:
        for name in (
            "_test_s0b_text_units",
            "_test_s0b_router_text_units",
            "_test_s0b_abbreviation_sentence_split",
            "_test_s0b_numeric_abbreviation_split",
            "_test_s0b_date_prefix_sentence_split",
            "_test_s0b_chinese_sentence_split",
            "_test_s0b_bracket_fragment_warning",
            "_test_s0b_text_units_from_evidence",
            "_test_s0b_base_plaintext_short_opener",
            "_test_s0b_book_plaintext_short_opener",
            "_test_s0b_chinese_plaintext_short_opener",
            "_test_s0b_profile_not_inferred_from_adapter",
            "_test_s0b_explicit_diary_profile",
            "_test_s0b_cli_profile_override",
            "_test_s0b_base_plaintext_explicit_front_matter",
        ):
            workspace = ROOT / "users" / name
            if workspace.exists():
                shutil.rmtree(workspace)

    def test_section_policy_writes_paragraph_and_sentence_text_units(self):
        workspace = self.make_workspace(
            "_test_s0b_text_units",
            "Mira wrote a note. She sent it today.\n\nThe client replied quickly.",
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:text-units",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )

        text_units_path = workspace / "raw" / "organization" / "text_units.jsonl"
        self.assertTrue(text_units_path.exists())
        units = [json.loads(line) for line in text_units_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        paragraphs = [unit for unit in units if unit["unit_type"] == "paragraph"]
        sentences = [unit for unit in units if unit["unit_type"] == "sentence"]

        self.assertGreaterEqual(len(paragraphs), 1)
        self.assertEqual(len(sentences), 3)
        first_paragraph = paragraphs[0]
        first_sentence = sentences[0]
        second_sentence = sentences[1]

        self.assertEqual(first_sentence["parent_text_unit_id"], first_paragraph["text_unit_id"])
        self.assertIn(first_sentence["text_unit_id"], first_paragraph["child_text_unit_ids"])
        self.assertEqual(first_sentence["next_text_unit_id"], second_sentence["text_unit_id"])
        self.assertEqual(second_sentence["previous_text_unit_id"], first_sentence["text_unit_id"])
        self.assertLess(first_sentence["char_start"], first_sentence["char_end"])
        self.assertEqual(first_sentence["raw_backpointer"]["locator"]["kind"], "text_span")
        self.assertEqual(first_sentence["segmentation_method"], "script_rule")

    def test_base_plaintext_does_not_treat_short_opening_paragraph_as_front_matter(self):
        workspace = self.make_workspace(
            "_test_s0b_base_plaintext_short_opener",
            "Quick note.\n\nI lost my job this month and need to update the project plan.",
            {"max_segment_chars": 20},
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:base-plaintext-profile",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertTrue(all(section["text_structure_profile"] == "base_plaintext" for section in sections))
        self.assertEqual([section["final_section_type"] for section in sections], ["body", "body"])
        self.assertTrue(all(section["s1_storage_policy"] == "ordinary_evidence" for section in sections))

    def test_book_plaintext_profile_keeps_short_opening_front_matter_rule(self):
        workspace = self.make_workspace(
            "_test_s0b_book_plaintext_short_opener",
            "A SHORT TITLE\n\nJan. 1, 1660. I went to the office.",
            {"max_segment_chars": 20, "text_structure_profile": "book_plaintext"},
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:book-profile",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(sections[0]["text_structure_profile"], "book_plaintext")
        self.assertEqual(sections[0]["final_section_type"], "front_matter")
        self.assertEqual(sections[1]["final_section_type"], "body")

    def test_plaintext_subprofile_is_not_inferred_from_source_type(self):
        workspace = self.make_workspace(
            "_test_s0b_profile_not_inferred_from_adapter",
            "Short title\n\nI wrote a useful project note today.",
            {"max_segment_chars": 20},
        )
        raw_path = workspace / "raw" / "organization" / "raw_sources.jsonl"
        raw_source = json.loads(raw_path.read_text(encoding="utf-8").strip())
        raw_source["source_type"] = "diary"
        raw_path.write_text(json.dumps(raw_source, ensure_ascii=False) + "\n", encoding="utf-8")

        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:no-adapter-profile-inference",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertTrue(all(section["text_structure_profile"] == "base_plaintext" for section in sections))
        self.assertEqual([section["final_section_type"] for section in sections], ["body", "body"])

    def test_explicit_plaintext_subprofile_is_honored(self):
        workspace = self.make_workspace(
            "_test_s0b_explicit_diary_profile",
            "Short title\n\nI wrote a useful project note today.",
            {"max_segment_chars": 20, "text_structure_profile": "diary_plaintext"},
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:explicit-diary-profile",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertTrue(all(section["text_structure_profile"] == "diary_plaintext" for section in sections))
        self.assertEqual([section["final_section_type"] for section in sections], ["body", "body"])

    def test_cli_text_structure_profile_override_is_honored(self):
        workspace = self.make_workspace(
            "_test_s0b_cli_profile_override",
            "A SHORT TITLE\n\nJan. 1, 1660. I went to the office.",
            {"max_segment_chars": 20},
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:cli-profile-override",
                max_segment_chars=1200,
                text_structure_profile="book_plaintext",
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertTrue(all(section["text_structure_profile"] == "book_plaintext" for section in sections))
        self.assertEqual(sections[0]["final_section_type"], "front_matter")

    def test_base_plaintext_explicit_front_matter_marker_is_not_ordinary_evidence(self):
        workspace = self.make_workspace(
            "_test_s0b_base_plaintext_explicit_front_matter",
            "FIXTURE FRONT MATTER\nThis synthetic fixture is for workflow testing and should not become ordinary evidence.\n\n"
            "Mira Chen wrote that she wants architecture-first reviews before implementation.",
            {"max_segment_chars": 140},
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:base-explicit-front-matter",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        text_units = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "text_units.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(sections[0]["text_structure_profile"], "base_plaintext")
        self.assertEqual(sections[0]["final_section_type"], "front_matter")
        self.assertEqual(sections[0]["s1_storage_policy"], "recoverable_raw_only")
        self.assertEqual(sections[0]["s2_policy"], "blocked_from_portrait")
        self.assertEqual(sections[1]["final_section_type"], "body")
        self.assertTrue(
            all(
                unit["s1_storage_policy"] != "ordinary_evidence"
                for unit in text_units
                if unit["raw_span_id"] == sections[0]["raw_span_id"]
            )
        )

    def test_chinese_base_plaintext_short_opening_reaches_s1_router(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_chinese_plaintext_short_opener",
            "谢谢，先把这个中文路由实验跑通。\n\n我这个月失业了，正在准备开一家舞蹈工作室。",
        )
        sections = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "section_map.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual([section["final_section_type"] for section in sections], ["body"])
        self.assertEqual(sections[0]["text_structure_profile"], "base_plaintext")
        result = run_router(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                output_dir=str(workspace / "routing" / "s1_router"),
                run_id="memory-proposal-router:test:chinese-base-plaintext",
                policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                target_tasks="s1_memory_candidate",
                duplicate_policy="fail",
            )
        )
        decisions = [
            json.loads(line)
            for line in Path(result["route_decisions"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(route["recommended_route"] != "skip_or_background_only" for row in decisions for route in row["task_routes"]))

    def test_router_consumes_sentence_text_units_before_raw_section_spans(self):
        workspace = self.make_workspace(
            "_test_s0b_router_text_units",
            "Mira wrote a note. She sent it today. The client replied quickly.",
        )
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id="s0b-section-policy:test:router-text-units",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        result = run_router(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                output_dir=None,
                run_id="memory-proposal-router:test:text-units",
                policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                target_tasks="s1_memory_candidate",
                duplicate_policy="fail",
            )
        )

        manifest = json.loads(Path(result["route_run_manifest"]).read_text(encoding="utf-8"))
        decisions = [
            json.loads(line)
            for line in Path(result["route_decisions"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(manifest["input_source"], "text_units_sentence_first")
        self.assertEqual(len(decisions), 3)
        self.assertTrue(all(decision["source_layer"] == "s0b_text_unit" for decision in decisions))
        self.assertTrue(all(decision["unit_type"] == "sentence" for decision in decisions))
        self.assertTrue(all(route["recommended_route"] != "split_or_segment_first" for decision in decisions for route in decision["task_routes"]))

    def read_sentence_units(self, workspace: Path) -> list[dict]:
        units = [
            json.loads(line)
            for line in (workspace / "raw" / "organization" / "text_units.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [unit for unit in units if unit["unit_type"] == "sentence"]

    def run_policy_for_text(self, name: str, text: str) -> Path:
        workspace = self.make_workspace(name, text)
        run_section_policy(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                run_id=f"s0b-section-policy:test:{name}",
                max_segment_chars=1200,
                duplicate_policy="fail",
            )
        )
        return workspace

    def test_abbreviation_aware_splitter_keeps_st_john_together(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_abbreviation_sentence_split",
            "Crew told me that my Lord St. John is for a free Parliament. I went home.",
        )
        sentences = self.read_sentence_units(workspace)

        self.assertEqual([sentence["text"] for sentence in sentences], [
            "Crew told me that my Lord St. John is for a free Parliament.",
            "I went home.",
        ])
        self.assertTrue(all("ends_with_non_terminal_abbreviation" not in sentence["warnings"] for sentence in sentences))

    def test_abbreviation_aware_splitter_handles_common_english_abbreviations(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_numeric_abbreviation_split",
            "Mr. Pepys read No. 3 on p. 55. He wrote later. The value was 3.14.",
        )
        sentences = self.read_sentence_units(workspace)

        self.assertEqual([sentence["text"] for sentence in sentences], [
            "Mr. Pepys read No. 3 on p. 55.",
            "He wrote later.",
            "The value was 3.14.",
        ])

    def test_sentence_splitter_keeps_date_prefix_with_sentence(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_date_prefix_sentence_split",
            "2026-04-01. Mira wrote a note.\n\nJan. 1, 1660. I went to the office.",
        )
        sentences = self.read_sentence_units(workspace)

        self.assertEqual([sentence["text"] for sentence in sentences], [
            "2026-04-01. Mira wrote a note.",
            "Jan. 1, 1660. I went to the office.",
        ])

    def test_sentence_splitter_handles_chinese_sentence_endings(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_chinese_sentence_split",
            "他说今天来。她已经走了！还要等吗？",
        )
        sentences = self.read_sentence_units(workspace)

        self.assertEqual([sentence["text"] for sentence in sentences], [
            "他说今天来。",
            "她已经走了！",
            "还要等吗？",
        ])

    def test_bracket_fragments_emit_segmentation_warnings(self):
        workspace = self.run_policy_for_text(
            "_test_s0b_bracket_fragment_warning",
            "] And how that he is quite ashamed of himself. He went home.",
        )
        sentences = self.read_sentence_units(workspace)

        self.assertIn("starts_with_closing_bracket", sentences[0]["warnings"])
        self.assertEqual(sentences[0]["segmentation_confidence"], "low")
        self.assertEqual(sentences[0]["raw_backpointer"]["locator"]["kind"], "text_span")

    def test_text_units_from_evidence_split_turns_and_preserve_dialogue_metadata(self):
        evidence = {
            "schema_version": "step1.evidence_item.v1",
            "evidence_ref": "evidence:locomo:test:D1:7",
            "canonical_evidence_ref": "evidence:locomo:test:D1:7",
            "source_type": "conversation",
            "subject_role": "target",
            "text": 'Thanks! We just did a contemporary piece called "Finding Freedom." It was really emotional.',
            "locator": {
                "kind": "conversation_turn",
                "record_id": "test-conv",
                "session": "session_1",
                "turn_index": 7,
                "speaker": "Gina",
            },
            "metadata": {"modeled_subject_id": "Gina"},
        }

        units = build_text_units([evidence], "_test_s0b_text_units_from_evidence")
        turn = units[0]
        sentences = [unit for unit in units if unit["unit_type"] == "sentence"]

        self.assertEqual(turn["unit_type"], "turn")
        self.assertEqual(turn["speaker"], "Gina")
        self.assertEqual(turn["subject_role"], "target")
        self.assertEqual(turn["target_participant"], "Gina")
        self.assertEqual(turn["evidence_ref"], "evidence:locomo:test:D1:7")
        self.assertEqual([sentence["text"] for sentence in sentences], [
            "Thanks!",
            'We just did a contemporary piece called "Finding Freedom."',
            "It was really emotional.",
        ])
        self.assertEqual(sentences[0]["parent_text_unit_id"], turn["text_unit_id"])
        self.assertEqual(sentences[0]["raw_backpointer"]["locator"]["kind"], "conversation_turn_sentence")
        self.assertEqual(sentences[0]["raw_backpointer"]["locator"]["char_start"], 0)
        self.assertEqual(sentences[0]["raw_backpointer"]["locator"]["char_end"], len("Thanks!"))
        self.assertEqual(sentences[1]["previous_text_unit_id"], sentences[0]["text_unit_id"])
        self.assertEqual(sentences[1]["next_text_unit_id"], sentences[2]["text_unit_id"])
        self.assertTrue(all(sentence["evidence_ref"] == "evidence:locomo:test:D1:7" for sentence in sentences))
        self.assertTrue(all(sentence["speaker"] == "Gina" for sentence in sentences))


if __name__ == "__main__":
    unittest.main()
