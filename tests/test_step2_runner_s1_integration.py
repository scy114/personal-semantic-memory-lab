import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "fixtures" / "public_locomo_conv_30_jon"
RUNNER = ROOT / "tools" / "step2" / "query_runner.py"
S23_RUNNER = ROOT / "tools" / "step2" / "s23_answer_context_runner.py"


class Step2RunnerStep1IntegrationTests(unittest.TestCase):
    def test_runner_uses_step1_toolbox_for_evidence_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cmd = [
                sys.executable,
                str(RUNNER),
                "--workspace",
                str(WORKSPACE),
                "--question",
                "Jon is starting a dance studio",
                "--query-id",
                "s1-integration",
                "--lexical-only",
                "--output",
                str(output),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)

            run_dir = output / "s1-integration"
            checks = [
                json.loads(line)
                for line in (run_dir / "evidence_checks.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(checks)
            self.assertTrue(any(check.get("support_check_source") == "step1_toolbox_rule" for check in checks))
            self.assertTrue(any(check.get("support_strength") == "direct" for check in checks))
            self.assertTrue(any(check.get("support_checked") is True for check in checks))

            packet = json.loads((run_dir / "dynamic_assistance_packet.json").read_text(encoding="utf-8"))
            self.assertEqual(packet["retrieval_methods"]["step1_support_check"], "step1_toolbox_rule")
            self.assertEqual(packet["retrieval_methods"]["step1_toolbox"]["status"], "available")

            dual_package = json.loads((run_dir / "dual_branch_retrieval_package.json").read_text(encoding="utf-8"))
            self.assertIn("step1_evidence_retrieval", dual_package)
            self.assertIn("step2_user_model_retrieval", dual_package)
            self.assertEqual(dual_package["graph_branch"]["graph_status"], "disabled")
            self.assertEqual(dual_package["analysis_layer"]["s3_status"], "not_invoked")
            self.assertEqual(dual_package["support_status"], "not_checked")
            self.assertEqual(packet["dual_branch_retrieval_package_ref"], "dual_branch_retrieval_package.json")
            self.assertIn("dual_branch_summary", packet)
            self.assertNotIn("dual_branch_retrieval_package", packet)

            interpretation = json.loads((run_dir / "memory_context_interpretation.json").read_text(encoding="utf-8"))
            self.assertEqual(interpretation["dual_branch_retrieval_package_ref"], "dual_branch_retrieval_package.json")
            self.assertIn("dual_branch_summary", interpretation)
            self.assertNotIn("dual_branch_retrieval_package", interpretation)

            answer_context = json.loads((run_dir / "s23_answer_context.json").read_text(encoding="utf-8"))
            self.assertEqual(answer_context["mode"], "answer")
            self.assertTrue(answer_context["answer_context"])
            self.assertEqual(answer_context["audit_pointer"]["source_package_path"], "dual_branch_retrieval_package.json")
            for item in answer_context["answer_context"]:
                self.assertIn("text", item)
                self.assertIn("type", item)
                self.assertIn("use_policy", item)
                self.assertIn("support_status", item)
                self.assertIn("evidence_support_status", item)
                self.assertIn("query_support_status", item)
                self.assertNotIn("score_components", item)
                self.assertNotIn("score", item)

            prompt_context = (run_dir / "s23_prompt_context.md").read_text(encoding="utf-8")
            self.assertIn("# Step 2.3 Prompt Context", prompt_context)
            self.assertIn("evidence:checked", prompt_context)
            self.assertIn("query:checked", prompt_context)
            self.assertIn("retrieved_by:", prompt_context)
            self.assertNotIn("audit_pointer", prompt_context)
            self.assertNotIn("audit_summary", prompt_context)
            self.assertNotIn("source_candidate_id", prompt_context)
            self.assertNotIn("schema_version", prompt_context)
            self.assertNotIn("score_components", prompt_context)

    def test_runner_emits_step1_retrieval_and_query_claim_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            cmd = [
                sys.executable,
                str(RUNNER),
                "--workspace",
                str(WORKSPACE),
                "--question",
                "Jon is starting a dance studio",
                "--query-id",
                "s1-retrieval",
                "--lexical-only",
                "--step1-retrieval-mode",
                "direct_scan",
                "--target-subject-id",
                "Jon",
                "--output",
                str(output),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)

            run_dir = output / "s1-retrieval"
            s1_results = [
                json.loads(line)
                for line in (run_dir / "step1_retrieval_results.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(s1_results)
            self.assertEqual(s1_results[0]["retrieval_mode"], "direct_scan")
            self.assertEqual(s1_results[0]["support_status"], "not_checked")

            claim_check = json.loads((run_dir / "query_claim_support_check.json").read_text(encoding="utf-8"))
            self.assertEqual(claim_check["support_status"], "checked")
            self.assertIn(claim_check["support_strength"], {"direct", "partial", "weak", "unknown", "contradicts"})
            self.assertTrue(claim_check["evidence_refs"])

            packet = json.loads((run_dir / "dynamic_assistance_packet.json").read_text(encoding="utf-8"))
            self.assertIn("step1_retrieval_results", packet)
            self.assertIn("query_claim_support_check", packet)
            self.assertEqual(packet["retrieval_methods"]["step1_retrieval"]["retrieval_mode"], "direct_scan")

            dual_package = json.loads((run_dir / "dual_branch_retrieval_package.json").read_text(encoding="utf-8"))
            s1_branch = dual_package["step1_evidence_retrieval"]
            s2_branch = dual_package["step2_user_model_retrieval"]
            self.assertIn("raw_evidence_candidates", s1_branch)
            self.assertTrue(s1_branch["raw_evidence_candidates"])
            self.assertEqual(s1_branch["support_status"], claim_check["support_status"])
            self.assertIn("selected_context_ids", s2_branch)
            self.assertIn("excluded_context_ids", s2_branch)
            self.assertIn("factual_answer", dual_package["recommended_use_policy"])

            rebuilt_path = output / "rebuilt_s23_answer_context.json"
            subprocess.run(
                [
                    sys.executable,
                    str(S23_RUNNER),
                    "--query-dir",
                    str(run_dir),
                    "--output",
                    str(rebuilt_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rebuilt = json.loads(rebuilt_path.read_text(encoding="utf-8"))
            self.assertEqual(rebuilt["schema_version"], "s2.s23_answer_context.v1")
            self.assertTrue(rebuilt["answer_context"])
            rebuilt_prompt = output / "rebuilt_s23_prompt_context.md"
            subprocess.run(
                [
                    sys.executable,
                    str(S23_RUNNER),
                    "--query-dir",
                    str(run_dir),
                    "--output",
                    str(rebuilt_path),
                    "--prompt-output",
                    str(rebuilt_prompt),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            prompt = rebuilt_prompt.read_text(encoding="utf-8")
            self.assertIn("# Step 2.3 Prompt Context", prompt)
            self.assertNotIn("audit_pointer", prompt)


if __name__ == "__main__":
    unittest.main()
