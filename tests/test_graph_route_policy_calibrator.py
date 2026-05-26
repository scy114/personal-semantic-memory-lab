import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_route_policy_calibrator import calibrate_graph_route_policy


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class GraphRoutePolicyCalibratorTests(unittest.TestCase):
    def test_calibrates_policy_from_available_proxy_datasets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            dataset_root = project_root / "external_references" / "graph_route_calibration_datasets"
            output_dir = project_root / "calibration"
            policy_output = project_root / "configs" / "routing" / "graph_package_router" / "graph_route_policy.v0.3.candidate.json"

            carb_dev = dataset_root / "CaRB" / "data" / "gold" / "dev.tsv"
            carb_dev.parent.mkdir(parents=True, exist_ok=True)
            carb_dev.write_text(
                "Mira uses a graph notebook.\tuses\tMira\tgraph notebook\n",
                encoding="utf-8",
            )
            carb_test = dataset_root / "CaRB" / "data" / "gold" / "test.tsv"
            carb_test.write_text(
                "Jon works on routing calibration.\tworks on\tJon\trouting calibration\n",
                encoding="utf-8",
            )
            dialog_row = [
                ["Speaker 1: I changed the graph route.", "Speaker 2: Why did it affect retrieval?"],
                [{"x": "Speaker 1", "y": "graph route", "r": ["project:updates"], "rid": [1]}],
            ]
            write_json(dataset_root / "DialogRE" / "data_v2" / "en" / "data" / "train.json", [dialog_row])
            write_json(dataset_root / "DialogRE" / "data_v2" / "en" / "data" / "dev.json", [dialog_row])
            write_json(dataset_root / "DialogRE" / "data_v2" / "en" / "data" / "test.json", [dialog_row])
            write_json(dataset_root / "DialogRE" / "data_v2" / "cn" / "data" / "train.json", [dialog_row])
            write_json(dataset_root / "DialogRE" / "data_v2" / "cn" / "data" / "dev.json", [dialog_row])
            write_json(dataset_root / "DialogRE" / "data_v2" / "cn" / "data" / "test.json", [dialog_row])
            duie = dataset_root / "DuIE-mirror-Bert-In-Relation-Extraction" / "duie_dev.json"
            duie.parent.mkdir(parents=True, exist_ok=True)
            duie.write_text(
                json.dumps(
                    {
                        "text": "项目使用图路由。",
                        "spo_list": [{"subject": "项目", "predicate": "使用", "object": {"@value": "图路由"}}],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (duie.parent / "train.json").write_text(duie.read_text(encoding="utf-8"), encoding="utf-8")
            retacred_root = dataset_root / "Re-TACRED" / "Re-TACRED"
            retacred_root.mkdir(parents=True, exist_ok=True)
            write_json(retacred_root / "train_id2label.json", {"a": "no_relation", "b": "per:title"})

            manifest = calibrate_graph_route_policy(
                project_root,
                dataset_root=dataset_root,
                output_dir=output_dir,
                policy_output=policy_output,
                limit_per_dataset=6,
            )

            self.assertTrue(policy_output.exists())
            self.assertTrue((output_dir / "graph_route_calibration_rows.jsonl").exists())
            self.assertTrue((output_dir / "graph_route_calibration_rows.csv").exists())
            self.assertEqual(manifest["schema_version"], "graph_v03.route_policy_calibration.v0.1")
            self.assertGreater(manifest["counts"]["training_example_count"], 0)
            policy = json.loads(policy_output.read_text(encoding="utf-8"))
            self.assertEqual(policy["target_task"], "graph_relation_candidate")
            self.assertFalse(policy["guardrails"]["llm_calls_executed"])


if __name__ == "__main__":
    unittest.main()
