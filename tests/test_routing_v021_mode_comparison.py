import json
import tempfile
import unittest
from pathlib import Path

from tools.routing_v021_mode_comparison import run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class RoutingV021ModeComparisonTests(unittest.TestCase):
    def test_run_writes_read_only_comparison_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix = root / "v021_feature_matrix.jsonl"
            rows = [
                {
                    "workspace_id": "fixture",
                    "source_layer": "s0b_text_unit",
                    "unit_id": "high1",
                    "unit_type": "sentence",
                    "text_preview": "Alice changed the prototype deadline because the project evidence was risky.",
                    "route_snapshot": {"s1_memory_candidate": "skip_or_background_only"},
                    "feature_status": {"chinese_dimlex": "available"},
                    "axis_scores": {
                        "value_score": 6.0,
                        "risk_score": 6.0,
                        "complexity_score": 5.5,
                        "entity_salience_score": 4.0,
                        "keyphrase_score": 5.0,
                        "low_value_score": 0.0,
                        "domain_term_score": 2.0,
                    },
                },
                {
                    "workspace_id": "fixture",
                    "source_layer": "s0b_text_unit",
                    "unit_id": "low1",
                    "unit_type": "sentence",
                    "text_preview": "OK thanks.",
                    "route_snapshot": {"s1_memory_candidate": "weak_llm_proposal"},
                    "feature_status": {"chinese_dimlex": "available"},
                    "axis_scores": {
                        "value_score": 0.5,
                        "risk_score": 0.0,
                        "complexity_score": 0.0,
                        "entity_salience_score": 0.0,
                        "keyphrase_score": 0.0,
                        "low_value_score": 8.0,
                        "domain_term_score": 0.0,
                    },
                },
            ]
            write_jsonl(matrix, rows)
            before = matrix.read_text(encoding="utf-8")

            outputs = run(
                type(
                    "Args",
                    (),
                    {
                        "matrix": str(matrix),
                        "output_dir": str(root / "out"),
                        "per_bucket": 5,
                        "max_samples": 20,
                    },
                )()
            )

            self.assertEqual(matrix.read_text(encoding="utf-8"), before)
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)
            compared = [json.loads(line) for line in Path(outputs["rows"]).read_text(encoding="utf-8").splitlines()]
            by_id = {row["unit_id"]: row for row in compared}
            self.assertIn("old_skipped_but_high_v021_value", by_id["high1"]["sample_buckets"])
            self.assertIn("high_risk_and_high_value", by_id["high1"]["sample_buckets"])
            self.assertEqual(by_id["high1"]["suggested_routes"]["s1_memory_candidate"], "strong_llm_proposal")
            self.assertIn("old_llm_but_low_v021_value", by_id["low1"]["sample_buckets"])
            self.assertEqual(by_id["low1"]["suggested_routes"]["s1_memory_candidate"], "skip_or_background_only")
            manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["provider_calls"])
            self.assertFalse(manifest["route_decisions_replaced"])
            self.assertFalse(manifest["write_permission"])


if __name__ == "__main__":
    unittest.main()
