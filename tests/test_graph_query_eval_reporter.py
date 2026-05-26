import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_query_eval_reporter import analyze_query_dir


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GraphQueryEvalReporterTests(unittest.TestCase):
    def test_flags_noisy_evidence_paths_separately_from_clean_graph_gain(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            query_dir = Path(temp_dir) / "q"
            write_json(query_dir / "query_manifest.json", {"query": {"question": "How are Paris and studio related?"}})
            write_json(query_dir / "route_decision.json", {"query_id": "q", "route": "graph_assisted"})
            write_jsonl(query_dir / "selected_context.jsonl", [{"evidence_refs": ["evidence:a"]}])
            write_jsonl(
                query_dir / "graph_context.jsonl",
                [
                    {
                        "summary": "Marley flooring --supports--> dance studio",
                        "evidence_refs": ["evidence:b"],
                        "warnings": [],
                    }
                ],
            )
            write_json(
                query_dir / "graph_answer_context.json",
                {
                    "supported_graph_relations": [
                        {"summary": "Marley flooring --supports--> dance studio", "evidence_refs": ["evidence:b"]}
                    ],
                    "cautious_graph_relations": [],
                    "evidence_paths": [],
                    "noisy_navigation_paths": [{"summary": "we -> this -> Jon -> Paris"}],
                    "unresolved_or_unsafe_graph_items": [],
                },
            )
            write_json(
                query_dir / "graph_retrieval_package.json",
                {
                    "ranked_items": [{"unit_id": "edge:e1"}],
                    "branches": {"bm25_graph_units": [{}], "graph_neighborhood_bfs_1hop": [{}]},
                },
            )

            row = analyze_query_dir(query_dir)

            self.assertEqual(row["question"], "How are Paris and studio related?")
            self.assertEqual(row["graph_value"], "adds_evidence")
            self.assertEqual(row["noisy_evidence_path_count"], 1)


if __name__ == "__main__":
    unittest.main()
