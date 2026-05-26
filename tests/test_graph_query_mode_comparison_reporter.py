import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_query_mode_comparison_reporter import run_comparison


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_query_run(root: Path, query_id: str, *, embedding: bool, supported: int) -> None:
    qdir = root / query_id
    write_json(qdir / "route_decision.json", {"query_id": query_id, "route": "graph_assisted"})
    write_json(qdir / "query_manifest.json", {"query": {"question": "How is the graph branch doing?"}})
    write_jsonl(qdir / "selected_context.jsonl", [{"evidence_refs": ["evidence:a"]}])
    graph_context = [
        {
            "summary": f"relation {idx}",
            "evidence_refs": [f"evidence:g{idx}"],
            "warnings": [],
        }
        for idx in range(supported)
    ]
    write_jsonl(qdir / "graph_context.jsonl", graph_context)
    write_json(
        qdir / "graph_answer_context.json",
        {
            "supported_graph_relations": graph_context,
            "cautious_graph_relations": [],
            "evidence_paths": [],
            "noisy_navigation_paths": [],
            "unresolved_or_unsafe_graph_items": [],
        },
    )
    branches = {"bm25_graph_units": [{}]}
    if embedding:
        branches["embedding_graph_units"] = [{}, {}]
    write_json(qdir / "graph_retrieval_package.json", {"ranked_items": [{}], "branches": branches})


class GraphQueryModeComparisonReporterTests(unittest.TestCase):
    def test_mode_comparison_reports_embedding_branch_and_deltas(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            lexical = root / "lexical"
            hybrid = root / "hybrid"
            output = root / "comparison"
            write_query_run(lexical, "q1", embedding=False, supported=1)
            write_query_run(hybrid, "q1", embedding=True, supported=2)

            result = run_comparison(
                [f"lexical={lexical}", f"hybrid={hybrid}"],
                output,
                "lexical",
            )

            self.assertEqual(result["mode_count"], 2)
            summary = json.loads((output / "graph_query_mode_comparison_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["mode_summary"]["hybrid"]["embedding_graph_unit_total"], 2)
            rows = [
                json.loads(line)
                for line in (output / "graph_query_mode_comparison_rows.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            hybrid_row = next(row for row in rows if row["mode_label"] == "hybrid")
            self.assertEqual(hybrid_row["supported_relation_delta_vs_baseline"], 1)


if __name__ == "__main__":
    unittest.main()
