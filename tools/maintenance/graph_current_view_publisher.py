"""Publish a v0.4 current derived graph view for query consumers.

This is the real merge/publish layer after candidate graph latest-view
generation. It copies a validated latest-view bundle into a stable
``workspace/graph_current`` directory so query runners can consume the current
derived candidate graph by default.

The published view is still not graph truth. It is an auditable current
candidate view with provenance and hashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_text
from tools.maintenance.graph_dependency_map_builder import run_graph_dependency_map_build
from tools.maintenance.graph_query_index_invalidation_runner import run_graph_query_index_invalidation
from tools.maintenance.graph_visual_review_incremental_refresher import run_graph_visual_review_incremental_refresh


SCHEMA_VERSION = "maintenance.graph_current_view_publish.v0.4"
MANIFEST_FILENAME = "graph_current_manifest.json"
REPORT_FILENAME = "graph_current_report.md"
DEFAULT_CURRENT_DIR_NAME = "graph_current"

REQUIRED_LATEST_FILES = [
    "graph_nodes_latest_view.jsonl",
    "graph_edges_latest_view.jsonl",
    "graph_claims_latest_view.jsonl",
    "evidence_links_latest_view.jsonl",
    "graph_nodes_latest_view_excluded.jsonl",
    "graph_edges_latest_view_excluded.jsonl",
    "graph_claims_latest_view_excluded.jsonl",
    "evidence_links_latest_view_excluded.jsonl",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def latest_view_manifest_path(source_dir: Path) -> Path:
    path = source_dir / "graph_candidate_latest_view_manifest.json"
    if path.exists():
        return path
    return source_dir / "graph_current_manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_source_latest_view(source_dir: Path) -> None:
    missing = [name for name in REQUIRED_LATEST_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Source latest-view bundle is missing required files: {missing}")


def count_jsonl(path: Path) -> int:
    return len(read_jsonl(path))


def table_counts(root: Path) -> dict[str, dict[str, int]]:
    return {
        "nodes": {
            "active_count": count_jsonl(root / "graph_nodes_latest_view.jsonl"),
            "excluded_count": count_jsonl(root / "graph_nodes_latest_view_excluded.jsonl"),
        },
        "edges": {
            "active_count": count_jsonl(root / "graph_edges_latest_view.jsonl"),
            "excluded_count": count_jsonl(root / "graph_edges_latest_view_excluded.jsonl"),
        },
        "claims": {
            "active_count": count_jsonl(root / "graph_claims_latest_view.jsonl"),
            "excluded_count": count_jsonl(root / "graph_claims_latest_view_excluded.jsonl"),
        },
        "evidence_links": {
            "active_count": count_jsonl(root / "evidence_links_latest_view.jsonl"),
            "excluded_count": count_jsonl(root / "evidence_links_latest_view_excluded.jsonl"),
        },
    }


def copy_latest_view_files(source_dir: Path, target_dir: Path) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_hashes: dict[str, str] = {}
    for name in REQUIRED_LATEST_FILES:
        source = source_dir / name
        target = target_dir / name
        shutil.copy2(source, target)
        copied_hashes[name] = file_hash(target) or ""

    source_latest_alias = source_dir / "latest_views"
    if source_latest_alias.exists():
        shutil.copytree(source_latest_alias, target_dir / "latest_views", dirs_exist_ok=True)
    else:
        latest_dir = target_dir / "latest_views"
        latest_dir.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_LATEST_FILES:
            shutil.copy2(target_dir / name, latest_dir / name)
    return copied_hashes


def archive_previous_current(current_dir: Path, history_dir: Path, slug: str) -> Path | None:
    if not current_dir.exists():
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = history_dir / f"graph_current_previous_{slug}"
    shutil.copytree(current_dir, archive_dir)
    return archive_dir


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Graph Current View Publish",
        "",
        f"- published_at: {manifest['published_at']}",
        f"- workspace: `{manifest['workspace']}`",
        f"- source_latest_view_dir: `{manifest['inputs']['source_latest_view_dir']}`",
        f"- current_graph_dir: `{manifest['current_graph_dir']}`",
        f"- previous_current_archive_dir: `{manifest.get('previous_current_archive_dir') or ''}`",
        "",
        "## Counts",
        "",
    ]
    for table, counts in manifest["counts"].items():
        lines.extend(
            [
                f"### {table}",
                "",
                f"- active: {counts['active_count']}",
                f"- excluded: {counts['excluded_count']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Maintenance Artifacts",
            "",
            f"- dependency_map: `{manifest.get('maintenance_artifacts', {}).get('dependency_map', '')}`",
            f"- changed_graph_units: `{manifest.get('maintenance_artifacts', {}).get('changed_graph_units', '')}`",
            f"- stale_graph_query_assets: `{manifest.get('maintenance_artifacts', {}).get('stale_graph_query_assets', '')}`",
            f"- incremental_visual_review: `{manifest.get('maintenance_artifacts', {}).get('incremental_visual_review_manifest', '')}`",
            "",
            "## Boundary",
            "",
            "- This is a real publish of the current derived candidate graph view.",
            "- Query consumers may use this directory by default.",
            "- Dependency and invalidation artifacts are maintenance signals only.",
            "- It is not graph truth and not durable memory.",
            "- graph_is_not_proof=true.",
            "",
        ]
    )
    return "\n".join(lines)


def run_graph_current_view_publish(
    *,
    workspace: Path,
    source_latest_view_dir: Path,
    current_dir: Path | None = None,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    source_latest_view_dir = source_latest_view_dir.resolve()
    current_dir = (current_dir or workspace / DEFAULT_CURRENT_DIR_NAME).resolve()
    history_dir = (history_dir or workspace / "maintenance" / "published_views").resolve()
    validate_source_latest_view(source_latest_view_dir)

    slug = timestamp_slug()
    temp_dir = workspace / f".graph_current_tmp_{slug}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    copied_hashes = copy_latest_view_files(source_latest_view_dir, temp_dir)
    source_manifest = latest_view_manifest_path(source_latest_view_dir)
    source_manifest_hash = file_hash(source_manifest)
    source_manifest_payload = read_json(source_manifest)
    previous_manifest_hash = file_hash(current_dir / MANIFEST_FILENAME)

    previous_archive = archive_previous_current(current_dir, history_dir, slug)
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.move(str(temp_dir), str(current_dir))

    dependency_manifest = run_graph_dependency_map_build(graph_dir=current_dir)
    invalidation_manifest = run_graph_query_index_invalidation(
        current_graph_dir=current_dir,
        previous_graph_dir=previous_archive,
        dependency_map=Path(dependency_manifest["outputs"]["dependency_map"]),
    )
    visual_manifest = run_graph_visual_review_incremental_refresh(
        workspace=workspace,
        graph_current_dir=current_dir,
        changed_graph_units=Path(invalidation_manifest["outputs"]["changed_graph_units"]),
        dependency_map=Path(dependency_manifest["outputs"]["dependency_map"]),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "published_at": now_iso(),
        "workspace": str(workspace),
        "inputs": {
            "source_latest_view_dir": str(source_latest_view_dir),
            "source_latest_view_manifest": str(source_manifest) if source_manifest.exists() else "",
        },
        "input_hashes": {
            "source_latest_view_manifest": source_manifest_hash,
            "previous_current_manifest": previous_manifest_hash,
        },
        "source_latest_view_schema_version": source_manifest_payload.get("schema_version", ""),
        "current_graph_dir": str(current_dir),
        "previous_current_archive_dir": str(previous_archive) if previous_archive else "",
        "history_dir": str(history_dir),
        "copied_file_hashes": copied_hashes,
        "counts": table_counts(current_dir),
        "paths": {
            "graph_nodes_latest_view": str(current_dir / "graph_nodes_latest_view.jsonl"),
            "graph_edges_latest_view": str(current_dir / "graph_edges_latest_view.jsonl"),
            "graph_claims_latest_view": str(current_dir / "graph_claims_latest_view.jsonl"),
            "evidence_links_latest_view": str(current_dir / "evidence_links_latest_view.jsonl"),
            "latest_views_dir": str(current_dir / "latest_views"),
            "graph_dependency_map_manifest": str(current_dir / "dependency_map" / "graph_dependency_map_manifest.json"),
            "graph_query_index_invalidation_manifest": str(current_dir / "query_index_invalidation" / "graph_query_index_invalidation_manifest.json"),
            "graph_visual_review_incremental_manifest": visual_manifest["outputs"]["manifest"],
            "graph_visual_review_incremental_html": visual_manifest["outputs"]["html_index"],
            "manifest": str(current_dir / MANIFEST_FILENAME),
            "report": str(current_dir / REPORT_FILENAME),
        },
        "maintenance_artifacts": {
            "dependency_map": dependency_manifest["outputs"]["dependency_map"],
            "dependency_reverse_index": dependency_manifest["outputs"]["reverse_index"],
            "changed_graph_units": invalidation_manifest["outputs"]["changed_graph_units"],
            "stale_graph_query_assets": invalidation_manifest["outputs"]["stale_graph_query_assets"],
            "incremental_visual_review_manifest": visual_manifest["outputs"]["manifest"],
            "incremental_visual_review_html": visual_manifest["outputs"]["html_index"],
        },
        "maintenance_counts": {
            "dependency_rows": sum(count["dependency_rows"] for count in dependency_manifest["counts"].values()),
            "changed_graph_units": invalidation_manifest["counts"]["changed_graph_units"],
            "stale_asset_rows": invalidation_manifest["counts"]["stale_asset_rows"],
            "visual_review_slices": visual_manifest["counts"]["slices"],
            "visual_review_skipped_changes": visual_manifest["counts"]["skipped_changes"],
        },
        "policies": {
            "current_view_publish_executed": True,
            "query_default_ready": True,
            "stable_current_dir": DEFAULT_CURRENT_DIR_NAME,
            "published_view_kind": "current_derived_candidate_graph_view",
            "graph_is_not_proof": True,
            "graph_truth_written": False,
            "durable_memory_written": False,
            "query_indexes_refreshed": False,
            "visual_review_refreshed": visual_manifest["policies"]["visual_review_refreshed"],
            "visual_review_artifact_written": visual_manifest["policies"]["visual_review_artifact_written"],
            "visual_review_no_changes_detected": visual_manifest["counts"]["no_changes_detected"],
            "support_checker_authority": False,
        },
        "boundary": "published_current_candidate_graph_view_not_graph_truth",
        "write_permission": False,
        "publish_executed": True,
        "query_default_ready": True,
        "graph_is_not_proof": True,
        "graph_truth_written": False,
        "durable_writes_executed": False,
    }
    write_json(current_dir / MANIFEST_FILENAME, manifest)
    write_text(current_dir / REPORT_FILENAME, render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-latest-view-dir", required=True, type=Path)
    parser.add_argument("--current-dir", type=Path)
    parser.add_argument("--history-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_graph_current_view_publish(
        workspace=args.workspace,
        source_latest_view_dir=args.source_latest_view_dir,
        current_dir=args.current_dir,
        history_dir=args.history_dir,
    )
    print(
        json.dumps(
            {
                "manifest": manifest["paths"]["manifest"],
                "current_graph_dir": manifest["current_graph_dir"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
