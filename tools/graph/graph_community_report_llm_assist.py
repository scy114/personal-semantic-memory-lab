"""On-demand LLM assist overlay for v0.3 graph community reports.

The base profile/community builder stays extractive and full-coverage. This
tool only enhances explicitly selected communities or communities activated by
query runs. It writes an overlay, not a replacement for the base profile dir.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.proposals.proposal_runner import SUPPORTED_API_MODES, load_dotenv
from tools.graph.graph_construction_packet_builder import read_jsonl, write_json, write_jsonl, write_text
from tools.graph.graph_profile_community_builder import (
    DEFAULT_COMMUNITY_REPORT_PROMPT,
    SUPPORTED_REPORT_PROVIDERS,
    apply_provider_community_reports,
    resolve_project_path,
)


SCHEMA_VERSION = "graph_v03.community_report_llm_assist.v0.1"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_community_report_assists"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def unique_ordered(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item or "").strip()
                if text and text not in out:
                    out.append(text)
            continue
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def parse_community_ids(values: list[str] | None) -> list[str]:
    raw: list[str] = []
    for value in values or []:
        raw.extend(part.strip() for part in str(value).split(","))
    return unique_ordered(raw)


def community_ids_from_graph_context(path: Path) -> list[str]:
    ids: list[str] = []
    for row in read_jsonl(path):
        if row.get("kind") == "v03_ranked_community_report":
            ids.append(str(row.get("community_id") or ""))
    return unique_ordered(ids)


def community_ids_from_package(path: Path) -> list[str]:
    package = read_json(path, default={})
    rows = ((package.get("graph_branch") or {}).get("activated_communities") or [])
    ids: list[str] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row, dict) else None
        if isinstance(payload, dict):
            ids.append(str(payload.get("community_id") or ""))
        if isinstance(row, dict):
            ids.append(str(row.get("community_id") or ""))
    return unique_ordered(ids)


def community_ids_from_query_run(path: Path) -> list[str]:
    path = path.resolve()
    ids: list[str] = []
    if path.is_file():
        if path.name == "graph_context.jsonl":
            return community_ids_from_graph_context(path)
        if path.name == "graph_retrieval_package.json":
            return community_ids_from_package(path)
        return []
    for graph_context_path in sorted(path.rglob("graph_context.jsonl")):
        ids.extend(community_ids_from_graph_context(graph_context_path))
    for package_path in sorted(path.rglob("graph_retrieval_package.json")):
        ids.extend(community_ids_from_package(package_path))
    return unique_ordered(ids)


def select_reports(
    reports: list[dict[str, Any]],
    *,
    explicit_ids: list[str],
    query_run_paths: list[Path],
    top_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(row.get("community_id") or ""): row for row in reports if row.get("community_id")}
    requested_ids = list(explicit_ids)
    for path in query_run_paths:
        requested_ids.extend(community_ids_from_query_run(path))
    requested_ids = unique_ordered(requested_ids)

    if requested_ids:
        selected_ids = requested_ids[:top_n] if top_n > 0 else requested_ids
    else:
        ranked = sorted(reports, key=lambda row: (-float(row.get("rank") or 0.0), str(row.get("community_id") or "")))
        selected_ids = [str(row.get("community_id") or "") for row in ranked[:top_n]]

    missing_ids = [community_id for community_id in selected_ids if community_id not in by_id]
    selected_reports = [by_id[community_id] for community_id in selected_ids if community_id in by_id]
    return selected_reports, {
        "requested_ids": requested_ids,
        "selected_ids": [str(row.get("community_id") or "") for row in selected_reports],
        "missing_ids": missing_ids,
        "selection_source": "explicit_or_query_activated" if requested_ids else "top_ranked",
        "top_n": top_n,
    }


def write_markdown_report(path: Path, manifest: dict[str, Any], assists: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 Community Report LLM Assist",
        "",
        f"- profile_dir: `{manifest['profile_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- provider: `{manifest['provider']['provider']}`",
        f"- selected communities: {manifest['counts']['selected_community_count']}",
        f"- assist reports: {manifest['counts']['assist_report_count']}",
        f"- failures: {manifest['counts']['failure_count']}",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## Assist Reports",
        "",
    ]
    for row in assists:
        lines.extend(
            [
                f"### {row.get('title', '')}",
                "",
                f"- community_id: `{row.get('community_id', '')}`",
                f"- provider: `{row.get('provider', '')}`",
                f"- model: `{row.get('model_id', '')}`",
                f"- evidence refs: {len(row.get('evidence_refs') or [])}",
                f"- summary: {row.get('summary', '')}",
                "",
            ]
        )
    if failures:
        lines.extend(["## Failures / Review", ""])
        counts = Counter(str(row.get("failure_kind") or "unknown") for row in failures)
        for key, count in sorted(counts.items()):
            lines.append(f"- `{key}`: {count}")
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- This overlay is optional query/review assist material.",
            "- It does not replace extractive community reports.",
            "- It does not write graph truth, durable memory, or support proof.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def run_assist(
    *,
    profile_dir: Path,
    output_dir: Path | None = None,
    project_root: Path | None = None,
    community_ids: list[str] | None = None,
    from_query_run: list[Path] | None = None,
    top_n: int = 5,
    provider: str = "openai",
    api_mode: str | None = None,
    allow_live_api: bool = False,
    model_id: str | None = None,
    prompt_path: Path | str = DEFAULT_COMMUNITY_REPORT_PROMPT,
    env_file: Path | str = ".env",
    external_model_outputs_path: Path | None = None,
) -> dict[str, Any]:
    project_root = (project_root or Path(".")).resolve()
    env_path = resolve_project_path(project_root, env_file)
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=True)
    profile_dir = profile_dir.resolve()
    output_dir = (output_dir or profile_dir.parent / DEFAULT_OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if provider not in SUPPORTED_REPORT_PROVIDERS - {"extractive"}:
        raise ValueError("LLM assist provider must be external_jsonl or openai")
    api_mode = api_mode or os.environ.get("OPENAI_API_MODE") or "responses"
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    model_id = model_id or os.environ.get("OPENAI_MODEL_WEAK") or "gpt-4o-mini"

    reports_path = profile_dir / "graph_community_reports.jsonl"
    reports = read_jsonl(reports_path)
    if not reports:
        raise FileNotFoundError(f"No graph_community_reports.jsonl rows found at: {reports_path}")
    explicit_ids = parse_community_ids(community_ids)
    selected_reports, selection = select_reports(
        reports,
        explicit_ids=explicit_ids,
        query_run_paths=[path.resolve() for path in from_query_run or []],
        top_n=top_n,
    )

    assisted_reports, model_inputs, model_results, failures, provider_status = apply_provider_community_reports(
        selected_reports,
        project_root=project_root,
        provider=provider,
        api_mode=api_mode,
        allow_live_api=allow_live_api,
        model_id=model_id,
        prompt_path=prompt_path,
        external_model_outputs_path=external_model_outputs_path.resolve() if external_model_outputs_path else None,
        max_provider_reports=len(selected_reports),
    )
    assists = [row for row in assisted_reports if row.get("provider_report_status") == "community_report_candidate"]
    review_rows = [row for row in assisted_reports if row.get("provider_report_status") and row.get("provider_report_status") != "community_report_candidate"]

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "profile_dir": str(profile_dir),
        "output_dir": str(output_dir),
        "source_reports": str(reports_path),
        "selection": selection,
        "provider": provider_status,
        "counts": {
            "source_report_count": len(reports),
            "selected_community_count": len(selected_reports),
            "assist_report_count": len(assists),
            "review_or_reject_row_count": len(review_rows),
            "failure_count": len(failures),
            "model_call_inputs": len(model_inputs),
            "model_call_results": len(model_results),
        },
        "outputs": {
            "graph_community_report_assists": str(output_dir / "graph_community_report_assists.jsonl"),
            "graph_community_report_assist_review_rows": str(output_dir / "graph_community_report_assist_review_rows.jsonl"),
            "graph_community_report_assist_failures": str(output_dir / "graph_community_report_assist_failures.jsonl"),
            "graph_community_report_assist_model_call_inputs": str(output_dir / "graph_community_report_assist_model_call_inputs.jsonl"),
            "graph_community_report_assist_model_call_results": str(output_dir / "graph_community_report_assist_model_call_results.jsonl"),
        },
        "boundary": {
            "overlay_not_replacement": True,
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
        },
    }
    write_jsonl(output_dir / "graph_community_report_assists.jsonl", assists)
    write_jsonl(output_dir / "graph_community_report_assist_review_rows.jsonl", review_rows)
    write_jsonl(output_dir / "graph_community_report_assist_failures.jsonl", failures)
    write_jsonl(output_dir / "graph_community_report_assist_model_call_inputs.jsonl", model_inputs)
    write_jsonl(output_dir / "graph_community_report_assist_model_call_results.jsonl", model_results)
    write_json(output_dir / "graph_community_report_assist_manifest.json", manifest)
    write_markdown_report(output_dir / "graph_community_report_assist_report.md", manifest, assists, failures)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate on-demand LLM assist overlay for v0.3 community reports.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--community-ids", action="append", default=None, help="Comma-separated community ids. Repeatable.")
    parser.add_argument("--from-query-run", action="append", default=None, help="Query run dir or query subdir containing graph_context.jsonl.")
    parser.add_argument("--top-n", type=int, default=5, help="Limit selected communities. With no explicit/query ids, selects top-ranked communities.")
    parser.add_argument("--provider", default="openai", choices=sorted(SUPPORTED_REPORT_PROVIDERS - {"extractive"}))
    parser.add_argument("--api-mode", default=None, choices=sorted(SUPPORTED_API_MODES))
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompt-path", default=DEFAULT_COMMUNITY_REPORT_PROMPT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_assist(
        profile_dir=Path(args.profile_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        project_root=Path(args.project_root),
        community_ids=args.community_ids,
        from_query_run=[Path(path) for path in args.from_query_run or []],
        top_n=args.top_n,
        provider=args.provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        model_id=args.model_id,
        prompt_path=args.prompt_path,
        env_file=args.env_file,
        external_model_outputs_path=Path(args.external_model_outputs) if args.external_model_outputs else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
