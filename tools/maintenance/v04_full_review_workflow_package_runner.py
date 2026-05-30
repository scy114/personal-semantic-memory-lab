"""Create and run a v0.4 full incremental workflow package with human review.

This is different from the post-review smoke runner. It has two explicit
stages:

1. prepare: create a fresh package with review queue and human-review WebUI
   instructions. No apply/publish is executed.
2. finalize: require finalized review decisions, then run apply/publish through
   S1, S2, graph current, invalidation, and visual review refresh.

Tests may use ``auto_review_for_test`` to simulate the human clicking through
the review queue, but normal packages must pass through the WebUI artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_text
from tools.maintenance.graph_candidate_latest_view_runner import run_graph_candidate_latest_view
from tools.maintenance.graph_current_view_publisher import run_graph_current_view_publish
from tools.maintenance.incremental_review_decision_applier import build_incremental_review_apply_bundle_from_files
from tools.maintenance.incremental_review_webui import (
    build_review_decision,
    build_review_session_manifest,
)
from tools.maintenance.s1_current_view_publisher import run_s1_current_view_publish
from tools.maintenance.s1_review_overlay_consumer import build_s1_review_latest_view_from_files
from tools.maintenance.s2_current_view_publisher import run_s2_current_view_publish


SCHEMA_VERSION = "maintenance.v04_full_review_workflow_package.v0.4"
DEFAULT_PACKAGE_DIR_NAME = "maintenance_v04_full_review_workflow"
PREPARE_MANIFEST_FILENAME = "v04_full_review_workflow_prepare_manifest.json"
FINAL_MANIFEST_FILENAME = "v04_full_review_workflow_final_manifest.json"
FINAL_REPORT_FILENAME = "v04_full_review_workflow_final_report.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def review_item(
    item_id: str,
    *,
    layer: str,
    review_stage: str,
    patch_action: str,
    allowed: list[str],
    default: str,
    summary: str,
    new_id: str,
    new_text: str,
    old_id: str = "",
    old_text: str = "",
    evidence_refs: list[str] | None = None,
    display: dict[str, Any] | None = None,
    new_display: dict[str, Any] | None = None,
    old_display: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence_refs = evidence_refs or []
    cards = [
        {
            "role": "new_s1_candidate",
            "object_id": new_id,
            "overlay_status": "candidate",
            "text": new_text,
            "evidence_refs": evidence_refs,
            "warnings": ["candidate_not_truth"],
            "review_display": new_display
            or {
                "role_zh": "新增候选",
                "status_zh": "待审核",
                "text_zh": new_text,
                "evidence_note_zh": "证据引用见下方。",
            },
        }
    ]
    if old_id:
        cards.append(
            {
                "role": "existing_s1_candidate",
                "object_id": old_id,
                "overlay_status": "current_or_historical_candidate",
                "text": old_text,
                "evidence_refs": ["evidence:old"],
                "warnings": [],
                "review_display": old_display
                or {
                    "role_zh": "已有内容",
                    "status_zh": "当前或历史候选",
                    "text_zh": old_text,
                    "evidence_note_zh": "旧证据引用见下方。",
                },
            }
        )
    return {
        "schema_version": "maintenance.incremental_review_queue_item.v0.4",
        "review_item_id": item_id,
        "patch_id": f"patch:{item_id}",
        "patch_hash": f"hash:{item_id}",
        "source_context_hash": f"source:{item_id}",
        "operation_id": "op-full-review-smoke",
        "review_stage": review_stage,
        "layer": layer,
        "patch_action": patch_action,
        "queue_priority": "high",
        "allowed_review_actions": allowed,
        "default_recommendation": default,
        "review_status": "pending_review",
        "review_summary": summary,
        "review_display": display or {},
        "risk_flags": ["candidate_not_truth", *([] if layer != "graph" else ["graph_is_not_proof"])],
        "review_rubric": [
            "Is the proposed action supported by evidence?",
            "Should this affect current view, historical view, or no view?",
        ],
        "context_cards": cards,
        "evidence_refs": evidence_refs,
        "write_permission": False,
        "apply_executed": False,
        "durable_writes_executed": False,
        "graph_truth_written": False,
    }


def seed_review_queue(package_dir: Path) -> list[dict[str, Any]]:
    queue = [
        review_item(
            "review-s1-1",
            layer="s1",
            review_stage="s1_review",
            patch_action="append_s1_contradiction_candidate",
            allowed=["split_current_vs_historical", "reject_patch", "defer", "needs_more_evidence"],
            default="split_current_vs_historical",
            summary="S1: new evidence updates Jon's current performance preference from contemporary dance to salsa.",
            new_id="memory-new",
            new_text="Jon currently prefers salsa for the performance.",
            old_id="memory-old",
            old_text="Jon prefers contemporary dance for the performance.",
            evidence_refs=["evidence:new"],
            display={
                "title_zh": "S1 审核：当前偏好发生变化",
                "summary_zh": "新证据表示 Jon 现在表演时优先选择 salsa；旧内容表示他偏好 contemporary dance。这里需要决定：是否把新内容作为当前候选，并把旧内容保留为历史/过期内容。",
                "recommended_action_zh": "建议拆分为当前事实和历史事实。",
                "why_review_zh": "这是可能覆盖旧偏好的增量更新，不能静默替换，需要人类确认。",
                "decision_hint_zh": "如果新证据可信，点“通过推荐处理”；如果证据不足，点“需要更多证据”；如果判断不应进入系统，点“拒绝”。",
            },
            new_display={
                "role_zh": "新增 S1 候选",
                "status_zh": "待进入当前视图",
                "text_zh": "Jon 当前表演偏好是 salsa。",
                "evidence_note_zh": "来自新增证据 evidence:new。",
            },
            old_display={
                "role_zh": "已有 S1 内容",
                "status_zh": "可能需要转为历史/过期",
                "text_zh": "Jon 之前表演偏好是 contemporary dance。",
                "evidence_note_zh": "来自旧证据 evidence:old。",
            },
        ),
        review_item(
            "review-s2-1",
            layer="s2",
            review_stage="s2_review",
            patch_action="propose_new_s2_unit",
            allowed=["route_s2_build", "reject_patch", "defer", "needs_more_evidence"],
            default="route_s2_build",
            summary="S2: materialize the reviewed current preference into the current portrait projection.",
            new_id="memory-new",
            new_text="Jon currently prefers salsa for the performance.",
            old_id="s2-old",
            old_text="Jon prefers contemporary dance for the performance.",
            evidence_refs=["evidence:new"],
            display={
                "title_zh": "S2 审核：把 S1 更新投影到画像层",
                "summary_zh": "S1 已出现“当前偏好为 salsa”的候选更新。S2 层需要决定是否生成/刷新画像单元，让当前画像反映这个变化。",
                "recommended_action_zh": "建议进入 S2 构建/刷新。",
                "why_review_zh": "S2 是用户画像投影层，比原始证据更接近可查询记忆，因此需要单独审核。",
                "decision_hint_zh": "如果同意该画像更新，点“通过推荐处理”；如果 S1 还不稳，点“需要更多证据”或“暂缓”。",
            },
            new_display={
                "role_zh": "新增 S2 候选",
                "status_zh": "待画像层 materialize",
                "text_zh": "当前画像候选：Jon 表演时偏好 salsa。",
                "evidence_note_zh": "来自新增证据 evidence:new，并承接 S1 审核。",
            },
            old_display={
                "role_zh": "已有 S2 单元",
                "status_zh": "可能需要从当前视图移出",
                "text_zh": "旧画像单元：Jon 表演时偏好 contemporary dance。",
                "evidence_note_zh": "来自旧证据 evidence:old。",
            },
        ),
        review_item(
            "review-graph-1",
            layer="graph",
            review_stage="graph_review",
            patch_action="review_graph_patch_candidate",
            allowed=["route_graph_extraction", "reject_patch", "defer", "needs_more_evidence"],
            default="route_graph_extraction",
            summary="Graph: refresh the relation neighborhood for Jon's performance preference.",
            new_id="graph-candidate:salsa",
            new_text="Jon --prefers_for_performance--> salsa",
            old_id="edge:jon-performance-preference",
            old_text="Jon --prefers_for_performance--> contemporary dance",
            evidence_refs=["evidence:new"],
            display={
                "title_zh": "图审核：刷新偏好关系邻域",
                "summary_zh": "图层看到一条候选关系：Jon 在表演场景下偏好 salsa；旧关系指向 contemporary dance。这里审核的是图候选和关系邻域刷新，不是把图当事实证明。",
                "recommended_action_zh": "建议进入图抽取/图刷新。",
                "why_review_zh": "图关系会影响后续关系查询、邻域扩展和可视化，所以需要和 S1/S2 分开审核。",
                "decision_hint_zh": "如果同意继续建图，点“通过推荐处理”；如果关系方向或证据不清楚，点“需要更多证据”。",
            },
            new_display={
                "role_zh": "新增图关系候选",
                "status_zh": "待图抽取/刷新",
                "text_zh": "Jon --表演偏好--> salsa。",
                "evidence_note_zh": "候选关系来自 evidence:new；图不是证明。",
            },
            old_display={
                "role_zh": "已有图关系",
                "status_zh": "可能需要历史化或刷新",
                "text_zh": "Jon --表演偏好--> contemporary dance。",
                "evidence_note_zh": "旧关系来自 evidence:old；图不是证明。",
            },
        ),
    ]
    write_jsonl(package_dir / "review_queue" / "incremental_review_queue.jsonl", queue)
    return queue


def seed_s1_inputs(package_dir: Path) -> dict[str, str]:
    input_dir = package_dir / "inputs"
    active = input_dir / "s1_active_before_review.jsonl"
    incremental = input_dir / "s1_incremental_candidates.jsonl"
    write_jsonl(
        active,
        [
            {
                "memory_id": "memory-old",
                "content": "Jon prefers contemporary dance for the performance.",
                "evidence_refs": ["evidence:old"],
                "raw_backpointer_refs": [{"display_ref": "D1:1"}],
            },
            {
                "memory_id": "memory-keep",
                "content": "Jon is preparing a performance.",
                "evidence_refs": ["evidence:keep"],
                "raw_backpointer_refs": [{"display_ref": "D1:2"}],
            },
        ],
    )
    write_jsonl(
        incremental,
        [
            {
                "memory_id": "memory-new",
                "content": "Jon currently prefers salsa for the performance.",
                "evidence_refs": ["evidence:new"],
                "raw_backpointer_refs": [{"display_ref": "D2:1"}],
            }
        ],
    )
    return {"active": str(active), "incremental": str(incremental)}


def seed_s2_inputs(package_dir: Path) -> dict[str, str]:
    input_dir = package_dir / "inputs"
    base = input_dir / "s2_base_reviewed_units.jsonl"
    candidates = input_dir / "s2_incremental_candidates.jsonl"
    affected = input_dir / "s2_affected_units.jsonl"
    write_jsonl(
        base,
        [
            {
                "schema_version": "s2.reviewed_portrait_unit.v1",
                "unit_id": "s2-old",
                "user_id": "Jon",
                "type": "preference",
                "content": "Jon prefers contemporary dance for the performance.",
                "evidence_refs": ["evidence:old"],
                "status": "active",
            }
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "schema_version": "maintenance.s2_incremental_candidate.v0.4",
                "candidate_status": "candidate_ready_for_review",
                "candidate_text": "Jon currently prefers salsa for the performance.",
                "candidate_type": "preference",
                "proposal_confidence": "high",
                "inference_level": "explicit",
                "proposal_id": "s2p-new",
                "proposal_input_id": "s2pi-new",
                "proposal_run_id": "run-full-review-smoke",
                "provider": "synthetic",
                "model_id": "human-review-smoke-fixture",
                "output_kind": "portrait_fact_candidate",
                "route_used": "review_queue_full_smoke",
                "review_item_id": "review-s2-1",
                "patch_id": "patch:review-s2-1",
                "operation_id": "op-full-review-smoke",
                "source_s1_memory_id": "memory-new",
                "refresh_reason": "split_current_and_historical_overlay",
                "evidence_refs": ["evidence:new"],
                "raw_backpointer_refs": [{"display_ref": "D2:1"}],
                "source_text_quote": "For the performance now, salsa is my first choice.",
                "proposal_row": {
                    "target_participant": "Jon",
                    "subject_id": "Jon",
                    "privacy_class": "personal_low",
                    "candidate_text": "Jon currently prefers salsa for the performance.",
                    "proposal_confidence": "high",
                    "inference_level": "explicit",
                },
            }
        ],
    )
    write_jsonl(
        affected,
        [
            {
                "new_s1_unit_id": "memory-new",
                "affected_s2_unit_id": "s2-old",
                "s2_delta_signal": "contradicts_profile",
                "subject_scope_existing_s2_unit_ids": ["s2-old"],
            }
        ],
    )
    return {"base": str(base), "candidates": str(candidates), "affected": str(affected)}


def seed_graph_dir(graph_dir: Path, *, target_node_id: str, target_label: str, evidence_ref: str, quote: str) -> None:
    write_jsonl(
        graph_dir / "graph_nodes_table.jsonl",
        [
            {"node_id": "node:jon", "label": "Jon", "entity_type": "person", "entity_quality_hint": "stable", "evidence_refs": [evidence_ref], "graph_is_not_proof": True},
            {"node_id": target_node_id, "label": target_label, "entity_type": "activity", "entity_quality_hint": "stable", "evidence_refs": [evidence_ref], "graph_is_not_proof": True},
        ],
    )
    write_jsonl(
        graph_dir / "graph_edges_table.jsonl",
        [
            {
                "edge_id": "edge:jon-performance-preference",
                "source_node_id": "node:jon",
                "target_node_id": target_node_id,
                "source_label": "Jon",
                "target_label": target_label,
                "relation_type": "prefers_for_performance",
                "generic_relation_review_hint": "not_generic",
                "evidence_refs": [evidence_ref],
                "source_text_quotes": [quote],
                "source_packet_ids": ["packet:full-review-smoke"],
                "candidate_ids": ["candidate:graph-full-review-smoke"],
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(
        graph_dir / "graph_claims_table.jsonl",
        [
            {
                "claim_id": "claim:jon-performance-preference",
                "subject_node_id": "node:jon",
                "subject_endpoint_status": "local_entity_id",
                "evidence_refs": [evidence_ref],
                "source_packet_id": "packet:full-review-smoke",
                "candidate_id": "candidate:claim-full-review-smoke",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(
        graph_dir / "evidence_links.jsonl",
        [
            {
                "link_id": "link:jon-performance-preference",
                "owner_kind": "edge",
                "owner_id": "edge:jon-performance-preference",
                "evidence_ref": evidence_ref,
                "source_packet_id": "packet:full-review-smoke",
                "candidate_id": "candidate:evidence-link-full-review-smoke",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_json(graph_dir / "graph_consolidation_manifest.json", {"schema_version": "synthetic.graph_consolidation_full_review_smoke"})


def seed_graph_inputs(package_dir: Path) -> dict[str, str]:
    base = package_dir / "inputs" / "graph_base_consolidated"
    incremental = package_dir / "inputs" / "graph_incremental_consolidated"
    seed_graph_dir(
        base,
        target_node_id="node:contemporary",
        target_label="contemporary dance",
        evidence_ref="evidence:old",
        quote="Jon prefers contemporary dance for the performance.",
    )
    seed_graph_dir(
        incremental,
        target_node_id="node:salsa",
        target_label="salsa",
        evidence_ref="evidence:new",
        quote="For the performance now, salsa is my first choice.",
    )
    return {"base": str(base), "incremental": str(incremental)}


def write_human_review_instructions(package_dir: Path, *, host: str, port: int) -> str:
    queue = package_dir / "review_queue" / "incremental_review_queue.jsonl"
    review_dir = package_dir / "human_review"
    text = f"""# v0.4 增量人工审核包

## 第一步：启动审核 WebUI

```powershell
python -m tools.maintenance.incremental_review_webui --review-queue-jsonl "{queue}" --output-dir "{review_dir}" --host {host} --port {port}
```

打开：

```text
http://{host}:{port}
```

逐条审核 S1 / S2 / graph 项目。页面会优先显示中文说明，但内部机器字段仍然保留在队列中，供后续 apply/finalize 使用。

审核完成后，点击页面右上方“提交本轮审核”。这一步只写：

```text
human_review/review_decisions.jsonl
human_review/review_session_manifest.json
```

## 第二步：审核后 finalize

```powershell
python -m tools.maintenance.v04_full_review_workflow_package_runner finalize --workspace "{package_dir.parent}" --package-dir "{package_dir}"
```

finalize 会读取人工审核结果，然后生成 apply plan，并发布 S1 current view、S2 current view、graph current、dependency/invalidation 和增量可视化。

边界：

- 人工审核是必须 gate；普通模式不会自动批准。
- current view 不是 durable memory truth。
- graph current 不是 graph truth。
- graph_is_not_proof=true 仍然成立。
"""
    path = package_dir / "HUMAN_REVIEW_INSTRUCTIONS.md"
    write_text(path, text)
    return str(path)


def prepare_full_review_workflow_package(
    *,
    workspace: Path,
    package_dir: Path | None = None,
    reset: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    package_dir = (package_dir or workspace / DEFAULT_PACKAGE_DIR_NAME).resolve()
    if package_dir.exists():
        if not reset:
            raise FileExistsError(f"Package already exists; pass reset=True to rebuild: {package_dir}")
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    review_dir = package_dir / "human_review"
    review_dir.mkdir(parents=True)
    write_jsonl(review_dir / "review_decisions.jsonl", [])

    queue = seed_review_queue(package_dir)
    inputs = {
        "s1": seed_s1_inputs(package_dir),
        "s2": seed_s2_inputs(package_dir),
        "graph": seed_graph_inputs(package_dir),
    }
    instructions = write_human_review_instructions(package_dir, host=host, port=port)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepared_for_human_review",
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "package_dir": str(package_dir),
        "review_required": True,
        "review_completed": False,
        "inputs": inputs,
        "outputs": {
            "review_queue": str(package_dir / "review_queue" / "incremental_review_queue.jsonl"),
            "review_decisions": str(review_dir / "review_decisions.jsonl"),
            "review_session_manifest": str(review_dir / "review_session_manifest.json"),
            "instructions": instructions,
        },
        "webui": {
            "host": host,
            "port": port,
            "url": f"http://{host}:{port}",
            "start_command": f"python -m tools.maintenance.incremental_review_webui --review-queue-jsonl \"{package_dir / 'review_queue' / 'incremental_review_queue.jsonl'}\" --output-dir \"{review_dir}\" --host {host} --port {port}",
        },
        "counts": {
            "review_queue_items": len(queue),
            "s1_review_items": sum(1 for item in queue if item.get("review_stage") == "s1_review"),
            "s2_review_items": sum(1 for item in queue if item.get("review_stage") == "s2_review"),
            "graph_review_items": sum(1 for item in queue if item.get("review_stage") == "graph_review"),
        },
        "boundary": {
            "provider_calls_executed": False,
            "apply_executed": False,
            "publish_executed": False,
            "durable_memory_written": False,
            "graph_truth_written": False,
        },
    }
    write_json(package_dir / PREPARE_MANIFEST_FILENAME, manifest)
    return manifest


def auto_review_for_test(package_dir: Path) -> dict[str, str]:
    queue_path = package_dir / "review_queue" / "incremental_review_queue.jsonl"
    review_dir = package_dir / "human_review"
    queue = read_jsonl(queue_path)
    decisions = [
        build_review_decision(item, review_action="", human_decision="approve_recommended")
        for item in queue
    ]
    session = build_review_session_manifest(queue, decisions, allow_partial_finalize=False)
    write_jsonl(review_dir / "review_decisions.jsonl", decisions)
    write_json(review_dir / "review_session_manifest.json", session)
    return {
        "review_decisions": str(review_dir / "review_decisions.jsonl"),
        "review_session_manifest": str(review_dir / "review_session_manifest.json"),
    }


def render_final_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Full Review Workflow Final Report",
        "",
        f"- finalized_at: {manifest['finalized_at']}",
        f"- package_dir: `{manifest['package_dir']}`",
        f"- acceptance_status: `{manifest['acceptance_status']}`",
        f"- review_decision_count: {manifest['counts']['review_decision_count']}",
        "",
        "## Acceptance Checks",
        "",
    ]
    for key, value in manifest["acceptance_checks"].items():
        lines.append(f"- {key}: {str(value).lower()}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
        ]
    )
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This package required human-review artifacts before finalize.",
            "- It does not call a live provider.",
            "- It does not write durable memory or graph truth.",
            "- Graph visualization remains audit support only.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize_full_review_workflow_package(
    *,
    workspace: Path,
    package_dir: Path | None = None,
    auto_review_for_test_mode: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    package_dir = (package_dir or workspace / DEFAULT_PACKAGE_DIR_NAME).resolve()
    if not (package_dir / PREPARE_MANIFEST_FILENAME).exists():
        raise FileNotFoundError(f"Prepare manifest not found: {package_dir / PREPARE_MANIFEST_FILENAME}")
    if auto_review_for_test_mode:
        auto_review_for_test(package_dir)

    review_queue = package_dir / "review_queue" / "incremental_review_queue.jsonl"
    review_decisions = package_dir / "human_review" / "review_decisions.jsonl"
    review_session = package_dir / "human_review" / "review_session_manifest.json"
    if not review_session.exists():
        raise FileNotFoundError(
            "Human review is not finalized. Start the WebUI, review items, and submit the review session first."
        )

    apply_dir = package_dir / "review_apply"
    apply_bundle = build_incremental_review_apply_bundle_from_files(
        review_queue_jsonl=review_queue,
        review_decisions_jsonl=review_decisions,
        review_session_manifest_json=review_session,
        output_dir=apply_dir,
    )

    s1_after_review = package_dir / "s1_after_review"
    s1_bundle = build_s1_review_latest_view_from_files(
        active_s1_jsonl=package_dir / "inputs" / "s1_active_before_review.jsonl",
        incremental_s1_jsonl=package_dir / "inputs" / "s1_incremental_candidates.jsonl",
        s1_review_overlay_jsonl=Path(apply_bundle["paths"]["s1_overlay"]),
        output_dir=s1_after_review,
    )
    s1_publish = run_s1_current_view_publish(workspace=package_dir, source_dir=s1_after_review)

    s2_publish = run_s2_current_view_publish(
        workspace=package_dir,
        base_reviewed_units_jsonl=package_dir / "inputs" / "s2_base_reviewed_units.jsonl",
        candidate_outputs_jsonl=package_dir / "inputs" / "s2_incremental_candidates.jsonl",
        review_decisions_jsonl=review_decisions,
        affected_s2_jsonl=package_dir / "inputs" / "s2_affected_units.jsonl",
    )

    graph_latest_base = package_dir / "graph_latest_base"
    run_graph_candidate_latest_view(
        incremental_graph_dir=package_dir / "inputs" / "graph_base_consolidated",
        output_dir=graph_latest_base,
    )
    run_graph_current_view_publish(workspace=package_dir, source_latest_view_dir=graph_latest_base)

    graph_latest_after_increment = package_dir / "graph_latest_after_increment"
    graph_latest = run_graph_candidate_latest_view(
        base_graph_dir=package_dir / "inputs" / "graph_base_consolidated",
        incremental_graph_dir=package_dir / "inputs" / "graph_incremental_consolidated",
        output_dir=graph_latest_after_increment,
    )
    graph_publish = run_graph_current_view_publish(workspace=package_dir, source_latest_view_dir=graph_latest_after_increment)
    visual_manifest = json.loads(Path(graph_publish["maintenance_artifacts"]["incremental_visual_review_manifest"]).read_text(encoding="utf-8-sig"))

    review_rows = read_jsonl(review_decisions)
    acceptance_checks = {
        "human_review_session_finalized": json.loads(review_session.read_text(encoding="utf-8-sig")).get("review_session_status") == "finalized",
        "review_decisions_nonzero": len(review_rows) > 0,
        "apply_plan_generated": apply_bundle["report"]["apply_plan_count"] == len(review_rows),
        "s1_current_has_active_rows": s1_publish["counts"]["current_active_count"] >= 1,
        "s2_current_has_active_rows": s2_publish["counts"]["current_active_count"] >= 1,
        "graph_changed_units_nonzero": graph_publish["maintenance_counts"]["changed_graph_units"] > 0,
        "graph_visual_slices_nonzero": visual_manifest["counts"]["slices"] > 0,
        "no_durable_memory_write": graph_publish.get("durable_writes_executed") is False,
        "no_graph_truth_write": graph_publish.get("graph_truth_written") is False,
    }
    acceptance_status = "pass" if all(acceptance_checks.values()) else "fail"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "finalized_after_human_review",
        "finalized_at": now_iso(),
        "workspace": str(workspace),
        "package_dir": str(package_dir),
        "acceptance_status": acceptance_status,
        "acceptance_checks": acceptance_checks,
        "counts": {
            "review_decision_count": len(review_rows),
            "apply_plan_count": apply_bundle["report"]["apply_plan_count"],
            "s1_current_active": s1_publish["counts"]["current_active_count"],
            "s2_current_active": s2_publish["counts"]["current_active_count"],
            "graph_changed_units": graph_publish["maintenance_counts"]["changed_graph_units"],
            "visual_review_slices": visual_manifest["counts"]["slices"],
        },
        "stages": {
            "review_apply": {"manifest": apply_bundle["paths"]["report_json"], "counts": apply_bundle["report"]},
            "s1_after_review": {"manifest": s1_bundle["paths"]["manifest"], "counts": s1_bundle["manifest"]},
            "s1_current_publish": {"manifest": s1_publish["paths"]["manifest"], "counts": s1_publish["counts"]},
            "s2_current_publish": {"manifest": s2_publish["paths"]["manifest"], "counts": s2_publish["counts"]},
            "graph_latest_view": {"manifest": graph_latest["paths"]["manifest"], "counts": graph_latest["counts"]},
            "graph_current_publish": {"manifest": graph_publish["paths"]["manifest"], "counts": graph_publish["maintenance_counts"]},
            "graph_visual_review": {"manifest": graph_publish["maintenance_artifacts"]["incremental_visual_review_manifest"], "counts": visual_manifest["counts"]},
        },
        "outputs": {
            "manifest": str(package_dir / FINAL_MANIFEST_FILENAME),
            "report": str(package_dir / FINAL_REPORT_FILENAME),
            "review_queue": str(review_queue),
            "review_decisions": str(review_decisions),
            "review_session_manifest": str(review_session),
            "apply_plan": apply_bundle["paths"]["apply_plan"],
            "s1_current": s1_publish["paths"]["s1_current"],
            "s2_current": s2_publish["paths"]["s2_current"],
            "graph_current": graph_publish["current_graph_dir"],
            "changed_graph_units": graph_publish["maintenance_artifacts"]["changed_graph_units"],
            "incremental_visual_review_html": graph_publish["maintenance_artifacts"]["incremental_visual_review_html"],
        },
        "source_hashes": {
            "review_queue": file_hash(review_queue),
            "review_decisions": file_hash(review_decisions),
            "review_session_manifest": file_hash(review_session),
        },
        "boundary": {
            "human_review_required": True,
            "auto_review_for_test_mode": auto_review_for_test_mode,
            "provider_calls_executed": False,
            "durable_memory_written": False,
            "graph_truth_written": False,
            "support_checker_authority": False,
        },
    }
    write_json(package_dir / FINAL_MANIFEST_FILENAME, manifest)
    write_text(package_dir / FINAL_REPORT_FILENAME, render_final_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workspace", required=True, type=Path)
    prepare.add_argument("--package-dir", type=Path)
    prepare.add_argument("--reset", action="store_true")
    prepare.add_argument("--host", default="127.0.0.1")
    prepare.add_argument("--port", type=int, default=8765)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--workspace", required=True, type=Path)
    finalize.add_argument("--package-dir", type=Path)
    finalize.add_argument("--auto-review-for-test", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_full_review_workflow_package(
            workspace=args.workspace,
            package_dir=args.package_dir,
            reset=args.reset,
            host=args.host,
            port=args.port,
        )
    else:
        manifest = finalize_full_review_workflow_package(
            workspace=args.workspace,
            package_dir=args.package_dir,
            auto_review_for_test_mode=args.auto_review_for_test,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if manifest.get("acceptance_status", "pass") != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
