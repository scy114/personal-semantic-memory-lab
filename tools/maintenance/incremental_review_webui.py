"""Local WebUI for v0.4 incremental human review.

The server reads an incremental review queue and writes review decisions to an
independent JSONL file. It never applies patches or mutates canonical memory,
portrait, graph, or durable memory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


SCHEMA_VERSION = "maintenance.incremental_review_decision.v0.4"
SESSION_SCHEMA_VERSION = "maintenance.incremental_review_session.v0.4"
DEFAULT_DECISIONS_FILENAME = "review_decisions.jsonl"
DEFAULT_SESSION_MANIFEST_FILENAME = "review_session_manifest.json"
ALLOWED_REVIEW_STATUSES = {
    "approved_for_apply",
    "approved_noop",
    "approved_with_edit",
    "rejected",
    "deferred",
    "needs_more_evidence",
}
ACTION_TO_STATUS = {
    "accept_noop": "approved_noop",
    "accept_patch": "approved_for_apply",
    "accept_with_edit": "approved_with_edit",
    "reject_patch": "rejected",
    "defer": "deferred",
    "needs_more_evidence": "needs_more_evidence",
    "split_current_vs_historical": "approved_with_edit",
    "mark_stale_only": "approved_with_edit",
    "route_s2_build": "approved_for_apply",
    "route_graph_extraction": "approved_for_apply",
}
BATCH_SAFE_ACTIONS = {"accept_noop"}
HUMAN_DECISIONS = {
    "approve_recommended",
    "approve_with_edit",
    "needs_more_evidence",
    "reject",
    "defer",
}
HUMAN_DECISION_DIRECT_ACTION = {
    "needs_more_evidence": "needs_more_evidence",
    "reject": "reject_patch",
    "defer": "defer",
}

ACTION_LABELS_ZH = {
    "accept_noop": "确认无需修改",
    "accept_patch": "批准修改",
    "accept_with_edit": "编辑后批准",
    "reject_patch": "拒绝修改",
    "defer": "暂缓",
    "needs_more_evidence": "需要更多证据",
    "split_current_vs_historical": "拆分为当前事实和历史事实",
    "mark_stale_only": "仅标记旧内容过期",
    "route_s2_build": "进入 S2 构建",
    "route_graph_extraction": "进入图抽取",
    "record_duplicate_noop": "记录重复且不修改",
    "append_s1_candidate": "追加 S1 候选",
    "append_s1_contradiction_candidate": "追加 S1 矛盾候选",
    "record_s2_duplicate_noop": "记录 S2 重复且不修改",
    "propose_s2_contradiction_review": "提出 S2 矛盾审核",
    "propose_new_s2_unit": "提出新 S2 单元",
    "review_graph_patch_candidate": "审核图补丁候选",
    "propose_graph_contradiction_refresh": "提出图矛盾刷新",
    "route_graph_extraction_for_new_candidate": "为新候选进入图抽取",
}

HUMAN_DECISION_LABELS_ZH = {
    "approve_recommended": "通过推荐处理",
    "approve_with_edit": "通过但手动编辑",
    "needs_more_evidence": "需要更多证据",
    "reject": "拒绝",
    "defer": "暂缓",
}

STATUS_LABELS_ZH = {
    "pending_review": "待审核",
    "approved_for_apply": "已批准待应用",
    "approved_noop": "已确认无需修改",
    "approved_with_edit": "已编辑批准",
    "rejected": "已拒绝",
    "deferred": "已暂缓",
    "needs_more_evidence": "需要更多证据",
}

PRIORITY_LABELS_ZH = {
    "high": "高优先级",
    "medium": "中优先级",
    "low": "低优先级",
}

LAYER_LABELS_ZH = {
    "s1": "S1 证据记忆",
    "s2": "S2 画像候选",
    "graph": "逻辑图候选",
}

REVIEW_STAGE_LABELS = {
    "s1_review": "S1 review",
    "s2_review": "S2 review",
    "graph_review": "Graph review",
    "incremental_review": "Incremental review",
}

RISK_LABELS_ZH = {
    "candidate_not_truth": "候选，不是真相",
    "graph_is_not_proof": "图不是证明",
    "high_impact_update": "高影响更新",
    "s2_materialization_pending": "S2 尚未落成",
    "graph_materialization_pending": "图尚未落成",
    "missing_context": "缺少上下文",
    "missing_evidence_refs": "缺少证据引用",
}

RUBRIC_LABELS_ZH = {
    "Is the proposed action supported by evidence?": "这个修改是否被证据支持？",
    "Is the target subject and attribution correct?": "主体和归因是否正确？",
    "Should this affect current view, historical view, or no view?": "它应该影响当前视图、历史视图，还是不进入视图？",
    "Does the patch require edit, split, or more evidence before apply?": "应用前是否需要编辑、拆分或补充证据？",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must be object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def queue_by_id(queue_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("review_item_id") or ""): item for item in queue_items if item.get("review_item_id")}


def decisions_by_item(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in decisions:
        item_id = str(row.get("review_item_id") or "")
        if item_id:
            latest[item_id] = row
    return latest


def validate_review_decision(queue_item: dict[str, Any], review_action: str, *, batch: bool = False) -> list[str]:
    errors: list[str] = []
    allowed = set(queue_item.get("allowed_review_actions") or [])
    if review_action not in allowed:
        errors.append("review_action_not_allowed_for_item")
    if review_action not in ACTION_TO_STATUS:
        errors.append("unknown_review_action")
    if batch and review_action not in BATCH_SAFE_ACTIONS:
        errors.append("batch_action_not_safe")
    if batch and queue_item.get("queue_priority") != "low":
        errors.append("batch_only_low_priority")
    if batch and queue_item.get("patch_action") not in {"record_duplicate_noop", "record_s2_duplicate_noop"}:
        errors.append("batch_only_duplicate_noop_patch")
    return errors


def recommended_action(queue_item: dict[str, Any]) -> str:
    action = str(queue_item.get("default_recommendation") or "")
    if action:
        return action
    allowed = [str(value) for value in queue_item.get("allowed_review_actions") or []]
    for candidate in allowed:
        if candidate not in {"reject_patch", "defer", "needs_more_evidence"}:
            return candidate
    return allowed[0] if allowed else ""


def resolve_human_decision(queue_item: dict[str, Any], human_decision: str, review_action: str = "") -> tuple[str, str, list[str]]:
    errors: list[str] = []
    human = human_decision or "legacy_review_action"
    if human == "legacy_review_action":
        internal_action = review_action
    elif human in {"approve_recommended", "approve_with_edit"}:
        internal_action = recommended_action(queue_item)
    elif human in HUMAN_DECISION_DIRECT_ACTION:
        internal_action = HUMAN_DECISION_DIRECT_ACTION[human]
    else:
        internal_action = ""
        errors.append("unknown_human_decision")
    if human != "legacy_review_action" and human not in HUMAN_DECISIONS:
        errors.append("unknown_human_decision")
    if not internal_action:
        errors.append("missing_internal_review_action")
    else:
        errors.extend(validate_review_decision(queue_item, internal_action))
    return human, internal_action, errors


def build_review_decision(
    queue_item: dict[str, Any],
    *,
    review_action: str,
    human_decision: str = "",
    review_notes: str = "",
    reviewer_id: str = "local_user",
    edited_payload: dict[str, Any] | None = None,
    batch: bool = False,
    reviewed_at: str | None = None,
) -> dict[str, Any]:
    human, internal_action, errors = resolve_human_decision(queue_item, human_decision, review_action)
    if batch:
        errors.extend(validate_review_decision(queue_item, internal_action, batch=True))
    if errors:
        raise ValueError(";".join(errors))
    status = "approved_with_edit" if human == "approve_with_edit" else ACTION_TO_STATUS[internal_action]
    if status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError("invalid_review_status")
    return {
        "schema_version": SCHEMA_VERSION,
        "review_item_id": queue_item["review_item_id"],
        "patch_id": queue_item["patch_id"],
        "patch_hash": queue_item["patch_hash"],
        "source_context_hash": queue_item["source_context_hash"],
        "operation_id": queue_item.get("operation_id") or "",
        "review_stage": queue_item.get("review_stage") or "",
        "layer": queue_item.get("layer") or "",
        "patch_action": queue_item.get("patch_action") or "",
        "human_decision": human,
        "approved_recommended_action": internal_action if human in {"approve_recommended", "approve_with_edit"} else "",
        "internal_review_action": internal_action,
        "review_action": internal_action,
        "review_status": status,
        "review_notes": review_notes,
        "edited_payload": edited_payload or {},
        "reviewer_id": reviewer_id,
        "batch_decision": batch,
        "reviewed_at": reviewed_at or now_iso(),
        "write_permission": False,
        "durable_writes_executed": False,
        "apply_executed": False,
    }


def upsert_review_decision(decisions: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, Any]]:
    item_id = str(decision.get("review_item_id") or "")
    if not item_id:
        raise ValueError("decision_missing_review_item_id")
    result: list[dict[str, Any]] = []
    replaced = False
    for row in decisions:
        if str(row.get("review_item_id") or "") == item_id:
            result.append(decision)
            replaced = True
        else:
            result.append(row)
    if not replaced:
        result.append(decision)
    return result


def build_batch_decisions(queue_items: list[dict[str, Any]], *, review_action: str, reviewer_id: str = "local_user") -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for item in queue_items:
        errors = validate_review_decision(item, review_action, batch=True)
        if errors:
            continue
        decisions.append(build_review_decision(item, review_action=review_action, reviewer_id=reviewer_id, batch=True))
    return decisions


def build_review_session_manifest(
    queue_items: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    finalized_by: str = "local_user",
    allow_partial_finalize: bool = True,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    queue_ids = [str(item.get("review_item_id") or "") for item in queue_items if item.get("review_item_id")]
    latest = decisions_by_item(decisions)
    decided_ids = [item_id for item_id in queue_ids if item_id in latest]
    undecided_ids = [item_id for item_id in queue_ids if item_id not in latest]
    if undecided_ids and not allow_partial_finalize:
        raise ValueError("review_session_has_undecided_items")
    if not decided_ids:
        raise ValueError("review_session_has_no_decisions")
    latest_decisions = [latest[item_id] for item_id in decided_ids]
    return {
        "schema_version": SESSION_SCHEMA_VERSION,
        "review_session_status": "finalized",
        "finalized_at": finalized_at or now_iso(),
        "finalized_by": finalized_by,
        "allow_partial_finalize": allow_partial_finalize,
        "queue_item_count": len(queue_ids),
        "decision_count": len(decided_ids),
        "undecided_count": len(undecided_ids),
        "decided_review_item_ids": decided_ids,
        "undecided_review_item_ids": undecided_ids,
        "queue_hash": stable_hash(queue_items),
        "decisions_hash": stable_hash(latest_decisions),
        "write_permission": False,
        "apply_executed": False,
        "durable_writes_executed": False,
    }


def item_status(item: dict[str, Any], decision: dict[str, Any] | None) -> str:
    return str((decision or {}).get("review_status") or item.get("review_status") or "pending_review")


def label_action(action: Any) -> str:
    text = str(action or "")
    zh = ACTION_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_human_decision(decision: Any) -> str:
    text = str(decision or "")
    zh = HUMAN_DECISION_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_status(status: Any) -> str:
    text = str(status or "")
    zh = STATUS_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_priority(priority: Any) -> str:
    text = str(priority or "")
    zh = PRIORITY_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_layer(layer: Any) -> str:
    text = str(layer or "")
    zh = LAYER_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_review_stage(stage: Any) -> str:
    text = str(stage or "")
    label = REVIEW_STAGE_LABELS.get(text)
    return f"{label} ({text})" if label else text


def label_risk(flag: Any) -> str:
    text = str(flag or "")
    zh = RISK_LABELS_ZH.get(text)
    return f"{zh} ({text})" if zh else text


def label_rubric(line: Any) -> str:
    text = str(line or "")
    zh = RUBRIC_LABELS_ZH.get(text)
    return f"{zh} / {text}" if zh else text


def read_translations(path: Path | None) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row.get("review_item_id") or "")
        if item_id:
            result[item_id] = row
    return result


def context_translation_map(translation: dict[str, Any] | None) -> dict[str, str]:
    if not translation:
        return {}
    result: dict[str, str] = {}
    value = translation.get("context_translations") or []
    if isinstance(value, dict):
        for key, text in value.items():
            result[str(key)] = str(text)
    elif isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                object_id = str(row.get("object_id") or "")
                text = str(row.get("text_zh") or row.get("translation_zh") or "")
                if object_id and text:
                    result[object_id] = text
    return result


def available_human_decisions(queue_item: dict[str, Any]) -> list[str]:
    allowed = set(queue_item.get("allowed_review_actions") or [])
    result: list[str] = []
    if recommended_action(queue_item) in allowed:
        result.extend(["approve_recommended", "approve_with_edit"])
    for human, internal in HUMAN_DECISION_DIRECT_ACTION.items():
        if internal in allowed:
            result.append(human)
    return result


def filter_items(items: list[dict[str, Any]], decisions: dict[str, dict[str, Any]], params: dict[str, list[str]]) -> list[dict[str, Any]]:
    layer = (params.get("layer") or [""])[0]
    stage = (params.get("stage") or [""])[0]
    priority = (params.get("priority") or [""])[0]
    status = (params.get("status") or [""])[0]
    action = (params.get("action") or [""])[0]
    query = ((params.get("q") or [""])[0]).lower().strip()
    result: list[dict[str, Any]] = []
    for item in items:
        decision = decisions.get(item["review_item_id"])
        if stage and item.get("review_stage") != stage:
            continue
        if layer and item.get("layer") != layer:
            continue
        if priority and item.get("queue_priority") != priority:
            continue
        if action and item.get("patch_action") != action:
            continue
        if status and item_status(item, decision) != status:
            continue
        if query and query not in json.dumps(item, ensure_ascii=False).lower():
            continue
        result.append(item)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(result, key=lambda row: (priority_order.get(str(row.get("queue_priority")), 9), str(row.get("layer")), str(row.get("patch_action"))))


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def render_context_cards(cards: list[dict[str, Any]], translations: dict[str, str] | None = None) -> str:
    chunks: list[str] = []
    translations = translations or {}
    for card in cards:
        object_id = str(card.get("object_id") or "")
        refs = ", ".join(str(ref) for ref in card.get("evidence_refs") or [])
        warnings = ", ".join(str(w) for w in card.get("warnings") or [])
        translation = translations.get(object_id)
        translation_html = (
            f"""
              <aside class="translation">
                <strong>中文辅助翻译</strong>
                <p>{esc(translation)}</p>
                <small>翻译仅帮助阅读，不作为证据；审核仍以原文、证据引用和 hash 为准。</small>
              </aside>
            """
            if translation
            else ""
        )
        chunks.append(
            f"""
            <section class="context-card">
              <div class="meta">{esc(card.get('role'))} · {esc(object_id)} · {esc(card.get('overlay_status'))}</div>
              <p class="source-text">{esc(card.get('text'))}</p>
              {translation_html}
              <details><summary>证据引用</summary><pre>{esc(refs)}</pre></details>
              {f'<details><summary>警告</summary><pre>{esc(warnings)}</pre></details>' if warnings else ''}
            </section>
            """
        )
    return "\n".join(chunks)


def render_detail_html(
    selected: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    translation: dict[str, Any] | None = None,
    *,
    saved: bool = False,
) -> str:
    saved_banner = '<div class="notice saved">已保存审核决定。</div>' if saved else ""
    if not selected:
        return "<p>没有可审核条目。</p>"
    translation_summary = str((translation or {}).get("review_summary_zh") or "")
    translation_map = context_translation_map(translation)
    translation_note = (
        f"""
        <aside class="translation top-translation">
          <strong>中文辅助说明</strong>
          <p>{esc(translation_summary)}</p>
          <small>translation_is_not_evidence=true；这段说明不作为证据，也不替代原文证据。</small>
        </aside>
        """
        if translation_summary
        else ""
    )
    buttons = "\n".join(
        f'<button name="human_decision" value="{esc(decision_code)}">{esc(label_human_decision(decision_code))}</button>'
        for decision_code in available_human_decisions(selected)
    )
    risks = ", ".join(label_risk(flag) for flag in selected.get("risk_flags") or [])
    rubric = "".join(f"<li>{esc(label_rubric(line))}</li>" for line in selected.get("review_rubric") or [])
    default_action = recommended_action(selected)
    internal_actions = ", ".join(label_action(action) for action in selected.get("allowed_review_actions") or [])
    decision_summary = ""
    if decision:
        decision_summary = f"""
        <div class="notice decision">
          <strong>当前已保存决策</strong>
          <p>人类动作：{esc(label_human_decision(decision.get('human_decision')))}</p>
          <p>内部执行码：{esc(label_action(decision.get('internal_review_action') or decision.get('review_action')))}</p>
          <p>状态：{esc(label_status(decision.get('review_status')))}</p>
        </div>
        """
    return f"""
    {saved_banner}
    <h2>{esc(label_action(selected.get('patch_action')))}</h2>
    <div class="chips">
      <span>{esc(label_priority(selected.get('queue_priority')))}</span>
      <span>{esc(label_review_stage(selected.get('review_stage')))}</span>
      <span>{esc(label_layer(selected.get('layer')))}</span>
      <span>{esc(label_status(item_status(selected, decision)))}</span>
      <span>推荐处理：{esc(label_action(default_action))}</span>
    </div>
    <p class="summary">{esc(selected.get('review_summary'))}</p>
    {translation_note}
    {decision_summary}
    <div class="grid">
      <section>
        <h3>上下文原文</h3>
        {render_context_cards(selected.get('context_cards') or [], translation_map)}
      </section>
      <section>
        <h3>审核</h3>
        <div class="recommendation">
          <h4>推荐处理方案</h4>
          <p>{esc(label_action(default_action))}</p>
          <small>主按钮是人类审核动作；内部执行码只作为后续 apply plan 的输入。</small>
          <details><summary>可用内部执行码</summary><pre>{esc(internal_actions)}</pre></details>
        </div>
        <form method="POST" action="/decide">
          <input type="hidden" name="review_item_id" value="{esc(selected['review_item_id'])}">
          <label>审核备注</label>
          <textarea name="review_notes" placeholder="可选：写下为什么批准、拒绝、暂缓或需要更多证据">{esc((decision or {}).get('review_notes') or '')}</textarea>
          <label>编辑后的 payload JSON（可选）</label>
          <textarea name="edited_payload" placeholder='{{}}'>{esc(json.dumps((decision or {}).get('edited_payload') or {}, ensure_ascii=False, indent=2))}</textarea>
          <div class="buttons">{buttons}</div>
        </form>
        <h3>风险标签</h3>
        <pre>{esc(risks)}</pre>
        <h3>审核准则</h3>
        <ul>{rubric}</ul>
        <h3>Hashes</h3>
        <pre>patch_hash: {esc(selected.get('patch_hash'))}
source_context_hash: {esc(selected.get('source_context_hash'))}</pre>
      </section>
    </div>
    """


def render_session_controls(
    items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    *,
    finalized: bool = False,
) -> str:
    queue_ids = [str(item.get("review_item_id") or "") for item in items if item.get("review_item_id")]
    decided_ids = [item_id for item_id in queue_ids if item_id in decisions]
    undecided_count = len(queue_ids) - len(decided_ids)
    current_decisions_hash = stable_hash([decisions[item_id] for item_id in decided_ids])
    current_queue_hash = stable_hash(items)
    manifest_status = ""
    if manifest:
        stale = manifest.get("queue_hash") != current_queue_hash or manifest.get("decisions_hash") != current_decisions_hash
        if stale:
            manifest_status = '<span class="session-warning">审核决定已变更，需要重新提交</span>'
        else:
            manifest_status = '<span class="session-finalized">本轮审核已提交</span>'
    if finalized:
        manifest_status = '<span class="session-finalized">本轮审核已提交</span>'
    disabled = "disabled" if not decided_ids else ""
    return f"""
    <div class="session-controls">
      <span>已审 {len(decided_ids)} / {len(queue_ids)}，未审 {undecided_count}</span>
      {manifest_status}
      <form method="POST" action="/finalize-review">
        <input type="hidden" name="allow_partial_finalize" value="true">
        <button {disabled}>提交本轮审核</button>
      </form>
    </div>
    """


def render_page(
    items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    params: dict[str, list[str]],
    translations: dict[str, dict[str, Any]] | None = None,
    session_manifest: dict[str, Any] | None = None,
) -> str:
    selected_id = (params.get("item") or [""])[0]
    filtered = filter_items(items, decisions, params)
    selected = next((item for item in items if item.get("review_item_id") == selected_id), None) or (filtered[0] if filtered else None)
    priority_filter = (params.get("priority") or [""])[0]
    layer_filter = (params.get("layer") or [""])[0]
    stage_filter = (params.get("stage") or [""])[0]
    status_filter = (params.get("status") or [""])[0]
    query_filter = (params.get("q") or [""])[0]
    translations = translations or {}
    session_html = render_session_controls(items, decisions, session_manifest, finalized=(params.get("finalized") or [""])[0] == "1")
    priority_options = ["high", "medium", "low"]
    stage_options = ["s1_review", "s2_review", "graph_review"]
    layer_options = ["s1", "s2", "graph"]
    status_options = ["pending_review", "approved_for_apply", "approved_noop", "approved_with_edit", "rejected", "deferred", "needs_more_evidence"]
    list_html = "\n".join(
        f"""
        <a class="item {esc(item.get('queue_priority'))} {'decided' if decisions.get(item['review_item_id']) else ''} {'selected' if selected and item['review_item_id'] == selected['review_item_id'] else ''}"
           href="/?item={esc(item['review_item_id'])}&stage={esc(stage_filter)}&layer={esc(layer_filter)}&priority={esc(priority_filter)}&status={esc(status_filter)}"
           data-item-id="{esc(item['review_item_id'])}">
          <strong>{esc(label_priority(item.get('queue_priority')))}</strong> · {esc(label_review_stage(item.get('review_stage')))} · {esc(label_layer(item.get('layer')))} · {esc(label_action(item.get('patch_action')))}
          <span>{esc(item.get('review_summary'))}</span>
          <em>{esc(label_status(item_status(item, decisions.get(item['review_item_id']))))}{' · 已处理' if decisions.get(item['review_item_id']) else ''}</em>
          {f'<small class="decision-badge">已处理</small>' if decisions.get(item['review_item_id']) else ''}
        </a>
        """
        for item in filtered
    )
    detail_html = render_detail_html(None, None)
    if selected:
        decision = decisions.get(selected["review_item_id"])
        translation = translations.get(str(selected["review_item_id"]) or "")
        detail_html = render_detail_html(selected, decision, translation, saved=(params.get("saved") or [""])[0] == "1")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>增量审核队列</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f5f7fa; }}
    header {{ padding: 14px 18px; background: #102033; color: white; display: flex; align-items: center; gap: 14px; }}
    header h1 {{ font-size: 18px; margin: 0; }}
    header form {{ margin-left: auto; display: flex; gap: 8px; align-items: center; }}
    .session-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; }}
    .session-controls form {{ margin-left: 0; }}
    .session-controls button {{ color: #102033; border-color: #b8d3bf; background: #eaf6ef; }}
    .session-controls button:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    .session-finalized {{ color: #bdf0ca; font-weight: 600; }}
    .session-warning {{ color: #ffd37a; font-weight: 600; }}
    select, input, textarea, button {{ font: inherit; }}
    select, input {{ padding: 6px 8px; border: 1px solid #c7d0dc; border-radius: 4px; }}
    main {{ display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 58px); }}
    nav {{ border-right: 1px solid #d8e0ea; background: white; overflow: auto; }}
    .item {{ display: block; padding: 12px 14px; border-bottom: 1px solid #edf1f5; color: inherit; text-decoration: none; }}
    .item strong {{ text-transform: uppercase; font-size: 12px; }}
    .item span {{ display: block; margin-top: 6px; font-size: 13px; line-height: 1.35; color: #405265; }}
    .item em {{ display: block; margin-top: 4px; font-size: 12px; color: #63758a; font-style: normal; }}
    .item.high {{ border-left: 5px solid #c93c37; }}
    .item.medium {{ border-left: 5px solid #c58a1c; }}
    .item.low {{ border-left: 5px solid #3c8b62; }}
    .item.decided {{ background: #f0f7f3; }}
    .item.decided em {{ color: #236844; font-weight: 600; }}
    .decision-badge {{ display: inline-block; margin-top: 8px; padding: 2px 6px; border-radius: 999px; background: #d9efdf; color: #236844; font-size: 12px; }}
    .item.selected {{ background: #eef5ff; }}
    .item.selected.decided {{ background: #e7f3ec; }}
    article {{ padding: 20px 24px; overflow: auto; }}
    article.loading {{ opacity: 0.55; transition: opacity 120ms ease; }}
    h2 {{ margin: 0 0 10px; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .chips span {{ padding: 4px 8px; border-radius: 4px; background: #e8edf3; font-size: 12px; }}
    .summary {{ font-size: 15px; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 0.9fr); gap: 18px; align-items: start; }}
    section {{ background: white; border: 1px solid #d8e0ea; border-radius: 6px; padding: 14px; }}
    .context-card {{ margin: 0 0 12px; }}
    .context-card .meta {{ color: #63758a; font-size: 12px; margin-bottom: 8px; }}
    .source-text {{ line-height: 1.55; }}
    .translation {{ border-left: 4px solid #8aa4c0; background: #f6f9fc; padding: 10px 12px; margin: 10px 0; }}
    .translation strong {{ display: block; margin-bottom: 4px; }}
    .translation p {{ margin: 4px 0; line-height: 1.55; }}
    .translation small {{ color: #63758a; }}
    .top-translation {{ margin-bottom: 14px; }}
    .recommendation {{ border: 1px solid #d8e0ea; background: #f9fbfd; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; }}
    .recommendation h4 {{ margin: 0 0 6px; }}
    .recommendation p {{ margin: 0 0 6px; font-weight: 600; }}
    .recommendation small {{ color: #63758a; }}
    .notice {{ border-radius: 6px; padding: 10px 12px; margin: 0 0 14px; }}
    .notice.saved {{ background: #eaf6ef; border: 1px solid #7bb18f; color: #1f5f3b; }}
    .notice.decision {{ background: #fff8e7; border: 1px solid #d6b35f; }}
    .notice.decision p {{ margin: 4px 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f2f5f8; padding: 10px; border-radius: 4px; }}
    textarea {{ width: 100%; min-height: 90px; box-sizing: border-box; border: 1px solid #c7d0dc; border-radius: 4px; padding: 8px; margin: 6px 0 12px; }}
    .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ border: 1px solid #8ca0b5; background: #fff; border-radius: 4px; padding: 7px 10px; cursor: pointer; }}
    button:hover {{ background: #edf4ff; }}
    .batch button {{ background: #eaf6ef; border-color: #7bb18f; }}
  </style>
</head>
<body>
  <!-- legacy-stage-label: S1 review (s1_review) -->
  <header>
    <h1>增量审核队列</h1>
    <span>{len(filtered)} / {len(items)} 条</span>
    {session_html}
    <form method="GET" action="/">
      <input name="q" placeholder="搜索" value="{esc(query_filter)}">
      <select name="priority"><option value="">全部优先级</option>{''.join(f'<option value="{p}" {"selected" if priority_filter == p else ""}>{esc(label_priority(p))}</option>' for p in priority_options)}</select>
      <select name="stage"><option value="">All review stages</option>{''.join(f'<option value="{p}" {"selected" if stage_filter == p else ""}>{esc(label_review_stage(p))}</option>' for p in stage_options)}</select>
      <select name="layer"><option value="">全部层级</option>{''.join(f'<option value="{p}" {"selected" if layer_filter == p else ""}>{esc(label_layer(p))}</option>' for p in layer_options)}</select>
      <select name="status"><option value="">全部状态</option>{''.join(f'<option value="{p}" {"selected" if status_filter == p else ""}>{esc(label_status(p))}</option>' for p in status_options)}</select>
      <button>筛选</button>
    </form>
    <form class="batch" method="POST" action="/batch-accept-noop">
      <button>批量确认低风险无需修改</button>
    </form>
  </header>
  <main>
    <nav>{list_html}</nav>
    <article id="review-detail">{detail_html}</article>
  </main>
  <script>
    const detail = document.getElementById("review-detail");
    document.querySelectorAll("nav a.item").forEach((link) => {{
      link.addEventListener("click", async (event) => {{
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        document.querySelectorAll("nav a.item").forEach((item) => item.classList.remove("selected"));
        link.classList.add("selected");
        detail.classList.add("loading");
        const url = new URL(link.href);
        url.searchParams.set("partial", "detail");
        try {{
          const response = await fetch(url);
          if (!response.ok) throw new Error(await response.text());
          detail.innerHTML = await response.text();
          history.pushState(null, "", link.href);
        }} catch (error) {{
          window.location.href = link.href;
        }} finally {{
          detail.classList.remove("loading");
        }}
      }});
    }});
  </script>
</body>
</html>"""


ACTION_LABELS_ZH.update(
    {
        "accept_noop": "确认无需修改",
        "accept_patch": "批准修改",
        "accept_with_edit": "编辑后批准",
        "reject_patch": "拒绝修改",
        "defer": "暂缓",
        "needs_more_evidence": "需要更多证据",
        "split_current_vs_historical": "拆分为当前事实和历史事实",
        "mark_stale_only": "仅标记旧内容过期",
        "route_s2_build": "进入 S2 构建",
        "route_graph_extraction": "进入图抽取",
        "record_duplicate_noop": "记录重复且不修改",
        "append_s1_candidate": "追加 S1 候选",
        "append_s1_contradiction_candidate": "追加 S1 矛盾候选",
        "record_s2_duplicate_noop": "记录 S2 重复且不修改",
        "propose_s2_contradiction_review": "提出 S2 矛盾审核",
        "propose_new_s2_unit": "提出新 S2 单元",
        "review_graph_patch_candidate": "审核图补丁候选",
        "propose_graph_contradiction_refresh": "提出图矛盾刷新",
        "route_graph_extraction_for_new_candidate": "为新候选进入图抽取",
    }
)
HUMAN_DECISION_LABELS_ZH.update(
    {
        "approve_recommended": "通过推荐处理",
        "approve_with_edit": "通过但手动编辑",
        "needs_more_evidence": "需要更多证据",
        "reject": "拒绝",
        "defer": "暂缓",
    }
)
STATUS_LABELS_ZH.update(
    {
        "pending_review": "待审核",
        "approved_for_apply": "已批准待应用",
        "approved_noop": "已确认无需修改",
        "approved_with_edit": "已编辑批准",
        "rejected": "已拒绝",
        "deferred": "已暂缓",
        "needs_more_evidence": "需要更多证据",
    }
)
PRIORITY_LABELS_ZH.update({"high": "高优先级", "medium": "中优先级", "low": "低优先级"})
LAYER_LABELS_ZH.update({"s1": "S1 证据记忆", "s2": "S2 画像候选", "graph": "逻辑图候选"})
REVIEW_STAGE_LABELS.update({"s1_review": "S1 审核", "s2_review": "S2 审核", "graph_review": "图审核", "incremental_review": "增量审核"})
RISK_LABELS_ZH.update(
    {
        "candidate_not_truth": "候选，不是真相",
        "graph_is_not_proof": "图不是证明",
        "high_impact_update": "高影响更新",
        "s2_materialization_pending": "S2 尚未落成",
        "graph_materialization_pending": "图尚未落成",
        "missing_context": "缺少上下文",
        "missing_evidence_refs": "缺少证据引用",
    }
)
RUBRIC_LABELS_ZH.update(
    {
        "Is the proposed action supported by evidence?": "这个修改是否被证据支持？",
        "Is the target subject and attribution correct?": "主体和归因是否正确？",
        "Should this affect current view, historical view, or no view?": "它应该影响当前视图、历史视图，还是不进入视图？",
        "Does the patch require edit, split, or more evidence before apply?": "应用前是否需要编辑、拆分或补充证据？",
    }
)


def item_display(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("review_display") or {}
    return value if isinstance(value, dict) else {}


def card_display(card: dict[str, Any]) -> dict[str, Any]:
    value = card.get("review_display") or {}
    return value if isinstance(value, dict) else {}


def display_text(item: dict[str, Any], key: str, fallback: Any = "") -> str:
    value = item_display(item).get(key)
    return str(value if value not in {None, ""} else fallback)


def render_context_cards(cards: list[dict[str, Any]], translations: dict[str, str] | None = None) -> str:
    chunks: list[str] = []
    translations = translations or {}
    for card in cards:
        object_id = str(card.get("object_id") or "")
        refs = [str(ref) for ref in card.get("evidence_refs") or []]
        warnings = [label_risk(w) for w in card.get("warnings") or []]
        display = card_display(card)
        text_zh = str(display.get("text_zh") or translations.get(object_id) or card.get("text") or "")
        role_zh = str(display.get("role_zh") or card.get("role") or "")
        status_zh = str(display.get("status_zh") or card.get("overlay_status") or "")
        evidence_note = str(display.get("evidence_note_zh") or "证据引用见下方。")
        warnings_html = (
            f"<div class=\"warn-line\">{'；'.join(esc(w) for w in warnings)}</div>"
            if warnings
            else ""
        )
        chunks.append(
            f"""
            <section class="context-card">
              <div class="meta">{esc(role_zh)} · {esc(status_zh)} · {esc(object_id)}</div>
              <p class="source-text">{esc(text_zh)}</p>
              <p class="evidence-note">{esc(evidence_note)}</p>
              {warnings_html}
              <details><summary>查看原始文本和证据引用</summary>
                <p>{esc(card.get('text'))}</p>
                <pre>{esc(', '.join(refs))}</pre>
              </details>
            </section>
            """
        )
    return "\n".join(chunks)


def render_detail_html(
    selected: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    translation: dict[str, Any] | None = None,
    *,
    saved: bool = False,
) -> str:
    saved_banner = '<div class="notice saved">已保存审核决定。</div>' if saved else ""
    if not selected:
        return "<p>没有可审核条目。</p>"
    translation_summary = str((translation or {}).get("review_summary_zh") or "")
    title = display_text(selected, "title_zh", label_action(selected.get("patch_action")))
    summary = display_text(selected, "summary_zh", selected.get("review_summary") or "")
    recommendation_zh = display_text(selected, "recommended_action_zh", label_action(recommended_action(selected)))
    why_review = display_text(selected, "why_review_zh", "")
    decision_hint = display_text(selected, "decision_hint_zh", "")
    translation_map = context_translation_map(translation)
    buttons = "\n".join(
        f'<button name="human_decision" value="{esc(decision_code)}">{esc(label_human_decision(decision_code))}</button>'
        for decision_code in available_human_decisions(selected)
    )
    risks = "；".join(label_risk(flag) for flag in selected.get("risk_flags") or [])
    rubric = "".join(f"<li>{esc(label_rubric(line))}</li>" for line in selected.get("review_rubric") or [])
    default_action = recommended_action(selected)
    internal_actions = ", ".join(label_action(action) for action in selected.get("allowed_review_actions") or [])
    translation_note = (
        f"""
        <aside class="translation top-translation">
          <strong>中文辅助说明</strong>
          <p>{esc(translation_summary)}</p>
          <small>translation_is_not_evidence=true；这段说明不作为证据，也不替代原文证据。</small>
        </aside>
        """
        if translation_summary
        else ""
    )
    decision_summary = ""
    if decision:
        decision_summary = f"""
        <div class="notice decision">
          <strong>当前已保存决策</strong>
          <p>人类动作：{esc(label_human_decision(decision.get('human_decision')))}</p>
          <p>内部执行码：{esc(label_action(decision.get('internal_review_action') or decision.get('review_action')))}</p>
          <p>状态：{esc(label_status(decision.get('review_status')))}</p>
        </div>
        """
    why_html = f"<p><strong>为什么需要审核：</strong>{esc(why_review)}</p>" if why_review else ""
    hint_html = f"<p><strong>怎么判断：</strong>{esc(decision_hint)}</p>" if decision_hint else ""
    return f"""
    {saved_banner}
    <h2>{esc(title)}</h2>
    <div class="chips">
      <span>{esc(label_priority(selected.get('queue_priority')))}</span>
      <span>{esc(label_review_stage(selected.get('review_stage')))}</span>
      <span>{esc(label_layer(selected.get('layer')))}</span>
      <span>{esc(label_status(item_status(selected, decision)))}</span>
    </div>
    <section class="human-summary">
      <p>{esc(summary)}</p>
      <p><strong>推荐处理：</strong>{esc(recommendation_zh)}</p>
      {why_html}
      {hint_html}
      <small>页面展示是审核辅助；最终 apply 仍读取下方保留的机器字段和证据引用。</small>
    </section>
    {translation_note}
    {decision_summary}
    <div class="grid">
      <section>
        <h3>审核上下文</h3>
        {render_context_cards(selected.get('context_cards') or [], translation_map)}
      </section>
      <section>
        <h3>审核动作</h3>
        <div class="recommendation">
          <h4>推荐处理方案</h4>
          <p>{esc(label_action(default_action))}</p>
          <small>按钮是人类审核动作；内部执行码只作为后续 apply plan 输入。</small>
          <details><summary>查看可用内部执行码</summary><pre>{esc(internal_actions)}</pre></details>
        </div>
        <form method="POST" action="/decide">
          <input type="hidden" name="review_item_id" value="{esc(selected['review_item_id'])}">
          <label>审核备注</label>
          <textarea name="review_notes" placeholder="可选：写下为什么批准、拒绝、暂缓或需要更多证据">{esc((decision or {}).get('review_notes') or '')}</textarea>
          <details>
            <summary>高级：编辑 payload JSON</summary>
            <textarea name="edited_payload" placeholder='{{}}'>{esc(json.dumps((decision or {}).get('edited_payload') or {}, ensure_ascii=False, indent=2))}</textarea>
          </details>
          <div class="buttons">{buttons}</div>
        </form>
        <h3>风险标签</h3>
        <p>{esc(risks)}</p>
        <h3>审核准则</h3>
        <ul>{rubric}</ul>
        <details><summary>机器字段和 hash</summary>
          <pre>review_item_id: {esc(selected.get('review_item_id'))}
patch_action: {esc(selected.get('patch_action'))}
patch_hash: {esc(selected.get('patch_hash'))}
source_context_hash: {esc(selected.get('source_context_hash'))}</pre>
        </details>
      </section>
    </div>
    """


def render_session_controls(
    items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    *,
    finalized: bool = False,
) -> str:
    queue_ids = [str(item.get("review_item_id") or "") for item in items if item.get("review_item_id")]
    decided_ids = [item_id for item_id in queue_ids if item_id in decisions]
    undecided_count = len(queue_ids) - len(decided_ids)
    current_decisions_hash = stable_hash([decisions[item_id] for item_id in decided_ids])
    current_queue_hash = stable_hash(items)
    manifest_status = ""
    if manifest:
        stale = manifest.get("queue_hash") != current_queue_hash or manifest.get("decisions_hash") != current_decisions_hash
        manifest_status = (
            '<span class="session-warning">审核决定已变化，需要重新提交</span>'
            if stale
            else '<span class="session-finalized">本轮审核已提交</span>'
        )
    if finalized:
        manifest_status = '<span class="session-finalized">本轮审核已提交</span>'
    disabled = "disabled" if not decided_ids else ""
    return f"""
    <div class="session-controls">
      <span>已审 {len(decided_ids)} / {len(queue_ids)}，未审 {undecided_count}</span>
      {manifest_status}
      <form method="POST" action="/finalize-review">
        <input type="hidden" name="allow_partial_finalize" value="true">
        <button {disabled}>提交本轮审核</button>
      </form>
    </div>
    """


def render_page(
    items: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    params: dict[str, list[str]],
    translations: dict[str, dict[str, Any]] | None = None,
    session_manifest: dict[str, Any] | None = None,
) -> str:
    selected_id = (params.get("item") or [""])[0]
    filtered = filter_items(items, decisions, params)
    selected = next((item for item in items if item.get("review_item_id") == selected_id), None) or (filtered[0] if filtered else None)
    priority_filter = (params.get("priority") or [""])[0]
    layer_filter = (params.get("layer") or [""])[0]
    stage_filter = (params.get("stage") or [""])[0]
    status_filter = (params.get("status") or [""])[0]
    query_filter = (params.get("q") or [""])[0]
    translations = translations or {}
    session_html = render_session_controls(items, decisions, session_manifest, finalized=(params.get("finalized") or [""])[0] == "1")
    priority_options = ["high", "medium", "low"]
    stage_options = ["s1_review", "s2_review", "graph_review"]
    layer_options = ["s1", "s2", "graph"]
    status_options = ["pending_review", "approved_for_apply", "approved_noop", "approved_with_edit", "rejected", "deferred", "needs_more_evidence"]
    list_html = "\n".join(
        f"""
        <a class="item {esc(item.get('queue_priority'))} {'decided' if decisions.get(item['review_item_id']) else ''} {'selected' if selected and item['review_item_id'] == selected['review_item_id'] else ''}"
           href="/?item={esc(item['review_item_id'])}&stage={esc(stage_filter)}&layer={esc(layer_filter)}&priority={esc(priority_filter)}&status={esc(status_filter)}"
           data-item-id="{esc(item['review_item_id'])}">
          <strong>{esc(label_priority(item.get('queue_priority')))}</strong> · {esc(label_review_stage(item.get('review_stage')))} · {esc(label_layer(item.get('layer')))}
          <span>{esc(display_text(item, 'title_zh', item.get('review_summary')))}</span>
          <em>{esc(label_status(item_status(item, decisions.get(item['review_item_id']))))}{' · 已处理' if decisions.get(item['review_item_id']) else ''}</em>
          {f'<small class="decision-badge">已处理</small>' if decisions.get(item['review_item_id']) else ''}
        </a>
        """
        for item in filtered
    )
    detail_html = render_detail_html(None, None)
    if selected:
        decision = decisions.get(selected["review_item_id"])
        translation = translations.get(str(selected["review_item_id"]) or "")
        detail_html = render_detail_html(selected, decision, translation, saved=(params.get("saved") or [""])[0] == "1")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>增量审核队列</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17212b; background: #f5f7fa; }}
    header {{ padding: 14px 18px; background: #102033; color: white; display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
    header h1 {{ font-size: 18px; margin: 0; }}
    header form {{ display: flex; gap: 8px; align-items: center; }}
    .session-controls {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; }}
    .session-controls button {{ color: #102033; border-color: #b8d3bf; background: #eaf6ef; }}
    .session-controls button:disabled {{ opacity: 0.55; cursor: not-allowed; }}
    .session-finalized {{ color: #bdf0ca; font-weight: 600; }}
    .session-warning {{ color: #ffd37a; font-weight: 600; }}
    select, input, textarea, button {{ font: inherit; }}
    select, input {{ padding: 6px 8px; border: 1px solid #c7d0dc; border-radius: 4px; }}
    main {{ display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 58px); }}
    nav {{ border-right: 1px solid #d8e0ea; background: white; overflow: auto; }}
    .item {{ display: block; padding: 12px 14px; border-bottom: 1px solid #edf1f5; color: inherit; text-decoration: none; }}
    .item strong {{ text-transform: uppercase; font-size: 12px; }}
    .item span {{ display: block; margin-top: 6px; font-size: 13px; line-height: 1.35; color: #405265; }}
    .item em {{ display: block; margin-top: 4px; font-size: 12px; color: #63758a; font-style: normal; }}
    .item.high {{ border-left: 5px solid #c93c37; }}
    .item.medium {{ border-left: 5px solid #c58a1c; }}
    .item.low {{ border-left: 5px solid #3c8b62; }}
    .item.decided {{ background: #f0f7f3; }}
    .item.decided em {{ color: #236844; font-weight: 600; }}
    .decision-badge {{ display: inline-block; margin-top: 8px; padding: 2px 6px; border-radius: 999px; background: #d9efdf; color: #236844; font-size: 12px; }}
    .item.selected {{ background: #eef5ff; }}
    .item.selected.decided {{ background: #e7f3ec; }}
    article {{ padding: 20px 24px; overflow: auto; }}
    article.loading {{ opacity: 0.55; transition: opacity 120ms ease; }}
    h2 {{ margin: 0 0 10px; }}
    .chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .chips span {{ padding: 4px 8px; border-radius: 4px; background: #e8edf3; font-size: 12px; }}
    .human-summary {{ background: white; border: 1px solid #d8e0ea; border-radius: 6px; padding: 14px; margin-bottom: 14px; }}
    .human-summary p {{ line-height: 1.55; }}
    .grid {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 0.9fr); gap: 18px; align-items: start; }}
    section {{ background: white; border: 1px solid #d8e0ea; border-radius: 6px; padding: 14px; }}
    .context-card {{ margin: 0 0 12px; }}
    .context-card .meta {{ color: #63758a; font-size: 12px; margin-bottom: 8px; }}
    .source-text {{ line-height: 1.55; font-size: 15px; }}
    .evidence-note {{ color: #405265; font-size: 13px; }}
    .warn-line {{ color: #9b442f; font-size: 13px; }}
    .translation {{ border-left: 4px solid #8aa4c0; background: #f6f9fc; padding: 10px 12px; margin: 10px 0; }}
    .recommendation {{ border: 1px solid #d8e0ea; background: #f9fbfd; border-radius: 6px; padding: 10px 12px; margin-bottom: 12px; }}
    .notice {{ border-radius: 6px; padding: 10px 12px; margin: 0 0 14px; }}
    .notice.saved {{ background: #eaf6ef; border: 1px solid #7bb18f; color: #1f5f3b; }}
    .notice.decision {{ background: #fff8e7; border: 1px solid #d6b35f; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f2f5f8; padding: 10px; border-radius: 4px; }}
    textarea {{ width: 100%; min-height: 90px; box-sizing: border-box; border: 1px solid #c7d0dc; border-radius: 4px; padding: 8px; margin: 6px 0 12px; }}
    .buttons {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    button {{ border: 1px solid #8ca0b5; background: #fff; border-radius: 4px; padding: 7px 10px; cursor: pointer; }}
    button:hover {{ background: #edf4ff; }}
    .batch button {{ background: #eaf6ef; border-color: #7bb18f; }}
  </style>
</head>
<body>
  <!-- legacy-stage-label: S1 review (s1_review) -->
  <header>
    <h1>增量审核队列</h1>
    <span>{len(filtered)} / {len(items)} 条</span>
    {session_html}
    <form method="GET" action="/">
      <input name="q" placeholder="搜索" value="{esc(query_filter)}">
      <select name="priority"><option value="">全部优先级</option>{''.join(f'<option value="{p}" {"selected" if priority_filter == p else ""}>{esc(label_priority(p))}</option>' for p in priority_options)}</select>
      <select name="stage"><option value="">全部审核阶段</option>{''.join(f'<option value="{p}" {"selected" if stage_filter == p else ""}>{esc(label_review_stage(p))}</option>' for p in stage_options)}</select>
      <select name="layer"><option value="">全部层级</option>{''.join(f'<option value="{p}" {"selected" if layer_filter == p else ""}>{esc(label_layer(p))}</option>' for p in layer_options)}</select>
      <select name="status"><option value="">全部状态</option>{''.join(f'<option value="{p}" {"selected" if status_filter == p else ""}>{esc(label_status(p))}</option>' for p in status_options)}</select>
      <button>筛选</button>
    </form>
    <form class="batch" method="POST" action="/batch-accept-noop">
      <button>批量确认低风险无需修改</button>
    </form>
  </header>
  <main>
    <nav>{list_html}</nav>
    <article id="review-detail">{detail_html}</article>
  </main>
  <script>
    const detail = document.getElementById("review-detail");
    document.querySelectorAll("nav a.item").forEach((link) => {{
      link.addEventListener("click", async (event) => {{
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        document.querySelectorAll("nav a.item").forEach((item) => item.classList.remove("selected"));
        link.classList.add("selected");
        detail.classList.add("loading");
        try {{
          const url = new URL(link.href);
          url.searchParams.set("partial", "detail");
          const response = await fetch(url);
          if (!response.ok) throw new Error(await response.text());
          detail.innerHTML = await response.text();
          history.pushState(null, "", link.href);
        }} catch (error) {{
          window.location.href = link.href;
        }} finally {{
          detail.classList.remove("loading");
        }}
      }});
    }});
  </script>
</body>
</html>"""


@dataclass
class ReviewAppState:
    review_queue_path: Path
    decisions_path: Path
    session_manifest_path: Path
    lock: threading.Lock
    translations_path: Path | None = None

    def load_items(self) -> list[dict[str, Any]]:
        return read_jsonl(self.review_queue_path)

    def load_decisions(self) -> list[dict[str, Any]]:
        return read_jsonl(self.decisions_path)

    def load_translations(self) -> dict[str, dict[str, Any]]:
        return read_translations(self.translations_path)

    def load_session_manifest(self) -> dict[str, Any] | None:
        return read_json(self.session_manifest_path)

    def save_decisions(self, rows: list[dict[str, Any]]) -> None:
        write_jsonl(self.decisions_path, rows)

    def save_session_manifest(self, manifest: dict[str, Any]) -> None:
        write_json(self.session_manifest_path, manifest)


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "IncrementalReviewWebUI/0.1"

    @property
    def state(self) -> ReviewAppState:
        return self.server.state  # type: ignore[attr-defined]

    def send_text(self, status: int, body: str, content_type: str = "text/plain; charset=utf-8") -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_text(HTTPStatus.OK, "ok")
            return
        if parsed.path != "/":
            self.send_text(HTTPStatus.NOT_FOUND, "not found")
            return
        with self.state.lock:
            items = self.state.load_items()
            decisions = decisions_by_item(self.state.load_decisions())
            translations = self.state.load_translations()
            session_manifest = self.state.load_session_manifest()
        params = parse_qs(parsed.query)
        if (params.get("partial") or [""])[0] == "detail":
            selected_id = (params.get("item") or [""])[0]
            selected = next((item for item in items if item.get("review_item_id") == selected_id), None)
            decision = decisions.get(selected_id) if selected else None
            translation = translations.get(selected_id) if selected else None
            body = render_detail_html(selected, decision, translation, saved=(params.get("saved") or [""])[0] == "1")
        else:
            body = render_page(items, decisions, params, translations, session_manifest)
        self.send_text(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        fields = {key: values[0] for key, values in parse_qs(payload).items()}
        try:
            with self.state.lock:
                items = self.state.load_items()
                decisions = self.state.load_decisions()
                by_id = queue_by_id(items)
                if parsed.path == "/decide":
                    item_id = fields.get("review_item_id", "")
                    item = by_id.get(item_id)
                    if not item:
                        raise ValueError("review_item_not_found")
                    edited_payload_raw = fields.get("edited_payload", "").strip()
                    edited_payload = json.loads(edited_payload_raw) if edited_payload_raw else {}
                    if not isinstance(edited_payload, dict):
                        raise ValueError("edited_payload_must_be_object")
                    decision = build_review_decision(
                        item,
                        review_action=fields.get("review_action", ""),
                        human_decision=fields.get("human_decision", ""),
                        review_notes=fields.get("review_notes", ""),
                        edited_payload=edited_payload,
                    )
                    self.state.save_decisions(upsert_review_decision(decisions, decision))
                    self.send_response(303)
                    self.send_header("Location", f"/?item={item_id}&saved=1")
                    self.end_headers()
                    return
                if parsed.path == "/batch-accept-noop":
                    new_decisions = build_batch_decisions(items, review_action="accept_noop")
                    merged = decisions
                    for decision in new_decisions:
                        merged = upsert_review_decision(merged, decision)
                    self.state.save_decisions(merged)
                    self.send_response(303)
                    self.send_header("Location", "/?priority=low")
                    self.end_headers()
                    return
                if parsed.path == "/finalize-review":
                    allow_partial = fields.get("allow_partial_finalize", "true").lower() == "true"
                    manifest = build_review_session_manifest(
                        items,
                        decisions,
                        allow_partial_finalize=allow_partial,
                    )
                    self.state.save_session_manifest(manifest)
                    self.send_response(303)
                    self.send_header("Location", "/?finalized=1")
                    self.end_headers()
                    return
        except Exception as exc:
            self.send_text(HTTPStatus.BAD_REQUEST, f"bad request: {exc}")
            return
        self.send_text(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: Any) -> None:
        return


def run_server(
    *,
    review_queue_jsonl: Path,
    output_dir: Path,
    host: str,
    port: int,
    review_translations_jsonl: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / DEFAULT_DECISIONS_FILENAME
    session_manifest_path = output_dir / DEFAULT_SESSION_MANIFEST_FILENAME
    if not decisions_path.exists():
        write_jsonl(decisions_path, [])
    state = ReviewAppState(
        review_queue_path=review_queue_jsonl,
        decisions_path=decisions_path,
        session_manifest_path=session_manifest_path,
        lock=threading.Lock(),
        translations_path=review_translations_jsonl,
    )
    server = ThreadingHTTPServer((host, port), ReviewHandler)
    server.state = state  # type: ignore[attr-defined]
    print(f"Review WebUI: http://{host}:{port}")
    print(f"Review decisions: {decisions_path}")
    print(f"Review session manifest: {session_manifest_path}")
    if review_translations_jsonl:
        print(f"Review translations: {review_translations_jsonl}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue-jsonl", required=True, type=Path)
    parser.add_argument("--review-translations-jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    run_server(
        review_queue_jsonl=args.review_queue_jsonl,
        review_translations_jsonl=args.review_translations_jsonl,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
