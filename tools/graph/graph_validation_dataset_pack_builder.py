"""Build cross-lingual v0.3 graph validation packs from external annotations.

This tool is a dataset suitability probe, not a production extractor. It uses
external gold/proxy annotations to create annotation-derived graph tables so we
can test whether candidate validation texts naturally form richer graph shapes.
It does not write graph truth, durable memory, or support-checker authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "graph_v03.validation_dataset_pack.v0.1"
NODE_SCHEMA_VERSION = "graph_v03.node_table.v0.2"
EDGE_SCHEMA_VERSION = "graph_v03.edge_table.v0.2"
DEFAULT_EXTERNAL_ROOT = Path("external_references/graph_v03_validation_datasets")
DEFAULT_OUTPUT_ROOT = Path("users")


@dataclass
class Entity:
    entity_id: str
    entity_type: str
    start: int
    end: int
    text: str


@dataclass
class Relation:
    relation_id: str
    relation_type: str
    arg1: str
    arg2: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}:{digest}"


def slug(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "-", value).strip("-").lower()
    return compact[:80] or "sample"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_brat(path: Path) -> tuple[list[Entity], list[Relation], list[str]]:
    entities: list[Entity] = []
    relations: list[Relation] = []
    coref_lines: list[str] = []
    if not path.exists():
        return entities, relations, coref_lines
    for line in read_text(path).splitlines():
        if line.startswith("T"):
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            meta = parts[1].split()
            if len(meta) < 3 or not meta[1].isdigit() or not meta[2].isdigit():
                continue
            entities.append(
                Entity(
                    entity_id=parts[0],
                    entity_type=meta[0],
                    start=int(meta[1]),
                    end=int(meta[2]),
                    text=parts[2],
                )
            )
        elif line.startswith("R"):
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            meta = parts[1].split()
            values: dict[str, str] = {}
            for item in meta[1:]:
                if ":" in item:
                    key, value = item.split(":", 1)
                    values[key] = value
            relations.append(
                Relation(
                    relation_id=parts[0],
                    relation_type=meta[0] if meta else "related",
                    arg1=values.get("Arg1", ""),
                    arg2=values.get("Arg2", ""),
                )
            )
        elif line.startswith("*"):
            coref_lines.append(line)
    return entities, relations, coref_lines


def sentence_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?。！？]\s+", text):
        end = match.end()
        if end > start:
            ranges.append((start, end))
        start = end
    if start < len(text):
        ranges.append((start, len(text)))
    return [(start, end) for start, end in ranges if text[start:end].strip()]


def label_key(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).lower()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relation_type_for_external(raw_type: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]+", "_", raw_type.strip()).strip("_").lower()
    return normalized or "external_relation"


def rows_to_graph_stats(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    degree: Counter[str] = Counter()
    for edge in edges:
        degree[str(edge.get("source_node_id") or "")] += 1
        degree[str(edge.get("target_node_id") or "")] += 1
    total_degree = sum(degree.values()) or 1
    top_share = max(degree.values(), default=0) / total_degree
    relation_counts = Counter(str(edge.get("relation_type") or "unknown") for edge in edges)
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "relation_type_counts": dict(relation_counts.most_common(20)),
        "top_degree_share": round(top_share, 4),
        "top_degree_node_id": degree.most_common(1)[0][0] if degree else "",
        "mesh_potential": "good" if len(nodes) >= 25 and len(edges) >= 40 and top_share <= 0.2 else "limited",
        "graph_is_not_proof": True,
        "support_status": "not_checked",
    }


def build_litbank_annotation_graph(sample_id: str, text_path: Path, entity_ann: Path, event_ann: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    text = read_text(text_path)
    entities, _, _ = parse_brat(entity_ann)
    events, _, _ = parse_brat(event_ann)
    ranges = sentence_ranges(text)
    node_by_key: dict[str, dict[str, Any]] = {}

    def add_node(kind: str, label: str, source_id: str, entity_type: str, evidence_ref: str, quote: str) -> str:
        key = f"{kind}:{label_key(label)}"
        if key not in node_by_key:
            node_id = stable_id("graph_node", f"{sample_id}:{key}")
            node_by_key[key] = {
                "schema_version": NODE_SCHEMA_VERSION,
                "node_id": node_id,
                "label": label,
                "normalized_key": key,
                "entity_type": entity_type,
                "entity_quality_hint": "external_annotation",
                "entity_quality_reasons": ["litbank_annotation_baseline"],
                "description": f"LitBank {kind} annotation baseline node.",
                "evidence_refs": [evidence_ref],
                "raw_backpointer_refs": [source_id],
                "source_refs": [sample_id],
                "source_text_quotes": [quote[:240]],
                "warnings": ["annotation_baseline_not_graph_truth"],
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        else:
            row = node_by_key[key]
            for field, value in (("evidence_refs", evidence_ref), ("raw_backpointer_refs", source_id)):
                if value not in row[field]:
                    row[field].append(value)
        return str(node_by_key[key]["node_id"])

    entity_refs_by_sentence: list[tuple[tuple[int, int], list[tuple[Entity, str]]]] = []
    for start, end in ranges:
        sentence_entities: list[tuple[Entity, str]] = []
        for entity in entities:
            if start <= entity.start < end:
                quote = text[max(start, entity.start - 120) : min(end, entity.end + 120)].strip()
                ref = f"litbank:{sample_id}:{entity.entity_id}"
                node_id = add_node("entity", entity.text, entity.entity_id, entity.entity_type, ref, quote)
                sentence_entities.append((entity, node_id))
        if sentence_entities:
            entity_refs_by_sentence.append(((start, end), sentence_entities))

    edges_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (start, end), sentence_entities in entity_refs_by_sentence:
        sentence = text[start:end].strip()
        events_in_sentence = [event for event in events if start <= event.start < end]
        unique_entities = []
        seen_entity_nodes: set[str] = set()
        for entity, node_id in sentence_entities:
            if node_id not in seen_entity_nodes:
                seen_entity_nodes.add(node_id)
                unique_entities.append((entity, node_id))
        for event in events_in_sentence[:8]:
            event_ref = f"litbank:{sample_id}:{event.entity_id}"
            event_node_id = add_node("event", event.text, event.entity_id, "event", event_ref, sentence)
            for entity, entity_node_id in unique_entities[:8]:
                edge_key = (entity_node_id, event_node_id, "participates_in_event_context")
                if edge_key not in edges_by_key:
                    edge_id = stable_id("graph_edge", f"{sample_id}:{edge_key}")
                    edges_by_key[edge_key] = {
                        "schema_version": EDGE_SCHEMA_VERSION,
                        "edge_id": edge_id,
                        "source_node_id": entity_node_id,
                        "target_node_id": event_node_id,
                        "source_label": entity.text,
                        "target_label": event.text,
                        "source_entity_type": entity.entity_type,
                        "target_entity_type": "event",
                        "relation_type": "participates_in",
                        "raw_relation_types": ["litbank_entity_event_same_sentence"],
                        "description": "Entity and event share a LitBank sentence; useful for validation mesh only.",
                        "evidence_refs": [f"litbank:{sample_id}:{entity.entity_id}", event_ref],
                        "raw_backpointer_refs": [entity.entity_id, event.entity_id],
                        "source_refs": [sample_id],
                        "source_text_quotes": [sentence[:360]],
                        "source_perspective": "narrator_or_text",
                        "attribution_status": "annotation_context_only",
                        "confidence_hint": "annotation_baseline",
                        "generic_relation_review_hint": "not_generic",
                        "generic_relation_review_reasons": [],
                        "evidence_count": 1,
                        "weight": 1.0,
                        "warnings": ["litbank_same_sentence_edge_not_relation_truth", "annotation_baseline_not_graph_truth"],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                else:
                    row = edges_by_key[edge_key]
                    if sentence[:360] not in row["source_text_quotes"]:
                        row["source_text_quotes"].append(sentence[:360])
                    row["evidence_count"] += 1

    nodes = list(node_by_key.values())
    edges = list(edges_by_key.values())
    metrics = rows_to_graph_stats(nodes, edges)
    metrics.update(
        {
            "sample_id": sample_id,
            "language": "en",
            "source_dataset": "litbank",
            "entity_annotation_count": len(entities),
            "event_annotation_count": len(events),
            "text_char_count": len(text),
        }
    )
    return nodes, edges, metrics


def build_chinese_re_annotation_graph(sample_id: str, text_path: Path, ann_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    text = read_text(text_path)
    entities, relations, coref_lines = parse_brat(ann_path)
    entity_by_id = {entity.entity_id: entity for entity in entities}
    relation_endpoint_ids = {relation.arg1 for relation in relations} | {relation.arg2 for relation in relations}
    node_by_entity: dict[str, dict[str, Any]] = {}
    for entity in entities:
        if entity.entity_id not in relation_endpoint_ids:
            continue
        node_id = stable_id("graph_node", f"{sample_id}:{entity.entity_id}:{entity.text}")
        quote = text[max(0, entity.start - 120) : min(len(text), entity.end + 120)].strip()
        node_by_entity[entity.entity_id] = {
            "schema_version": NODE_SCHEMA_VERSION,
            "node_id": node_id,
            "label": entity.text,
            "normalized_key": f"{entity.entity_type}:{label_key(entity.text)}",
            "entity_type": entity.entity_type,
            "entity_quality_hint": "external_annotation",
            "entity_quality_reasons": ["chinese_literature_ner_re_annotation"],
            "description": "Chinese literature NER/RE annotation baseline node.",
            "evidence_refs": [f"zh-lit:{sample_id}:{entity.entity_id}"],
            "raw_backpointer_refs": [entity.entity_id],
            "source_refs": [sample_id],
            "source_text_quotes": [quote[:240]],
            "warnings": ["annotation_baseline_not_graph_truth"],
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
        }
    edges: list[dict[str, Any]] = []
    for relation in relations:
        source = entity_by_id.get(relation.arg1)
        target = entity_by_id.get(relation.arg2)
        source_node = node_by_entity.get(relation.arg1)
        target_node = node_by_entity.get(relation.arg2)
        if not source or not target or not source_node or not target_node:
            continue
        span_start = max(0, min(source.start, target.start) - 160)
        span_end = min(len(text), max(source.end, target.end) + 160)
        quote = text[span_start:span_end].strip()
        raw_type = relation.relation_type
        relation_type = relation_type_for_external(raw_type)
        edges.append(
            {
                "schema_version": EDGE_SCHEMA_VERSION,
                "edge_id": stable_id("graph_edge", f"{sample_id}:{relation.relation_id}:{source.entity_id}:{target.entity_id}:{raw_type}"),
                "source_node_id": source_node["node_id"],
                "target_node_id": target_node["node_id"],
                "source_label": source.text,
                "target_label": target.text,
                "source_entity_type": source.entity_type,
                "target_entity_type": target.entity_type,
                "relation_type": relation_type,
                "raw_relation_types": [raw_type],
                "description": "Explicit relation annotation from Chinese literature NER/RE dataset.",
                "evidence_refs": [f"zh-lit:{sample_id}:{relation.relation_id}"],
                "raw_backpointer_refs": [relation.relation_id, source.entity_id, target.entity_id],
                "source_refs": [sample_id],
                "source_text_quotes": [quote[:360]],
                "source_perspective": "text_annotation",
                "attribution_status": "external_annotation",
                "confidence_hint": "gold_annotation",
                "generic_relation_review_hint": "not_generic",
                "generic_relation_review_reasons": [],
                "evidence_count": 1,
                "weight": 1.0,
                "warnings": ["annotation_baseline_not_graph_truth"],
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
    nodes = list(node_by_entity.values())
    metrics = rows_to_graph_stats(nodes, edges)
    metrics.update(
        {
            "sample_id": sample_id,
            "language": "zh",
            "source_dataset": "Chinese-Literature-NER-RE-Dataset",
            "entity_annotation_count": len(entities),
            "relation_annotation_count": len(relations),
            "unlinked_entity_annotation_count": max(0, len(entities) - len(node_by_entity)),
            "coreference_line_count": len(coref_lines),
            "text_char_count": len(text),
        }
    )
    return nodes, edges, metrics


def select_litbank_sample(external_root: Path) -> dict[str, Any]:
    entity_dir = external_root / "litbank" / "entities" / "brat"
    event_dir = external_root / "litbank" / "events" / "brat"
    candidates: list[dict[str, Any]] = []
    for text_path in entity_dir.glob("*_brat.txt"):
        sample_id = text_path.name.removesuffix("_brat.txt")
        entity_ann = entity_dir / f"{sample_id}_brat.ann"
        event_ann = event_dir / f"{sample_id}_brat.ann"
        if not entity_ann.exists() or not event_ann.exists():
            continue
        nodes, edges, metrics = build_litbank_annotation_graph(sample_id, text_path, entity_ann, event_ann)
        score = int(metrics["edge_count"]) + int(metrics["node_count"]) - int(float(metrics["top_degree_share"]) * 100)
        candidates.append(
            {
                "sample_id": sample_id,
                "text_path": text_path,
                "entity_ann": entity_ann,
                "event_ann": event_ann,
                "nodes": nodes,
                "edges": edges,
                "metrics": metrics,
                "score": score,
            }
        )
    if not candidates:
        raise FileNotFoundError("No usable LitBank BRAT candidates found.")
    return sorted(candidates, key=lambda row: (-row["score"], row["sample_id"]))[0]


def select_chinese_sample(external_root: Path) -> dict[str, Any]:
    relation_root = external_root / "Chinese-Literature-NER-RE-Dataset" / "relation_extraction"
    candidates: list[dict[str, Any]] = []
    for ann_path in relation_root.glob("*/*.ann"):
        text_path = ann_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        sample_id = f"{ann_path.parent.name.lower()}_{ann_path.stem}"
        nodes, edges, metrics = build_chinese_re_annotation_graph(sample_id, text_path, ann_path)
        score = int(metrics["edge_count"]) + int(metrics["node_count"]) - int(float(metrics["top_degree_share"]) * 100)
        if int(metrics["edge_count"]) < 20:
            continue
        candidates.append(
            {
                "sample_id": sample_id,
                "text_path": text_path,
                "ann_path": ann_path,
                "nodes": nodes,
                "edges": edges,
                "metrics": metrics,
                "score": score,
            }
        )
    if not candidates:
        raise FileNotFoundError("No usable Chinese NER/RE candidates found.")
    return sorted(candidates, key=lambda row: (-row["score"], row["sample_id"]))[0]


def write_workspace_pack(output_root: Path, name: str, sample: dict[str, Any], *, source_dataset: str, language: str) -> dict[str, Any]:
    workspace = (output_root / name).resolve()
    raw_dir = workspace / "raw"
    source_text_path = raw_dir / f"{slug(str(sample['sample_id']))}.txt"
    source_text_path.parent.mkdir(parents=True, exist_ok=True)
    source_text = read_text(sample["text_path"])
    source_text_path.write_text(source_text, encoding="utf-8")
    raw_source = {
        "schema_version": "s0b.raw_source.v0.1",
        "workspace_id": name,
        "raw_source_id": f"raw:{name}:{sample['sample_id']}",
        "bundle_id": f"bundle:{name}",
        "source_type": "external_validation_text",
        "modality": "text",
        "original_format": "txt",
        "local_path": str(source_text_path.relative_to(Path.cwd())),
        "original_uri_or_path": str(sample["text_path"]),
        "adapter_recommendation": "generic_text_adapter",
        "inclusion_decision": "include",
        "processing_status": "ready_for_s1_intake",
        "organization_degree": "medium",
        "privacy_class": "public_dataset",
        "quality_notes": "External literary validation sample; annotation-derived graph is a suitability baseline, not graph truth.",
        "coverage_notes": "Selected for higher graph mesh potential than two-person dialogue.",
        "perspective_notes": "Narrative/literary text; modeled subject should be explicit if used for S1/S2 provider experiments.",
        "source_specific_metadata": {
            "record_id": str(sample["sample_id"]),
            "title": str(sample["sample_id"]),
            "language": language,
            "source_dataset": source_dataset,
            "text_structure_profile": "book_plaintext",
            "source_hash": file_hash(sample["text_path"]),
            "annotation_baseline_available": True,
        },
    }
    write_jsonl(workspace / "raw" / "organization" / "raw_sources.jsonl", [raw_source])
    graph_dir = workspace / "graph_v03_annotation_baseline"
    write_jsonl(graph_dir / "graph_nodes_table.jsonl", sample["nodes"])
    write_jsonl(graph_dir / "graph_edges_table.jsonl", sample["edges"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "sample_id": str(sample["sample_id"]),
        "language": language,
        "source_dataset": source_dataset,
        "source_text_path": str(source_text_path),
        "source_annotation_paths": [str(path) for key, path in sample.items() if key.endswith("_ann") or key == "ann_path"],
        "graph_dir": str(graph_dir),
        "counts": sample["metrics"],
        "boundary": {
            "annotation_baseline_not_graph_truth": True,
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
        },
    }
    write_json(workspace / "validation_dataset_pack_manifest.json", manifest)
    return manifest


def write_report(path: Path, manifests: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 Cross-Lingual Graph Validation Dataset Pack",
        "",
        "本报告只验证数据集是否适合作为图结构泛化测试材料，不证明抽取质量。",
        "",
        "## Boundary",
        "",
        "- annotation-derived graph is a validation baseline, not graph truth;",
        "- graph_is_not_proof=true;",
        "- support_status=not_checked;",
        "- write_permission=false;",
        "- provider extraction is not run by this pack builder.",
        "",
        "## Selected Samples",
        "",
        "| language | dataset | sample | nodes | edges | relation types | top degree share | mesh potential | workspace |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for manifest in manifests:
        counts = manifest["counts"]
        lines.append(
            "| {language} | {dataset} | `{sample}` | {nodes} | {edges} | {rels} | {share} | {mesh} | `{workspace}` |".format(
                language=manifest["language"],
                dataset=manifest["source_dataset"],
                sample=manifest["sample_id"],
                nodes=counts["node_count"],
                edges=counts["edge_count"],
                rels=len(counts.get("relation_type_counts") or {}),
                share=counts["top_degree_share"],
                mesh=counts["mesh_potential"],
                workspace=manifest["workspace"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 如果 `top_degree_share` 明显低于当前 Jon/Gina 双 hub 图，说明新材料更适合网状可视化。",
            "- 英文 LitBank baseline 主要来自 entity/event 同句标注，是图潜力探针，不是关系真值。",
            "- 中文文学数据集包含显式 relation annotation，更适合做关系抽取泛化检查。",
            "- 下一步可对这些 workspace 运行 S0B/S1/S2/provider graph extraction，再和 annotation baseline 对比。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_validation_pack(external_root: Path, output_root: Path) -> list[dict[str, Any]]:
    external_root = external_root.resolve()
    output_root = output_root.resolve()
    english = select_litbank_sample(external_root)
    chinese = select_chinese_sample(external_root)
    date_part = datetime.now().strftime("%Y%m%d")
    manifests = [
        write_workspace_pack(
            output_root,
            f"_v03_validation_litbank_en_{date_part}",
            english,
            source_dataset="LitBank",
            language="en",
        ),
        write_workspace_pack(
            output_root,
            f"_v03_validation_chinese_literature_zh_{date_part}",
            chinese,
            source_dataset="Chinese-Literature-NER-RE-Dataset",
            language="zh",
        ),
    ]
    write_report(Path("reports") / "graph_v03_cross_lingual_dataset_validation.md", manifests)
    write_json(
        Path("reports") / "graph_v03_cross_lingual_dataset_validation.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_iso(),
            "external_root": str(external_root),
            "manifests": manifests,
        },
    )
    return manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-lingual graph validation dataset packs.")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = build_validation_pack(args.external_root, args.output_root)
    print(json.dumps({"status": "ok", "workspaces": [row["workspace"] for row in manifests]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
