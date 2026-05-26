"""Build graph relation normalization policy from downloaded external schemas.

This tool turns local external relation resources into a generated
normalization policy. It avoids maintaining graph relation aliases as a
hand-written project wordlist.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import file_hash, sha256_text, write_json, write_text


SCHEMA_VERSION = "graph_relation_type_normalization_policy.generated.v0.3"
DEFAULT_EXTERNAL_ROOT = "external_references/graph_relation_schemas"
DEFAULT_OUTPUT = "configs/graph/graph_relation_type_normalization.generated.v0.3.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_relation_label(value: Any) -> str:
    text = " ".join(str(value or "").split())
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    text = text.lower()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def relation_variants(label: str) -> set[str]:
    normalized = normalize_relation_label(label)
    variants = {label.strip().lower(), normalized, normalized.replace("_", "-"), normalized.replace("_", " ")}
    variants = {variant for variant in variants if variant}
    return variants


def source_category(source_id: str) -> str:
    if source_id.startswith("arf"):
        return "external_arf_fiction"
    if source_id.startswith("dialogre"):
        return "external_dialogue"
    if source_id.startswith("conceptnet"):
        return "external_commonsense"
    if source_id.startswith("wikidata"):
        return "external_wikidata_property"
    if source_id.startswith("dbpedia"):
        return "external_dbpedia_ontology"
    return "external_relation_schema"


def add_relation(
    *,
    canonical: dict[str, dict[str, Any]],
    alias_map: dict[str, str],
    relation_label: str,
    source_id: str,
    source_path: str,
    count: int | None = None,
    aliases: list[str] | None = None,
) -> None:
    normalized = normalize_relation_label(relation_label)
    if not normalized:
        return
    row = canonical.setdefault(
        normalized,
        {
            "category": source_category(source_id),
            "external_sources": [],
            "source_counts": {},
        },
    )
    if source_id not in row["external_sources"]:
        row["external_sources"].append(source_id)
    if count is not None:
        row["source_counts"][source_id] = int(row["source_counts"].get(source_id, 0)) + int(count)
    row.setdefault("source_paths", [])
    if source_path not in row["source_paths"]:
        row["source_paths"].append(source_path)

    for variant in relation_variants(relation_label):
        alias_map.setdefault(variant, normalized)
    for alias in aliases or []:
        for variant in relation_variants(alias):
            alias_map.setdefault(variant, normalized)


def load_arf(root: Path, canonical: dict[str, dict[str, Any]], alias_map: dict[str, str]) -> dict[str, Any]:
    summary_path = root / "arf" / "arf_relation_schema_summary.json"
    types_path = root / "arf" / "arf_relation_types.txt"
    counts: Counter[str] = Counter()
    if summary_path.exists():
        summary = read_json(summary_path)
        for label, count in summary.get("top_relation_types") or []:
            counts[str(label)] += int(count)
    if types_path.exists():
        for line in types_path.read_text(encoding="utf-8-sig").splitlines():
            label = line.strip()
            if label:
                counts.setdefault(label, 0)
    for label, count in counts.items():
        add_relation(
            canonical=canonical,
            alias_map=alias_map,
            relation_label=label,
            source_id="arf_fiction_relation_ontology",
            source_path=str(types_path),
            count=count,
        )
    return {"relation_types": len(counts), "relation_instances": int(sum(counts.values()))}


def load_dialogre(root: Path, canonical: dict[str, dict[str, Any]], alias_map: dict[str, str]) -> dict[str, Any]:
    summary_path = root / "dialogre" / "dialogre_relation_schema_summary.json"
    types_path = root / "dialogre" / "dialogre_relation_types.txt"
    counts: Counter[str] = Counter()
    if summary_path.exists():
        summary = read_json(summary_path)
        for label, count in summary.get("relation_label_counts") or []:
            label = str(label)
            if label == "unanswerable":
                continue
            counts[label] += int(count)
    elif types_path.exists():
        for line in types_path.read_text(encoding="utf-8-sig").splitlines():
            label = line.strip()
            if label and label != "unanswerable":
                counts.setdefault(label, 0)
    for label, count in counts.items():
        normalized_label = label.split(":", 1)[-1]
        add_relation(
            canonical=canonical,
            alias_map=alias_map,
            relation_label=normalized_label,
            source_id="dialogre_relation_schema",
            source_path=str(types_path),
            count=count,
            aliases=[label],
        )
    return {"relation_types": len(counts), "relation_instances": int(sum(counts.values()))}


def load_conceptnet(root: Path, canonical: dict[str, dict[str, Any]], alias_map: dict[str, str]) -> dict[str, Any]:
    path = root / "conceptnet" / "conceptnet5.wiki" / "Relations.md"
    labels: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            match = re.search(r"\| /r/([^ |]+)", line)
            if match:
                labels.append(match.group(1))
    for label in labels:
        add_relation(
            canonical=canonical,
            alias_map=alias_map,
            relation_label=label,
            source_id="conceptnet_relations",
            source_path=str(path),
            count=None,
            aliases=[f"/r/{label}"],
        )
    return {"relation_types": len(labels)}


def load_wikidata(root: Path, canonical: dict[str, dict[str, Any]], alias_map: dict[str, str]) -> dict[str, Any]:
    wikidata_dir = root / "wikidata"
    count = 0
    for path in sorted(wikidata_dir.glob("P*.json")):
        data = read_json(path)
        entity = data.get("entities", {}).get(path.stem, {})
        label = (entity.get("labels", {}).get("en", {}) or {}).get("value")
        aliases = [
            item.get("value")
            for item in (entity.get("aliases", {}).get("en", {}) or [])
            if isinstance(item, dict) and item.get("value")
        ]
        if not label:
            continue
        add_relation(
            canonical=canonical,
            alias_map=alias_map,
            relation_label=label,
            source_id="wikidata_selected_properties",
            source_path=str(path),
            count=None,
            aliases=[path.stem, *aliases],
        )
        count += 1
    return {"selected_properties": count}


def load_dbpedia(root: Path, canonical: dict[str, dict[str, Any]], alias_map: dict[str, str]) -> dict[str, Any]:
    path = root / "dbpedia" / "dbpedia_ontology_tbox.owl"
    labels: set[str] = set()
    if path.exists():
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        labels.update(re.findall(r"http://dbpedia.org/ontology/([A-Za-z][A-Za-z0-9_]+)", text))
    for label in labels:
        add_relation(
            canonical=canonical,
            alias_map=alias_map,
            relation_label=label,
            source_id="dbpedia_ontology",
            source_path=str(path),
            count=None,
        )
    return {"ontology_terms": len(labels)}


def build_generated_policy(external_root: Path) -> dict[str, Any]:
    canonical: dict[str, dict[str, Any]] = {}
    alias_map: dict[str, str] = {}
    source_counts = {
        "arf": load_arf(external_root, canonical, alias_map),
        "dialogre": load_dialogre(external_root, canonical, alias_map),
        "conceptnet": load_conceptnet(external_root, canonical, alias_map),
        "wikidata": load_wikidata(external_root, canonical, alias_map),
        "dbpedia": load_dbpedia(external_root, canonical, alias_map),
    }

    for row in canonical.values():
        if len(row.get("external_sources") or []) > 1:
            row["category"] = "external_multi_source"

    # Keep the fallback relation explicit, but do not use hand-written semantic
    # aliases to hide unmapped raw labels.
    canonical.setdefault(
        "related_to_generic",
        {
            "category": "generic",
            "external_sources": ["conceptnet_relations"],
            "source_counts": {},
            "source_paths": [str(external_root / "conceptnet" / "conceptnet5.wiki" / "Relations.md")],
        },
    )
    for alias in relation_variants("related_to_generic"):
        alias_map.setdefault(alias, "related_to_generic")

    source_paths = sorted(
        {
            source_path
            for row in canonical.values()
            for source_path in (row.get("source_paths") or [])
            if source_path
        }
    )

    policy_body = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": "graph_relation_type_normalization.generated.v0.3",
        "target_task": "graph_candidate_consolidation",
        "status": "generated_from_downloaded_external_resources",
        "generated_at": now_iso(),
        "external_root": str(external_root),
        "provenance": {
            "rule": "Generated from downloaded external relation schemas. Do not edit aliases by hand; regenerate from source artifacts.",
            "source_counts": source_counts,
            "external_sources": [
                "ARF / Artificial Relationships in Fiction",
                "DialogRE",
                "ConceptNet relations",
                "Wikidata selected properties",
                "DBpedia ontology T-BOX",
            ],
            "source_hashes": {source_path: file_hash(Path(source_path)) for source_path in source_paths},
        },
        "canonical_relation_types": dict(sorted(canonical.items())),
        "alias_map": dict(sorted(alias_map.items())),
        "generic_relation_types": ["related_to_generic"],
        "normalization_strategy": {
            "direct_alias_lookup": True,
            "canonical_direct_lookup": True,
            "generated_from_external_resources_only": True,
            "handwritten_project_relation_wordlist": False,
        },
        "graph_is_not_proof": True,
    }
    policy_body["policy_hash"] = sha256_text(
        json.dumps(
            {
                "canonical_relation_types": policy_body["canonical_relation_types"],
                "alias_map": policy_body["alias_map"],
                "provenance": policy_body["provenance"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return policy_body


def build_policy_file(external_root: Path, output_path: Path) -> dict[str, Any]:
    policy = build_generated_policy(external_root)
    write_json(output_path, policy)
    report = "\n".join(
        [
            "# Generated Graph Relation Normalization Policy",
            "",
            f"- output: `{output_path}`",
            f"- canonical relation types: {len(policy['canonical_relation_types'])}",
            f"- aliases: {len(policy['alias_map'])}",
            f"- policy_hash: `{policy['policy_hash']}`",
            "",
            "This policy is generated from downloaded external resources and is not graph truth.",
            "",
        ]
    )
    write_text(output_path.with_suffix(".report.md"), report)
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build relation normalization policy from downloaded external schemas.")
    parser.add_argument("--external-root", type=Path, default=Path(DEFAULT_EXTERNAL_ROOT))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = build_policy_file(args.external_root, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "canonical_relation_types": len(policy["canonical_relation_types"]),
                "aliases": len(policy["alias_map"]),
                "policy_hash": policy["policy_hash"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
