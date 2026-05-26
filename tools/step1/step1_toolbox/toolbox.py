"""Minimal local implementation of the Step 1 Query Toolbox contract."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .scoring import bm25_score, cosine_similarity, lexical_score, tokens
from .store import LocalStep1Store, read_json, read_jsonl


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


class LocalStep1Toolbox:
    """Read-only local Step 1 toolbox.

    The class is deliberately narrow:
    - no mutation of canonical assets;
    - no external services;
    - no hidden experiment JSON parsing;
    - no cache-as-truth behavior.
    """

    def __init__(self, root: str | Path):
        self.store = LocalStep1Store(root)

    def retrieve_evidence(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        route_policy: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return retrieval_result objects.

        Default behavior returns raw evidence only. Optional context layers
        such as memory units must be explicitly enabled through filters.
        """

        filters = filters or {}
        route_policy = route_policy or {}
        retrieval_mode = self._retrieval_mode(filters, route_policy)
        if retrieval_mode != "direct_scan":
            requested_object_types = ["raw_evidence"]
            requested_layers = filters.get("item_layers") or ["raw_evidence"]
            if filters.get("include_memory_units") or "memory_unit" in requested_layers:
                requested_object_types.append("memory_unit")
            if filters.get("include_summaries") or "doc_level_summary" in requested_layers:
                requested_object_types.append("doc_level_summary")
            indexed = self._retrieve_indexed(
                query=query,
                filters=filters,
                top_k=top_k,
                route_policy=route_policy,
                requested_object_types=requested_object_types,
                default_degraded=lambda warnings: self._retrieve_evidence_direct(
                    query, filters, top_k, route_policy, extra_warnings=warnings
                ),
            )
            if indexed is not None:
                return indexed
        return self._retrieve_evidence_direct(query, filters, top_k, route_policy)

    def _retrieve_evidence_direct(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        route_policy: dict[str, Any],
        extra_warnings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        route_policy = route_policy or {}
        route = route_policy.get("route", "mixed")
        requested_layers = filters.get("item_layers") or ["raw_evidence"]
        include_memory = bool(filters.get("include_memory_units")) or "memory_unit" in requested_layers

        results: list[dict[str, Any]] = []

        if "raw_evidence" in requested_layers:
            for item in self._filter_evidence(filters):
                score = lexical_score(query, item.get("text", ""))
                if score <= 0 and query:
                    continue
                source = self.store.source_for_evidence(item) or {}
                raw_source = self.store.raw_source_for_evidence(item) or {}
                warnings = sorted(set(self.store.s0_alignment_warnings(item) + (extra_warnings or [])))
                results.append(
                    {
                        "schema_version": "step1.retrieval_result.v1",
                        "result_id": f"rr:{item.get('evidence_ref')}",
                        "rank": 0,
                        "item_layer": "raw_evidence",
                        "source_type": item.get("source_type") or source.get("source_type", "unknown"),
                        "text": item.get("text", ""),
                        "source_refs": [item.get("source_id")] if item.get("source_id") else [],
                        "evidence_refs": [item.get("evidence_ref")] if item.get("evidence_ref") else [],
                        "score": score,
                        "method": route_policy.get("method", "direct_scan_lexical"),
                        "retrieval_method": "direct_scan",
                        "retrieval_mode": "degraded_direct_scan" if extra_warnings else "direct_scan",
                        "retrieval_status": "degraded" if extra_warnings else "success",
                        "retrieval_backend": "direct_scan",
                        "support_status": "not_checked",
                        "score_components": {
                            "direct_scan_lexical": score,
                            "bm25": None,
                            "embedding": None,
                        },
                        "route": route,
                        "provenance": {
                            "store": str(self.store.root),
                            "index_id": "local_jsonl_scan",
                            "cache_id": "",
                            "run_id": route_policy.get("run_id", ""),
                            "raw_source_id": item.get("raw_source_id") or source.get("raw_source_id", ""),
                            "adapter_run_id": item.get("adapter_run_id") or source.get("adapter_run_id", ""),
                            "source_specific_ref": item.get("source_specific_ref", ""),
                            "display_ref": item.get("display_ref", ""),
                            "organization_degree": raw_source.get("organization_degree", ""),
                            "adapter_recommendation": raw_source.get("adapter_recommendation", ""),
                        },
                        "warnings": warnings,
                        "speaker": item.get("speaker"),
                        "timestamp": item.get("timestamp"),
                        "raw_source_id": item.get("raw_source_id") or source.get("raw_source_id"),
                        "adapter_run_id": item.get("adapter_run_id") or source.get("adapter_run_id"),
                        "modality": item.get("modality") or raw_source.get("modality"),
                        "locator": item.get("locator"),
                        "source_specific_ref": item.get("source_specific_ref"),
                        "display_ref": item.get("display_ref"),
                        "content_hash": item.get("content_hash"),
                        "extraction_method": item.get("extraction_method"),
                        "extraction_confidence": item.get("extraction_confidence"),
                        "entity_matches": self._entity_matches(query, item),
                        "time_matches": [],
                        "event_matches": [],
                        "backpointer_refs": [],
                        "limitations": [],
                    }
                )

        if include_memory:
            memory_results = self.retrieve_memory_units(query, filters, top_k=top_k, route_policy=route_policy)
            for result in memory_results:
                result = dict(result)
                warnings = set(result.get("warnings", []))
                warnings.add("memory_unit_not_raw_evidence")
                if not result.get("backpointer_refs"):
                    warnings.add("memory_summary_without_backpointer")
                result["warnings"] = sorted(warnings)
                results.append(result)

        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        return self._rank(results[:top_k])

    def retrieve_memory_units(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        route_policy: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return retrieval_result objects for accepted memory units by default."""

        filters = filters or {}
        route_policy = route_policy or {}
        retrieval_mode = self._retrieval_mode(filters, route_policy)
        if retrieval_mode != "direct_scan":
            indexed = self._retrieve_indexed(
                query=query,
                filters=filters,
                top_k=top_k,
                route_policy=route_policy,
                requested_object_types=["memory_unit"],
                default_degraded=lambda warnings: self._retrieve_memory_units_direct(
                    query, filters, top_k, route_policy, extra_warnings=warnings
                ),
            )
            if indexed is not None:
                return indexed
        return self._retrieve_memory_units_direct(query, filters, top_k, route_policy)

    def _retrieve_memory_units_direct(
        self,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        route_policy: dict[str, Any],
        extra_warnings: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        route_policy = route_policy or {}
        allowed_statuses = filters.get("memory_statuses") or [
            "accepted_for_experiment",
            "accepted_by_user",
        ]
        route = route_policy.get("route", "mixed")

        results: list[dict[str, Any]] = []
        for item in self.store.memory_units:
            status = item.get("status", "unknown")
            if status not in allowed_statuses:
                continue
            score = lexical_score(query, item.get("content", ""))
            if score <= 0 and query:
                continue
            warnings = ["memory_unit_not_raw_evidence"]
            warnings.extend(extra_warnings or [])
            if status not in {"accepted_for_experiment", "accepted_by_user"}:
                warnings.append(f"non_default_memory_status:{status}")
            if not item.get("backpointer_refs"):
                warnings.append("memory_summary_without_backpointer")
            if item.get("inference_level") in {"weak_signal", "strong_inference"}:
                warnings.append(f"inference_level:{item.get('inference_level')}")

            results.append(
                {
                    "schema_version": "step1.retrieval_result.v1",
                    "result_id": f"rr:{item.get('memory_id')}",
                    "rank": 0,
                    "item_layer": "memory_unit",
                    "source_type": "generated_memory",
                    "text": item.get("content", ""),
                    "source_refs": item.get("source_refs", []),
                    "evidence_refs": item.get("evidence_refs", []),
                    "score": score,
                    "method": route_policy.get("method", "direct_scan_lexical"),
                    "retrieval_method": "direct_scan",
                    "retrieval_mode": "degraded_direct_scan" if extra_warnings else "direct_scan",
                    "retrieval_status": "degraded" if extra_warnings else "success",
                    "retrieval_backend": "direct_scan",
                    "support_status": "not_checked",
                    "score_components": {
                        "direct_scan_lexical": score,
                        "bm25": None,
                        "embedding": None,
                    },
                    "route": route,
                    "provenance": {
                        "store": str(self.store.root),
                        "memory_id": item.get("memory_id"),
                        "generation_method": item.get("generation_method"),
                        "run_id": route_policy.get("run_id", ""),
                    },
                    "warnings": warnings,
                    "memory_id": item.get("memory_id"),
                    "confidence": item.get("confidence"),
                    "status": status,
                    "backpointer_refs": item.get("backpointer_refs", []),
                    "limitations": ["Memory unit is context, not direct evidence."],
                }
            )

        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        return self._rank(results[:top_k])

    def resolve_ref(
        self,
        ref_id: str,
        ref_type: str = "unknown",
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve core refs and explicit extension refs."""

        options = options or {}
        if ref_type == "unknown":
            ref_type = self._guess_ref_type(ref_id)

        if ref_type == "evidence_ref":
            item = self.store.resolve_evidence(ref_id)
            if item:
                source = self.store.source_for_evidence(item) or {}
                raw_source = self.store.raw_source_for_evidence(item) or {}
                warnings = self.store.s0_alignment_warnings(item)
                return {
                    "schema_version": "step1.resolved_ref.v1",
                    "ref_id": ref_id,
                    "ref_type": ref_type,
                    "resolved": True,
                    "text": item.get("text", ""),
                    "metadata": {
                        **{
                            k: item.get(k)
                            for k in (
                                "source_id",
                                "record_id",
                                "turn_id",
                                "speaker",
                                "timestamp",
                                "item_layer",
                                "raw_source_id",
                                "adapter_run_id",
                                "modality",
                                "locator",
                                "source_specific_ref",
                                "display_ref",
                                "content_hash",
                                "extraction_method",
                                "extraction_confidence",
                            )
                        },
                        "source_type": item.get("source_type") or source.get("source_type", "unknown"),
                        "raw_source_processing_status": raw_source.get("processing_status"),
                        "raw_source_organization_degree": raw_source.get("organization_degree"),
                        "raw_source_adapter_recommendation": raw_source.get("adapter_recommendation"),
                    },
                    "source_refs": [item.get("source_id")] if item.get("source_id") else [],
                    "evidence_refs": [ref_id],
                    "resolution_warnings": warnings,
                }
            return self._unresolved(ref_id, ref_type, "ref_not_found")

        if ref_type == "memory_id":
            item = self.store.resolve_memory(ref_id)
            if item:
                return {
                    "schema_version": "step1.resolved_ref.v1",
                    "ref_id": ref_id,
                    "ref_type": ref_type,
                    "resolved": True,
                    "text": item.get("content", ""),
                    "metadata": {
                        "item_layer": "memory_unit",
                        "status": item.get("status"),
                        "generation_method": item.get("generation_method"),
                        "confidence": item.get("confidence"),
                        "backpointer_refs": item.get("backpointer_refs", []),
                    },
                    "source_refs": item.get("source_refs", []),
                    "evidence_refs": item.get("evidence_refs", []),
                }
            return self._unresolved(ref_id, ref_type, "ref_not_found")

        if ref_type == "source_ref":
            item = self.store.resolve_source(ref_id)
            if item:
                raw_source = self.store.resolve_raw_source(str(item.get("raw_source_id", ""))) if item.get("raw_source_id") else None
                return {
                    "schema_version": "step1.resolved_ref.v1",
                    "ref_id": ref_id,
                    "ref_type": ref_type,
                    "resolved": True,
                    "text": item.get("path_or_uri", ""),
                    "metadata": {
                        **item,
                        "raw_source": raw_source or {},
                    },
                    "source_refs": [ref_id],
                    "evidence_refs": [],
                }
            return self._unresolved(ref_id, ref_type, "ref_not_found")

        if ref_type in {"report_ref", "audit_ref", "cache_ref"}:
            if not options.get("allow_extension_routes", False):
                return self._unresolved(ref_id, ref_type, "extension_route_not_enabled", ["extension_resolution_route"])
            warning = {
                "report_ref": "report_context_not_evidence",
                "audit_ref": "audit_context_not_evidence",
                "cache_ref": "cache_not_truth",
            }[ref_type]
            return {
                "schema_version": "step1.resolved_ref.v1",
                "ref_id": ref_id,
                "ref_type": ref_type,
                "resolved": False,
                "text": "",
                "metadata": {"failure_reason": "extension_store_not_configured"},
                "source_refs": [],
                "evidence_refs": [],
                "failure_reason": "extension_store_not_configured",
                "resolution_warnings": ["extension_resolution_route", warning],
            }

        return self._unresolved(ref_id, ref_type, "unsupported_ref_type")

    def check_claim_support(
        self,
        claim: str,
        evidence_refs: list[str],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rule-based support check that trusts raw evidence refs by default."""

        options = options or {}
        checked_refs: list[str] = []
        supporting_refs: list[str] = []
        conflict_refs: list[str] = []
        missing_refs: list[str] = []
        raw_texts: list[str] = []
        resolved_ref_ids: list[str] = []

        for ref in evidence_refs:
            ref_type = self._guess_ref_type(ref)
            if ref_type == "unknown" and options.get("default_ref_type"):
                ref_type = str(options["default_ref_type"])
            if ref_type != "evidence_ref" and not options.get("allow_backpointer_resolution", False):
                missing_refs.append("raw_evidence_ref_required")
                checked_refs.append(ref)
                continue
            resolved = self.resolve_ref(ref, ref_type)
            checked_refs.append(ref)
            if not resolved.get("resolved"):
                missing_refs.append(ref)
                continue
            raw_texts.append(resolved.get("text", ""))
            resolved_ref_ids.append(ref)

        claim_tokens = tokens(claim)
        positive = False
        contradiction = False
        for ref, text in zip(resolved_ref_ids, raw_texts):
            text_tokens = tokens(text)
            overlap = claim_tokens & text_tokens
            if overlap:
                positive = True
                supporting_refs.append(ref)
            if self._looks_contradictory(claim, text):
                contradiction = True
                conflict_refs.append(ref)

        if contradiction:
            strength = "contradicts"
            supporting_refs = []
        elif positive and len(claim_tokens & set().union(*(tokens(text) for text in raw_texts))) >= max(2, min(4, len(claim_tokens) // 2)):
            strength = "direct"
        elif positive:
            strength = "partial"
        elif missing_refs:
            strength = "unknown"
        else:
            strength = "weak" if raw_texts else "unknown"

        return {
            "schema_version": "step1.support_check.v1",
            "check_id": _stable_id("sc", {"claim": claim, "evidence_refs": evidence_refs}),
            "claim": claim,
            "support_strength": strength,
            "supporting_refs": supporting_refs,
            "checked_refs": checked_refs,
            "created_at": _now_iso(),
            "method": "rule",
            "checker": {
                "type": "rule",
                "model_id": "",
                "prompt_or_policy": "raw_evidence_only_default_v1",
                "reviewer": "step1_toolbox",
                "notes": "Lexical overlap and simple negation check; not a semantic judge.",
            },
            "missing_refs": missing_refs,
            "conflict_refs": conflict_refs,
            "entity_mismatch": False,
            "temporal_mismatch": False,
            "event_mismatch": False,
            "notes": "",
        }

    def evaluate_grounding(
        self,
        target: dict[str, Any],
        refs: list[str],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate grounding for a small target object."""

        options = options or {}
        target_type = options.get("target_type", target.get("target_type", "unknown"))
        claims = _as_list(target.get("claims") or target.get("checked_claims"))
        checks = [self.check_claim_support(claim, refs, options=options) for claim in claims]
        unsupported = [check["claim"] for check in checks if check["support_strength"] in {"weak", "unknown"}]
        contradictions = [ref for check in checks for ref in check.get("conflict_refs", [])]

        if contradictions:
            overall = "conflicting"
        elif not claims:
            overall = "unknown"
        elif not unsupported:
            overall = "grounded"
        elif len(unsupported) < len(claims):
            overall = "partially_grounded"
        else:
            overall = "ungrounded"

        return {
            "schema_version": "step1.grounding_report.v1",
            "grounding_id": _stable_id("gr", {"target_type": target_type, "claims": claims, "refs": refs}),
            "target_type": target_type,
            "target_ref": target.get("target_ref", ""),
            "checked_claims": claims,
            "unsupported_claims": unsupported,
            "contradictions": contradictions,
            "overall_grounding": overall,
            "created_at": _now_iso(),
            "overclaim_risks": [],
            "missing_evidence": unsupported,
            "used_memory_without_evidence": [],
            "used_cache_as_truth": [],
            "notes": "",
        }

    def adapt_step1_results_for_step2(
        self,
        query_context: dict[str, Any],
        components: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create by-ref or by-value Step1-to-Step2 handoff packet."""

        options = options or {}
        packet_mode = options.get("packet_mode", "by_ref")
        retrieval_results = components.get("retrieval_results", [])
        support_checks = components.get("support_checks", [])
        grounding_reports = components.get("grounding_reports", [])
        resolved_refs = components.get("resolved_refs", [])

        return {
            "schema_version": "step1.handoff_packet.v1",
            "packet_id": options.get("packet_id", f"s1_to_s2:{query_context.get('query_id', 'unknown')}"),
            "packet_mode": packet_mode,
            "query_or_task": {
                "query_id": query_context.get("query_id", ""),
                "query_text": query_context.get("query_text", ""),
            },
            "created_at": _now_iso(),
            "route": query_context.get("route_hint", "unknown"),
            "retrieval_results": [
                self._packet_result(result, packet_mode)
                for result in retrieval_results
            ],
            "resolved_refs": [
                self._packet_resolved_ref(ref, packet_mode)
                for ref in resolved_refs
            ],
            "support_checks": [
                self._packet_support_check(check)
                for check in support_checks
            ],
            "grounding_reports": [
                {
                    "grounding_id": report.get("grounding_id"),
                    "overall_grounding": report.get("overall_grounding"),
                }
                for report in grounding_reports
            ],
            "warnings": sorted({warning for result in retrieval_results for warning in result.get("warnings", [])}),
            "filters": options.get("filters", {}),
            "cache_status": options.get("cache_status", "not_used"),
            "known_failure_modes": options.get("known_failure_modes", []),
        }

    def _retrieval_mode(self, filters: dict[str, Any], route_policy: dict[str, Any]) -> str:
        mode = (
            route_policy.get("retrieval_mode")
            or filters.get("retrieval_mode")
            or route_policy.get("mode")
            or filters.get("mode")
            or "auto"
        )
        allowed = {
            "direct_scan",
            "auto",
            "bm25_indexed",
            "embedding_indexed",
            "hybrid_indexed",
            "degraded_direct_scan",
        }
        return str(mode) if str(mode) in allowed else "auto"

    def _retrieve_indexed(
        self,
        *,
        query: str,
        filters: dict[str, Any],
        top_k: int,
        route_policy: dict[str, Any],
        requested_object_types: list[str],
        default_degraded: Any,
    ) -> list[dict[str, Any]] | None:
        mode = self._retrieval_mode(filters, route_policy)
        index_root = self._index_root(filters, route_policy)
        bm25_state = self._load_bm25_state(index_root)
        embedding_state = self._load_embedding_state(index_root)

        bm25_fresh, bm25_warnings = self._index_fresh(bm25_state.get("manifest"))
        embedding_fresh, embedding_warnings = self._index_fresh(embedding_state.get("manifest"))
        embedding_usable, embedding_use_warnings = self._embedding_usable(embedding_state, route_policy)

        if mode == "auto":
            if bm25_fresh and embedding_fresh and embedding_usable:
                mode = "hybrid_indexed"
            elif bm25_fresh:
                mode = "bm25_indexed"
            elif embedding_fresh and embedding_usable:
                mode = "embedding_indexed"
            else:
                return default_degraded(
                    ["degraded_direct_scan", "index_missing_or_stale"]
                    + bm25_warnings
                    + embedding_warnings
                    + embedding_use_warnings
                )

        requested_mode = self._retrieval_mode(filters, route_policy)
        if mode == "bm25_indexed" and not bm25_fresh:
            return default_degraded(["degraded_direct_scan", "bm25_index_missing_or_stale"] + bm25_warnings)
        if mode == "embedding_indexed" and (not embedding_fresh or not embedding_usable):
            return default_degraded(
                ["degraded_direct_scan", "embedding_index_missing_stale_or_unusable"]
                + embedding_warnings
                + embedding_use_warnings
            )
        if mode == "hybrid_indexed" and not (bm25_fresh or (embedding_fresh and embedding_usable)):
            return default_degraded(
                ["degraded_direct_scan", "hybrid_indexes_missing_or_stale"]
                + bm25_warnings
                + embedding_warnings
                + embedding_use_warnings
            )

        bm25_hits: dict[str, dict[str, Any]] = {}
        embedding_hits: dict[str, dict[str, Any]] = {}
        if mode in {"bm25_indexed", "hybrid_indexed"} and bm25_fresh:
            bm25_hits = self._bm25_hits(query, bm25_state, requested_object_types)
        if mode in {"embedding_indexed", "hybrid_indexed"} and embedding_fresh and embedding_usable:
            query_vector = self._query_vector(query, embedding_state, route_policy)
            if query_vector:
                embedding_hits = self._embedding_hits(query_vector, embedding_state, requested_object_types)
            elif mode == "embedding_indexed":
                return default_degraded(["degraded_direct_scan", "embedding_query_vector_unavailable"])

        if mode == "bm25_indexed":
            combined_ids = set(bm25_hits)
        elif mode == "embedding_indexed":
            combined_ids = set(embedding_hits)
        else:
            combined_ids = set(bm25_hits) | set(embedding_hits)

        if not combined_ids:
            return default_degraded(["degraded_direct_scan", f"{mode}_returned_no_hits"])

        max_bm25 = max((hit["bm25_score"] for hit in bm25_hits.values()), default=0.0)
        max_embedding = max((hit["embedding_score"] for hit in embedding_hits.values()), default=0.0)
        weights = self._fusion_weights(route_policy)
        if mode == "hybrid_indexed":
            if not bm25_hits and embedding_hits:
                weights = {"bm25": 0.0, "embedding": 1.0}
            elif bm25_hits and not embedding_hits:
                weights = {"bm25": 1.0, "embedding": 0.0}
        target_subject_id = self._target_subject_id(filters, route_policy)
        embedding_runtime = self._embedding_runtime_metadata(embedding_state.get("manifest") or {}, route_policy)
        candidates: list[dict[str, Any]] = []
        for entry_id in combined_ids:
            bm25_hit = bm25_hits.get(entry_id)
            embedding_hit = embedding_hits.get(entry_id)
            entry = (bm25_hit or embedding_hit or {}).get("entry", {})
            text = self._entry_text(entry)
            bm25_raw = bm25_hit["bm25_score"] if bm25_hit else None
            embedding_raw = embedding_hit["embedding_score"] if embedding_hit else None
            bm25_norm = (bm25_raw / max_bm25) if bm25_raw is not None and max_bm25 > 0 else 0.0
            embedding_norm = (embedding_raw / max_embedding) if embedding_raw is not None and max_embedding > 0 else 0.0
            if mode == "bm25_indexed":
                retrieval_score = bm25_norm
                method = "bm25_indexed"
                active_backends = ["bm25"]
                missing_backends: list[str] = []
            elif mode == "embedding_indexed":
                retrieval_score = embedding_norm
                method = "embedding_indexed"
                active_backends = ["embedding"]
                missing_backends = []
            else:
                retrieval_score = weights["bm25"] * bm25_norm + weights["embedding"] * embedding_norm
                method = "hybrid_indexed"
                active_backends = []
                missing_backends = []
                if bm25_hit:
                    active_backends.append("bm25")
                else:
                    missing_backends.append("bm25")
                if embedding_hit:
                    active_backends.append("embedding")
                else:
                    missing_backends.append("embedding")
            active_warnings: list[str] = []
            if method in {"bm25_indexed", "hybrid_indexed"} and bm25_hit:
                active_warnings.extend(bm25_warnings)
            if method in {"embedding_indexed", "hybrid_indexed"} and embedding_hit:
                active_warnings.extend(embedding_warnings + embedding_use_warnings)
            if method == "hybrid_indexed" and not bm25_hits:
                active_warnings.append("hybrid_bm25_path_unavailable")
            if method == "hybrid_indexed" and not embedding_hits:
                active_warnings.append("hybrid_embedding_path_unavailable")
            subject_metadata = self._subject_metadata(entry, target_subject_id)
            subject_adjustment = self._subject_adjustment(subject_metadata, route_policy)
            if subject_adjustment["factor"] < 1.0:
                active_warnings.append("subject_downranked")
            score = retrieval_score * subject_adjustment["factor"]
            result = self._indexed_result(
                entry=entry,
                text=text,
                score=score,
                retrieval_score_before_subject_adjustment=retrieval_score,
                bm25_raw_score=bm25_raw,
                bm25_normalized_score=bm25_norm if bm25_hit else None,
                embedding_raw_score=embedding_raw,
                embedding_normalized_score=embedding_norm if embedding_hit else None,
                method=method,
                requested_mode=requested_mode,
                active_backends=active_backends,
                missing_backends=missing_backends,
                route_policy=route_policy,
                warnings=active_warnings,
                fusion_policy=weights if method == "hybrid_indexed" else None,
                target_subject_id=target_subject_id,
                subject_metadata=subject_metadata,
                subject_adjustment=subject_adjustment,
                embedding_runtime=embedding_runtime,
                active_index_manifests=self._active_index_manifests(
                    method=method,
                    bm25_hit=bool(bm25_hit),
                    embedding_hit=bool(embedding_hit),
                    bm25_manifest=bm25_state.get("manifest") or {},
                    embedding_manifest=embedding_state.get("manifest") or {},
                ),
            )
            candidates.append(result)

        candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return self._rank(candidates[:top_k])

    def _index_root(self, filters: dict[str, Any], route_policy: dict[str, Any]) -> Path:
        value = route_policy.get("index_root") or filters.get("index_root") or (self.store.root / "indexes")
        path = Path(value)
        if not path.is_absolute():
            if path.exists():
                path = path.resolve()
            else:
                path = self.store.root / path
        return path

    def _load_bm25_state(self, index_root: Path) -> dict[str, Any]:
        return {
            "manifest": read_json(index_root / "step1_bm25_manifest.json"),
            "entries": read_jsonl(index_root / "step1_bm25_entries.jsonl"),
            "index": read_json(index_root / "step1_bm25_index.json"),
        }

    def _load_embedding_state(self, index_root: Path) -> dict[str, Any]:
        return {
            "manifest": read_json(index_root / "step1_embedding_manifest.json"),
            "entries": read_jsonl(index_root / "step1_embedding_entries.jsonl"),
            "sidecar": read_json(index_root / "step1_embedding_vectors.json"),
        }

    def _index_fresh(self, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
        if not manifest:
            return False, ["index_manifest_missing"]
        expected = manifest.get("source_hashes") or {}
        current = self.store.source_hashes()
        stale = [
            name
            for name, expected_hash in expected.items()
            if current.get(name) != expected_hash
        ]
        if stale:
            return False, [f"stale_index_source_hash:{name}" for name in stale]
        return True, []

    def _embedding_usable(self, state: dict[str, Any], route_policy: dict[str, Any]) -> tuple[bool, list[str]]:
        manifest = state.get("manifest") or {}
        sidecar = state.get("sidecar") or {}
        backend = manifest.get("embedding_backend") or sidecar.get("embedding_backend")
        if not manifest or not sidecar:
            return False, ["embedding_index_missing"]
        if backend == "hash" and not route_policy.get("allow_hash_embedding_for_tests", False):
            return False, ["hash_embedding_backend_test_only"]
        entry_ids = [entry.get("index_entry_id") for entry in state.get("entries", [])]
        sidecar_ids = sidecar.get("index_entry_ids") or []
        vectors = sidecar.get("vectors") or []
        if entry_ids != sidecar_ids or len(vectors) != len(entry_ids):
            return False, ["embedding_sidecar_alignment_failed"]
        return True, []

    def _bm25_hits(
        self,
        query: str,
        state: dict[str, Any],
        requested_object_types: list[str],
    ) -> dict[str, dict[str, Any]]:
        hits: dict[str, dict[str, Any]] = {}
        bm25_index = state.get("index") or {}
        for entry in state.get("entries", []):
            if entry.get("source_object_type") not in requested_object_types:
                continue
            text = self._entry_text(entry)
            score = bm25_score(query, text, bm25_index, str(entry.get("index_entry_id")))
            if score <= 0:
                continue
            hits[str(entry["index_entry_id"])] = {"entry": entry, "bm25_score": score}
        return hits

    def _embedding_hits(
        self,
        query_vector: list[float],
        state: dict[str, Any],
        requested_object_types: list[str],
    ) -> dict[str, dict[str, Any]]:
        entries = state.get("entries", [])
        sidecar = state.get("sidecar") or {}
        ids = sidecar.get("index_entry_ids") or []
        vectors = sidecar.get("vectors") or []
        by_id = {entry.get("index_entry_id"): entry for entry in entries}
        hits: dict[str, dict[str, Any]] = {}
        for entry_id, vector in zip(ids, vectors):
            entry = by_id.get(entry_id)
            if not entry or entry.get("source_object_type") not in requested_object_types:
                continue
            score = cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            hits[str(entry_id)] = {"entry": entry, "embedding_score": score}
        return hits

    def _query_vector(self, query: str, state: dict[str, Any], route_policy: dict[str, Any]) -> list[float] | None:
        supplied = route_policy.get("query_vector")
        if isinstance(supplied, list):
            return [float(value) for value in supplied]
        manifest = state.get("manifest") or {}
        backend = manifest.get("embedding_backend")
        if backend == "hash":
            return None
        if backend == "qwen_local":
            try:
                return self._qwen_query_vector(
                    query=query,
                    model_id=str(manifest.get("embedding_model") or "Qwen/Qwen3-Embedding-0.6B"),
                    device=route_policy.get("embedding_device"),
                )
            except Exception:
                return None
        return None

    def _qwen_query_vector(self, query: str, model_id: str, device: str | None = None) -> list[float]:
        import torch
        from transformers import AutoModel, AutoTokenizer

        selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, local_files_only=True, trust_remote_code=True)
        model.to(selected_device)
        model.eval()
        encoded = tokenizer([query], padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(selected_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return [float(value) for value in pooled.detach().cpu().tolist()[0]]

    def _fusion_weights(self, route_policy: dict[str, Any]) -> dict[str, float]:
        if "bm25_weight" in route_policy or "embedding_weight" in route_policy:
            bm25 = float(route_policy.get("bm25_weight", 0.5))
            embedding = float(route_policy.get("embedding_weight", 0.5))
        else:
            route = str(route_policy.get("route", "mixed"))
            if route in {"fact", "evidence_first", "exact"}:
                bm25, embedding = 0.55, 0.45
            elif route in {"semantic", "exploratory"}:
                bm25, embedding = 0.25, 0.75
            else:
                bm25, embedding = 0.35, 0.65
        total = max(bm25 + embedding, 1e-9)
        return {"bm25": bm25 / total, "embedding": embedding / total}

    def _target_subject_id(self, filters: dict[str, Any], route_policy: dict[str, Any]) -> str:
        value = route_policy.get("target_subject_id") or filters.get("target_subject_id")
        if value:
            return str(value)
        for key in ("subject_ids", "participant_ids"):
            values = filters.get(key)
            if isinstance(values, list) and values:
                return str(values[0])
        return ""

    def _source_object_for_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        object_type = entry.get("source_object_type")
        object_ref = str(entry.get("source_object_ref") or entry.get("source_object_id") or "")
        if object_type == "raw_evidence":
            return self.store.resolve_evidence(object_ref) or {}
        if object_type == "memory_unit":
            return self.store.resolve_memory(object_ref) or {}
        if object_type == "doc_level_summary":
            return self.store.resolve_summary(object_ref) or {}
        return {}

    def _subject_metadata(self, entry: dict[str, Any], target_subject_id: str) -> dict[str, Any]:
        source_object = self._source_object_for_entry(entry)
        locator = entry.get("locator") or source_object.get("locator") or {}
        speaker = locator.get("speaker") or source_object.get("speaker") or ""
        subject_scope = entry.get("subject_scope") or source_object.get("subject_scope") or entry.get("scope") or "unknown"
        subject_ids = {
            str(value).lower()
            for value in (source_object.get("subject_ids") or entry.get("subject_ids") or [])
            if value
        }
        target = target_subject_id.lower()

        if not target_subject_id:
            status = "unknown"
            warning = ""
        elif subject_scope == "mixed_participant":
            status = "mixed_participant"
            warning = "mixed_participant_context_requires_subject_lock"
        elif speaker and speaker.lower() == target:
            status = "target_match"
            warning = ""
        elif speaker:
            status = "other_participant"
            warning = "subject_mismatch_risk"
        elif subject_ids and target in subject_ids:
            status = "target_match"
            warning = ""
        elif subject_ids:
            status = "other_participant"
            warning = "subject_mismatch_risk"
        else:
            status = "unknown"
            warning = ""

        return {
            "target_subject_id": target_subject_id,
            "subject_role": "target_subject" if target_subject_id else "unknown",
            "subject_scope": subject_scope,
            "subject_match_status": status,
            "subject_risk_warning": warning,
        }

    def _subject_adjustment(self, subject_metadata: dict[str, Any], route_policy: dict[str, Any]) -> dict[str, Any]:
        """Return a route-aware subject ranking adjustment.

        This is ranking discipline only. It does not turn subject matches into
        factual support and it does not delete subject-adjacent context.
        """

        if not subject_metadata.get("target_subject_id"):
            return {
                "factor": 1.0,
                "policy": "no_target_subject_no_adjustment",
                "reason": "",
            }

        status = subject_metadata.get("subject_match_status", "unknown")
        route = str(route_policy.get("route", "mixed"))
        exact_routes = {"fact", "evidence_first", "exact"}
        semantic_routes = {"semantic", "exploratory"}

        if status == "target_match":
            factor = 1.0
            reason = "target subject match"
        elif status == "mixed_participant":
            factor = 0.75 if route in exact_routes else 0.85
            reason = "mixed participant context requires subject lock"
        elif status == "other_participant":
            factor = 0.25 if route in exact_routes else 0.55
            reason = "other participant context cannot outrank target-subject evidence"
        else:
            factor = 0.65 if route in exact_routes else 0.80
            reason = "unknown subject match is kept but downweighted when target subject is known"

        return {
            "factor": factor,
            "policy": "route_aware_subject_downranking_v0.1",
            "route": route,
            "reason": reason,
        }

    def _hybrid_status(self, method: str, active_backends: list[str], missing_backends: list[str]) -> str:
        if method != "hybrid_indexed":
            return "not_applicable"
        if "bm25" in active_backends and "embedding" in active_backends and not missing_backends:
            return "full"
        if active_backends:
            return "partial"
        return "unavailable"

    def _embedding_runtime_metadata(self, manifest: dict[str, Any], route_policy: dict[str, Any]) -> dict[str, Any]:
        backend = str(manifest.get("embedding_backend") or "")
        if route_policy.get("query_vector") is not None:
            model_load_behavior = "query_vector_supplied_no_model_load"
        elif backend == "qwen_local":
            model_load_behavior = "process_local_model_load"
        elif backend == "hash":
            model_load_behavior = "test_only_no_runtime_encoder"
        else:
            model_load_behavior = "unknown"
        return {
            "embedding_backend": backend,
            "embedding_model": manifest.get("embedding_model", ""),
            "embedding_model_revision": manifest.get("embedding_model_revision", ""),
            "embedding_dimension": manifest.get("embedding_dimension"),
            "embedding_device": route_policy.get("embedding_device") or manifest.get("embedding_device") or "auto",
            "model_load_behavior": model_load_behavior,
            "cache_policy": "none",
        }

    def _indexed_result(
        self,
        *,
        entry: dict[str, Any],
        text: str,
        score: float,
        retrieval_score_before_subject_adjustment: float,
        bm25_raw_score: float | None,
        bm25_normalized_score: float | None,
        embedding_raw_score: float | None,
        embedding_normalized_score: float | None,
        method: str,
        requested_mode: str,
        active_backends: list[str],
        missing_backends: list[str],
        route_policy: dict[str, Any],
        warnings: list[str],
        fusion_policy: dict[str, float] | None,
        target_subject_id: str,
        subject_metadata: dict[str, Any],
        subject_adjustment: dict[str, Any],
        embedding_runtime: dict[str, Any],
        active_index_manifests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        item_warnings = sorted(set((entry.get("warnings") or []) + warnings + ["retrieval_hit_not_support_check"]))
        if entry.get("source_object_type") == "doc_level_summary":
            item_warnings.append("summary_hit_context_only")
        if entry.get("source_object_type") == "memory_unit":
            item_warnings.append("memory_unit_not_raw_evidence")
        if subject_metadata["subject_risk_warning"]:
            item_warnings.append(subject_metadata["subject_risk_warning"])
        return {
            "schema_version": "step1.retrieval_result.v1",
            "result_id": f"rr:{entry.get('index_entry_id')}",
            "rank": 0,
            "item_layer": entry.get("item_layer") or entry.get("source_object_type"),
            "source_type": "indexed_s1_asset",
            "text": text,
            "source_refs": entry.get("source_refs", []),
            "evidence_refs": entry.get("evidence_refs", []),
            "score": score,
            "method": method,
            "retrieval_method": method,
            "retrieval_mode": method,
            "requested_retrieval_mode": requested_mode,
            "retrieval_status": "partial" if missing_backends and active_backends else "success",
            "retrieval_backend": method.replace("_indexed", ""),
            "hybrid_status": self._hybrid_status(method, active_backends, missing_backends),
            "active_backends": active_backends,
            "missing_backends": missing_backends,
            "degraded_components": missing_backends,
            "active_index_manifests": active_index_manifests,
            "stale_check_status": "fresh",
            "degraded_reason": "",
            "support_status": "not_checked",
            "route": route_policy.get("route", "mixed"),
            "score_components": {
                "bm25": bm25_normalized_score,
                "bm25_raw_score": bm25_raw_score,
                "bm25_normalized_score": bm25_normalized_score,
                "embedding": embedding_normalized_score,
                "embedding_raw_score": embedding_raw_score,
                "embedding_normalized_score": embedding_normalized_score,
                "fusion_policy": fusion_policy,
                "fusion_weights": fusion_policy,
                "retrieval_score_before_subject_adjustment": retrieval_score_before_subject_adjustment,
                "subject_adjustment_factor": subject_adjustment["factor"],
                "subject_adjustment_policy": subject_adjustment["policy"],
                "subject_adjustment_reason": subject_adjustment["reason"],
                "final_score": score,
                "tie_breaking_rule": "stable sort by subject-adjusted final score descending; input order breaks exact ties",
                "normalization_scope": "current_hit_set",
                "normalization_warning": "scores are query-local and not comparable across queries",
            },
            "embedding_runtime": embedding_runtime,
            "provenance": {
                "store": str(self.store.root),
                "index_entry_id": entry.get("index_entry_id"),
                "source_object_id": entry.get("source_object_id"),
                "source_object_ref": entry.get("source_object_ref"),
                "source_object_type": entry.get("source_object_type"),
                "source_object_hash": entry.get("source_object_hash"),
                "source_object_version": entry.get("source_object_version"),
                "raw_source_id": entry.get("raw_source_id"),
                "source_specific_ref": entry.get("source_specific_ref"),
                "display_ref": entry.get("display_ref"),
            },
            "warnings": sorted(set(item_warnings)),
            "raw_source_id": entry.get("raw_source_id"),
            "modality": entry.get("modality"),
            "locator": entry.get("locator"),
            "source_specific_ref": entry.get("source_specific_ref"),
            "display_ref": entry.get("display_ref"),
            "content_hash": entry.get("source_object_hash"),
            "truth_status": entry.get("truth_status"),
            "target_subject_id": subject_metadata["target_subject_id"],
            "subject_role": subject_metadata["subject_role"],
            "subject_scope": subject_metadata["subject_scope"],
            "subject_match_status": subject_metadata["subject_match_status"],
            "subject_risk_warning": subject_metadata["subject_risk_warning"],
            "source_object_type": entry.get("source_object_type"),
            "memory_id": entry.get("source_object_id") if entry.get("source_object_type") == "memory_unit" else None,
            "confidence": entry.get("confidence"),
            "status": entry.get("status"),
            "backpointer_refs": entry.get("evidence_refs", []) if entry.get("source_object_type") == "memory_unit" else [],
            "limitations": ["Retrieval hit is not a support check."],
        }

    def _active_index_manifests(
        self,
        *,
        method: str,
        bm25_hit: bool,
        embedding_hit: bool,
        bm25_manifest: dict[str, Any],
        embedding_manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        manifests: list[dict[str, Any]] = []
        if method in {"bm25_indexed", "hybrid_indexed"} and bm25_hit:
            manifests.append(
                {
                    "index_type": "bm25",
                    "index_id": bm25_manifest.get("index_id", ""),
                    "run_id": bm25_manifest.get("run_id", ""),
                    "manifest": "step1_bm25_manifest.json",
                }
            )
        if method in {"embedding_indexed", "hybrid_indexed"} and embedding_hit:
            manifests.append(
                {
                    "index_type": "embedding",
                    "index_id": embedding_manifest.get("index_id", ""),
                    "run_id": embedding_manifest.get("run_id", ""),
                    "manifest": "step1_embedding_manifest.json",
                    "embedding_backend": embedding_manifest.get("embedding_backend", ""),
                    "embedding_model": embedding_manifest.get("embedding_model", ""),
                }
            )
        return manifests

    def _entry_text(self, entry: dict[str, Any]) -> str:
        object_type = entry.get("source_object_type")
        object_ref = str(entry.get("source_object_ref") or entry.get("source_object_id") or "")
        if object_type == "raw_evidence":
            item = self.store.resolve_evidence(object_ref)
            return str((item or {}).get("text") or (item or {}).get("derived_text") or "")
        if object_type == "memory_unit":
            item = self.store.resolve_memory(object_ref)
            return str((item or {}).get("content") or (item or {}).get("candidate_text") or "")
        if object_type == "doc_level_summary":
            item = self.store.resolve_summary(object_ref)
            if not item:
                return ""
            structured = item.get("structured_summary") or {}
            snippets = []
            for key in ("main_events", "explicit_commitments"):
                for event in structured.get(key, []) or []:
                    if event.get("text"):
                        snippets.append(str(event["text"]))
            return " ".join([str(item.get("summary_text", ""))] + snippets)
        return ""

    def _filter_evidence(self, filters: dict[str, Any]) -> list[dict[str, Any]]:
        rows = self.store.evidence
        if filters.get("source_ids"):
            allowed = set(filters["source_ids"])
            rows = [row for row in rows if row.get("source_id") in allowed]
        if filters.get("subject_ids"):
            allowed = set(filters["subject_ids"])
            rows = [row for row in rows if allowed & set(row.get("subject_ids", []))]
        if filters.get("participant_ids"):
            allowed = set(filters["participant_ids"])
            rows = [row for row in rows if allowed & set(row.get("participant_ids", []))]
        return rows

    def _entity_matches(self, query: str, item: dict[str, Any]) -> list[str]:
        query_tokens = tokens(query)
        matches = []
        speaker = item.get("speaker")
        if speaker and speaker.lower() in query_tokens:
            matches.append(speaker)
        for subject in item.get("subject_ids", []):
            if subject.lower() in query_tokens and subject not in matches:
                matches.append(subject)
        return matches

    def _rank(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for index, row in enumerate(rows, 1):
            row["rank"] = index
        return rows

    def _guess_ref_type(self, ref_id: str) -> str:
        if ref_id.startswith("evidence:"):
            return "evidence_ref"
        if self.store.resolve_evidence(ref_id):
            return "evidence_ref"
        if ref_id.startswith("memory:"):
            return "memory_id"
        if ref_id.startswith("source:"):
            return "source_ref"
        if ref_id.startswith("report:"):
            return "report_ref"
        if ref_id.startswith("audit:"):
            return "audit_ref"
        if ref_id.startswith("cache:"):
            return "cache_ref"
        return "unknown"

    def _unresolved(self, ref_id: str, ref_type: str, reason: str, warnings: list[str] | None = None) -> dict[str, Any]:
        return {
            "schema_version": "step1.resolved_ref.v1",
            "ref_id": ref_id,
            "ref_type": ref_type,
            "resolved": False,
            "text": "",
            "metadata": {"failure_reason": reason},
            "source_refs": [],
            "evidence_refs": [],
            "failure_reason": reason,
            "resolution_warnings": warnings or ["unresolved_ref"],
        }

    def _looks_contradictory(self, claim: str, text: str) -> bool:
        claim_tokens = tokens(claim)
        text_tokens = tokens(text)
        if not (claim_tokens & text_tokens):
            return False
        negators = {"not", "no", "never", "none"}
        return bool(text_tokens & negators) and not bool(claim_tokens & negators)

    def _packet_result(self, result: dict[str, Any], packet_mode: str) -> dict[str, Any]:
        use_policy = self._derive_use_policy(
            item_layer=result.get("item_layer"),
            warnings=result.get("warnings", []),
            support_strength=None,
        )
        base = {
            "result_id": result.get("result_id"),
            "use_policy": use_policy,
            "summary": self._summary(result.get("text", "")),
        }
        if packet_mode == "by_value":
            base.update(
                {
                    "text": result.get("text", ""),
                    "source_refs": result.get("source_refs", []),
                    "evidence_refs": result.get("evidence_refs", []),
                    "warnings": result.get("warnings", []),
                }
            )
        return base

    def _packet_resolved_ref(self, ref: dict[str, Any], packet_mode: str) -> dict[str, Any]:
        base = {
            "ref_id": ref.get("ref_id"),
            "resolved": ref.get("resolved", False),
        }
        if packet_mode == "by_value":
            base["text"] = ref.get("text", "")
        return base

    def _packet_support_check(self, check: dict[str, Any]) -> dict[str, Any]:
        support_strength = check.get("support_strength")
        return {
            "check_id": check.get("check_id"),
            "claim": check.get("claim"),
            "support_strength": support_strength,
            "supporting_refs": check.get("supporting_refs", []),
            "use_policy": self._derive_use_policy(
                item_layer="raw_evidence",
                warnings=[],
                support_strength=support_strength,
            ),
        }

    def _derive_use_policy(self, item_layer: str | None, warnings: list[str], support_strength: str | None) -> str:
        warning_set = set(warnings)
        if "cache_not_truth" in warning_set or "cache_stale_warning" in warning_set:
            return "do_not_use"
        if support_strength == "direct":
            return "assert"
        if support_strength == "partial":
            return "cautious_assert"
        if support_strength in {"weak", "unknown", "contradicts"}:
            return "do_not_assert"
        if item_layer == "raw_evidence":
            return "assert"
        if item_layer == "memory_unit":
            return "background_only"
        return "do_not_assert"

    def _summary(self, text: str, limit: int = 160) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."
