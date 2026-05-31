"""Shared pre-build routing orchestration for Build runners.

This module intentionally composes the existing router and proposal runner.
It does not implement routing logic, provider logic, prompt logic, or schema
validation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import memory_proposal_router
from tools.proposals import proposal_runner


PRE_BUILD_ROUTE_MODES = {"none", "route_only", "route_and_propose"}
PRE_BUILD_PROVIDERS = {"mock", "external_jsonl", "openai"}
DEFAULT_PRE_BUILD_ROUTE_MODE = "route_and_propose"
DEFAULT_ROUTE_POLICY = "configs/routing/memory_proposal_router/heuristic_salience_v0.21.candidate.yaml"
DEFAULT_PROPOSAL_PROVIDER = "openai"
DEFAULT_S1_PROPOSAL_PROFILE = "configs/proposals/s1_memory_candidate_proposal.v0.1.json"
DEFAULT_S2_PROPOSAL_PROFILE = "configs/proposals/s2_portrait_proposal.v0.2.json"


@dataclass
class PreBuildRoutingOptions:
    mode: str
    step_name: str
    target_task: str
    route_policy: str
    proposal_profile: str
    proposal_provider: str
    api_mode: str
    allow_live_api: bool
    env_file: str | None
    external_model_outputs: str | None
    max_items: int | None
    item_offset: int
    sample_stride: int
    duplicate_policy: str
    provider_concurrency: int


def safe_run_part(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "run")).strip("-")
    compact = compact.replace(":", "-")
    return compact or "run"


def get_arg(args: argparse.Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def options_from_args(
    args: argparse.Namespace,
    *,
    step_name: str,
    target_task: str,
    default_profile: str,
) -> PreBuildRoutingOptions:
    mode = str(get_arg(args, "pre_build_route_mode", DEFAULT_PRE_BUILD_ROUTE_MODE) or DEFAULT_PRE_BUILD_ROUTE_MODE)
    if mode not in PRE_BUILD_ROUTE_MODES:
        raise ValueError(f"Unsupported pre_build_route_mode: {mode}")
    provider = str(get_arg(args, "proposal_provider", DEFAULT_PROPOSAL_PROVIDER) or DEFAULT_PROPOSAL_PROVIDER)
    if provider not in PRE_BUILD_PROVIDERS:
        raise ValueError(f"Unsupported proposal_provider: {provider}")
    return PreBuildRoutingOptions(
        mode=mode,
        step_name=step_name,
        target_task=target_task,
        route_policy=str(get_arg(args, "route_policy", DEFAULT_ROUTE_POLICY) or DEFAULT_ROUTE_POLICY),
        proposal_profile=str(get_arg(args, "proposal_profile", default_profile) or default_profile),
        proposal_provider=provider,
        api_mode=str(get_arg(args, "api_mode", None) or os.environ.get("OPENAI_API_MODE") or "responses"),
        allow_live_api=bool(get_arg(args, "allow_live_api", False)),
        env_file=get_arg(args, "env_file", ".env"),
        external_model_outputs=get_arg(args, "external_model_outputs", None),
        max_items=get_arg(args, "max_items", None),
        item_offset=max(0, int(get_arg(args, "item_offset", 0) or 0)),
        sample_stride=max(1, int(get_arg(args, "sample_stride", 1) or 1)),
        duplicate_policy=str(get_arg(args, "duplicate_policy", "fail") or "fail"),
        provider_concurrency=max(1, int(get_arg(args, "provider_concurrency", 1) or 1)),
    )


def route_dir(output_workspace: Path, step_name: str, run_id: str) -> Path:
    return output_workspace / "routing" / f"prebuild_{step_name}_{safe_run_part(run_id)}"


def proposal_dir(output_workspace: Path, step_name: str, run_id: str) -> Path:
    return output_workspace / "proposals" / f"prebuild_{step_name}_{safe_run_part(run_id)}"


def run_router_to_dir(
    *,
    project_root: Path,
    route_workspace: Path,
    output_dir: Path,
    run_id: str,
    target_task: str,
    policy_path: str,
    duplicate_policy: str,
) -> dict[str, Any]:
    policy = memory_proposal_router.load_policy(project_root, policy_path)
    blocked = policy.blocked_target_tasks.intersection({target_task})
    if blocked:
        raise ValueError(f"Blocked target task(s): {', '.join(sorted(blocked))}")
    inputs = memory_proposal_router.RouterInputs(
        project_root=project_root,
        workspace=route_workspace,
        output_dir=output_dir,
        duplicate_policy=duplicate_policy,
        route_run_id=run_id,
        target_tasks=[target_task],
        policy=policy,
    )
    memory_proposal_router.prepare_outputs(output_dir, duplicate_policy)
    decisions, manifest = memory_proposal_router.build_route_decisions(inputs)
    memory_proposal_router.write_jsonl(output_dir / "route_decisions.jsonl", decisions)
    memory_proposal_router.write_json(output_dir / "route_run_manifest.json", manifest)
    (output_dir / "route_summary.md").write_text(
        memory_proposal_router.render_summary(manifest, decisions),
        encoding="utf-8",
    )
    return {
        "workspace": str(route_workspace),
        "output_dir": str(output_dir),
        "route_decisions": str(output_dir / "route_decisions.jsonl"),
        "route_run_manifest": str(output_dir / "route_run_manifest.json"),
        "route_summary": str(output_dir / "route_summary.md"),
        "router_policy_id": manifest["router_policy_id"],
        "router_policy_hash": manifest["router_policy_hash"],
        "counts": manifest["counts"],
    }


def resolve_path(project_root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_env_file(project_root: Path, env_file: str | None) -> str | None:
    if not env_file or proposal_runner.load_dotenv is None:
        return None
    env_path = resolve_path(project_root, env_file)
    if env_path is None:
        return None
    proposal_runner.load_dotenv(env_path)
    return str(env_path)


def load_runner_inputs(
    *,
    project_root: Path,
    proposal_workspace: Path,
    output_dir: Path,
    route_decisions_path: Path,
    run_id: str,
    options: PreBuildRoutingOptions,
) -> proposal_runner.RunnerInputs:
    load_env_file(project_root, options.env_file)
    profile = proposal_runner.load_profile(project_root, options.proposal_profile)
    live_api_enabled, live_api_unlock_source = proposal_runner.resolve_live_api(
        options.proposal_provider,
        options.allow_live_api,
    )
    weak_prompt_profile = profile.prompt_policies["weak"]
    strong_prompt_profile = profile.prompt_policies["strong"]
    return proposal_runner.RunnerInputs(
        project_root=project_root,
        workspace=proposal_workspace,
        output_dir=output_dir,
        profile=profile,
        duplicate_policy=options.duplicate_policy,
        proposal_run_id=run_id,
        provider=options.proposal_provider,
        api_mode=options.api_mode,
        live_api_enabled=live_api_enabled,
        live_api_unlock_source=live_api_unlock_source,
        weak_model=os.environ.get("OPENAI_MODEL_WEAK") or "mock-weak-model",
        strong_model=os.environ.get("OPENAI_MODEL_STRONG") or "mock-strong-model",
        weak_prompt=proposal_runner.load_prompt(
            project_root,
            weak_prompt_profile["path"],
            weak_prompt_profile["policy_id"],
        ),
        strong_prompt=proposal_runner.load_prompt(
            project_root,
            strong_prompt_profile["path"],
            strong_prompt_profile["policy_id"],
        ),
        route_decisions_path=route_decisions_path,
        max_items=options.max_items,
        item_offset=options.item_offset,
        sample_stride=options.sample_stride,
        text_units_path=proposal_workspace / "raw" / "organization" / "text_units.jsonl",
        evidence_path=proposal_workspace / "evidence" / "evidence.jsonl",
        section_map_path=proposal_workspace / "raw" / "organization" / "section_map.jsonl",
        memory_candidates_path=proposal_workspace / "memory" / "memory_candidates.jsonl",
        preprocessing_decisions_path=proposal_workspace / "memory" / "preprocessing_decisions.jsonl",
        external_model_outputs_path=resolve_path(project_root, options.external_model_outputs),
        provider_concurrency=options.provider_concurrency,
    )


def run_proposal_to_dir(
    *,
    project_root: Path,
    proposal_workspace: Path,
    output_dir: Path,
    route_decisions_path: Path,
    run_id: str,
    options: PreBuildRoutingOptions,
) -> dict[str, Any]:
    inputs = load_runner_inputs(
        project_root=project_root,
        proposal_workspace=proposal_workspace,
        output_dir=output_dir,
        route_decisions_path=route_decisions_path,
        run_id=run_id,
        options=options,
    )
    proposal_runner.prepare_outputs(inputs)
    packets, packet_warnings = proposal_runner.build_input_packets(inputs)
    proposals, human_queue, failures, model_call_inputs = proposal_runner.process_packets(inputs, packets)
    proposal_runner.validate_outputs(inputs, proposals, human_queue, failures)
    proposal_runner.write_jsonl(proposal_runner.output_path(inputs, "proposals"), proposals)
    proposal_runner.write_jsonl(proposal_runner.output_path(inputs, "human_review_queue"), human_queue)
    proposal_runner.write_jsonl(proposal_runner.output_path(inputs, "model_output_failures"), failures)
    proposal_runner.write_jsonl(proposal_runner.model_call_inputs_path(inputs), model_call_inputs)
    proposal_runner.empty_review_log(proposal_runner.output_path(inputs, "review_log"))
    manifest = proposal_runner.build_manifest(
        inputs,
        packets,
        packet_warnings,
        proposals,
        human_queue,
        failures,
        model_call_inputs,
    )
    proposal_runner.write_json(proposal_runner.output_path(inputs, "manifest"), manifest)
    proposal_runner.write_text(proposal_runner.output_path(inputs, "report"), proposal_runner.render_report(manifest))
    return {
        "workspace": str(proposal_workspace),
        "output_dir": str(output_dir),
        "proposal_run_id": inputs.proposal_run_id,
        "proposal_profile_id": inputs.profile.profile_id,
        "proposal_profile_hash": inputs.profile.profile_hash,
        "provider": options.proposal_provider,
        "env_file": str(resolve_path(project_root, options.env_file)) if options.env_file else None,
        "outputs": {
            "proposals": str(proposal_runner.output_path(inputs, "proposals")),
            "human_review_queue": str(proposal_runner.output_path(inputs, "human_review_queue")),
            "model_output_failures": str(proposal_runner.output_path(inputs, "model_output_failures")),
            "model_call_inputs": str(proposal_runner.model_call_inputs_path(inputs)),
            "proposal_review_log": str(proposal_runner.output_path(inputs, "review_log")),
            "proposal_run_manifest": str(proposal_runner.output_path(inputs, "manifest")),
            "proposal_run_report": str(proposal_runner.output_path(inputs, "report")),
        },
        "counts": manifest["counts"],
    }


def run_prebuild_routing(
    *,
    project_root: Path,
    route_workspace: Path,
    proposal_workspace: Path,
    output_workspace: Path,
    base_run_id: str,
    options: PreBuildRoutingOptions,
) -> dict[str, Any] | None:
    if options.mode == "none":
        return None
    route_run_id = f"prebuild-route:{options.step_name}:{base_run_id}"
    proposal_run_id = f"prebuild-proposal:{options.step_name}:{base_run_id}"
    routes = run_router_to_dir(
        project_root=project_root,
        route_workspace=route_workspace,
        output_dir=route_dir(output_workspace, options.step_name, base_run_id),
        run_id=route_run_id,
        target_task=options.target_task,
        policy_path=options.route_policy,
        duplicate_policy=options.duplicate_policy,
    )
    proposals = None
    if options.mode == "route_and_propose":
        proposals = run_proposal_to_dir(
            project_root=project_root,
            proposal_workspace=proposal_workspace,
            output_dir=proposal_dir(output_workspace, options.step_name, base_run_id),
            route_decisions_path=Path(routes["route_decisions"]),
            run_id=proposal_run_id,
            options=options,
        )
    return {
        "schema_version": "prebuild_routing_result.v0.1",
        "mode": options.mode,
        "step_name": options.step_name,
        "target_task": options.target_task,
        "route": routes,
        "proposal": proposals,
        "canonical_writes_executed": False,
    }


def add_prebuild_arguments(parser: argparse.ArgumentParser, *, default_profile: str) -> None:
    parser.add_argument("--pre-build-route-mode", default=DEFAULT_PRE_BUILD_ROUTE_MODE, choices=sorted(PRE_BUILD_ROUTE_MODES))
    parser.add_argument("--route-policy", default=DEFAULT_ROUTE_POLICY)
    parser.add_argument("--proposal-provider", default=DEFAULT_PROPOSAL_PROVIDER, choices=sorted(PRE_BUILD_PROVIDERS))
    parser.add_argument("--proposal-profile", default=default_profile)
    parser.add_argument("--api-mode", default=None)
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--provider-concurrency", type=int, default=1)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
