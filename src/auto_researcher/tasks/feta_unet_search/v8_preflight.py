"""Fail-closed planning and launch-readiness preflight for V8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from auto_researcher.agents.models import ResearchLandscapeEvidence
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_direct.model import (
    architecture_identity,
    create_unet_model,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    V8_DYNUNET_ARCHITECTURE_BUDGET,
    V8_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
    V8_MAXIMUM_TRAINABLE_PARAMETERS,
    V8_MINIMUM_TRAINABLE_PARAMETERS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask

V8_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v8-planning-preflight-v1"
V8_PORTFOLIO_VERSION = "feta-unet-v8-exploitation-44-30-18-8-4-3-v1"
V8_FIDELITY_TARGETS = {10: 44, 15: 30, 25: 18, 50: 8, 100: 4, 150: 3}
V8_OPERATOR_LIMITS = {"OPTUNA": 26, "OPENEVOLVE": 10, "DIRECT": 8}
V8_INITIAL_ALLOCATION = {
    "v7_structural_children": 8,
    "dynunet_roots": 4,
    "branch_local_optuna": 26,
    "controlled_direct_ablations": 4,
    "structural_wildcards": 2,
}
V8_DURATION_SECONDS = 32 * 60 * 60
V8_FINALISATION_RESERVE_SECONDS = 6 * 60 * 60
V8_RESEARCH_DIRECTOR_MODEL = "claude-opus-5"
V8_RESEARCH_DIRECTOR_MAXIMUM_CALLS = 8


def _mapping(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(reason) from exc
    if not isinstance(value, dict):
        raise ValueError(reason)
    return value


def _integer_mapping(raw: object, reason: str) -> dict[int, int]:
    if not isinstance(raw, dict):
        raise ValueError(reason)
    try:
        return {int(key): int(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(reason) from exc


def _sha256_value(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _research_director_evidence_bound(options: dict[str, Any]) -> bool:
    raw = options.get("research_director_evidence")
    manifest = options.get("research_director_evidence_manifest_sha256")
    if not isinstance(raw, list) or not raw or not _sha256_value(manifest):
        return False
    try:
        evidence = [ResearchLandscapeEvidence.model_validate(item) for item in raw]
    except (TypeError, ValueError):
        return False
    required = {"V7", "REQ11", "ENSEMBLE", "RUNTIME", "FAILURE"}
    if {item.evidence_type for item in evidence} != required:
        return False
    return payload_hash([item.model_dump(mode="json") for item in evidence]) == manifest


def build_v8_preflight_plan(
    task_config_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Lock the V8 envelope and report every remaining launch blocker."""

    raw_config = _mapping(
        task_config_path, "feta_unet_v8_preflight_configuration_invalid"
    )
    raw_runtime = raw_config.get("runtime")
    if not isinstance(raw_runtime, dict):
        raise ValueError("feta_unet_v8_preflight_configuration_invalid")
    options = raw_runtime.get("options")
    if not isinstance(options, dict):
        raise ValueError("feta_unet_v8_preflight_configuration_invalid")
    portfolio = options.get("campaign_portfolio")
    if not isinstance(portfolio, dict):
        raise ValueError("feta_unet_v8_portfolio_invalid")

    contract = ResearchContract.model_validate(
        _mapping(contract_path, "feta_unet_v8_preflight_contract_invalid")
    )
    FeTAUNetSearchTask().validate_contract(contract)
    agents = raw_config.get("agents")
    if not isinstance(agents, dict):
        raise ValueError("feta_unet_v8_research_director_invalid")
    director = agents.get("research_director")
    agent_budget = agents.get("budget")
    if (
        not isinstance(director, dict)
        or not isinstance(agent_budget, dict)
        or director.get("provider") != "anthropic"
        or director.get("model_id") != V8_RESEARCH_DIRECTOR_MODEL
        or director.get("temperature") is not None
        or director.get("thinking") != {"type": "adaptive"}
        or director.get("effort") != "xhigh"
        or director.get("maximum_output_tokens") != 64_000
        or director.get("maximum_attempts") != 2
        or director.get("maximum_cost_per_call") != 5.0
        or agent_budget.get("maximum_research_director_calls_total")
        != V8_RESEARCH_DIRECTOR_MAXIMUM_CALLS
        or agent_budget.get("maximum_research_director_output_tokens") != 64_000
        or agent_budget.get("maximum_research_director_cost_per_call") != 5.0
        or contract.maximum_cost != 150.0
    ):
        raise ValueError("feta_unet_v8_research_director_invalid")
    if (
        contract.constraints.get("campaign_duration_seconds") != V8_DURATION_SECONDS
        or contract.constraints.get("campaign_finalisation_reserve_seconds")
        != V8_FINALISATION_RESERVE_SECONDS
        or options.get("campaign_finalisation_reserve_seconds")
        != float(V8_FINALISATION_RESERVE_SECONDS)
        or options.get("maximum_peak_gpu_memory_bytes")
        != V8_MAXIMUM_PEAK_GPU_MEMORY_BYTES
    ):
        raise ValueError("feta_unet_v8_time_or_memory_envelope_invalid")

    allocation = portfolio.get("initial_candidate_allocation")
    operator_limits = portfolio.get("operator_limits")
    if (
        portfolio.get("version") != V8_PORTFOLIO_VERSION
        or _integer_mapping(
            portfolio.get("fidelity_targets"), "feta_unet_v8_fidelity_targets_invalid"
        )
        != V8_FIDELITY_TARGETS
        or operator_limits != V8_OPERATOR_LIMITS
        or allocation != V8_INITIAL_ALLOCATION
        or sum(V8_INITIAL_ALLOCATION.values()) != V8_FIDELITY_TARGETS[10]
        or portfolio.get("architecture_change_target") != 14
        or portfolio.get("independent_confirmation_count") != 1
    ):
        raise ValueError("feta_unet_v8_portfolio_invalid")

    lineage = portfolio.get("lineage_rules")
    if lineage != {
        "cross_family_mutation": False,
        "branch_local_optuna": True,
        "optuna_fixed_architecture": True,
        "openevolve_generation_zero_reuse": True,
        "duplicate_scientific_identity_policy": "reuse_verified_result",
    }:
        raise ValueError("feta_unet_v8_lineage_rules_invalid")
    dynunet_gate = portfolio.get("dynunet_gate")
    if dynunet_gate != {
        "comparison_fidelity": 25,
        "absolute_score_gap_maximum": 0.015,
        "alternative_evidence": [
            "superior_trajectory_slope",
            "req11_priority_gain",
            "ensemble_complementarity",
        ],
        "minimum_alternative_evidence_count": 2,
        "maximum_promotions_to_50": 2,
        "cross_family_mutation": False,
    }:
        raise ValueError("feta_unet_v8_dynunet_gate_invalid")

    raw_roots = portfolio.get("dynunet_root_configurations")
    if not isinstance(raw_roots, list) or len(raw_roots) != 4:
        raise ValueError("feta_unet_v8_dynunet_roots_invalid")
    roots: list[dict[str, Any]] = []
    identities: set[str] = set()
    trajectories: set[str] = set()
    for index, raw_root in enumerate(raw_roots):
        configuration = FeTAUNetSearchConfiguration.model_validate(raw_root)
        model = create_unet_model(configuration)
        parameters = trainable_parameter_count(model)
        identity = architecture_identity(configuration)
        trajectory = trajectory_identity(configuration)
        del model
        if (
            configuration.architecture_budget != V8_DYNUNET_ARCHITECTURE_BUDGET
            or configuration.maximum_epochs != 10
            or not V8_MINIMUM_TRAINABLE_PARAMETERS
            <= parameters
            <= V8_MAXIMUM_TRAINABLE_PARAMETERS
            or identity in identities
            or trajectory in trajectories
        ):
            raise ValueError("feta_unet_v8_dynunet_roots_invalid")
        identities.add(identity)
        trajectories.add(trajectory)
        roots.append(
            {
                "root_index": index,
                "feature_width": configuration.feature_width,
                "architecture_identity": identity,
                "trajectory_identity": trajectory,
                "trainable_parameters": parameters,
            }
        )

    raw_experiment = raw_config.get("experiment")
    if not isinstance(raw_experiment, dict) or raw_config.get("search") is not None:
        raise ValueError("feta_unet_v8_direct_launch_shape_invalid")
    initial = dict(raw_experiment)
    openevolve = initial.pop("openevolve", None)
    if not isinstance(openevolve, dict):
        raise ValueError("feta_unet_v8_openevolve_controls_invalid")
    initial_configuration = FeTAUNetSearchConfiguration.model_validate(initial)
    first_root = FeTAUNetSearchConfiguration.model_validate(raw_roots[0])
    if initial_configuration != first_root:
        raise ValueError("feta_unet_v8_initial_candidate_invalid")

    parent_selection = portfolio.get("parent_selection")
    if not isinstance(parent_selection, dict):
        raise ValueError("feta_unet_v8_parent_selection_invalid")
    selected = parent_selection.get("selected_parents")
    if not isinstance(selected, list) or len(selected) > 3:
        raise ValueError("feta_unet_v8_parent_selection_invalid")
    parent_ids: set[str] = set()
    for parent in selected:
        if (
            not isinstance(parent, dict)
            or not isinstance(parent.get("experiment_id"), str)
            or not parent["experiment_id"].startswith("experiment-")
            or parent["experiment_id"] in parent_ids
            or not isinstance(parent.get("configuration"), dict)
            or not isinstance(parent.get("score"), (int, float))
        ):
            raise ValueError("feta_unet_v8_parent_selection_invalid")
        candidate = FeTAUNetSearchConfiguration.model_validate(parent["configuration"])
        if candidate.model_variant != "structural_basic_unet":
            raise ValueError("feta_unet_v8_parent_selection_invalid")
        parent_ids.add(parent["experiment_id"])

    blockers: list[str] = []
    if len(selected) < int(parent_selection.get("required_parent_count", -1)):
        blockers.append("v7_parent_selection_pending")
    if not _sha256_value(options.get("v7_parent_manifest_sha256")):
        blockers.append("v7_parent_manifest_pending")
    if options.get("campaign_runtime_rates_finalised") is not True:
        blockers.append("runtime_coefficients_pending")
    if options.get("campaign_portfolio_controller_implemented") is not True:
        blockers.append("v8_portfolio_controller_pending")
    if options.get("research_director_controller_implemented") is not True:
        blockers.append("research_director_controller_pending")
    if not _research_director_evidence_bound(options):
        blockers.append("research_director_evidence_binding_pending")
    if options.get("research_director_shadow_evaluation_passed") is not True:
        blockers.append("research_director_shadow_evaluation_pending")
    if not _sha256_value(options.get("research_director_live_smoke_sha256")):
        blockers.append("research_director_live_smoke_pending")
    if options.get("research_director_resume_replay_passed") is not True:
        blockers.append("research_director_resume_replay_pending")
    if not _sha256_value(options.get("cuda_preflight_sha256")):
        blockers.append("real_cuda_preflight_pending")
    if options.get("launch_gate") != "passed":
        blockers.append("launch_gate_not_passed")

    plan_payload = {
        "portfolio_version": V8_PORTFOLIO_VERSION,
        "fidelity_targets": V8_FIDELITY_TARGETS,
        "operator_limits": V8_OPERATOR_LIMITS,
        "initial_candidate_allocation": V8_INITIAL_ALLOCATION,
        "duration_seconds": V8_DURATION_SECONDS,
        "finalisation_reserve_seconds": V8_FINALISATION_RESERVE_SECONDS,
        "maximum_peak_gpu_memory_bytes": V8_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
        "research_director": {
            "model_id": V8_RESEARCH_DIRECTOR_MODEL,
            "thinking": "adaptive",
            "effort": "xhigh",
            "maximum_calls": V8_RESEARCH_DIRECTOR_MAXIMUM_CALLS,
            "finalisation_reserve_suppressed": True,
        },
        "dynunet_gate": dynunet_gate,
        "lineage_rules": lineage,
    }
    return {
        "schema_version": V8_PREFLIGHT_SCHEMA_VERSION,
        "contract_id": contract.contract_id,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        **plan_payload,
        "dynunet_roots": roots,
        "selected_v7_parent_count": len(selected),
        "planning_locked": True,
        "launch_ready": not blockers,
        "blockers": blockers,
        "model_calls_performed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    value = build_v8_preflight_plan(args.task_config, args.contract)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
