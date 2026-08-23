"""Fail-closed planning and launch-readiness preflight for V8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V7_REQ11_PANEL_IDENTITY,
    V8_FIDELITY_TARGETS,
    V8_INITIAL_ALLOCATION,
    V8_PORTFOLIO_VERSION,
    V8PortfolioPolicy,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext

V8_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v8-planning-preflight-v1"
V8_OPERATOR_LIMITS = {"OPTUNA": 26, "OPENEVOLVE": 10, "DIRECT": 8}
V8_DURATION_SECONDS = 32 * 60 * 60
V8_FINALISATION_RESERVE_SECONDS = 6 * 60 * 60
V8_RESEARCH_DIRECTOR_MODEL = "claude-opus-5"
V8_RESEARCH_DIRECTOR_MAXIMUM_CALLS = 8
V8_A6000_EPOCH_WORK = {"structural_basic_unet": 1_075, "dynunet": 245}


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
    if any(
        item.evidence_hash != payload_hash(dict(item.safe_payload)) for item in evidence
    ):
        return False
    by_type = {item.evidence_type: item for item in evidence}
    v7 = by_type["V7"].safe_payload
    req11 = by_type["REQ11"].safe_payload
    ensemble = by_type["ENSEMBLE"].safe_payload
    runtime = by_type["RUNTIME"].safe_payload
    failure = by_type["FAILURE"].safe_payload
    if (
        v7.get("sealed_holdout_evaluations") != 0
        or req11.get("panel_identity") != V7_REQ11_PANEL_IDENTITY
        or req11.get("objective_role") != "parent_selection_and_close_tie_evidence_only"
        or ensemble.get("sealed_holdout_evaluations") != 0
        or not isinstance(ensemble.get("mean_subject_macro_dice"), (int, float))
        or not isinstance(ensemble.get("best_single_score"), (int, float))
        or ensemble["mean_subject_macro_dice"] <= ensemble["best_single_score"]
        or runtime.get("gpu") != "NVIDIA RTX A6000"
        or runtime.get("measured_full_step_passed") is not True
        or runtime.get("runtime_coefficients_finalised") is not False
        or failure.get("successful_evidence_affected") is not False
        or not isinstance(failure.get("trainable_parameters"), int)
        or not isinstance(failure.get("maximum_trainable_parameters"), int)
        or failure["trainable_parameters"] <= failure["maximum_trainable_parameters"]
    ):
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
        or portfolio.get("independent_confirmation_execution")
        != "l4_sidecar_after_champion_freeze"
        or portfolio.get("local_optuna_allocation")
        != {"structural_basic_unet": 22, "dynunet": 4}
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
        "maximum_promotions_to_50": 1,
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
    required_parent_count = parent_selection.get("required_parent_count")
    optional_parent_count = parent_selection.get("optional_parent_count")
    if (
        not isinstance(selected, list)
        or not isinstance(required_parent_count, int)
        or not isinstance(optional_parent_count, int)
        or len(selected) > required_parent_count + optional_parent_count
    ):
        raise ValueError("feta_unet_v8_parent_selection_invalid")
    parent_ids: set[str] = set()
    parent_trajectories: set[str] = set()
    for parent in selected:
        if (
            not isinstance(parent, dict)
            or not isinstance(parent.get("experiment_id"), str)
            or not parent["experiment_id"].startswith("experiment-")
            or parent["experiment_id"] in parent_ids
            or not isinstance(parent.get("configuration"), dict)
            or not isinstance(parent.get("score"), (int, float))
            or not math.isfinite(float(parent["score"]))
            or not isinstance(parent.get("trajectory_identity"), str)
            or not _sha256_value(parent["trajectory_identity"])
            or parent["trajectory_identity"] in parent_trajectories
            or not isinstance(parent.get("v8_seed_trajectory_identity"), str)
            or parent.get("selection_role") not in {"mandatory", "optional"}
        ):
            raise ValueError("feta_unet_v8_parent_selection_invalid")
        candidate = FeTAUNetSearchConfiguration.model_validate(parent["configuration"])
        if (
            candidate.model_variant != "structural_basic_unet"
            or candidate.maximum_epochs != 150
            or trajectory_identity(candidate) != parent["v8_seed_trajectory_identity"]
        ):
            raise ValueError("feta_unet_v8_parent_selection_invalid")
        parent_ids.add(parent["experiment_id"])
        parent_trajectories.add(parent["trajectory_identity"])

    blockers: list[str] = []
    if len(selected) < required_parent_count:
        blockers.append("v7_parent_selection_pending")
    if options.get("v7_parent_manifest_sha256") != payload_hash(selected):
        blockers.append("v7_parent_manifest_pending")
    raw_rates = options.get("campaign_seconds_per_epoch_by_model_variant")
    if not isinstance(raw_rates, dict):
        raise ValueError("feta_unet_v8_runtime_rates_invalid")
    raw_structural_rate = raw_rates.get("structural_basic_unet")
    raw_dynunet_rate = raw_rates.get("dynunet")
    raw_reporting_reserve = options.get("campaign_reporting_reserve_seconds")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (
            raw_structural_rate,
            raw_dynunet_rate,
            raw_reporting_reserve,
        )
    ):
        raise ValueError("feta_unet_v8_runtime_rates_invalid")
    try:
        structural_rate = float(raw_structural_rate)
        dynunet_rate = float(raw_dynunet_rate)
        reporting_reserve = float(raw_reporting_reserve)
    except (TypeError, ValueError) as exc:
        raise ValueError("feta_unet_v8_runtime_rates_invalid") from exc
    planned_training_seconds = (
        V8_A6000_EPOCH_WORK["structural_basic_unet"] * structural_rate
        + V8_A6000_EPOCH_WORK["dynunet"] * dynunet_rate
    )
    graduation_seconds = 100 * structural_rate + 50 * dynunet_rate
    runtime_envelope_valid = (
        structural_rate > 0
        and dynunet_rate > 0
        and dynunet_rate >= structural_rate
        and reporting_reserve >= 0
        and planned_training_seconds + reporting_reserve <= V8_DURATION_SECONDS
        and graduation_seconds <= V8_FINALISATION_RESERVE_SECONDS
    )
    if (
        options.get("campaign_runtime_rates_finalised") is not True
        or not runtime_envelope_valid
    ):
        blockers.append("runtime_coefficients_pending")
    if options.get("campaign_portfolio_controller_implemented") is not True:
        blockers.append("v8_portfolio_controller_pending")
    else:
        V8PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=options))
    if options.get("v8_parent_reuse_imported") is not True:
        blockers.append("v8_parent_reuse_import_pending")
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
        "a6000_epoch_work": V8_A6000_EPOCH_WORK,
        "planned_training_seconds": planned_training_seconds,
        "planned_training_hours": planned_training_seconds / 3_600,
        "graduation_seconds": graduation_seconds,
        "graduation_hours": graduation_seconds / 3_600,
        "runtime_envelope_valid": runtime_envelope_valid,
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
