"""Fail-closed offline readiness checks for the frozen V10 campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks import TaskRuntimeContext, default_task_registry
from auto_researcher.tasks.feta_unet_search.configuration import (
    V10_CONFIGURATION_SCHEMA_VERSION,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V10_FIDELITY_TARGETS,
    V10_PORTFOLIO_VERSION,
    V10PortfolioPolicy,
    apply_portfolio_policy,
)

V10_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v10-static-preflight-v1"
V10_BOUND_EVIDENCE_SCHEMA_VERSION = "feta-unet-v10-bound-evidence-v1"


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("feta_unet_v10_input_invalid") from exc
    if not isinstance(raw, dict):
        raise ValueError("feta_unet_v10_input_invalid")
    return raw


def _validate_evidence(raw: dict[str, Any]) -> tuple[str, ...]:
    if (
        raw.get("schema_version") != V10_BOUND_EVIDENCE_SCHEMA_VERSION
        or raw.get("development_fold") != 0
        or raw.get("sealed_holdout_evaluations") != 0
    ):
        raise ValueError("feta_unet_v10_evidence_identity_invalid")
    postmortem = raw.get("v9_postmortem")
    ensemble = raw.get("v9_ensemble")
    policy = raw.get("evidence_policy")
    cards = raw.get("literature_cards")
    directives = raw.get("research_director_directives")
    if (
        not isinstance(postmortem, dict)
        or postmortem.get("verified_execution_count") != 58
        or postmortem.get("champion_experiment_id") != "experiment-73ea1c554b176a67"
        or postmortem.get("champion_score") != 0.8246828360878596
        or postmortem.get("best_bound_prior_score") != 0.8260410594691577
        or not isinstance(ensemble, dict)
        or ensemble.get("mean_subject_macro_dice") != 0.8306509760526394
        or ensemble.get("sealed_holdout_evaluations") != 0
        or not isinstance(cards, list)
        or len(cards) != 4
        or not isinstance(directives, list)
        or len(directives) != 6
        or not isinstance(policy, dict)
        or policy.get("experiment_authority_exercised") is not False
        or policy.get("planner_allocation")
        != "deterministic-campaign-portfolio-compiler"
    ):
        raise ValueError("feta_unet_v10_evidence_invalid")
    blockers = raw.get("launch_blockers")
    if raw.get("launch_ready") is not False or not isinstance(blockers, list):
        raise ValueError("feta_unet_v10_launch_boundary_invalid")
    return tuple(str(item) for item in blockers)


def static_v10_preflight(
    *, config_path: Path, contract_path: Path, evidence_path: Path
) -> dict[str, Any]:
    configuration = _load(config_path)
    evidence = _load(evidence_path)
    blockers = _validate_evidence(evidence)
    prior_manifest = _load(evidence_path.with_name("v10-prior-artifacts.json"))
    cuda_smoke = _load(evidence_path.with_name("v10-cuda-mechanism-smoke.json"))
    if (
        cuda_smoke.get("schema_version") != "feta-unet-v10-cuda-mechanism-smoke-v1"
        or cuda_smoke.get("gpu") != "NVIDIA RTX A6000"
        or cuda_smoke.get("loss_variant") != "generalized_dice_focal"
        or cuda_smoke.get("sampling_policy") != "weak_tissue_balanced"
        or cuda_smoke.get("loss_finite") is not True
        or cuda_smoke.get("passed") is not True
        or cuda_smoke.get("holdout_subjects_evaluated") != 0
        or cuda_smoke.get("model_calls_performed") != 0
        or not (0 < float(cuda_smoke.get("peak_gpu_memory_gib", 0)) <= 44)
    ):
        raise ValueError("feta_unet_v10_cuda_smoke_invalid")
    artifacts = prior_manifest.get("artifacts")
    if (
        prior_manifest.get("schema_version") != "feta-unet-v10-prior-artifacts-v1"
        or prior_manifest.get("sealed_holdout_evaluations") != 0
        or not isinstance(artifacts, list)
        or len(artifacts) != 4
        or not all(
            isinstance(item, dict)
            and str(item.get("experiment_id", "")).startswith("experiment-")
            and all(
                isinstance(item.get(name), str) and len(item[name]) == 64
                for name in (
                    "experiment_spec_sha256",
                    "evaluation_result_sha256",
                    "best_checkpoint_sha256",
                )
            )
            for item in artifacts
        )
    ):
        raise ValueError("feta_unet_v10_prior_artifacts_invalid")
    options = configuration.get("runtime", {}).get("options")
    agents = configuration.get("agents", {})
    if not isinstance(options, dict):
        raise ValueError("feta_unet_v10_runtime_options_invalid")
    policy = V10PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=options))
    if (
        options.get("launch_gate") != "blocked_pending_v10_action_preflight"
        or options.get("planner_allocation_mode")
        != "deterministic_campaign_portfolio_compiler"
        or options.get("v10_fidelity_ladder") != [30, 50, 100, 150]
        or options.get("campaign_prior_results") != 30
        or policy.fidelity_targets != V10_FIDELITY_TARGETS
        or agents.get("budget", {}).get(
            "maximum_research_director_valid_decisions_total"
        )
        != 20
        or agents.get("research_director", {}).get("model_id") != "claude-opus-5"
        or agents.get("research_director", {}).get("effort") != "xhigh"
    ):
        raise ValueError("feta_unet_v10_configuration_invalid")
    contract = ResearchContract.model_validate(_load(contract_path))
    task = default_task_registry().get("feta_unet_search")
    task.validate_contract(contract)
    original = SearchRequest(
        request_id="v10-preflight-request",
        hypothesis_id="v10-preflight-hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 30},
        experiment_budget=1,
        rationale="deterministic controller preflight",
    )
    projected = apply_portfolio_policy(
        original,
        run_id="v10-preflight",
        cycle=1,
        events=(),
        runtime_context=TaskRuntimeContext(task_options=options),
    )
    first_root = FeTAUNetSearchConfiguration.model_validate(policy.roots[0])
    if (
        projected is None
        or projected.search_type != SearchType.DIRECT
        or projected.search_space
        != {
            name: first_root.model_dump(mode="json")[name]
            for name in projected.search_space
        }
        or not projected.rationale.startswith(V10_PORTFOLIO_VERSION)
    ):
        raise ValueError("feta_unet_v10_controller_replay_invalid")
    return {
        "schema_version": V10_PREFLIGHT_SCHEMA_VERSION,
        "configuration_schema_version": V10_CONFIGURATION_SCHEMA_VERSION,
        "config_sha256": payload_hash(configuration),
        "contract_sha256": payload_hash(_load(contract_path)),
        "bound_evidence_sha256": payload_hash(evidence),
        "prior_artifact_manifest_sha256": payload_hash(prior_manifest),
        "cuda_mechanism_smoke_sha256": payload_hash(cuda_smoke),
        "root_count": len(policy.roots),
        "screening_target": policy.fidelity_targets[30],
        "promotion_targets": {
            str(key): value
            for key, value in policy.fidelity_targets.items()
            if key > 30
        },
        "research_director_valid_decision_budget": 20,
        "planner_allocation_mode": options["planner_allocation_mode"],
        "first_request_id": projected.request_id,
        "holdout_subjects_evaluated": 0,
        "model_calls_performed": 0,
        "launch_ready": False,
        "launch_blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = static_v10_preflight(
        config_path=arguments.config,
        contract_path=arguments.contract,
        evidence_path=arguments.evidence,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
