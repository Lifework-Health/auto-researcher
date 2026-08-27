"""Fail-closed offline readiness checks for V11 five-fold confirmation."""

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
    V11_CONFIGURATION_SCHEMA_VERSION,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V11_PORTFOLIO_VERSION,
    V11PortfolioPolicy,
    apply_portfolio_policy,
)

V11_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v11-static-preflight-v1"
V11_BOUND_EVIDENCE_SCHEMA_VERSION = "feta-unet-v11-bound-evidence-v1"
V11_SELECTED_EXPERIMENTS = (
    "experiment-fd7420c452e1982d",
    "experiment-fc2d8d2a371ddba0",
)


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("feta_unet_v11_input_invalid") from exc
    if not isinstance(raw, dict):
        raise TypeError("feta_unet_v11_input_invalid")
    return raw


def _validate_evidence(raw: dict[str, Any]) -> tuple[str, ...]:
    selected = raw.get("selected_candidates")
    prior_ensemble = raw.get("prior_ensemble")
    duplicate = raw.get("excluded_duplicate")
    if (
        raw.get("schema_version") != V11_BOUND_EVIDENCE_SCHEMA_VERSION
        or raw.get("selection_scope") != "fold-0-development-only"
        or raw.get("confirmation_scope") != "five-fold-development-oof"
        or raw.get("sealed_holdout_evaluations") != 0
        or not isinstance(selected, list)
        or len(selected) != 2
        or tuple(item.get("experiment_id") for item in selected)
        != V11_SELECTED_EXPERIMENTS
        or not all(
            isinstance(item, dict)
            and item.get("panel_order") == index
            and all(
                isinstance(item.get(name), str) and len(item[name]) == 64
                for name in (
                    "experiment_spec_sha256",
                    "evaluation_result_sha256",
                    "best_checkpoint_sha256",
                )
            )
            for index, item in enumerate(selected, start=1)
        )
        or not isinstance(prior_ensemble, dict)
        or prior_ensemble.get("fold0_mean_subject_macro_dice") != 0.8288165856904497
        or prior_ensemble.get("sealed_holdout_evaluations") != 0
        or not isinstance(duplicate, dict)
        or duplicate.get("experiment_id") != "experiment-73ea1c554b176a67"
        or duplicate.get("duplicate_of") != "experiment-f7626c9939c6e2be"
    ):
        raise ValueError("feta_unet_v11_evidence_invalid")
    blockers = raw.get("launch_blockers")
    if raw.get("launch_ready") is not False or not isinstance(blockers, list):
        raise ValueError("feta_unet_v11_launch_boundary_invalid")
    return tuple(str(item) for item in blockers)


def static_v11_preflight(
    *, config_path: Path, contract_path: Path, evidence_path: Path
) -> dict[str, Any]:
    configuration = _load(config_path)
    evidence = _load(evidence_path)
    blockers = _validate_evidence(evidence)
    options = configuration.get("runtime", {}).get("options")
    if not isinstance(options, dict):
        raise TypeError("feta_unet_v11_runtime_options_invalid")
    policy = V11PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=options))
    if (
        options.get("launch_gate") != "blocked_pending_v11_action_preflight"
        or options.get("sealed_test_gate")
        != "blocked_until_five_fold_models_and_ensemble_frozen"
        or options.get("planner_allocation_mode")
        != "deterministic_campaign_portfolio_compiler"
        or options.get("hypothesis_mode")
        != "deterministic_campaign_portfolio_confirmation"
        or options.get("v11_confirmation_scope") != "five-fold-development-oof"
        or options.get("continue_after_failed_candidate") is not False
    ):
        raise ValueError("feta_unet_v11_configuration_invalid")
    contract = ResearchContract.model_validate(_load(contract_path))
    task = default_task_registry().get("feta_unet_search")
    task.validate_contract(contract)
    original = SearchRequest(
        request_id="v11-preflight-request",
        hypothesis_id="v11-preflight-hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={},
        experiment_budget=1,
        rationale="deterministic confirmation controller preflight",
    )
    context = TaskRuntimeContext(task_options=options)
    projected = apply_portfolio_policy(
        original,
        run_id="v11-preflight",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    first = FeTAUNetSearchConfiguration.model_validate(policy.roots[0])
    if (
        projected is None
        or projected.search_type != SearchType.DIRECT
        or dict(projected.search_space) != first.model_dump(mode="json")
        or not projected.rationale.startswith(V11_PORTFOLIO_VERSION)
        or task.estimate_search_duration_seconds(projected, context) != 45_000.0
    ):
        raise ValueError("feta_unet_v11_controller_replay_invalid")
    return {
        "schema_version": V11_PREFLIGHT_SCHEMA_VERSION,
        "configuration_schema_version": V11_CONFIGURATION_SCHEMA_VERSION,
        "config_sha256": payload_hash(configuration),
        "contract_sha256": payload_hash(_load(contract_path)),
        "bound_evidence_sha256": payload_hash(evidence),
        "portfolio_version": V11_PORTFOLIO_VERSION,
        "root_count": len(policy.roots),
        "fold_count_per_root": 5,
        "oof_development_subjects": 68,
        "estimated_seconds_per_root": 45_000.0,
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
    report = static_v11_preflight(
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
