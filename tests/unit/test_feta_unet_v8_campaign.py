from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime

import pytest
import yaml

from auto_researcher.contracts.enums import EventType, ProvenanceKind, SearchType
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.tasks.feta_unet_direct.model import (
    architecture_identity,
    create_unet_model,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    V8_MAXIMUM_TRAINABLE_PARAMETERS,
    V8_MINIMUM_TRAINABLE_PARAMETERS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.v8_preflight import (
    V8_FIDELITY_TARGETS,
    V8_PORTFOLIO_VERSION,
    build_v8_preflight_plan,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V8PortfolioPolicy,
    _evidence,
    _tree_candidates,
    apply_portfolio_policy,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext

ROOT = Path(__file__).parents[2]
TASK_CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-32h-v8-template.yaml"
CONTRACT = ROOT / "examples/tasks/feta_unet_search/contract-32h-v8.yaml"
BOUND_EVIDENCE = ROOT / "examples/tasks/feta_unet_search/v8-bound-evidence.yaml"


def _options() -> dict:
    return yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))["runtime"]["options"]


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 10},
        experiment_budget=1,
        rationale="controller replaces this request",
    )


def _planned(index: int, request: SearchRequest) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"planned-{index}",
        run_id="v8-run",
        cycle=index,
        event_type=EventType.SEARCH_PLANNED,
        actor="planner",
        input_references=(request.hypothesis_id,),
        output_references=(
            request.request_id,
            f"search_type:{request.search_type.value}",
            *(
                f"evidence_reference:{reference}"
                for reference in request.evidence_references
            ),
        ),
        rationale=request.rationale,
        timestamp=datetime(2026, 8, 23, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


def _prepared(index: int, request: SearchRequest, experiment_id: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"prepared-{index}-{experiment_id}",
        run_id="v8-run",
        cycle=index,
        event_type=EventType.EXPERIMENT_PREPARED,
        actor="search",
        input_references=(request.request_id,),
        output_references=(experiment_id,),
        rationale="prepared",
        timestamp=datetime(2026, 8, 23, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


def _verified(
    index: int,
    experiment_id: str,
    configuration: dict,
    score: float,
    search_type: SearchType,
) -> DecisionEvent:
    fidelity = int(configuration["maximum_epochs"])
    return DecisionEvent(
        event_id=f"verified-{index}-{experiment_id}",
        run_id="v8-run",
        cycle=index,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        input_references=(experiment_id,),
        output_references=(
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
            f"score:{score}",
            f"search_type:{search_type.value}",
        ),
        rationale="verified aggregate evidence",
        timestamp=datetime(2026, 8, 23, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "primary_score": score,
                "validation_history": [
                    {"epoch": max(5, fidelity - 5), "validation_score": score - 0.01},
                    {"epoch": fidelity, "validation_score": score},
                ],
            },
        },
    )


def _stage(request: SearchRequest) -> str:
    return next(
        reference.split(":", 1)[1]
        for reference in request.evidence_references
        if reference.startswith("tree-stage:")
    )


def _simulated_configurations(
    request: SearchRequest, *, request_index: int
) -> tuple[dict, ...]:
    stage = _stage(request)
    if request.search_type == SearchType.OPENEVOLVE:
        raw = dict(
            request.search_space["campaign_context"]["incumbent_training_policy"]
        )
        raw.pop("policy_version", None)
        base = FeTAUNetSearchConfiguration(maximum_epochs=10, **raw).model_dump(
            mode="json"
        )
        if stage == "v8-structural-child":
            changes = (
                ("kernel_profile", "large_front"),
                (
                    "deep_supervision_heads",
                    1 if base["deep_supervision_heads"] != 1 else 0,
                ),
                ("stage_block_profile", "bottleneck_heavy"),
                ("residual_profile", "encoder_only"),
            )
            return tuple(
                FeTAUNetSearchConfiguration(**{**base, name: value}).model_dump(
                    mode="json"
                )
                for name, value in changes[: request.experiment_budget - 1]
            )
        assert stage == "v8-structural-wildcard"
        value = 3 if base["convolutions_per_stage"] != 3 else 1
        return (
            FeTAUNetSearchConfiguration(
                **{**base, "convolutions_per_stage": value}
            ).model_dump(mode="json"),
        )
    if request.search_type == SearchType.OPTUNA:
        fixed = dict(request.search_space["fixed"])
        parameters = request.search_space["parameters"]
        rows = []
        for offset in range(request.experiment_budget):
            fraction = (offset + 1) / (request.experiment_budget + 1)
            sampled: dict = {}
            for name, specification in parameters.items():
                if "choices" in specification:
                    choices = specification["choices"]
                    sampled[name] = choices[(request_index + offset) % len(choices)]
                elif "low" in specification and "high" in specification:
                    sampled[name] = specification["low"] + fraction * (
                        specification["high"] - specification["low"]
                    )
            sampled["learning_rate"] *= 1.0 + request_index * 1e-5
            rows.append(
                FeTAUNetSearchConfiguration(**fixed, **sampled).model_dump(mode="json")
            )
        return tuple(rows)
    return (
        FeTAUNetSearchConfiguration.model_validate(request.search_space).model_dump(
            mode="json"
        ),
    )


def _replay_v8_portfolio() -> tuple[list[DecisionEvent], list[SearchRequest]]:
    context = TaskRuntimeContext(task_options=_options())
    events: list[DecisionEvent] = []
    requests: list[SearchRequest] = []
    experiment_index = 0
    for cycle in range(1, 200):
        request = apply_portfolio_policy(
            _original(),
            run_id="v8-run",
            cycle=cycle,
            events=tuple(events),
            runtime_context=context,
        )
        if request is None:
            break
        requests.append(request)
        events.append(_planned(cycle, request))
        configurations = _simulated_configurations(request, request_index=len(requests))
        for configuration in configurations:
            experiment_index += 1
            experiment_id = f"experiment-v8-{experiment_index:03d}"
            score = 0.70 + experiment_index / 10_000
            events.extend(
                (
                    _prepared(cycle, request, experiment_id),
                    _verified(
                        cycle,
                        experiment_id,
                        configuration,
                        score,
                        request.search_type,
                    ),
                )
            )
    else:
        raise AssertionError("V8 deterministic controller did not terminate")
    return events, requests


def test_v8_planning_preflight_locks_envelope_but_blocks_launch():
    plan = build_v8_preflight_plan(TASK_CONFIG, CONTRACT)

    assert plan["planning_locked"] is True
    assert plan["launch_ready"] is False
    assert plan["portfolio_version"] == V8_PORTFOLIO_VERSION
    assert plan["fidelity_targets"] == V8_FIDELITY_TARGETS
    assert plan["operator_limits"] == {
        "OPTUNA": 26,
        "OPENEVOLVE": 10,
        "DIRECT": 8,
    }
    assert plan["initial_candidate_allocation"] == {
        "v7_structural_children": 8,
        "dynunet_roots": 4,
        "branch_local_optuna": 26,
        "controlled_direct_ablations": 4,
        "structural_wildcards": 2,
    }
    assert plan["blockers"] == [
        "runtime_coefficients_pending",
        "v8_parent_reuse_import_pending",
        "research_director_shadow_evaluation_pending",
        "research_director_live_smoke_pending",
        "research_director_resume_replay_pending",
        "real_cuda_preflight_pending",
        "launch_gate_not_passed",
    ]
    assert plan["model_calls_performed"] == 0
    assert plan["selected_v7_parent_count"] == 2
    assert plan["research_director"] == {
        "model_id": "claude-opus-5",
        "thinking": "adaptive",
        "effort": "xhigh",
        "maximum_calls": 8,
        "finalisation_reserve_suppressed": True,
    }


def test_v8_bound_evidence_matches_parent_and_director_manifests():
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    bound = yaml.safe_load(BOUND_EVIDENCE.read_text(encoding="utf-8"))
    options = raw["runtime"]["options"]
    selected = options["campaign_portfolio"]["parent_selection"]["selected_parents"]
    evidence = options["research_director_evidence"]

    assert options["v7_parent_manifest_sha256"] == bound["v7_parent_manifest_sha256"]
    assert [item["experiment_id"] for item in selected] == [
        item["experiment_id"] for item in bound["selected_parents"]
    ]
    assert (
        options["research_director_evidence_manifest_sha256"]
        == bound["research_director_evidence_manifest_sha256"]
    )
    assert {item["evidence_type"]: item["evidence_hash"] for item in evidence} == (
        bound["research_director_evidence_hashes"]
    )
    assert bound["sealed_holdout_evaluations"] == 0


def test_v8_preflight_rejects_rehashed_but_semantically_invalid_evidence(
    tmp_path: Path,
):
    from auto_researcher.runtime.identity import payload_hash

    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    evidence = raw["runtime"]["options"]["research_director_evidence"]
    ensemble = next(item for item in evidence if item["evidence_type"] == "ENSEMBLE")
    ensemble["safe_payload"]["sealed_holdout_evaluations"] = 1
    ensemble["evidence_hash"] = payload_hash(ensemble["safe_payload"])
    raw["runtime"]["options"]["research_director_evidence_manifest_sha256"] = (
        payload_hash(evidence)
    )
    changed = tmp_path / "campaign.yaml"
    changed.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_v8_preflight_plan(changed, CONTRACT)
    assert "research_director_evidence_binding_pending" in plan["blockers"]


def test_v8_dynunet_roots_are_unique_and_inside_parameter_envelope():
    plan = build_v8_preflight_plan(TASK_CONFIG, CONTRACT)

    assert len(plan["dynunet_roots"]) == 4
    assert len({item["architecture_identity"] for item in plan["dynunet_roots"]}) == 4
    assert all(
        V8_MINIMUM_TRAINABLE_PARAMETERS
        <= item["trainable_parameters"]
        <= V8_MAXIMUM_TRAINABLE_PARAMETERS
        for item in plan["dynunet_roots"]
    )


def test_v8_supports_real_ten_and_fifteen_epoch_trajectories():
    ten = FeTAUNetSearchConfiguration(maximum_epochs=10)
    fifteen = FeTAUNetSearchConfiguration(maximum_epochs=15)

    assert ten.maximum_epochs == 10
    assert fifteen.maximum_epochs == 15
    assert ten.model_dump(exclude={"maximum_epochs"}) == fifteen.model_dump(
        exclude={"maximum_epochs"}
    )


def test_v8_stage_patterns_materially_change_structural_architecture():
    common = {
        "model_variant": "structural_basic_unet",
        "feature_width": "v7_balanced_5",
        "architecture_budget": "basicunet-structural-15m-150m-v1",
        "kernel_profile": "standard",
        "residual_blocks": True,
        "convolutions_per_stage": 2,
    }
    uniform = FeTAUNetSearchConfiguration(**common)
    staged = FeTAUNetSearchConfiguration(
        **common,
        stage_block_profile="bottleneck_heavy",
        residual_profile="deep_only",
    )

    uniform_model = create_unet_model(uniform)
    staged_model = create_unet_model(staged)
    assert architecture_identity(uniform) != architecture_identity(staged)
    assert trainable_parameter_count(uniform_model) != trainable_parameter_count(
        staged_model
    )


def test_v8_preflight_rejects_cross_family_mutation(tmp_path: Path):
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["runtime"]["options"]["campaign_portfolio"]["lineage_rules"][
        "cross_family_mutation"
    ] = True
    changed = tmp_path / "campaign.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="feta_unet_v8_lineage_rules_invalid"):
        build_v8_preflight_plan(changed, CONTRACT)


def test_v8_preflight_rejects_more_candidates_than_locked(tmp_path: Path):
    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    raw["runtime"]["options"]["campaign_portfolio"]["fidelity_targets"]["10"] = 45
    changed = tmp_path / "campaign.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="feta_unet_v8_portfolio_invalid"):
        build_v8_preflight_plan(changed, CONTRACT)


def test_v8_agent_context_exposes_both_families_but_only_evolves_basicunet():
    contract = ResearchContract.model_validate(
        yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    )
    context = FeTAUNetSearchTask().create_agent_context(
        contract,
        TaskRuntimeContext(task_options={"openevolve_fidelity": 10}),
        {},
    )

    assert context.direct_configuration_schema["model_variant"] == [
        "structural_basic_unet",
        "dynunet",
    ]
    assert context.openevolve_space_summary["mutable_policy"]["model_variant"] == [
        "structural_basic_unet"
    ]
    assert context.openevolve_space_summary["mutable_policy"][
        "stage_block_profile"
    ] == ["uniform", "shallow_to_deep", "deep_to_shallow", "bottleneck_heavy"]


def test_v8_controller_replays_exact_44_to_3_envelope_and_confirmation():
    events, requests = _replay_v8_portfolio()
    candidates = _tree_candidates(tuple(events), _evidence(tuple(events)))

    stage_counts = {
        stage: len(
            {
                item.evidence.trajectory_identity
                for item in candidates
                if item.stage == stage
            }
        )
        for stage in {
            "v8-structural-child",
            "v8-dynunet-root",
            "v8-local-optuna",
            "v8-direct-ablation",
            "v8-structural-wildcard",
            "v8-promote-15",
            "v8-promote-25",
            "v8-promote-50",
            "v8-promote-100",
            "v8-promote-150",
            "v8-confirmation-150",
        }
    }
    assert stage_counts == {
        "v8-structural-child": 8,
        "v8-dynunet-root": 4,
        "v8-local-optuna": 26,
        "v8-direct-ablation": 4,
        "v8-structural-wildcard": 2,
        "v8-promote-15": 30,
        "v8-promote-25": 18,
        "v8-promote-50": 8,
        "v8-promote-100": 4,
        "v8-promote-150": 3,
        "v8-confirmation-150": 1,
    }
    rung10 = {
        item.evidence.trajectory_identity
        for item in candidates
        if item.evidence.fidelity == 10
    }
    assert len(rung10) == 44
    assert requests[-1].evidence_references[-1] == "confirmation-seed:20260824"
    assert requests[-1].search_space["seed"] == 20260824


def test_v8_local_optuna_replay_fixes_every_architectural_field():
    events, _ = _replay_v8_portfolio()
    candidates = _tree_candidates(tuple(events), _evidence(tuple(events)))
    by_identity = {item.evidence.trajectory_identity: item for item in candidates}
    architecture_fields = (
        "model_variant",
        "feature_width",
        "features",
        "architecture_budget",
        "upsample",
        "kernel_profile",
        "residual_blocks",
        "deep_supervision_heads",
        "convolutions_per_stage",
        "stage_block_profile",
        "residual_profile",
        "dilation_profile",
        "skip_fusion",
        "downsample",
    )
    local = [item for item in candidates if item.stage == "v8-local-optuna"]
    assert len(local) == 26
    for child in local:
        parent = by_identity[child.parent_trajectory]
        assert all(
            child.evidence.configuration[field] == parent.evidence.configuration[field]
            for field in architecture_fields
        )


def test_v8_policy_binds_two_external_parents_and_four_dynunet_roots():
    policy = V8PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=_options()))
    assert len(policy.selected_parents) == 2
    assert len(policy.dynunet_roots) == 4
    assert policy.fidelity_targets == V8_FIDELITY_TARGETS
    assert all(
        trajectory_identity(
            FeTAUNetSearchConfiguration.model_validate(parent["configuration"])
        )
        == parent["trajectory_identity"]
        for parent in policy.selected_parents
    )
