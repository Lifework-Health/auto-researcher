from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from auto_researcher.contracts.enums import EventType, ProvenanceKind, SearchType
from auto_researcher.contracts.models import (
    DecisionEvent,
    ResearchContract,
    SearchRequest,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    V7_ARCHITECTURE_BUDGET,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V7_MECHANISM_PORTFOLIO_VERSION,
    V7_REQ11_PANEL_IDENTITY,
    V7_REQ11_PRIORITIES,
    V7MechanismPortfolioPolicy,
    apply_portfolio_policy,
    apply_v7_deadline_graduation_policy,
)
from auto_researcher.tasks.feta_unet_search.openevolve import (
    FeTAUNetEvolvableComponent,
    policy_from_configuration,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.feta_unet_search.v7_preflight import (
    build_v7_preflight_plan,
)
from auto_researcher.tasks.models import ExperimentMetadata, TaskRuntimeContext


ROOT = Path(__file__).resolve().parents[2]


def _options() -> dict:
    return yaml.safe_load(
        (
            ROOT / "examples/tasks/feta_unet_search/campaign-22h-v7-template.yaml"
        ).read_text()
    )["runtime"]["options"]


def _contract() -> ResearchContract:
    return ResearchContract.model_validate(
        yaml.safe_load(
            (ROOT / "examples/tasks/feta_unet_search/contract-22h-v7.yaml").read_text()
        )
    )


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 25},
        experiment_budget=1,
        rationale="controller replaces this request",
    )


def _planned(index: int, request: SearchRequest) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"plan-{index}",
        run_id="v7-run",
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
        timestamp=datetime(2026, 8, 22, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


def _prepared(index: int, request: SearchRequest, experiment_id: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"prepared-{index}-{experiment_id}",
        run_id="v7-run",
        cycle=index,
        event_type=EventType.EXPERIMENT_PREPARED,
        actor="search",
        input_references=(request.request_id,),
        output_references=(experiment_id,),
        rationale="prepared",
        timestamp=datetime(2026, 8, 22, tzinfo=UTC),
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
        run_id="v7-run",
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
        timestamp=datetime(2026, 8, 22, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "primary_score": score,
                "validation_history": [{"epoch": fidelity, "validation_score": score}],
            },
        },
    )


def test_v7_contract_and_frozen_mechanism_roots_are_valid():
    FeTAUNetSearchTask().validate_contract(_contract())
    policy = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    )
    assert len(policy.structural_roots) == 4
    assert policy.mutations_per_root == 3
    assert policy.optuna_trials_per_parent == 2
    assert len(policy.v6_parent_evidence) == 2
    assert policy.req11_diagnostic["panel_identity"] == V7_REQ11_PANEL_IDENTITY
    assert tuple(policy.req11_diagnostic["priorities"]) == V7_REQ11_PRIORITIES
    assert policy.promotion_targets == {50: 8, 100: 4, 150: 2}
    assert all(
        FeTAUNetSearchConfiguration.model_validate(item).architecture_budget
        == V7_ARCHITECTURE_BUDGET
        for item in policy.structural_roots
    )


def test_v7_static_preflight_freezes_four_roots_and_memory_ceiling():
    plan = build_v7_preflight_plan(
        ROOT / "examples/tasks/feta_unet_search/campaign-22h-v7-template.yaml",
        ROOT / "examples/tasks/feta_unet_search/contract-22h-v7.yaml",
    )
    assert plan["root_count"] == 4
    assert plan["maximum_peak_gpu_memory_bytes"] == 44 * 1024**3
    assert len({item["architecture_identity"] for item in plan["roots"]}) == 4
    assert all(
        15_000_000 <= item["trainable_parameters"] <= 150_000_000
        for item in plan["roots"]
    )
    assert plan["model_calls_performed"] == 0
    assert plan["graduating_finalist_count"] == 2
    assert plan["required_graduation_reserve_seconds"] == 24_300
    assert plan["configured_graduation_reserve_seconds"] == 24_300
    assert plan["inference_calibration"]["enabled"] is True
    assert plan["req11_diagnostic_bound"] is True
    assert plan["req11_panel_identity"] == V7_REQ11_PANEL_IDENTITY
    assert tuple(plan["req11_priorities"]) == V7_REQ11_PRIORITIES
    assert plan["static_preflight_passed"] is True
    assert plan["cuda_preflight_passed"] is False


def test_v7_static_preflight_rejects_reserve_that_cannot_finish_two_finalists(
    tmp_path: Path,
):
    configuration = yaml.safe_load(
        (
            ROOT / "examples/tasks/feta_unet_search/campaign-22h-v7-template.yaml"
        ).read_text()
    )
    configuration["runtime"]["options"]["campaign_finalisation_reserve_seconds"] = (
        24_299
    )
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    with pytest.raises(ValueError, match="graduation_budget_invalid"):
        build_v7_preflight_plan(
            path,
            ROOT / "examples/tasks/feta_unet_search/contract-22h-v7.yaml",
        )


def test_v7_starts_with_direct_mechanism_roots_then_structural_evolution():
    context = TaskRuntimeContext(task_options=_options())
    policy = V7MechanismPortfolioPolicy.from_runtime(context)
    events: list[DecisionEvent] = []
    for index, configuration in enumerate(policy.structural_roots):
        request = apply_portfolio_policy(
            _original(),
            run_id="v7-run",
            cycle=index + 1,
            events=tuple(events),
            runtime_context=context,
        )
        assert request is not None
        assert request.search_type == SearchType.DIRECT
        assert request.search_space == configuration
        experiment_id = f"root-{index}"
        events.extend(
            (
                _planned(index, request),
                _prepared(index, request, experiment_id),
                _verified(
                    index,
                    experiment_id,
                    configuration,
                    0.72 + index / 100,
                    SearchType.DIRECT,
                ),
            )
        )

    request = apply_portfolio_policy(
        _original(),
        run_id="v7-run",
        cycle=5,
        events=tuple(events),
        runtime_context=context,
    )
    assert request is not None
    assert request.search_type == SearchType.OPENEVOLVE
    assert request.experiment_budget == 4
    campaign = request.search_space["campaign_context"]
    assert campaign["required_model_variant"] == "structural_basic_unet"
    assert campaign["required_architecture_budget"] == V7_ARCHITECTURE_BUDGET
    assert "structural" in campaign["mutation_objective"]
    assert campaign["req11_diagnostic_evidence"] == policy.req11_diagnostic
    assert "topology continuity" in campaign["mutation_objective"]
    prior = campaign["prior_verified_results"]
    assert (
        sum(item.get("evidence_role") == "v6_parent_not_retrained" for item in prior)
        == 2
    )
    assert (
        FeTAUNetSearchTask().estimate_search_duration_seconds(request, context)
        == 3 * 25 * 90.0
    )


def test_v7_rejects_unbound_req11_panel_identity():
    options = _options()
    options["campaign_portfolio"]["req11_diagnostic"]["panel_identity"] = "0" * 64
    with pytest.raises(ValueError, match="req11_diagnostic_invalid"):
        V7MechanismPortfolioPolicy.from_runtime(
            TaskRuntimeContext(task_options=options)
        )


def test_v7_rejects_changed_req11_candidate_delta():
    options = _options()
    options["campaign_portfolio"]["req11_diagnostic"]["candidates"][0][
        "mean_macro_dice_delta"
    ] = 0.01
    with pytest.raises(ValueError, match="req11_diagnostic_invalid"):
        V7MechanismPortfolioPolicy.from_runtime(
            TaskRuntimeContext(task_options=options)
        )


def test_v7_openevolve_rejects_training_only_mutation():
    root = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    ).structural_roots[0]
    policy = policy_from_configuration(root)
    component = FeTAUNetEvolvableComponent(
        maximum_epochs=25,
        seed_policy=policy,
    )
    assert component.component_spec().task_mutation_context[
        "bounded_model_variants"
    ] == ["structural_basic_unet"]
    request = SearchRequest(
        request_id="structural-evolution",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="mean_subject_macro_dice",
        search_space={
            "campaign_context": {
                "incumbent_training_policy": policy.model_dump(mode="json"),
                "required_model_variant": "structural_basic_unet",
                "required_architecture_budget": V7_ARCHITECTURE_BUDGET,
            }
        },
        experiment_budget=2,
        rationale="structural evolution",
    )
    training_only = policy.model_copy(
        update={"learning_rate": policy.learning_rate * 1.05}
    )
    with pytest.raises(ValueError, match="v7_structural_mutation_required"):
        component.candidate_to_experiment(
            SimpleNamespace(generation=1, candidate_id="candidate-training-only"),
            SimpleNamespace(
                generated_configuration=training_only.model_dump(mode="json")
            ),
            request,
            _contract(),
            ExperimentMetadata(
                evaluator_id="feta-basic-unet-search-evaluator",
                code_version="test",
                dataset_version="test",
                provenance=ProvenanceKind.REAL,
            ),
            run_id="v7-run",
        )


def test_v7_deadline_switches_to_direct_150_epoch_graduation():
    context = TaskRuntimeContext(task_options=_options())
    policy = V7MechanismPortfolioPolicy.from_runtime(context)
    events: list[DecisionEvent] = []
    for index, configuration in enumerate(policy.structural_roots[:2]):
        request = SearchRequest(
            request_id=f"root-request-{index}",
            hypothesis_id="hypothesis",
            search_type=SearchType.DIRECT,
            target="mean_subject_macro_dice",
            search_space=configuration,
            experiment_budget=1,
            rationale="root",
            evidence_references=("tree-stage:root", "tree-action:DIRECT"),
        )
        experiment_id = f"root-{index}"
        events.extend(
            (
                _planned(index, request),
                _prepared(index, request, experiment_id),
                _verified(
                    index,
                    experiment_id,
                    configuration,
                    0.80 + index / 100,
                    SearchType.DIRECT,
                ),
            )
        )

    completion = apply_v7_deadline_graduation_policy(
        _original(),
        run_id="v7-run",
        cycle=3,
        events=tuple(events),
        runtime_context=context,
    )
    assert completion is not None
    assert completion.search_type == SearchType.DIRECT
    assert completion.search_space["maximum_epochs"] == 150
    assert "promotion-from-epoch:25" in completion.evidence_references
    assert "graduation-mode:protected-deadline" in completion.evidence_references


def test_v7_local_optuna_preserves_structural_configuration():
    task = FeTAUNetSearchTask()
    root = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    ).structural_roots[1]
    tuned = {
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "lr_schedule",
        "loss_variant",
        "augmentation_policy",
    }
    request = SearchRequest(
        request_id="local-optuna",
        hypothesis_id="hypothesis",
        search_type=SearchType.OPTUNA,
        target="mean_subject_macro_dice",
        search_space={
            "fixed": {key: value for key, value in root.items() if key not in tuned},
            "parameters": {
                "learning_rate": {"low": 0.00008, "high": 0.0002},
                "weight_decay": {"low": 0.000003, "high": 0.00003},
                "dropout": {"low": 0.02, "high": 0.14},
                "dice_weight": {"low": 1.1, "high": 1.4},
                "positive_negative_ratio": {"choices": ["1:1", "2:1", "3:1"]},
                "lr_schedule": {"choices": ["constant", "cosine", "polynomial"]},
                "loss_variant": {"choices": ["dice_ce", "dice_focal", "dice_tversky"]},
                "augmentation_policy": {
                    "choices": [
                        "reference_light",
                        "geometric",
                        "intensity",
                        "combined",
                    ]
                },
            },
        },
        experiment_budget=2,
        rationale="local refinement",
    )
    study = task.create_optuna_study_spec(_contract(), request)
    fixed = study.fixed_configuration
    assert fixed["features"] == root["features"]
    assert fixed["kernel_profile"] == root["kernel_profile"]
    assert fixed["residual_blocks"] == root["residual_blocks"]
    assert fixed["deep_supervision_heads"] == root["deep_supervision_heads"]
    assert {item.name for item in study.parameters} == {
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
        "positive_negative_ratio",
        "lr_schedule",
        "loss_variant",
        "augmentation_policy",
    }


def test_v7_configuration_identity_includes_mechanism_axes():
    policy = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    )
    first = FeTAUNetSearchConfiguration.model_validate(policy.structural_roots[0])
    changed = first.model_copy(update={"residual_blocks": True})
    changed = FeTAUNetSearchConfiguration.model_validate(
        changed.model_dump(mode="json")
    )
    assert trajectory_identity(first) != trajectory_identity(changed)
    assert V7_MECHANISM_PORTFOLIO_VERSION in _options()["campaign_portfolio"]["version"]


def test_v7_structural_basicunet_roots_have_finite_forward_contracts():
    torch = pytest.importorskip("torch")
    from auto_researcher.tasks.feta_unet_direct.model import create_unet_model

    policy = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    )
    for raw in policy.structural_roots:
        configuration = FeTAUNetSearchConfiguration.model_validate(raw)
        model = create_unet_model(configuration).eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 1, 32, 32, 32))
        assert output.shape == (1, 8, 32, 32, 32)
        assert bool(torch.isfinite(output).all())


def test_deep_supervision_loss_weights_all_heads():
    torch = pytest.importorskip("torch")
    from auto_researcher.tasks.feta_unet_direct.trainer import (
        deep_supervision_training_loss,
    )

    configuration = FeTAUNetSearchConfiguration(
        maximum_epochs=25,
        model_variant="structural_basic_unet",
        feature_width="v7_compact_5",
        architecture_budget=V7_ARCHITECTURE_BUDGET,
        kernel_profile="standard",
        residual_blocks=True,
        deep_supervision_heads=2,
    )
    prediction = torch.stack(
        (
            torch.full((1, 8, 2, 2, 2), 1.0),
            torch.full((1, 8, 2, 2, 2), 2.0),
            torch.full((1, 8, 2, 2, 2), 3.0),
        ),
        dim=1,
    )
    target = torch.zeros((1, 1, 2, 2, 2))
    loss = deep_supervision_training_loss(
        prediction,
        target,
        lambda logits, _: logits.mean(),
        configuration,
    )
    assert float(loss) == pytest.approx((1.0 + 1.0 + 0.75) / 1.75)
