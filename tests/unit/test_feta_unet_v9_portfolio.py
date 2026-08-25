from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import yaml

from auto_researcher.contracts.enums import EventType, ProvenanceKind, SearchType
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V9_FIDELITY_TARGETS,
    V9_PORTFOLIO_VERSION,
    V9PortfolioPolicy,
    apply_portfolio_policy,
)
from auto_researcher.tasks.feta_unet_search.openevolve import (
    policy_from_configuration,
)
from auto_researcher.tasks.models import TaskRuntimeContext

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-36h-v9-template.yaml"


def _options() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["runtime"]["options"]


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 15},
        experiment_budget=1,
        rationale="controller replaces this request",
    )


def _planned(index: int, request: SearchRequest) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"planned-{index}",
        run_id="v9-run",
        cycle=index,
        event_type=EventType.SEARCH_PLANNED,
        actor="planner",
        input_references=(request.hypothesis_id,),
        output_references=(
            request.request_id,
            f"search_type:{request.search_type.value}",
            *(f"evidence_reference:{item}" for item in request.evidence_references),
        ),
        rationale=request.rationale,
        timestamp=datetime(2026, 8, 24, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
    )


def _prepared(index: int, request: SearchRequest, experiment_id: str) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"prepared-{index}-{experiment_id}",
        run_id="v9-run",
        cycle=index,
        event_type=EventType.EXPERIMENT_PREPARED,
        actor="search",
        input_references=(request.request_id,),
        output_references=(experiment_id,),
        rationale="prepared",
        timestamp=datetime(2026, 8, 24, tzinfo=UTC),
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
        run_id="v9-run",
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
        timestamp=datetime(2026, 8, 24, tzinfo=UTC),
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


def _configurations(request: SearchRequest, request_index: int) -> tuple[dict, ...]:
    if request.search_type == SearchType.DIRECT:
        return (
            FeTAUNetSearchConfiguration.model_validate(request.search_space).model_dump(
                mode="json"
            ),
        )
    if request.search_type == SearchType.OPTUNA:
        rows = []
        for offset in range(request.experiment_budget):
            fixed = dict(request.search_space["fixed"])
            fraction = (request_index + offset + 1) / 100
            rows.append(
                FeTAUNetSearchConfiguration(
                    **fixed,
                    learning_rate=0.00008 + fraction * 0.0001,
                    weight_decay=0.000005 + fraction * 0.00001,
                    dropout=0.01 + fraction * 0.1,
                    dice_weight=1.15 + fraction,
                ).model_dump(mode="json")
            )
        return tuple(rows)
    parent = dict(request.search_space["campaign_context"]["incumbent_training_policy"])
    parent.pop("policy_version", None)
    changes = (
        {
            "feature_width": "v8_dyn_compact_5",
            "features": [48, 96, 192, 384, 768],
            "kernel_profile": "standard",
            "residual_blocks": False,
            "deep_supervision_heads": 0,
        },
        {
            "feature_width": "v8_dyn_compact_5",
            "features": [48, 96, 192, 384, 768],
            "kernel_profile": "large_front",
            "residual_blocks": True,
            "deep_supervision_heads": 1,
        },
        {
            "feature_width": "v8_dyn_balanced_5",
            "features": [64, 128, 256, 512, 768],
            "kernel_profile": "standard",
            "residual_blocks": False,
            "deep_supervision_heads": 0,
        },
        {
            "feature_width": "v8_dyn_balanced_5",
            "features": [64, 128, 256, 512, 768],
            "kernel_profile": "large_front",
            "residual_blocks": True,
            "deep_supervision_heads": 1,
        },
        {
            "feature_width": "v8_dyn_context_5",
            "features": [64, 96, 192, 480, 960],
            "kernel_profile": "standard",
            "residual_blocks": False,
            "deep_supervision_heads": 0,
        },
        {
            "feature_width": "v8_dyn_deep_6",
            "features": [40, 80, 160, 320, 640, 960],
            "kernel_profile": "standard",
            "residual_blocks": True,
            "deep_supervision_heads": 1,
        },
    )
    return tuple(
        FeTAUNetSearchConfiguration(**{**parent, **change}).model_dump(mode="json")
        for change in changes
    )


def test_v9_policy_validates_the_frozen_mixed_family_envelope():
    policy = V9PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=_options()))

    assert len(policy.roots) == 10
    assert policy.local_optuna_parent_count == 4
    assert policy.local_optuna_trials_per_parent == 2
    assert policy.openevolve_novel_children == 6
    assert policy.fidelity_targets == V9_FIDELITY_TARGETS


def test_v9_dynunet_parent_is_a_legal_openevolve_seed():
    root = _options()["v9_fixed_roots"][2]
    configuration = FeTAUNetSearchConfiguration.model_validate(root).model_dump(
        mode="json"
    )

    policy = policy_from_configuration(configuration)

    assert policy.model_variant == "dynunet"
    assert policy.architecture_budget == "dynunet-15m-150m-v1"


def test_v9_controller_replays_the_complete_24_to_3_ladder():
    context = TaskRuntimeContext(task_options=_options())
    events: list[DecisionEvent] = []
    requests: list[SearchRequest] = []
    experiment_index = 0
    for cycle in range(1, 200):
        request = apply_portfolio_policy(
            _original(),
            run_id="v9-run",
            cycle=cycle,
            events=tuple(events),
            runtime_context=context,
        )
        if request is None:
            break
        requests.append(request)
        events.append(_planned(cycle, request))
        for configuration in _configurations(request, len(requests)):
            experiment_index += 1
            experiment_id = f"experiment-v9-{experiment_index:03d}"
            score = 0.72 + experiment_index / 10_000
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
        raise AssertionError("V9 deterministic controller did not terminate")

    fidelities: dict[int, int] = {}
    for event in events:
        if event.event_type != EventType.EVIDENCE_VERIFIED:
            continue
        fidelity = event.safe_payload["configuration"]["maximum_epochs"]
        fidelities[fidelity] = fidelities.get(fidelity, 0) + 1
    assert fidelities == V9_FIDELITY_TARGETS
    assert sum(item.search_type == SearchType.OPTUNA for item in requests) == 4
    assert sum(item.search_type == SearchType.OPENEVOLVE for item in requests) == 1
    assert all(
        item.search_space.get("campaign_context", {}).get("required_model_variant")
        == "dynunet"
        for item in requests
        if item.search_type == SearchType.OPENEVOLVE
    )
    assert requests[0].rationale.startswith(V9_PORTFOLIO_VERSION)
