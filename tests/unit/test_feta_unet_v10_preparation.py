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
    V10_FIDELITY_TARGETS,
    V10_PORTFOLIO_VERSION,
    V10PortfolioPolicy,
    apply_portfolio_policy,
)
from auto_researcher.tasks.feta_unet_search.v10_preflight import (
    static_v10_preflight,
)
from auto_researcher.tasks.models import TaskRuntimeContext

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-36h-v10-template.yaml"
CONTRACT = ROOT / "examples/tasks/feta_unet_search/contract-36h-v10.yaml"
EVIDENCE = ROOT / "examples/tasks/feta_unet_search/v10-bound-evidence.json"


def _options() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["runtime"]["options"]


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 30},
        experiment_budget=1,
        rationale="controller replaces this request",
    )


def _verified(
    index: int,
    configuration: dict,
    score: float,
    search_type: SearchType,
) -> DecisionEvent:
    fidelity = int(configuration["maximum_epochs"])
    experiment_id = f"experiment-v10-{index:03d}"
    return DecisionEvent(
        event_id=f"verified-{index}",
        run_id="v10-run",
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
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "validation_history": [
                    {"epoch": fidelity - 5, "validation_score": score - 0.01},
                    {"epoch": fidelity, "validation_score": score},
                ]
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
                    loss_variant="generalized_dice_focal"
                    if offset % 2
                    else "dice_focal",
                    sampling_policy="weak_tissue_balanced"
                    if offset % 2
                    else "foreground",
                ).model_dump(mode="json")
            )
        return tuple(rows)
    parent = dict(request.search_space["campaign_context"]["incumbent_training_policy"])
    parent.pop("policy_version", None)
    profiles = (
        ("v8_dyn_compact_5", [48, 96, 192, 384, 768]),
        ("v8_dyn_balanced_5", [64, 128, 256, 512, 768]),
        ("v8_dyn_context_5", [64, 96, 192, 480, 960]),
        ("v8_dyn_deep_6", [40, 80, 160, 320, 640, 960]),
        ("v8_dyn_compact_5", [48, 96, 192, 384, 768]),
        ("v8_dyn_balanced_5", [64, 128, 256, 512, 768]),
    )
    return tuple(
        FeTAUNetSearchConfiguration(
            **{
                **parent,
                "maximum_epochs": 30,
                "feature_width": profile,
                "features": features,
                "learning_rate": parent["learning_rate"] * (1.01 + index / 100),
                "sampling_policy": "weak_tissue_balanced",
                "loss_variant": "generalized_dice_focal"
                if index % 2
                else "dice_tversky",
            }
        ).model_dump(mode="json")
        for index, (profile, features) in enumerate(profiles)
    )


def test_v10_static_preflight_is_complete_but_fail_closed():
    report = static_v10_preflight(
        config_path=CONFIG,
        contract_path=CONTRACT,
        evidence_path=EVIDENCE,
    )

    assert report["root_count"] == 6
    assert report["screening_target"] == 20
    assert report["promotion_targets"] == {"50": 10, "100": 6, "150": 4}
    assert report["planner_allocation_mode"] == (
        "deterministic_campaign_portfolio_compiler"
    )
    assert report["model_calls_performed"] == 0
    assert report["holdout_subjects_evaluated"] == 0
    assert report["launch_ready"] is False
    assert len(report["launch_blockers"]) == 1
    assert len(report["cuda_mechanism_smoke_sha256"]) == 64


def test_v10_policy_validates_mechanism_coverage():
    assert _options()["campaign_prior_results"] == 30
    policy = V10PortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=_options())
    )

    assert len(policy.roots) == 6
    assert policy.fidelity_targets == V10_FIDELITY_TARGETS
    assert {
        (item["loss_variant"], item["sampling_policy"]) for item in policy.roots
    }.issuperset(
        {
            ("dice_focal", "foreground"),
            ("dice_tversky", "weak_tissue_balanced"),
            ("generalized_dice_focal", "weak_tissue_balanced"),
        }
    )


def test_v10_controller_replays_complete_20_to_4_ladder():
    context = TaskRuntimeContext(task_options=_options())
    events: list[DecisionEvent] = []
    requests: list[SearchRequest] = []
    experiment_index = 0
    for cycle in range(1, 160):
        request = apply_portfolio_policy(
            _original(),
            run_id="v10-run",
            cycle=cycle,
            events=tuple(events),
            runtime_context=context,
        )
        if request is None:
            break
        requests.append(request)
        for configuration in _configurations(request, len(requests)):
            experiment_index += 1
            events.append(
                _verified(
                    experiment_index,
                    configuration,
                    0.72 + experiment_index / 10_000,
                    request.search_type,
                )
            )
    else:
        raise AssertionError("V10 deterministic controller did not terminate")

    fidelities: dict[int, int] = {}
    for event in events:
        fidelity = int(event.safe_payload["configuration"]["maximum_epochs"])
        fidelities[fidelity] = fidelities.get(fidelity, 0) + 1
    assert fidelities == V10_FIDELITY_TARGETS
    assert sum(item.search_type == SearchType.OPTUNA for item in requests) == 4
    assert sum(item.search_type == SearchType.OPENEVOLVE for item in requests) == 1
    assert requests[0].rationale.startswith(V10_PORTFOLIO_VERSION)
