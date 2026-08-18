from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import yaml

from auto_researcher.contracts.enums import (
    EventType,
    ProvenanceKind,
    SearchType,
)
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.continuation import (
    CONTINUATION_VERSION,
    find_resume_source,
    trajectory_identity,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    PortfolioPolicy,
    apply_portfolio_policy,
)
from auto_researcher.tasks.models import TaskRuntimeContext


def _options() -> dict:
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/tasks/feta_unet_search/campaign-20h-template.yaml"
    return yaml.safe_load(path.read_text())["runtime"]["options"]


def _original() -> SearchRequest:
    return SearchRequest(
        request_id="planner-request",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target="mean_subject_macro_dice",
        search_space={"maximum_epochs": 25},
        experiment_budget=1,
        rationale="planner proposal overridden by the controller-owned portfolio",
    )


def _event(
    index: int,
    search_type: SearchType,
    configuration: dict,
    score: float,
) -> DecisionEvent:
    fidelity = int(configuration["maximum_epochs"])
    return DecisionEvent(
        event_id=f"event-{index}",
        run_id="portfolio-run",
        cycle=index,
        event_type=EventType.EVIDENCE_VERIFIED,
        actor="verifier",
        input_references=(f"experiment-{index}",),
        output_references=(
            "evidence:SUPPORTED",
            "verified:true",
            "constraints:true",
            f"score:{score}",
            f"search_type:{search_type.value}",
        ),
        rationale="verified aggregate development evidence",
        timestamp=datetime(2026, 8, 18, tzinfo=UTC),
        code_version="test",
        provenance=ProvenanceKind.REAL,
        safe_payload={
            "configuration": configuration,
            "aggregate_metrics": {
                "primary_score": score,
                "validation_history": [
                    {"epoch": 5, "validation_score": score - 0.05},
                    {"epoch": fidelity, "validation_score": score},
                ],
            },
        },
    )


def _configuration(index: int, fidelity: int = 25) -> dict:
    return {
        "maximum_epochs": fidelity,
        "learning_rate": 0.00003 + index * 0.000005,
        "weight_decay": 0.000001,
        "dropout": 0.0,
        "dice_weight": 1.2,
        "positive_negative_ratio": "1:1",
        "augmentation_strength": "baseline",
    }


def test_portfolio_policy_requires_exact_screening_and_promotion_shape():
    policy = PortfolioPolicy.from_runtime(TaskRuntimeContext(task_options=_options()))
    assert policy is not None
    assert policy.screening == {
        SearchType.OPTUNA: 36,
        SearchType.OPENEVOLVE: 12,
        SearchType.DIRECT: 12,
    }
    assert policy.promotion_targets == {50: 18, 100: 7, 150: 2}
    assert policy.wildcard_counts == {50: 2, 100: 1, 150: 0}


def test_portfolio_controller_executes_60_18_7_2_with_continuations():
    context = TaskRuntimeContext(task_options=_options())
    original = _original()
    events: list[DecisionEvent] = []

    request = apply_portfolio_policy(
        original,
        run_id="portfolio-run",
        cycle=1,
        events=tuple(events),
        runtime_context=context,
    )
    assert request is not None
    assert request.search_type == SearchType.OPTUNA
    assert request.experiment_budget == 36
    assert request.search_space["fixed"]["maximum_epochs"] == 25

    for index in range(36):
        events.append(
            _event(index, SearchType.OPTUNA, _configuration(index), 0.5 + index / 1000)
        )
    request = apply_portfolio_policy(
        original,
        run_id="portfolio-run",
        cycle=2,
        events=tuple(events),
        runtime_context=context,
    )
    assert request is not None
    assert request.search_type == SearchType.OPENEVOLVE
    assert request.experiment_budget == 13
    assert request.search_space["campaign_context"]["incumbent_primary_score"] == (
        0.535
    )
    assert len(request.search_space["campaign_context"]["prior_verified_results"]) == 12

    for offset in range(12):
        index = 36 + offset
        events.append(
            _event(
                index, SearchType.OPENEVOLVE, _configuration(index), 0.6 + offset / 1000
            )
        )
    request = apply_portfolio_policy(
        original,
        run_id="portfolio-run",
        cycle=3,
        events=tuple(events),
        runtime_context=context,
    )
    assert request is not None
    assert request.search_type == SearchType.DIRECT
    assert request.search_space["maximum_epochs"] == 25

    for offset in range(12):
        index = 48 + offset
        events.append(
            _event(
                index, SearchType.DIRECT, _configuration(index), 0.55 + offset / 1000
            )
        )

    expected = ((50, 18), (100, 7), (150, 2))
    event_index = len(events)
    for fidelity, count in expected:
        seen: set[str] = set()
        for _ in range(count):
            request = apply_portfolio_policy(
                original,
                run_id="portfolio-run",
                cycle=event_index + 1,
                events=tuple(events),
                runtime_context=context,
            )
            assert request is not None
            assert request.search_type == SearchType.DIRECT
            assert request.search_space["maximum_epochs"] == fidelity
            assert any(
                reference.startswith("promotion-from-epoch:")
                for reference in request.evidence_references
            )
            candidate = FeTAUNetSearchConfiguration.model_validate(
                dict(request.search_space)
            )
            identity = trajectory_identity(candidate)
            assert identity not in seen
            seen.add(identity)
            events.append(
                _event(
                    event_index,
                    SearchType.DIRECT,
                    dict(request.search_space),
                    0.7 + event_index / 10_000,
                )
            )
            event_index += 1

    assert (
        apply_portfolio_policy(
            original,
            run_id="portfolio-run",
            cycle=64,
            events=tuple(events),
            runtime_context=context,
        )
        is None
    )


def test_trajectory_identity_excludes_only_fidelity():
    at_25 = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 25))
    at_50 = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 50))
    changed = FeTAUNetSearchConfiguration.model_validate(_configuration(2, 50))
    assert trajectory_identity(at_25) == trajectory_identity(at_50)
    assert trajectory_identity(at_25) != trajectory_identity(changed)


def test_resume_source_selects_highest_completed_lower_rung(tmp_path):
    requested = FeTAUNetSearchConfiguration.model_validate(_configuration(1, 150))
    namespace = tmp_path / "namespace"
    current = namespace / "experiment-current"
    for name, completed in (("experiment-25", 25), ("experiment-100", 100)):
        checkpoint_root = namespace / name / "checkpoints/fold-0"
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "continuation.json").write_text(
            json.dumps(
                {
                    "schema_version": CONTINUATION_VERSION,
                    "trajectory_identity": trajectory_identity(requested),
                    "completed_epoch": completed,
                }
            )
        )
    assert find_resume_source(namespace, current, requested) == (
        namespace / "experiment-100"
    )
