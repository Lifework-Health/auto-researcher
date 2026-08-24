from __future__ import annotations

from pathlib import Path
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import yaml

from auto_researcher.cli import _load_development_openevolve_runtime
from auto_researcher.contracts.enums import EventType, ProvenanceKind, SearchType
from auto_researcher.contracts.models import DecisionEvent, SearchRequest
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.secrets import SecretProviderKind, parse_secret_reference
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
    V8_A6000_EPOCH_WORK,
    V8_FIDELITY_TARGETS,
    V8_PORTFOLIO_VERSION,
    _cuda_preflight_valid,
    build_v8_preflight_plan,
    run_v8_cuda_preflight,
)
from auto_researcher.tasks.feta_unet_search.continuation import trajectory_identity
from auto_researcher.tasks.feta_unet_search import portfolio as portfolio_module
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


def test_v8_uses_one_persistent_keyring_reference_for_all_anthropic_calls():
    task_config = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    agent_credential = parse_secret_reference(task_config["agents"]["credential"])
    mutation_credential = parse_secret_reference(
        task_config["openevolve_development_mutation"]["credential"]
    )

    assert agent_credential.provider is SecretProviderKind.LINUX_KERNEL_KEYRING
    assert mutation_credential == agent_credential
    assert agent_credential.provider_identifier == "auto-researcher/anthropic-api-key"


def test_v8_development_openevolve_guardrails_fit_runtime_interface():
    task_config = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))

    runtime = _load_development_openevolve_runtime(task_config)

    assert runtime is not None
    assert runtime.maximum_model_calls == 100
    assert runtime.maximum_total_cost_usd == 50.0


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
        "v8_parent_reuse_import_pending",
        "research_director_shadow_evaluation_pending",
        "research_director_live_smoke_pending",
        "research_director_resume_replay_pending",
        "real_cuda_preflight_pending",
        "launch_gate_not_passed",
    ]
    assert plan["model_calls_performed"] == 0
    assert plan["a6000_epoch_work"] == {
        "structural_basic_unet": 1085,
        "dynunet_promotable": 225,
        "v8_dyn_context_5": 10,
    }
    assert plan["runtime_envelope_valid"] is True
    assert plan["runtime_calibration_valid"] is True
    assert plan["planned_training_seconds"] == 111_875.0
    assert plan["graduation_seconds"] == 13_250.0
    assert plan["cuda_preflight_valid"] is False
    assert plan["selected_v7_parent_count"] == 2
    assert plan["research_director"] == {
        "model_id": "claude-opus-5",
        "thinking": "adaptive",
        "effort": "xhigh",
        "maximum_calls": 8,
        "finalisation_reserve_suppressed": True,
        "gate_valid": False,
        "gate_sha256": None,
    }


def test_v8_preflight_accepts_hash_bound_measured_runtime_envelope(tmp_path):
    from auto_researcher.runtime.identity import payload_hash

    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    options = raw["runtime"]["options"]
    feature_widths = [
        "v8_dyn_compact_5",
        "v8_dyn_balanced_5",
        "v8_dyn_context_5",
        "v8_dyn_deep_6",
    ]
    roots = [
        {
            "root_index": index,
            "feature_width": feature_width,
            "trainable_parameters": 50_000_000 + index,
            "total_seconds_per_epoch": 96.5 + index,
            "peak_gpu_memory_gib": 10.0 + index,
            "holdout_subjects_evaluated": 0,
        }
        for index, feature_width in enumerate(feature_widths)
    ]
    calibration = {
        "schema_version": "feta-unet-v8-runtime-calibration-v1",
        "holdout_subjects_evaluated": 0,
        "structural_basic_unet": {
            "observed_p90_total_seconds_per_epoch": 76.37,
            "selected_seconds_per_epoch": 78.0,
        },
        "dynunet": {
            "roots": roots,
            "selected_seconds_per_epoch": 120.0,
            "non_promotable_feature_widths": ["v8_dyn_context_5"],
            "source_log_sha256": "a" * 64,
        },
    }
    options["campaign_seconds_per_epoch_by_model_variant"].update(
        {"structural_basic_unet": 78.0, "dynunet": 120.0}
    )
    options["campaign_seconds_per_epoch_by_feature_width"] = {
        "v8_dyn_compact_5": 100.0,
        "v8_dyn_balanced_5": 100.0,
        "v8_dyn_context_5": 145.0,
        "v8_dyn_deep_6": 120.0,
    }
    options["campaign_runtime_rates_finalised"] = True
    options["campaign_runtime_calibration"] = calibration
    options["campaign_runtime_calibration_sha256"] = payload_hash(calibration)
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    plan = build_v8_preflight_plan(config_path, CONTRACT)

    assert plan["runtime_calibration_valid"] is True
    assert plan["runtime_envelope_valid"] is True
    assert "runtime_coefficients_pending" not in plan["blockers"]
    assert plan["planned_training_seconds"] == 113_080.0
    assert plan["graduation_seconds"] == 13_800.0


def test_v8_cuda_preflight_measures_both_parents_and_all_dynunet_roots():
    from auto_researcher.runtime.identity import payload_hash

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(_device):
            total = 48 * 1024**3
            return total, total

        @staticmethod
        def get_device_name(_device):
            return "NVIDIA RTX A6000"

    report = run_v8_cuda_preflight(
        TASK_CONFIG,
        CONTRACT,
        torch_module=SimpleNamespace(cuda=FakeCuda()),
        step_runner=lambda _configuration: 2 * 1024**3,
    )
    options = _options()
    selected = options["campaign_portfolio"]["parent_selection"]["selected_parents"]

    assert report["passed"] is True
    assert report["model_calls_performed"] == 0
    assert report["holdout_subjects_evaluated"] == 0
    assert len(report["parents"]) == 2
    assert report["dynunet_root_indexes"] == [0, 1, 2, 3]
    assert len(report["dynunet_roots"]) == 4
    assert _cuda_preflight_valid(
        report,
        payload_hash(report),
        runtime_calibration_sha256=options["campaign_runtime_calibration_sha256"],
        selected_parent_ids={item["experiment_id"] for item in selected},
    )


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


def test_v8_dynunet_parameter_budget_is_registered_by_trusted_runner():
    from auto_researcher.tasks.feta_unet_direct.runner import (
        _architecture_parameter_bounds,
    )

    raw = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    root = FeTAUNetSearchConfiguration.model_validate(
        raw["runtime"]["options"]["campaign_portfolio"]["dynunet_root_configurations"][
            0
        ]
    )
    assert _architecture_parameter_bounds(root) == (
        V8_MINIMUM_TRAINABLE_PARAMETERS,
        V8_MAXIMUM_TRAINABLE_PARAMETERS,
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


def test_v8_controller_replays_exact_44_to_3_envelope():
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
    }
    rung10 = {
        item.evidence.trajectory_identity
        for item in candidates
        if item.evidence.fidelity == 10
    }
    assert len(rung10) == 44
    for fidelity, maximum_dynunet in ((15, 4), (25, 2), (50, 1), (100, 1), (150, 1)):
        assert (
            sum(
                item.evidence.configuration["model_variant"] == "dynunet"
                for item in candidates
                if item.evidence.fidelity == fidelity
            )
            <= maximum_dynunet
        )
    increments = {10: 10, 15: 5, 25: 10, 50: 25, 100: 50, 150: 50}
    measured_work = {
        "structural_basic_unet": 0,
        "dynunet_promotable": 0,
        "v8_dyn_context_5": 0,
    }
    for item in candidates:
        configuration = item.evidence.configuration
        if configuration["model_variant"] == "structural_basic_unet":
            family = "structural_basic_unet"
        elif configuration["feature_width"] == "v8_dyn_context_5":
            family = "v8_dyn_context_5"
        else:
            family = "dynunet_promotable"
        measured_work[family] += increments[item.evidence.fidelity]
    assert sum(measured_work.values()) == sum(V8_A6000_EPOCH_WORK.values())
    assert (
        measured_work["dynunet_promotable"] <= V8_A6000_EPOCH_WORK["dynunet_promotable"]
    )
    assert measured_work["v8_dyn_context_5"] == 10
    local = [item for item in candidates if item.stage == "v8-local-optuna"]
    assert (
        sum(item.evidence.configuration["model_variant"] == "dynunet" for item in local)
        == 3
    )
    assert (
        sum(
            item.evidence.configuration["model_variant"] == "structural_basic_unet"
            for item in local
        )
        == 23
    )
    assert _stage(requests[-1]) == "v8-promote-150"
    assert (
        _options()["campaign_portfolio"]["independent_confirmation_execution"]
        == "l4_sidecar_after_champion_freeze"
    )


def test_v8_controller_recovers_verified_duplicate_without_cherry_picking():
    context = TaskRuntimeContext(task_options=_options())
    first = apply_portfolio_policy(
        _original(),
        run_id="v8-run",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert first is not None
    configuration = _simulated_configurations(first, request_index=1)[0]
    events = (
        _planned(1, first),
        _prepared(1, first, "experiment-v8-original"),
        _verified(
            1,
            "experiment-v8-original",
            configuration,
            0.70,
            SearchType.OPENEVOLVE,
        ),
        _prepared(1, first, "experiment-v8-duplicate"),
        _verified(
            1,
            "experiment-v8-duplicate",
            configuration,
            0.71,
            SearchType.OPENEVOLVE,
        ),
    )

    candidates = _tree_candidates(events, _evidence(events))
    second = apply_portfolio_policy(
        _original(),
        run_id="v8-run",
        cycle=2,
        events=events,
        runtime_context=context,
    )

    assert [item.evidence.experiment_id for item in candidates] == [
        "experiment-v8-original"
    ]
    assert candidates[0].evidence.best_score == 0.70
    assert second is not None
    assert _stage(second) == "v8-structural-child"
    assert second.experiment_budget == 4


def test_v8_controller_skips_only_an_inapplicable_frozen_direct_ablation(
    monkeypatch: pytest.MonkeyPatch,
):
    original_ablation = portfolio_module._v8_controlled_direct_ablation
    unavailable = "replace_stagewise_blocks_with_uniform_blocks"

    def controlled_ablation(design, parents, existing):
        if design == unavailable:
            raise ValueError(f"feta_unet_v8_direct_design_unavailable:{design}")
        return original_ablation(design, parents, existing)

    monkeypatch.setattr(
        portfolio_module,
        "_v8_controlled_direct_ablation",
        controlled_ablation,
    )

    _, requests = _replay_v8_portfolio()
    direct_designs = [
        reference.removeprefix("direct-design:")
        for request in requests
        if _stage(request) == "v8-direct-ablation"
        for reference in request.evidence_references
        if reference.startswith("direct-design:")
    ]

    assert direct_designs == [
        "remove_deep_supervision_from_selected_parent",
        "replace_gated_skip_with_concat",
        "replace_stagewise_residuals_with_uniform_residuals",
    ]
    assert _stage(requests[-1]) == "v8-promote-150"


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
