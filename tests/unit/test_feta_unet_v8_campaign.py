from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext

ROOT = Path(__file__).parents[2]
TASK_CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-32h-v8-template.yaml"
CONTRACT = ROOT / "examples/tasks/feta_unet_search/contract-32h-v8.yaml"


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
        "v7_parent_selection_pending",
        "v7_parent_manifest_pending",
        "runtime_coefficients_pending",
        "v8_portfolio_controller_pending",
        "real_cuda_preflight_pending",
        "launch_gate_not_passed",
    ]
    assert plan["model_calls_performed"] == 0


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
