from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from auto_researcher.agents.models import AgentBudgetPolicy
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.research_intelligence import LiteratureScoutMode
from auto_researcher.tasks.feta_unet_search.configuration import (
    CANDIDATE_CONFIGURATION_FIELDS,
    V9_CONFIGURATION_SCHEMA_VERSION,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.v9_preflight import (
    run_v9_cuda_calibration,
    static_v9_preflight,
)
from auto_researcher.tasks.feta_unet_search.v9_research_intelligence import (
    build_v9_knowledge_library,
    build_v9_literature_brief,
    reviewed_v9_materials,
    v9_director_evidence,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-36h-v9-template.yaml"
EVIDENCE = ROOT / "examples/tasks/feta_unet_search/v9-bound-evidence.json"
CONTRACT = ROOT / "examples/tasks/feta_unet_search/contract-36h-v9.yaml"


def _root(feature_width: str) -> dict:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return next(
        item
        for item in raw["runtime"]["options"]["v9_fixed_roots"]
        if item["feature_width"] == feature_width
    )


@pytest.mark.parametrize(
    ("feature_width", "variant", "family"),
    (
        ("v9_attn_compact_5", "attention_unet", "AttentionUnet"),
        ("v9_attn_balanced_5", "attention_unet", "AttentionUnet"),
        ("v9_unetr_base_16", "unetr", "UNETR"),
        ("v9_swin_tiny_24", "swin_unetr", "SwinUNETR"),
    ),
)
def test_v9_architecture_pilot_profiles_are_explicit_and_valid(
    feature_width: str, variant: str, family: str
):
    candidate = FeTAUNetSearchConfiguration.model_validate(_root(feature_width))

    assert candidate.model_variant == variant
    assert candidate.network_family == family
    assert candidate.maximum_epochs == 15


def test_v9_architecture_profiles_fail_closed_on_cross_family_relabel():
    raw = _root("v9_unetr_base_16")
    raw["model_variant"] = "swin_unetr"

    with pytest.raises(ValueError, match="v9_transformer_architecture_invalid"):
        FeTAUNetSearchConfiguration.model_validate(raw)


def test_v9_primary_source_material_preserves_transfer_limitations():
    materials = reviewed_v9_materials()

    assert len(materials) == 4
    assert all(str(item.source.uri).startswith("https://") for item in materials)
    assert all(item.findings[0].limitations for item in materials)
    assert any("fetal MRI" in item.findings[0].limitations[0] for item in materials)


def test_v9_knowledge_and_literature_evidence_are_deterministic_and_advisory():
    first_library = build_v9_knowledge_library()
    second_library = build_v9_knowledge_library()
    first_brief = build_v9_literature_brief(mode=LiteratureScoutMode.LIVE)
    second_brief = build_v9_literature_brief(mode=LiteratureScoutMode.LIVE)

    assert first_library == second_library
    assert first_brief == second_brief
    assert len(first_library.cards) == 4
    assert len(first_brief.evidence) == 4
    evidence = v9_director_evidence()
    assert {item.evidence_type for item in evidence} == {
        "KNOWLEDGE_CARD_LIBRARY",
        "LITERATURE",
    }
    assert all(
        item.safe_payload.get("experiment_authority_exercised") is not True
        for item in evidence
    )


def test_v9_static_preflight_is_evidence_bound_and_launch_blocked():
    result = static_v9_preflight(config_path=CONFIG, evidence_path=EVIDENCE)

    assert result["configuration_schema_version"] == V9_CONFIGURATION_SCHEMA_VERSION
    assert result["root_count"] == 10
    assert result["root_model_variant_counts"] == {
        "dynunet": 6,
        "attention_unet": 2,
        "unetr": 1,
        "swin_unetr": 1,
    }
    assert result["research_director_valid_decision_budget"] == 16
    assert result["model_calls_performed"] == 0
    assert result["holdout_subjects_evaluated"] == 0
    assert result["launch_ready"] is False
    assert len(result["launch_blockers"]) == 4
    assert result["cuda_calibration_sha256"] == (
        "fc1e9dbe57e423e308674d81d32e72a9ecafcb5dceee0816f6654ab7ff384d73"
    )
    bound = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    counts = {
        item["feature_width"]: item["trainable_parameters"]
        for item in bound["v9_architecture_pilots"]["profiles"]
    }
    assert counts == {
        "v9_attn_compact_5": 36912564,
        "v9_attn_balanced_5": 53150880,
        "v9_unetr_base_16": 121350472,
        "v9_swin_tiny_24": 15703154,
    }


def test_v9_openevolve_runtime_stays_within_enforced_safety_caps():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    runtime = raw["openevolve_development_mutation"]

    assert runtime["maximum_model_calls"] == 100
    assert runtime["maximum_total_cost_usd"] == 50.0


def test_v9_static_preflight_rejects_evidence_tampering(tmp_path):
    raw = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    raw["v8_campaign"]["champion"]["best_score"] = 0.84
    altered = tmp_path / "v9-evidence.json"
    altered.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="prior_evidence_invalid"):
        static_v9_preflight(config_path=CONFIG, evidence_path=altered)


def test_valid_decision_budget_is_separate_from_raw_director_call_cap():
    policy = AgentBudgetPolicy(
        maximum_research_director_calls_total=64,
        maximum_research_director_valid_decisions_total=16,
        maximum_total_model_calls=768,
    )

    assert policy.maximum_research_director_calls_total == 64
    assert policy.maximum_research_director_valid_decisions_total == 16
    assert policy.maximum_total_model_calls == 768


def test_v9_contract_and_agent_context_keep_evolution_inside_dynunet():
    contract = ResearchContract.model_validate(
        yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    )
    task = FeTAUNetSearchTask()
    task.validate_contract(contract)
    context = task.create_agent_context(
        contract,
        TaskRuntimeContext(task_options={"openevolve_fidelity": 15}),
        {},
    )

    assert context.direct_configuration_schema["model_variant"] == [
        "dynunet",
        "attention_unet",
        "unetr",
        "swin_unetr",
    ]
    assert context.openevolve_space_summary["mutable_policy"]["model_variant"] == [
        "dynunet"
    ]
    assert context.openevolve_space_summary["mutable_policy"][
        "architecture_budget"
    ] == ["dynunet-15m-150m-v1"]
    assert 30 in context.direct_configuration_schema["maximum_epochs"]


def test_v9_branch_local_optuna_fixes_attention_architecture():
    contract = ResearchContract.model_validate(
        yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    )
    configuration = FeTAUNetSearchConfiguration.model_validate(
        _root("v9_attn_compact_5")
    ).model_dump(mode="json")
    tuned = {"learning_rate", "weight_decay", "dropout", "dice_weight"}
    request = SearchRequest(
        request_id="v9-attention-local",
        hypothesis_id="v9-hypothesis",
        search_type=SearchType.OPTUNA,
        target="mean_subject_macro_dice",
        search_space={
            "fixed": {
                key: value
                for key, value in configuration.items()
                if key in CANDIDATE_CONFIGURATION_FIELDS and key not in tuned
            },
            "parameters": {name: {} for name in tuned},
        },
        experiment_budget=4,
        rationale="Tune training policy while freezing the attention architecture.",
    )

    study = FeTAUNetSearchTask().create_optuna_study_spec(contract, request)

    assert study.fixed_configuration["model_variant"] == "attention_unet"
    assert study.fixed_configuration["feature_width"] == "v9_attn_compact_5"
    assert study.fixed_configuration["features"] == [40, 80, 160, 320, 640]
    assert {item.name for item in study.parameters} == {
        "learning_rate",
        "weight_decay",
        "dropout",
        "dice_weight",
    }


def test_v9_30_epoch_fidelity_is_valid_but_not_a_default_root():
    promoted = FeTAUNetSearchConfiguration.model_validate(
        {**_root("v9_attn_compact_5"), "maximum_epochs": 30}
    )

    assert promoted.maximum_epochs == 30
    assert all(
        item["maximum_epochs"] == 15
        for item in yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["runtime"][
            "options"
        ]["v9_fixed_roots"]
    )


def test_v9_cuda_calibration_covers_each_new_architecture_without_holdout():
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return (47 * 1024**3, 48 * 1024**3)

        @staticmethod
        def get_device_name(index):
            assert index == 0
            return "NVIDIA RTX A6000"

    class FakeTorch:
        cuda = FakeCuda()

    result = run_v9_cuda_calibration(
        config_path=CONFIG,
        torch_module=FakeTorch(),
        step_runner=lambda configuration: {
            "peak_gpu_memory_bytes": 8 * 1024**3,
            "amp_step_seconds": 1.0
            + (0.1 if configuration.model_variant == "swin_unetr" else 0.0),
        },
    )

    assert result["passed"] is True
    assert result["holdout_subjects_evaluated"] == 0
    assert result["model_calls_performed"] == 0
    assert {item["model_variant"] for item in result["pilots"]} == {
        "attention_unet",
        "unetr",
        "swin_unetr",
    }
    assert len(result["pilots"]) == 4
