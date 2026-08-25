from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from auto_researcher.agents.models import AgentBudgetPolicy
from auto_researcher.research_intelligence import LiteratureScoutMode
from auto_researcher.tasks.feta_unet_search.configuration import (
    V9_CONFIGURATION_SCHEMA_VERSION,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.v9_preflight import static_v9_preflight
from auto_researcher.tasks.feta_unet_search.v9_research_intelligence import (
    build_v9_knowledge_library,
    build_v9_literature_brief,
    reviewed_v9_materials,
    v9_director_evidence,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "examples/tasks/feta_unet_search/campaign-36h-v9-template.yaml"
EVIDENCE = ROOT / "examples/tasks/feta_unet_search/v9-bound-evidence.json"


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
    assert len(result["launch_blockers"]) == 5
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
