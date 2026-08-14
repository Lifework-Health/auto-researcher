from __future__ import annotations

from pathlib import Path

import pytest

from auto_researcher.search.openevolve.capabilities import (
    CAPABILITY_MANIFEST_VERSION,
    CapabilityClassification,
    verify_capability_manifest,
)
from auto_researcher.search.openevolve.upstream import DISABLED_UPSTREAM_FEATURES
from auto_researcher.search.openevolve.upstream_models import (
    UPSTREAM_COMMIT,
    UPSTREAM_PACKAGE_VERSION,
)

ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/capabilities/openevolve-0.3.2-capability-manifest.yaml"

REQUIRED_CAPABILITIES = {
    "adaptive_feature_scaling",
    "archive",
    "arbitrary_network_access",
    "arbitrary_package_installation",
    "best_program_tracking",
    "cascade_evaluation",
    "checkpoint",
    "diff_rewrites",
    "direct_provider_credential_access",
    "distributed_cross_node_evaluation",
    "diverse_program_inspiration",
    "double_selection",
    "durable_rng_state",
    "edit_distance_diversity",
    "embedding_novelty",
    "evaluation_feedback_to_mutation",
    "evaluation_retries_timeouts",
    "evaluator_artifacts",
    "evaluator_metrics",
    "evaluator_resource_limits",
    "evaluator_subprocess_semantics",
    "evolution_trace",
    "evolutionary_telemetry",
    "evolve_block",
    "exploration_exploitation",
    "feature_dimensions",
    "full_rewrites",
    "generation_history",
    "islands",
    "llm_ensemble",
    "map_elites_quality_diversity",
    "migration",
    "model_selection",
    "multiple_metrics",
    "native_evolution_controller",
    "native_generation_lifecycle",
    "native_stopping",
    "outer_budget_ceiling",
    "parallel_candidate_evaluation",
    "parent_selection",
    "pareto_multi_objective",
    "population",
    "program_database",
    "provider_network_access",
    "resume",
    "seeded_reproducibility",
    "semantic_scientific_dedup",
    "scientific_evaluator_verifier_boundary",
    "source_novelty_detection",
    "stochastic_prompt_templates",
    "task_owned_candidate_normalisation",
    "top_program_inspiration",
    "unrestricted_host_filesystem",
}

RUNTIME_PROBES = {
    "test_a3_semantic_dedup_and_feedback_regression",
    "test_evaluator_adapter_returns_verified_safe_metrics",
    "test_later_generation_prompt_contains_safe_evaluation_feedback",
    "test_map_elites_and_native_feature_dimensions_are_active",
    "test_native_and_outer_budget_ceiling_use_minimum",
    "test_native_checkpoint_resume_preserves_population_archive_and_reuse",
    "test_native_controller_owns_population_archive_and_selection",
    "test_native_full_rewrite_and_diff_modes",
    "test_native_islands_migrate_without_changing_scientific_identity",
    "test_native_template_stochasticity_is_seeded_and_traceable",
    "test_native_weighted_model_ensemble_uses_approved_adapters",
    "test_parallel_evaluations_use_three_simulated_gpus",
    "test_safe_embedding_adapter_never_receives_protected_context",
    "test_task_owned_normalizer_uses_hardened_component_projection",
    "test_task_owned_scientific_evaluator_invokes_verifier_and_evidence_sink",
}

ADAPTER_CONTRACTS = {
    "ApprovedModel",
    "AutoResearcherEvaluatorAdapter",
    "DurableOpenEvolveModelBridge",
    "HardenedDockerExecutor",
    "NativeEvolutionLimits",
    "ResourceBrokerParallelController",
    "SafeEmbeddingProvider",
    "SafeEvolutionFeedback",
    "TaskOwnedCandidateNormalizer",
    "TaskOwnedScientificEvaluator",
}


pytestmark = pytest.mark.upstream_openevolve


def test_exact_pin_capability_manifest_is_executable_and_complete() -> None:
    pytest.importorskip("openevolve")
    manifest = verify_capability_manifest(MANIFEST, repository_root=ROOT)

    assert manifest.manifest_version == CAPABILITY_MANIFEST_VERSION
    assert manifest.upstream_version == UPSTREAM_PACKAGE_VERSION == "0.3.2"
    assert manifest.upstream_commit == UPSTREAM_COMMIT
    assert REQUIRED_CAPABILITIES == {item.capability for item in manifest.capabilities}
    assert sum(manifest.counts().values()) == len(REQUIRED_CAPABILITIES)


def test_manifest_preservation_claims_name_real_probes_and_contracts() -> None:
    manifest = verify_capability_manifest(MANIFEST, repository_root=ROOT)

    for capability in manifest.capabilities:
        if capability.classification in {
            CapabilityClassification.PRESERVED_NATIVE,
            CapabilityClassification.PRESERVED_VIA_ADAPTER,
        }:
            assert capability.probe in RUNTIME_PROBES
        if capability.classification is CapabilityClassification.PRESERVED_VIA_ADAPTER:
            assert capability.adapter_contract in ADAPTER_CONTRACTS
        if capability.classification is CapabilityClassification.CURRENTLY_DISABLED:
            assert capability.justification


def test_preserved_capability_cannot_be_silently_rethinned() -> None:
    manifest = verify_capability_manifest(MANIFEST, repository_root=ROOT)
    disabled = {
        item.capability
        for item in manifest.capabilities
        if item.classification is CapabilityClassification.CURRENTLY_DISABLED
    }
    preserved = {
        item.capability
        for item in manifest.capabilities
        if item.classification
        in {
            CapabilityClassification.PRESERVED_NATIVE,
            CapabilityClassification.PRESERVED_VIA_ADAPTER,
        }
    }

    assert disabled == set(DISABLED_UPSTREAM_FEATURES)
    assert preserved.isdisjoint(DISABLED_UPSTREAM_FEATURES)
