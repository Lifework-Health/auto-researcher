"""Safe, aggregate-only model context for the iCCA NBS task."""

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.icca_nbs.bindings import ICCABindings
from auto_researcher.tasks.icca_nbs.configuration import (
    ICCA_DEFAULT_RESAMPLING_ITERATIONS,
    ICCA_MINIMUM_RESAMPLING_ITERATIONS,
)
from auto_researcher.tasks.models import DatasetManifest


def _enum_names(enum_type: type) -> list[str]:
    return sorted(
        str(getattr(member, "doc_name", getattr(member, "name", member)))
        for member in enum_type
    )


def create_icca_agent_context(
    contract: ResearchContract,
    manifest: DatasetManifest,
    search_capabilities: dict[SearchType, SearchCapability],
    bindings: ICCABindings,
) -> TaskAgentContext:
    networks = _enum_names(bindings.network_type)
    alignments = _enum_names(bindings.alignment_type)
    available = tuple(
        sorted(
            (
                item
                for item, capability in search_capabilities.items()
                if capability.available and item in contract.allowed_search_types
            ),
            key=lambda item: item.value,
        )
    )
    return TaskAgentContext(
        task_id="icca_nbs",
        task_version="1.0",
        display_name="iCCA Network Based Stratification",
        domain="cancer-subtyping",
        task_description=(
            "Mutation-only network propagation followed by consensus clustering "
            "and aggregate stability/eligibility evaluation."
        ),
        safe_scientific_vocabulary=(
            "network propagation",
            "consensus clustering",
            "alpha",
            "cluster count K",
            "stability objective",
            "eligibility gates",
        ),
        primary_metric_description=(
            "stability_objective is maximised only for configurations satisfying "
            "registered aggregate eligibility gates"
        ),
        scientific_constraint_summary=tuple(
            f"{key}={value}" for key, value in sorted(contract.constraints.items())
        ),
        dataset_summary={
            "dataset_version": manifest.dataset_version,
            "file_count": len(manifest.files),
            "objective_version": manifest.metadata.get(
                "objective_version", "unspecified"
            ),
            "contains_patient_identifiers": False,
        },
        available_search_types=available,
        direct_configuration_schema={
            "network": {"enum": networks},
            "alignment": {"enum": alignments},
            "alpha": {
                "type": "number",
                "minimum": bindings.alpha_bounds[0],
                "maximum": bindings.alpha_bounds[1],
            },
            "K": {
                "type": "integer",
                "minimum": bindings.k_bounds[0],
                "maximum": bindings.k_bounds[1],
            },
            "r": {
                "type": "integer",
                "minimum": ICCA_MINIMUM_RESAMPLING_ITERATIONS,
                "default": ICCA_DEFAULT_RESAMPLING_ITERATIONS,
                "recommended": ICCA_DEFAULT_RESAMPLING_ITERATIONS,
            },
        },
        optuna_space_summary={
            "optimised": {
                "alpha": {
                    "low": bindings.alpha_bounds[0],
                    "high": bindings.alpha_bounds[1],
                },
                "K": {"low": bindings.k_bounds[0], "high": bindings.k_bounds[1]},
            },
            "fixed_required": ["network", "alignment", "r"],
        },
        fixed_scientific_context={
            "networks": networks,
            "alignments": alignments,
            "data_modality": "mutation-only",
        },
        task_limitations=(
            "Only validated references from the configured knowledge bundle may be cited.",
            "Only aggregate evaluator outputs may be used as prior findings.",
            (
                "Consensus stability requires at least "
                f"{ICCA_MINIMUM_RESAMPLING_ITERATIONS} resampling iterations; "
                f"{ICCA_DEFAULT_RESAMPLING_ITERATIONS} is the recommended standard setting."
            ),
        ),
        safety_notes=(
            "Patient identifiers, mutation values, and clinical rows are prohibited.",
        ),
    )
