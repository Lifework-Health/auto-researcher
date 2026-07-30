"""Safe model-facing context for the synthetic reference task."""

from auto_researcher.agents.models import TaskAgentContext
from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.search.protocols import SearchCapability
from auto_researcher.tasks.models import DatasetManifest


def create_synthetic_agent_context(
    contract: ResearchContract,
    manifest: DatasetManifest,
    search_capabilities: dict[SearchType, SearchCapability],
) -> TaskAgentContext:
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
        task_id="synthetic",
        task_version="1.0",
        display_name="Deterministic synthetic landscape",
        domain="synthetic",
        task_description="A deterministic offline objective used to test research orchestration.",
        safe_scientific_vocabulary=(
            "model family",
            "complexity",
            "learning rate",
            "objective score",
            "stability",
            "runtime",
        ),
        primary_metric_description=(
            "objective_score is a deterministic scalar maximised subject to "
            "stability and runtime constraints"
        ),
        scientific_constraint_summary=tuple(
            f"{key}={value}" for key, value in sorted(contract.constraints.items())
        ),
        dataset_summary={
            "dataset_version": manifest.dataset_version,
            "generator": "deterministic",
            "contains_patient_data": False,
        },
        available_search_types=available,
        direct_configuration_schema={
            "model_family": {"enum": ["linear", "tree", "neural"]},
            "complexity": {"type": "integer", "minimum": 1, "maximum": 10},
            "learning_rate": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 1,
            },
        },
        optuna_space_summary={
            "model_family": {"choices": ["linear", "tree", "neural"]},
            "complexity": {"low": 1, "high": 10},
            "learning_rate": {"low": 0.001, "high": 1.0, "log": True},
        },
        fixed_scientific_context={"objective_kind": "offline_reference"},
        task_limitations=("Synthetic results are not real scientific evidence.",),
        safety_notes=("No external evidence grounding is available.",),
    )
