"""Attested metadata-only model exposure for sensitive evaluator datasets."""

from __future__ import annotations

import re
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from auto_researcher.search.openevolve.identity import openevolve_hash
from auto_researcher.search.openevolve.models import EvolvableComponentSpec
from auto_researcher.search.openevolve.upstream_models import MutationConstraints

UnderlyingDatasetClass: TypeAlias = Literal[
    "synthetic",
    "public_benchmark",
    "aura",
    "genuine_icca",
    "mri",
    "patient_data",
]
ModelExposureClass: TypeAlias = Literal["metadata_only"]

UNDERLYING_DATASET_CLASSES: Final[frozenset[str]] = frozenset(
    {"synthetic", "public_benchmark", "aura", "genuine_icca", "mri", "patient_data"}
)
MODEL_EXPOSURE_CLASSES: Final[frozenset[str]] = frozenset({"metadata_only"})
METADATA_ONLY_BOUNDARY_VERSION: Final = "metadata-only-model-boundary-v1"


class BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetadataOnlyMutationBoundary(BoundaryModel):
    """Task-owned declaration; it grants no approval by itself."""

    boundary_version: Literal["metadata-only-model-boundary-v1"] = (
        METADATA_ONLY_BOUNDARY_VERSION
    )
    underlying_dataset_class: UnderlyingDatasetClass
    exposure_class: Literal["metadata_only"] = "metadata_only"
    underlying_data_access: Literal[False] = False
    mri_access: Literal[False] = False
    patient_data_access: Literal[False] = False
    filesystem_access: Literal[False] = False
    network_access: Literal[False] = False
    evaluator_runtime_context_access: Literal[False] = False


class MetadataOnlyParentSource(BoundaryModel):
    id: str = Field(min_length=1)
    authoritative_candidate_id: str | None = Field(default=None, min_length=1)
    code: str = Field(min_length=1)
    generation: int = Field(ge=0)


class MetadataOnlyMutationRequest(BoundaryModel):
    protocol: Literal["upstream-adapter-mutation-request-v2"]
    parent: MetadataOnlyParentSource
    mutable_file: str = Field(pattern=r"^[A-Za-z0-9_.-]+\.py$")
    interface_contract: str = Field(min_length=1)
    maximum_source_bytes: int = Field(gt=0, le=1_000_000)
    mutation_constraints: MutationConstraints
    native_evolution_prompt: str | None = Field(
        default=None,
        min_length=1,
        max_length=500_000,
    )


_PROHIBITED_DYNAMIC_KEY_PARTS: Final = frozenset(
    {
        "case_record",
        "checkpoint",
        "data_dir",
        "dataset_path",
        "error_message",
        "exception",
        "file_path",
        "holdout",
        "mask",
        "mri",
        "patient",
        "prediction",
        "scan",
        "subject",
        "traceback",
        "voxel",
    }
)
_PROHIBITED_DYNAMIC_TEXT = re.compile(
    r"(?ix)("
    r"(?:^|[^a-z0-9_])(?:mri|voxel|mask|patient|subject|checkpoint|holdout|prediction)(?:s)?(?:$|[^a-z0-9_])"
    r"|(?:^|[^a-z0-9_])case[ _-]?(?:id|record|row|\d)"
    r"|data[_-]?dir"
    r"|sub-[a-z0-9_-]+"
    r"|/(?:users|home|private|mnt|data|protected)/"
    r"|[a-z]:\\\\"
    r"|(?:file|s3|gs)://"
    r")"
)


def mutation_constraints_for_component(
    component: EvolvableComponentSpec,
) -> MutationConstraints:
    return MutationConstraints(
        mutable_file=component.mutable_file,
        allowed_files=component.allowed_files,
        entry_point=component.entry_point,
        immutable_interface_contract=component.immutable_interface_contract,
        maximum_source_bytes=component.maximum_source_bytes,
        allowed_imports=component.allowed_imports,
        allowed_dependencies=component.allowed_dependencies,
        allowed_imports_display=(
            ", ".join(component.allowed_imports)
            if component.allowed_imports
            else "NONE"
        ),
        allowed_dependencies_display=(
            ", ".join(component.allowed_dependencies)
            if component.allowed_dependencies
            else "NONE"
        ),
        parameter_schema=component.parameter_schema,
        output_schema=component.output_schema,
    )


def metadata_only_model_exposure_identity(
    component: EvolvableComponentSpec,
) -> str:
    """Bind exactly the schema/context transported in the v2 model request."""

    if component.task_mutation_context:
        embedded = component.parameter_schema.get("mutation_context")
        if embedded != component.task_mutation_context:
            raise ValueError("metadata_only_mutation_context_unbound")
    constraints = mutation_constraints_for_component(component)
    assert_no_prohibited_dynamic_content(constraints.model_dump(mode="json"))
    return openevolve_hash(
        "openevolve-metadata-only-model-exposure-v1",
        constraints,
    )


def assert_no_prohibited_dynamic_content(value: object) -> None:
    """Reject untrusted request/response content outside the attested schema."""

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                normalised = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
                if any(part in normalised for part in _PROHIBITED_DYNAMIC_KEY_PARTS):
                    raise ValueError("metadata_only_prohibited_content")
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str) and _PROHIBITED_DYNAMIC_TEXT.search(item):
            raise ValueError("metadata_only_prohibited_content")

    visit(value)


def validate_metadata_only_request(
    request: dict,
    *,
    expected_exposure_identity: str,
) -> MetadataOnlyMutationRequest:
    parsed = MetadataOnlyMutationRequest.model_validate(request)
    actual_identity = openevolve_hash(
        "openevolve-metadata-only-model-exposure-v1",
        parsed.mutation_constraints,
    )
    if actual_identity != expected_exposure_identity:
        raise ValueError("metadata_only_model_exposure_identity_mismatch")
    assert_no_prohibited_dynamic_content(parsed.parent.model_dump(mode="json"))
    return parsed


__all__ = [
    "METADATA_ONLY_BOUNDARY_VERSION",
    "MODEL_EXPOSURE_CLASSES",
    "UNDERLYING_DATASET_CLASSES",
    "MetadataOnlyMutationBoundary",
    "MetadataOnlyMutationRequest",
    "ModelExposureClass",
    "UnderlyingDatasetClass",
    "assert_no_prohibited_dynamic_content",
    "metadata_only_model_exposure_identity",
    "mutation_constraints_for_component",
    "validate_metadata_only_request",
]
