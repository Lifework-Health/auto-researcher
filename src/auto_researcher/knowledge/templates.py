"""Fixed, versioned Cypher template registry and static safety lint."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auto_researcher.knowledge.models import KnowledgeTemplateRequest

FORBIDDEN_CYPHER = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b",
    re.IGNORECASE,
)
CALL_KEYWORD = re.compile(r"\bCALL\b", re.IGNORECASE)
CALL_PROCEDURE = re.compile(
    r"\bCALL\s+([A-Za-z][A-Za-z0-9_.]*)\s*\(",
    re.IGNORECASE,
)
ALLOWED_READ_PROCEDURES = frozenset({"db.labels", "db.relationshiptypes"})


class EntityLookupParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    curie: str = Field(min_length=3)
    limit: int = Field(ge=1, le=100)


class SchemaPreflightParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    limit: int = Field(default=1, ge=1, le=1)


class NetworkCatalogParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    codenames: tuple[str, ...] = Field(min_length=1, max_length=20)
    limit: int = Field(ge=1, le=100)


class GenePathwayParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    gene_curies: tuple[str, ...] = Field(min_length=1, max_length=100)
    limit: int = Field(ge=1, le=500)


class DiseaseContextParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    disease_curies: tuple[str, ...] = Field(min_length=1, max_length=20)
    limit: int = Field(ge=1, le=500)


class ImmuneBridgeParameters(DiseaseContextParameters):
    pass


@dataclass(frozen=True)
class KnowledgeQueryTemplate:
    template_id: str
    version: str
    cypher_path: Path
    parameter_model: type[BaseModel]
    output_schema_version: str
    allowed_labels: frozenset[str]
    allowed_relationships: frozenset[str]
    maximum_hops: int
    maximum_rows: int
    task_compatibility: frozenset[str]
    schema_compatibility: frozenset[str]

    @property
    def cypher(self) -> str:
        return self.cypher_path.read_text(encoding="utf-8")

    @property
    def cypher_sha256(self) -> str:
        return hashlib.sha256(self.cypher.encode()).hexdigest()


def lint_read_only_cypher(cypher: str) -> None:
    match = FORBIDDEN_CYPHER.search(cypher)
    if match:
        raise ValueError(f"forbidden Cypher clause: {match.group(1).upper()}")
    call_keywords = CALL_KEYWORD.findall(cypher)
    call_procedures = CALL_PROCEDURE.findall(cypher)
    if len(call_keywords) != len(call_procedures):
        raise ValueError("forbidden Cypher clause: CALL")
    for procedure in call_procedures:
        if procedure.casefold() not in ALLOWED_READ_PROCEDURES:
            raise ValueError(f"forbidden Cypher procedure: {procedure}")
    upper = cypher.upper()
    if "ORDER BY" not in upper or "LIMIT $LIMIT" not in upper:
        raise ValueError(
            "knowledge query requires explicit ORDER BY before LIMIT $limit"
        )
    if upper.index("ORDER BY") > upper.index("LIMIT $LIMIT"):
        raise ValueError("ORDER BY must precede LIMIT")


class KnowledgeQueryTemplateRegistry:
    def __init__(self) -> None:
        self._templates: dict[tuple[str, str], KnowledgeQueryTemplate] = {}

    def register(self, template: KnowledgeQueryTemplate) -> None:
        key = (template.template_id, template.version)
        if key in self._templates:
            raise ValueError(f"knowledge query template {key!r} already registered")
        if not re.fullmatch(r"[a-z0-9_.-]+", template.template_id):
            raise ValueError("knowledge template ID must be a safe identifier")
        if not re.fullmatch(r"\d+\.\d+\.\d+", template.version):
            raise ValueError("knowledge template version must be semantic")
        if not template.output_schema_version:
            raise ValueError("knowledge template requires an output schema")
        if template.maximum_rows < 1 or not 0 <= template.maximum_hops <= 6:
            raise ValueError("knowledge template limits are invalid")
        if not template.task_compatibility or not template.schema_compatibility:
            raise ValueError("knowledge template compatibility cannot be empty")
        lint_read_only_cypher(template.cypher)
        self._templates[key] = template

    def get(self, template_id: str, version: str) -> KnowledgeQueryTemplate:
        try:
            return self._templates[(template_id, version)]
        except KeyError as exc:
            raise KeyError(
                f"unknown knowledge query template {template_id}@{version}"
            ) from exc

    def validate_request(
        self,
        request: KnowledgeTemplateRequest,
        *,
        task_id: str,
        schema_version: str,
    ) -> tuple[KnowledgeQueryTemplate, dict[str, Any]]:
        template = self.get(request.template_id, request.template_version)
        if task_id not in template.task_compatibility:
            raise ValueError("query template is not compatible with the active task")
        if schema_version not in template.schema_compatibility:
            raise ValueError("query template is not compatible with the graph schema")
        if request.maximum_records > template.maximum_rows:
            raise ValueError("requested row limit exceeds registered template limit")
        try:
            parameters = template.parameter_model.model_validate(
                dict(request.parameters)
            ).model_dump(mode="python")
        except ValidationError as exc:
            raise ValueError("invalid knowledge query parameters") from exc
        if int(parameters["limit"]) > request.maximum_records:
            raise ValueError("query parameter limit exceeds request limit")
        return template, parameters

    def list_templates(self) -> tuple[KnowledgeQueryTemplate, ...]:
        return tuple(self._templates[key] for key in sorted(self._templates))


def default_template_registry() -> KnowledgeQueryTemplateRegistry:
    root = Path(__file__).parent / "queries"
    registry = KnowledgeQueryTemplateRegistry()
    definitions = (
        (
            "generic.schema_preflight",
            root / "generic/schema_preflight_v1.cypher",
            SchemaPreflightParameters,
            "knowledge-schema-preflight-row-v1",
            set(),
            set(),
            0,
            {"synthetic", "icca_nbs"},
            1,
        ),
        (
            "generic.entity_lookup",
            root / "generic/entity_lookup_v1.cypher",
            EntityLookupParameters,
            "knowledge-assertion-row-v1",
            {"Gene", "Pathway", "Disease", "Signature", "Network"},
            set(),
            0,
            {"synthetic", "icca_nbs"},
            100,
        ),
        (
            "icca_nbs.network_catalog",
            root / "icca_nbs/network_catalog_v1.cypher",
            NetworkCatalogParameters,
            "knowledge-assertion-row-v1",
            {"Network"},
            set(),
            0,
            {"icca_nbs"},
            100,
        ),
        (
            "icca_nbs.gene_signature_pathway",
            root / "icca_nbs/gene_signature_pathway_v1.cypher",
            GenePathwayParameters,
            "knowledge-assertion-row-v1",
            {"Gene", "Signature", "Pathway"},
            {"INCLUDES", "PARTICIPATES_IN", "PART_OF"},
            2,
            {"icca_nbs"},
            500,
        ),
        (
            "icca_nbs.disease_context",
            root / "icca_nbs/disease_context_v1.cypher",
            DiseaseContextParameters,
            "knowledge-assertion-row-v1",
            {"Disease", "Subtype", "Pathway", "ClinicalCovariate"},
            {"IS_A", "SUBTYPE_OF", "DEFINED_BY", "IMPLICATED_IN", "PROGNOSTIC_IN"},
            1,
            {"icca_nbs"},
            500,
        ),
        (
            "icca_nbs.immune_bridge",
            root / "icca_nbs/immune_bridge_v1.cypher",
            ImmuneBridgeParameters,
            "knowledge-assertion-row-v1",
            {"Disease", "Subtype", "Signature", "CellState"},
            {"HAS_IMMUNE_PHENOTYPE"},
            1,
            {"icca_nbs"},
            500,
        ),
    )
    for (
        template_id,
        path,
        params,
        output_schema,
        labels,
        rels,
        hops,
        tasks,
        rows,
    ) in definitions:
        registry.register(
            KnowledgeQueryTemplate(
                template_id=template_id,
                version="1.0.0",
                cypher_path=path,
                parameter_model=params,
                output_schema_version=output_schema,
                allowed_labels=frozenset(labels),
                allowed_relationships=frozenset(rels),
                maximum_hops=hops,
                maximum_rows=rows,
                task_compatibility=frozenset(tasks),
                schema_compatibility=frozenset(
                    {"knowledge-graph-auto-v0.1", "synthetic-v1"}
                ),
            )
        )
    return registry
