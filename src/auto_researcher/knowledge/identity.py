"""Canonical hashes and identifiers for replay-safe knowledge retrieval."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def bundle_content_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = dict(value)
    payload.pop("bundle_hash", None)
    return content_hash(payload)


def stable_identifier(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def query_plan_hash(plan: BaseModel) -> str:
    return content_hash(plan)


def retrieval_id(
    *,
    run_id: str,
    cycle: int,
    task_id: str,
    task_version: str,
    contract_id: str,
    provider_id: str,
    provider_version: str,
    graph_alias: str,
    schema_version: str,
    content_version: str,
    query_plan_version: str,
    plan_hash: str,
) -> str:
    return stable_identifier(
        "knowledge-retrieval",
        run_id,
        str(cycle),
        task_id,
        task_version,
        contract_id,
        provider_id,
        provider_version,
        graph_alias,
        schema_version,
        content_version,
        query_plan_version,
        plan_hash,
    )


def entity_id(curie: str, entity_type: str) -> str:
    return stable_identifier("entity", entity_type, curie)


def assertion_id(
    subject_curie: str,
    predicate: str,
    object_curie: str,
    source_references: tuple[str, ...],
) -> str:
    return stable_identifier(
        "assertion",
        subject_curie,
        predicate,
        object_curie,
        *sorted(source_references),
    )


def reference_id(assertion_identifier: str, bundle_id: str) -> str:
    return stable_identifier("knowledge-ref", assertion_identifier, bundle_id)
