"""Canonical hashes and identifiers for replay-safe knowledge retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class CanonicalizationError(ValueError):
    """A value cannot be represented without ambiguity as canonical JSON."""


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _mapping_key(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite mapping key")
        return _json_text(value)
    raise CanonicalizationError("unsupported mapping key type")


def canonicalize(value: Any) -> Any:
    """Recursively normalise supported values into unambiguous JSON data."""

    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalizationError("naive datetime")
        return (
            value.astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace(
                "+00:00",
                "Z",
            )
        )
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            text_key = _mapping_key(key)
            if text_key in normalised:
                raise CanonicalizationError("duplicate stringified mapping key")
            normalised[text_key] = canonicalize(item)
        return {key: normalised[key] for key in sorted(normalised)}
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        encoded: dict[str, Any] = {}
        for item in value:
            normalised = canonicalize(item)
            identity = _json_text(normalised)
            if identity in encoded:
                raise CanonicalizationError("duplicate canonical set element")
            encoded[identity] = normalised
        return [encoded[key] for key in sorted(encoded)]
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite float")
        return value
    raise CanonicalizationError("unsupported canonical value type")


def canonical_json(value: Any) -> str:
    return _json_text(canonicalize(value))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def domain_separated_hash(
    value: Any,
    *,
    hash_domain: str,
    hash_version: str,
    schema_version: str,
) -> str:
    return content_hash(
        {
            "hash_domain": hash_domain,
            "hash_version": hash_version,
            "schema_version": schema_version,
            "payload": value,
        }
    )


def bundle_content_hash(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
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
