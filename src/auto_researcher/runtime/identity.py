"""Canonical hashes for immutable execution and scientific identities."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python


def canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        # Preserve typed sets until this function can order them deterministically.
        # Pydantic's JSON mode converts a frozenset to a list using process-specific
        # hash iteration order before canonicalisation can see the collection type.
        return canonical_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = list(value)
        if items and all(
            isinstance(item, Enum) and type(item) is type(items[0]) for item in items
        ):
            declaration_order = {
                member: index for index, member in enumerate(type(items[0]))
            }
            return [
                canonical_value(item)
                for item in sorted(items, key=declaration_order.__getitem__)
            ]
        converted = [canonical_value(item) for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return to_jsonable_python(value)


def payload_hash(value: Any) -> str:
    payload = json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
