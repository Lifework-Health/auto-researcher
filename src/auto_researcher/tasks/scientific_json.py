"""Task-configurable finite JSON normalisation for scientific values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

SCIENTIFIC_JSON_ENCODING_VERSION = "scientific-json-v1"
UNAVAILABLE_NON_FINITE_ENCODING = "null_for_unavailable_non_finite_v1"


@dataclass(frozen=True)
class ScientificJsonPolicy:
    """Exact schema paths where an unavailable NaN is scientifically legitimate."""

    permitted_nan_paths: frozenset[str] = frozenset()
    unavailable_encoding: str = UNAVAILABLE_NON_FINITE_ENCODING


@dataclass(frozen=True)
class ScientificJsonNormalisationResult:
    """A JSON-safe value plus an audit of every non-finite input encountered."""

    value: Any
    unavailable_paths: tuple[str, ...]
    rejected_paths: tuple[str, ...]
    category_counts: dict[str, int]

    @property
    def valid(self) -> bool:
        return not self.rejected_paths


class ScientificJsonNormalisationError(ValueError):
    """Raised when a scientific result contains a forbidden non-finite value."""

    def __init__(
        self,
        result: ScientificJsonNormalisationResult,
        *,
        reason_code: str = "unexpected_non_finite_scientific_value",
    ) -> None:
        self.result = result
        self.reason_code = reason_code
        super().__init__(reason_code)


def _child_path(parent: str, key: str) -> str:
    return f"{parent}.{key}" if parent else key


def _index_path(parent: str, index: int) -> str:
    return f"{parent}[{index}]" if parent else f"[{index}]"


def normalise_scientific_json(
    value: Any,
    *,
    policy: ScientificJsonPolicy | None = None,
    root_path: str = "",
) -> ScientificJsonNormalisationResult:
    """Recursively convert accepted scientific containers to strict JSON values.

    Finite numbers are retained. Only NaN values at exact, task-declared paths are
    encoded as ``null``. All other NaN values and every infinity are rejected.
    Rejected values are replaced with ``null`` only in the diagnostic return value;
    callers must check ``valid`` before using it.
    """

    active_policy = policy or ScientificJsonPolicy()
    unavailable: list[str] = []
    rejected: list[str] = []
    counts = {
        "finite_numbers": 0,
        "unavailable_nan": 0,
        "rejected_nan": 0,
        "rejected_positive_infinity": 0,
        "rejected_negative_infinity": 0,
    }

    def visit(item: Any, path: str) -> Any:
        if isinstance(item, BaseModel):
            return visit(item.model_dump(mode="python"), path)
        if is_dataclass(item) and not isinstance(item, type):
            return visit(asdict(item), path)
        if isinstance(item, Enum):
            return visit(item.value, path)
        if isinstance(item, Path):
            return item.name
        if isinstance(item, Mapping):
            return {
                str(key): visit(child, _child_path(path, str(key)))
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(child, _index_path(path, index))
                for index, child in enumerate(item)
            ]
        if isinstance(item, (set, frozenset)):
            converted = [
                visit(child, _index_path(path, index))
                for index, child in enumerate(sorted(item, key=repr))
            ]
            return sorted(converted, key=repr)
        if item is None or isinstance(item, (str, bool)):
            return item
        if isinstance(item, int):
            counts["finite_numbers"] += 1
            return item
        if isinstance(item, float):
            if math.isfinite(item):
                counts["finite_numbers"] += 1
                return item
            if math.isnan(item):
                if path in active_policy.permitted_nan_paths:
                    unavailable.append(path)
                    counts["unavailable_nan"] += 1
                else:
                    rejected.append(path)
                    counts["rejected_nan"] += 1
                return None
            rejected.append(path)
            category = (
                "rejected_positive_infinity"
                if item > 0
                else "rejected_negative_infinity"
            )
            counts[category] += 1
            return None
        # NumPy arrays/scalars and already-supported pandas-like containers stay
        # optional dependencies behind their public conversion protocols.
        if hasattr(item, "tolist") and callable(item.tolist):
            return visit(item.tolist(), path)
        if hasattr(item, "item") and callable(item.item):
            return visit(item.item(), path)
        raise TypeError(f"unsupported scientific result type: {type(item).__name__}")

    normalised = visit(value, root_path)
    return ScientificJsonNormalisationResult(
        value=normalised,
        unavailable_paths=tuple(sorted(unavailable)),
        rejected_paths=tuple(sorted(rejected)),
        category_counts=counts,
    )


def require_valid_scientific_json(
    result: ScientificJsonNormalisationResult,
    *,
    reason_code: str = "unexpected_non_finite_scientific_value",
) -> Any:
    """Return a normalised value or fail with safe schema-path diagnostics."""

    if not result.valid:
        raise ScientificJsonNormalisationError(result, reason_code=reason_code)
    return result.value
