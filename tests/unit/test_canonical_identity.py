from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
import math
from pathlib import Path

import pytest

from auto_researcher.knowledge.identity import (
    CanonicalizationError,
    canonical_json,
    canonicalize,
    content_hash,
    domain_separated_hash,
)


class ExampleEnum(StrEnum):
    VALUE = "stable-value"


@dataclass(frozen=True)
class ExampleDataclass:
    label: str
    values: frozenset[int]


@dataclass(frozen=True, eq=False)
class AmbiguousSetValue:
    label: str


def test_mapping_and_unordered_collection_order_is_canonical():
    first = {
        "mapping": {"z": 1, "a": 2},
        "set": {"z", "a"},
        "frozen": frozenset({ExampleEnum.VALUE, "other"}),
    }
    second = {
        "frozen": frozenset(("other", ExampleEnum.VALUE)),
        "set": set(("a", "z")),
        "mapping": {"a": 2, "z": 1},
    }

    assert canonical_json(first) == canonical_json(second)
    assert content_hash(first) == content_hash(second)


def test_ordered_list_and_tuple_order_remains_identity_bearing():
    assert content_hash(["a", "b"]) != content_hash(["b", "a"])
    assert content_hash(("a", "b")) != content_hash(("b", "a"))


def test_datetime_unicode_dataclass_enum_and_path_are_platform_stable():
    instant = datetime(2026, 8, 3, 12, 30, 45, 123, tzinfo=UTC)
    offset = instant.astimezone(timezone(timedelta(hours=2)))
    payload = {
        "instant": instant,
        "label": "évidence-研究",
        "record": ExampleDataclass("fixture", frozenset({2, 1})),
        "enum": ExampleEnum.VALUE,
        "path": Path("safe/relative/path"),
    }
    changed_timezone = {**payload, "instant": offset}

    assert canonical_json(payload) == canonical_json(changed_timezone)
    assert "évidence-研究" in canonical_json(payload)
    assert "2026-08-03T12:30:45.000123Z" in canonical_json(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected(value):
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json({"value": value})


def test_naive_datetime_and_unsupported_values_are_rejected():
    with pytest.raises(CanonicalizationError, match="naive datetime"):
        canonical_json(datetime(2026, 8, 3, 12, 30))
    with pytest.raises(CanonicalizationError, match="unsupported"):
        canonical_json(object())


def test_duplicate_normalised_mapping_keys_and_set_values_are_rejected():
    with pytest.raises(CanonicalizationError, match="mapping key"):
        canonicalize({1: "numeric", "1": "text"})
    ambiguous = {AmbiguousSetValue("same"), AmbiguousSetValue("same")}
    with pytest.raises(CanonicalizationError, match="set element"):
        canonicalize(ambiguous)


def test_hash_envelopes_are_domain_separated():
    payload = {"same": "payload"}
    attestation = domain_separated_hash(
        payload,
        hash_domain="auto-researcher-read-safety-attestation",
        hash_version="1",
        schema_version="knowledge-read-safety-v2",
    )
    configuration = domain_separated_hash(
        payload,
        hash_domain="auto-researcher-read-safety-configuration",
        hash_version="1",
        schema_version="knowledge-read-safety-v2",
    )

    assert attestation != configuration
