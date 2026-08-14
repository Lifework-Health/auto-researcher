"""Executable capability contract for the exact pinned OpenEvolve release."""

from __future__ import annotations

import hashlib
import importlib.metadata
from enum import StrEnum
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_researcher.search.openevolve.upstream import installed_record_hash
from auto_researcher.search.openevolve.upstream_models import (
    UPSTREAM_COMMIT,
    UPSTREAM_INSTALLED_RECORD_HASH,
    UPSTREAM_PACKAGE_VERSION,
    UPSTREAM_WHEEL_SHA256,
)

CAPABILITY_MANIFEST_VERSION = "openevolve-capability-manifest-v1"


class CapabilityClassification(StrEnum):
    PRESERVED_NATIVE = "PRESERVED_NATIVE"
    PRESERVED_VIA_ADAPTER = "PRESERVED_VIA_ADAPTER"
    CURRENTLY_WEAKENED = "CURRENTLY_WEAKENED"
    CURRENTLY_DISABLED = "CURRENTLY_DISABLED"
    NOT_PRESENT_IN_PINNED_UPSTREAM = "NOT_PRESENT_IN_PINNED_UPSTREAM"


class CapabilityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str = Field(min_length=1)
    classification: CapabilityClassification
    upstream_evidence: str = Field(min_length=1)
    adapter_contract: str | None = None
    probe: str | None = None
    justification: str | None = None

    @model_validator(mode="after")
    def classification_has_evidence(self) -> "CapabilityRecord":
        if (
            self.classification is CapabilityClassification.PRESERVED_NATIVE
            and not self.probe
        ):
            raise ValueError("preserved native capability requires a probe")
        if self.classification is CapabilityClassification.PRESERVED_VIA_ADAPTER and (
            not self.adapter_contract or not self.probe
        ):
            raise ValueError("adapter-preserved capability requires contract and probe")
        if (
            self.classification
            in {
                CapabilityClassification.CURRENTLY_DISABLED,
                CapabilityClassification.CURRENTLY_WEAKENED,
            }
            and not self.justification
        ):
            raise ValueError("weakened or disabled capability requires justification")
        return self


class OpenEvolveCapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str = CAPABILITY_MANIFEST_VERSION
    upstream_package: str
    upstream_version: str
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    wheel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    installed_record_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_lock: str
    dependency_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: tuple[CapabilityRecord, ...]

    @model_validator(mode="after")
    def capability_names_are_unique(self) -> "OpenEvolveCapabilityManifest":
        names = [item.capability for item in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("capability names must be unique")
        return self

    def counts(self) -> dict[CapabilityClassification, int]:
        return {
            classification: sum(
                item.classification is classification for item in self.capabilities
            )
            for classification in CapabilityClassification
        }


def load_capability_manifest(path: Path) -> OpenEvolveCapabilityManifest:
    return OpenEvolveCapabilityManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def verify_capability_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
) -> OpenEvolveCapabilityManifest:
    manifest = load_capability_manifest(manifest_path)
    lock_path = repository_root / manifest.dependency_lock
    lock_hash = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if (
        importlib.metadata.version("openevolve") != UPSTREAM_PACKAGE_VERSION
        or manifest.upstream_version != UPSTREAM_PACKAGE_VERSION
        or manifest.upstream_commit != UPSTREAM_COMMIT
        or manifest.wheel_sha256 != UPSTREAM_WHEEL_SHA256
        or manifest.installed_record_hash != UPSTREAM_INSTALLED_RECORD_HASH
        or installed_record_hash() != UPSTREAM_INSTALLED_RECORD_HASH
        or manifest.dependency_lock_sha256 != lock_hash
    ):
        raise ValueError("openevolve_capability_manifest_pin_mismatch")
    return manifest
