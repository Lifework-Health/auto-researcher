"""Executable capability contract for the exact integrated Optuna release."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from importlib import metadata
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator


CAPABILITY_MANIFEST_VERSION = "optuna-capability-manifest-v1"
OPTUNA_PORTABLE_IDENTITY_SCHEME = "optuna-portable-wheel-content-identity-v1"
OPTUNA_4_9_0_PORTABLE_IDENTITY = (
    "e4fe2b30f50e56472c518dc839d67d96a2c703089b92efdc80f9800aa43eda8d"
)


class CapabilityClassification(StrEnum):
    PRESERVED_NATIVE = "PRESERVED_NATIVE"
    PRESERVED_VIA_ADAPTER = "PRESERVED_VIA_ADAPTER"
    CURRENTLY_WEAKENED = "CURRENTLY_WEAKENED"
    CURRENTLY_DISABLED = "CURRENTLY_DISABLED"
    NOT_PRESENT_IN_PINNED_UPSTREAM = "NOT_PRESENT_IN_PINNED_UPSTREAM"


class CapabilityItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: str = Field(min_length=1)
    classification: CapabilityClassification
    upstream_evidence: str = Field(min_length=1)
    probe: str | None = None
    probes: tuple[str, ...] = ()
    adapter_contract: str | None = None
    justification: str | None = None

    @model_validator(mode="after")
    def evidence_is_complete(self) -> "CapabilityItem":
        if (
            self.classification
            in {
                CapabilityClassification.PRESERVED_NATIVE,
                CapabilityClassification.PRESERVED_VIA_ADAPTER,
            }
            and not self.probe
        ):
            raise ValueError("preserved Optuna capabilities require a runtime probe")
        if self.probe is not None and self.probe in self.probes:
            raise ValueError("Optuna capability probes must be unique")
        if len(self.probes) != len(set(self.probes)):
            raise ValueError("Optuna capability probes must be unique")
        if (
            self.classification is CapabilityClassification.PRESERVED_VIA_ADAPTER
            and not self.adapter_contract
        ):
            raise ValueError(
                "adapter-preserved capabilities require an adapter contract"
            )
        if (
            self.classification
            in {
                CapabilityClassification.CURRENTLY_WEAKENED,
                CapabilityClassification.CURRENTLY_DISABLED,
            }
            and not self.justification
        ):
            raise ValueError(
                "weakened/disabled Optuna capabilities require justification"
            )
        return self


class OptunaCapabilityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str
    upstream_package: str
    upstream_version: str
    installed_record_hash: str = Field(min_length=64, max_length=64)
    dependency_lock: str
    dependency_lock_sha256: str = Field(min_length=64, max_length=64)
    public_samplers: tuple[str, ...]
    public_pruners: tuple[str, ...]
    capabilities: tuple[CapabilityItem, ...]

    @model_validator(mode="after")
    def capabilities_are_unique(self) -> "OptunaCapabilityManifest":
        names = [item.capability for item in self.capabilities]
        if len(names) != len(set(names)):
            raise ValueError("Optuna manifest capability names must be unique")
        return self

    def counts(self) -> dict[CapabilityClassification, int]:
        return {
            classification: sum(
                item.classification is classification for item in self.capabilities
            )
            for classification in CapabilityClassification
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_distribution_identity(files: object) -> str:
    """Hash only wheel-provided Optuna content, excluding environment artefacts."""

    rows: list[str] = []
    for item in files or ():  # type: ignore[union-attr]
        name = str(item)
        lowered = name.casefold()
        content_hash = getattr(item, "hash", None)
        if content_hash is None:
            continue
        if "__pycache__" in lowered.split("/") or lowered.endswith(".pyc"):
            continue
        if not (
            name.startswith("optuna/")
            or name.endswith((".dist-info/METADATA", ".dist-info/WHEEL"))
        ):
            continue
        rows.append(f"{name}|{content_hash.mode}|{content_hash.value}")
    if not rows:
        raise ValueError("portable Optuna distribution identity has no hashed rows")
    return hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def verify_capability_manifest(
    path: Path,
    *,
    repository_root: Path,
) -> OptunaCapabilityManifest:
    """Validate pin identity plus the exact public sampler/pruner inventory."""

    manifest = OptunaCapabilityManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    if manifest.manifest_version != CAPABILITY_MANIFEST_VERSION:
        raise ValueError("unsupported Optuna capability manifest version")
    distribution = metadata.distribution(manifest.upstream_package)
    if distribution.version != manifest.upstream_version:
        raise ValueError("installed Optuna version does not match capability manifest")
    portable_identity = portable_distribution_identity(distribution.files)
    if portable_identity != OPTUNA_4_9_0_PORTABLE_IDENTITY:
        raise ValueError("installed Optuna portable identity does not match manifest")
    lock_path = repository_root / manifest.dependency_lock
    if _sha256(lock_path) != manifest.dependency_lock_sha256:
        raise ValueError("Optuna dependency lock changed without manifest review")

    import optuna

    public_samplers = tuple(
        sorted(
            name
            for name in optuna.samplers.__all__
            if isinstance(getattr(optuna.samplers, name, None), type)
        )
    )
    public_pruners = tuple(
        sorted(
            name
            for name in optuna.pruners.__all__
            if isinstance(getattr(optuna.pruners, name, None), type)
        )
    )
    if public_samplers != tuple(sorted(manifest.public_samplers)):
        raise ValueError("Optuna public sampler inventory changed")
    if public_pruners != tuple(sorted(manifest.public_pruners)):
        raise ValueError("Optuna public pruner inventory changed")
    return manifest
