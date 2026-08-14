"""Secret identity, redacted runtime value, and bounded safe errors."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SecretProviderKind(StrEnum):
    ENVIRONMENT = "environment"
    GOOGLE_SECRET_MANAGER = "google_secret_manager"


class SecretReference(BaseModel):
    """Serializable non-sensitive identity for one runtime secret.

    A reference can safely enter operational configuration or audit metadata. It
    deliberately has no field capable of carrying the resolved secret value.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    logical_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    provider: SecretProviderKind
    provider_identifier: str | None = Field(default=None, max_length=512)
    version: str | None = Field(default=None, max_length=128)
    required: bool = True

    @field_validator("provider_identifier", "version")
    @classmethod
    def bounded_safe_metadata(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or value != value.strip():
            raise ValueError("secret reference metadata must be non-empty and trimmed")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("secret reference metadata contains control characters")
        return value

    @model_validator(mode="after")
    def provider_metadata_is_unambiguous(self) -> "SecretReference":
        if self.provider is SecretProviderKind.ENVIRONMENT:
            if self.provider_identifier is None:
                raise ValueError(
                    "environment secret references require an explicit identifier"
                )
            if self.version is not None:
                raise ValueError(
                    "environment secret references do not support versions"
                )
            if not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*",
                self.provider_identifier,
            ):
                raise ValueError(
                    "environment secret identifiers must be environment variable names"
                )
        if self.provider is SecretProviderKind.GOOGLE_SECRET_MANAGER:
            if self.provider_identifier is None:
                raise ValueError(
                    "Google secret references require a fully-qualified identifier"
                )
            if not re.fullmatch(
                r"projects/[A-Za-z0-9][A-Za-z0-9._:-]*/secrets/[A-Za-z0-9_-]+",
                self.provider_identifier,
            ):
                raise ValueError("Google secret identifier must be fully qualified")
        if (
            self.provider is SecretProviderKind.GOOGLE_SECRET_MANAGER
            and self.version is not None
            and not re.fullmatch(r"[A-Za-z0-9._-]+", self.version)
        ):
            raise ValueError("Google secret version is invalid")
        return self

    @property
    def selected_version(self) -> str | None:
        if self.provider is SecretProviderKind.GOOGLE_SECRET_MANAGER:
            return self.version or "latest"
        return None


def parse_secret_reference(value: Mapping[str, Any]) -> SecretReference:
    """Parse untrusted configuration without echoing rejected input values."""
    try:
        return SecretReference.model_validate(value)
    except Exception:
        raise ValueError("secret_reference_invalid") from None


class ResolvedSecret:
    """Opaque runtime-only value whose common representations are redacted.

    The explicit accessor makes the identity/value distinction visible at the
    narrow provider-construction boundary. The object is intentionally neither
    JSON- nor pickle-serialisable.
    """

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("resolved secret must be a non-empty string")
        self.__value = value

    def reveal(self) -> str:
        """Reveal the value only to code constructing the credential consumer."""
        return self.__value

    def __repr__(self) -> str:
        return "ResolvedSecret(<redacted>)"

    def __str__(self) -> str:
        return "<redacted>"

    def __reduce__(self) -> Any:
        raise TypeError("resolved secrets cannot be serialised")

    def __getstate__(self) -> Any:
        raise TypeError("resolved secrets cannot be serialised")


class SecretResolutionErrorCode(StrEnum):
    INVALID_REFERENCE = "invalid_reference"
    MISSING = "missing"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    PERMISSION_DENIED = "permission_denied"
    API_UNAVAILABLE = "api_unavailable"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_VALUE = "invalid_value"


class SecretResolutionError(RuntimeError):
    """Stable, bounded provider failure that never includes a secret value."""

    def __init__(
        self,
        code: SecretResolutionErrorCode,
        reference: SecretReference,
    ) -> None:
        self.code = code
        self.logical_name = reference.logical_name
        self.provider = reference.provider
        identifier = ""
        if (
            reference.provider is SecretProviderKind.ENVIRONMENT
            and reference.provider_identifier is not None
        ):
            identifier = f" identifier={reference.provider_identifier}"
        super().__init__(
            "secret_resolution_failed "
            f"code={code.value} provider={reference.provider.value} "
            f"logical_name={reference.logical_name}{identifier}"
        )
