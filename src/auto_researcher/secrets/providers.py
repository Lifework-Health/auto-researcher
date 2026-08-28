"""Narrow runtime secret providers with optional Google Cloud integration."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from auto_researcher.secrets.models import (
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
)
from auto_researcher.secrets.linux_keyring import LinuxKernelKeyringSecretProvider


@runtime_checkable
class SecretProvider(Protocol):
    def resolve(self, reference: SecretReference) -> ResolvedSecret | None: ...


def _raise_if_required(
    reference: SecretReference,
    code: SecretResolutionErrorCode,
) -> None:
    if reference.required:
        raise SecretResolutionError(code, reference) from None


class EnvironmentSecretProvider:
    """Resolve a logical secret from one explicitly named environment variable."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self.__environment = environment

    def __repr__(self) -> str:
        return "EnvironmentSecretProvider()"

    def resolve(self, reference: SecretReference) -> ResolvedSecret | None:
        if reference.provider is not SecretProviderKind.ENVIRONMENT:
            _raise_if_required(
                reference,
                SecretResolutionErrorCode.INVALID_REFERENCE,
            )
            return None
        variable = reference.provider_identifier
        if variable is None:
            _raise_if_required(
                reference,
                SecretResolutionErrorCode.INVALID_REFERENCE,
            )
            return None
        environment = (
            self.__environment if self.__environment is not None else os.environ
        )
        value = environment.get(variable)
        if not value:
            _raise_if_required(reference, SecretResolutionErrorCode.MISSING)
            return None
        return ResolvedSecret(value)


def _google_client_factory() -> Any:
    try:
        secretmanager = import_module("google.cloud.secretmanager")
    except ImportError:
        raise _MissingGoogleDependency from None
    return secretmanager.SecretManagerServiceClient()


class _MissingGoogleDependency(Exception):
    pass


def _google_error_code(exc: Exception) -> SecretResolutionErrorCode:
    name = type(exc).__name__.casefold()
    structured_reasons = {str(getattr(exc, "reason", "")).casefold()}
    errors = getattr(exc, "errors", ())
    if isinstance(errors, (list, tuple)):
        structured_reasons.update(
            str(item.get("reason", "")).casefold()
            for item in errors
            if isinstance(item, dict)
        )
    if structured_reasons & {"service_disabled", "api_disabled"}:
        return SecretResolutionErrorCode.API_UNAVAILABLE
    if isinstance(exc, _MissingGoogleDependency):
        return SecretResolutionErrorCode.DEPENDENCY_UNAVAILABLE
    if any(
        token in name
        for token in ("defaultcredential", "authentication", "unauthenticated")
    ):
        return SecretResolutionErrorCode.AUTHENTICATION_FAILED
    if "permissiondenied" in name or "forbidden" in name:
        return SecretResolutionErrorCode.PERMISSION_DENIED
    if "notfound" in name:
        return SecretResolutionErrorCode.NOT_FOUND
    if any(token in name for token in ("deadlineexceeded", "timeout")):
        return SecretResolutionErrorCode.TIMEOUT
    if any(token in name for token in ("failedprecondition", "methodnotallowed")):
        return SecretResolutionErrorCode.API_UNAVAILABLE
    if any(
        token in name
        for token in (
            "serviceunavailable",
            "connection",
            "internalserver",
            "resourceexhausted",
        )
    ):
        return SecretResolutionErrorCode.UNAVAILABLE
    return SecretResolutionErrorCode.UNAVAILABLE


class GoogleSecretManagerProvider:
    """Resolve secrets with ADC/attached identity through Secret Manager.

    Importing this module never imports Google Cloud packages. The optional
    dependency is loaded only when a non-injected client is first required.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("secret provider timeout must be positive")
        self.__client = client
        self.__client_factory = client_factory or _google_client_factory
        self.__timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return "GoogleSecretManagerProvider()"

    def _resource_name(self, reference: SecretReference) -> str:
        identifier = reference.provider_identifier
        if identifier is None:
            raise SecretResolutionError(
                SecretResolutionErrorCode.INVALID_REFERENCE,
                reference,
            ) from None
        return f"{identifier}/versions/{reference.selected_version}"

    def resolve(self, reference: SecretReference) -> ResolvedSecret | None:
        if reference.provider is not SecretProviderKind.GOOGLE_SECRET_MANAGER:
            _raise_if_required(
                reference,
                SecretResolutionErrorCode.INVALID_REFERENCE,
            )
            return None
        name = self._resource_name(reference)
        failure: SecretResolutionErrorCode | None = None
        try:
            if self.__client is None:
                self.__client = self.__client_factory()
            response = self.__client.access_secret_version(
                request={"name": name},
                timeout=self.__timeout_seconds,
            )
        except SecretResolutionError:
            raise
        except Exception as exc:
            code = _google_error_code(exc)
            if not reference.required and code is SecretResolutionErrorCode.NOT_FOUND:
                return None
            failure = code
        if failure is not None:
            raise SecretResolutionError(failure, reference) from None
        invalid_payload = False
        try:
            data = response.payload.data
            if not isinstance(data, bytes):
                raise TypeError
            value = data.decode("utf-8")
        except (AttributeError, TypeError, UnicodeDecodeError):
            invalid_payload = True
            value = ""
        if invalid_payload:
            raise SecretResolutionError(
                SecretResolutionErrorCode.INVALID_VALUE,
                reference,
            ) from None
        if not value:
            raise SecretResolutionError(
                SecretResolutionErrorCode.INVALID_VALUE,
                reference,
            ) from None
        return ResolvedSecret(value)


def provider_for_reference(reference: SecretReference) -> SecretProvider:
    """Construct the configured provider without resolving or recording a value."""
    if reference.provider is SecretProviderKind.ENVIRONMENT:
        return EnvironmentSecretProvider()
    if reference.provider is SecretProviderKind.GOOGLE_SECRET_MANAGER:
        return GoogleSecretManagerProvider()
    if reference.provider is SecretProviderKind.LINUX_KERNEL_KEYRING:
        return LinuxKernelKeyringSecretProvider()
    raise SecretResolutionError(
        SecretResolutionErrorCode.INVALID_REFERENCE,
        reference,
    ) from None
