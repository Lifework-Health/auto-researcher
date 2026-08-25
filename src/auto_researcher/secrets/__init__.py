"""Task-agnostic runtime secret identities and provider abstractions."""

from auto_researcher.secrets.models import (
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    parse_secret_reference,
)
from auto_researcher.secrets.providers import (
    EnvironmentSecretProvider,
    GoogleSecretManagerProvider,
    SecretProvider,
    provider_for_reference,
)
from auto_researcher.secrets.linux_keyring import (
    LinuxKernelKeyringSecretProvider,
    LinuxUserKeyring,
)

__all__ = [
    "EnvironmentSecretProvider",
    "GoogleSecretManagerProvider",
    "LinuxKernelKeyringSecretProvider",
    "LinuxUserKeyring",
    "ResolvedSecret",
    "SecretProvider",
    "SecretProviderKind",
    "SecretReference",
    "SecretResolutionError",
    "SecretResolutionErrorCode",
    "parse_secret_reference",
    "provider_for_reference",
]
