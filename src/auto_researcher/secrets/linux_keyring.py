"""Linux user-keyring lifecycle without secret serialization."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from auto_researcher.secrets.models import (
    ResolvedSecret,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
)

DEFAULT_KEYCTL_PATH = "/usr/bin/keyctl"
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
MAXIMUM_TIMEOUT_SECONDS = 30 * 24 * 60 * 60
USER_ONLY_PERMISSIONS = "0x3f3f0000"


@dataclass(frozen=True)
class KeyringCommandResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


KeyringRunner = Callable[..., KeyringCommandResult]


def _subprocess_runner(
    command: Sequence[str],
    *,
    input: bytes | None,
    timeout: float,
) -> KeyringCommandResult:
    completed = subprocess.run(
        list(command),
        input=input,
        stdin=subprocess.DEVNULL if input is None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    return KeyringCommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


class LinuxUserKeyring:
    """Narrow keyctl wrapper that never places secret values in argv or errors."""

    def __init__(
        self,
        *,
        keyctl_path: str = DEFAULT_KEYCTL_PATH,
        keyring_identifier: str = "@u",
        session_keyring_identifier: str = "@s",
        runner: KeyringRunner = _subprocess_runner,
        command_timeout_seconds: float = 5.0,
    ) -> None:
        if not keyctl_path.startswith("/"):
            raise ValueError("keyctl path must be absolute")
        if command_timeout_seconds <= 0:
            raise ValueError("keyctl command timeout must be positive")
        if keyring_identifier != "@u":
            raise ValueError("linux user keyring identifier must be @u")
        if session_keyring_identifier != "@s":
            raise ValueError("linux session keyring identifier must be @s")
        self._keyctl_path = keyctl_path
        self._keyring_identifier = keyring_identifier
        self._session_keyring_identifier = session_keyring_identifier
        self._runner = runner
        self._command_timeout_seconds = command_timeout_seconds

    def __repr__(self) -> str:
        return "LinuxUserKeyring()"

    def _run(
        self,
        *arguments: str,
        input: bytes | None = None,
    ) -> KeyringCommandResult:
        try:
            return self._runner(
                (self._keyctl_path, *arguments),
                input=input,
                timeout=self._command_timeout_seconds,
            )
        except FileNotFoundError:
            raise RuntimeError("linux_keyring_dependency_unavailable") from None
        except subprocess.TimeoutExpired:
            raise RuntimeError("linux_keyring_command_timeout") from None

    @staticmethod
    def _numeric_identifier(result: KeyringCommandResult) -> str | None:
        try:
            value = result.stdout.decode("ascii").strip()
        except UnicodeDecodeError:
            return None
        return value if value.isdigit() else None

    def find(self, identifier: str) -> str | None:
        result = self._run(
            "search",
            self._keyring_identifier,
            "user",
            identifier,
        )
        if result.returncode == 1:
            return None
        key_id = self._numeric_identifier(result)
        if result.returncode != 0 or key_id is None:
            raise RuntimeError("linux_keyring_search_failed")
        return key_id

    def resolve(self, identifier: str) -> ResolvedSecret | None:
        key_id = self.find(identifier)
        if key_id is None:
            return None
        result = self._run("pipe", key_id)
        if result.returncode != 0:
            raise RuntimeError("linux_keyring_read_failed")
        try:
            value = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeError("linux_keyring_value_invalid") from None
        if not value or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise RuntimeError("linux_keyring_value_invalid")
        return ResolvedSecret(value)

    def store(
        self,
        identifier: str,
        value: str,
        *,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> str:
        if not value or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("linux_keyring_value_invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 3_600 <= timeout_seconds <= MAXIMUM_TIMEOUT_SECONDS
        ):
            raise ValueError("linux_keyring_timeout_invalid")
        existing = self.find(identifier)
        if existing is None:
            result = self._run(
                "padd",
                "user",
                identifier,
                self._session_keyring_identifier,
                input=value.encode("utf-8"),
            )
            key_id = self._numeric_identifier(result)
            if result.returncode != 0 or key_id is None:
                raise RuntimeError("linux_keyring_store_failed")
        else:
            key_id = existing
            result = self._run("update", key_id, input=value.encode("utf-8"))
            if result.returncode != 0:
                raise RuntimeError("linux_keyring_update_failed")
        if self._run("setperm", key_id, USER_ONLY_PERMISSIONS).returncode != 0:
            self._discard(key_id)
            raise RuntimeError("linux_keyring_permission_failed")
        if self._run("timeout", key_id, str(timeout_seconds)).returncode != 0:
            self._discard(key_id)
            raise RuntimeError("linux_keyring_timeout_failed")
        if existing is None:
            if self._run("link", key_id, self._keyring_identifier).returncode != 0:
                self._discard(key_id)
                raise RuntimeError("linux_keyring_link_failed")
            if (
                self._run("unlink", key_id, self._session_keyring_identifier).returncode
                != 0
            ):
                self._discard(key_id)
                raise RuntimeError("linux_keyring_unlink_failed")
        return key_id

    def _discard(self, key_id: str) -> None:
        self._run("revoke", key_id)
        self._run("unlink", key_id, self._session_keyring_identifier)
        self._run("unlink", key_id, self._keyring_identifier)

    def remove(self, identifier: str) -> bool:
        key_id = self.find(identifier)
        if key_id is None:
            return False
        if self._run("revoke", key_id).returncode != 0:
            raise RuntimeError("linux_keyring_revoke_failed")
        return True


class LinuxKernelKeyringSecretProvider:
    """Resolve one typed reference from the Linux user keyring."""

    def __init__(self, *, keyring: LinuxUserKeyring | None = None) -> None:
        self._keyring = keyring or LinuxUserKeyring()

    def __repr__(self) -> str:
        return "LinuxKernelKeyringSecretProvider()"

    def resolve(self, reference: SecretReference) -> ResolvedSecret | None:
        if reference.provider is not SecretProviderKind.LINUX_KERNEL_KEYRING:
            if reference.required:
                raise SecretResolutionError(
                    SecretResolutionErrorCode.INVALID_REFERENCE,
                    reference,
                ) from None
            return None
        identifier = reference.provider_identifier
        if identifier is None:
            if reference.required:
                raise SecretResolutionError(
                    SecretResolutionErrorCode.INVALID_REFERENCE,
                    reference,
                ) from None
            return None
        try:
            resolved = self._keyring.resolve(identifier)
        except RuntimeError as exc:
            code = (
                SecretResolutionErrorCode.DEPENDENCY_UNAVAILABLE
                if str(exc) == "linux_keyring_dependency_unavailable"
                else SecretResolutionErrorCode.TIMEOUT
                if str(exc) == "linux_keyring_command_timeout"
                else SecretResolutionErrorCode.INVALID_VALUE
                if str(exc) == "linux_keyring_value_invalid"
                else SecretResolutionErrorCode.UNAVAILABLE
            )
            raise SecretResolutionError(code, reference) from None
        if resolved is None and reference.required:
            raise SecretResolutionError(
                SecretResolutionErrorCode.MISSING,
                reference,
            ) from None
        return resolved
