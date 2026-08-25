from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from auto_researcher.secrets import (
    LinuxKernelKeyringSecretProvider,
    LinuxUserKeyring,
    SecretProviderKind,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    parse_secret_reference,
    provider_for_reference,
)
from auto_researcher.secret_cli import main
from auto_researcher.secrets.linux_keyring import KeyringCommandResult

SECRET = "test-anthropic-secret-value"
IDENTIFIER = "auto-researcher/anthropic-api-key"


@dataclass
class FakeKeyctl:
    present: bool = False
    session_present: bool = False
    value: bytes = b""
    key_id: str = "24680"
    fail_action: str | None = None
    commands: list[tuple[str, ...]] = field(default_factory=list)
    inputs: list[bytes | None] = field(default_factory=list)

    def __call__(self, command, *, input, timeout):
        del timeout
        command = tuple(command)
        self.commands.append(command)
        self.inputs.append(input)
        action = command[1]
        if action == self.fail_action:
            return KeyringCommandResult(2)
        if action == "search":
            return (
                KeyringCommandResult(0, f"{self.key_id}\n".encode())
                if self.present
                else KeyringCommandResult(1)
            )
        if action == "pipe":
            return KeyringCommandResult(0, self.value)
        if action == "padd":
            self.session_present = True
            self.value = input or b""
            return KeyringCommandResult(0, f"{self.key_id}\n".encode())
        if action == "update":
            self.value = input or b""
            return KeyringCommandResult(0)
        if action in {"setperm", "timeout"}:
            return KeyringCommandResult(0)
        if action == "link":
            self.present = True
            return KeyringCommandResult(0)
        if action == "unlink":
            if command[-1] == "@s":
                self.session_present = False
            if command[-1] == "@u":
                self.present = False
            return KeyringCommandResult(0)
        if action == "revoke":
            self.present = False
            self.session_present = False
            self.value = b""
            return KeyringCommandResult(0)
        raise AssertionError(action)


def _reference(*, required: bool = True) -> SecretReference:
    return SecretReference(
        logical_name="anthropic_api_key",
        provider=SecretProviderKind.LINUX_KERNEL_KEYRING,
        provider_identifier=IDENTIFIER,
        required=required,
    )


def test_linux_keyring_reference_is_typed_and_safe():
    parsed = parse_secret_reference(_reference().model_dump(mode="json"))
    assert parsed.provider is SecretProviderKind.LINUX_KERNEL_KEYRING
    assert parsed.provider_identifier == IDENTIFIER
    assert isinstance(provider_for_reference(parsed), LinuxKernelKeyringSecretProvider)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "logical_name": "anthropic_api_key",
            "provider": "linux_kernel_keyring",
        },
        {
            "logical_name": "anthropic_api_key",
            "provider": "linux_kernel_keyring",
            "provider_identifier": "unsafe key name",
        },
        {
            "logical_name": "anthropic_api_key",
            "provider": "linux_kernel_keyring",
            "provider_identifier": IDENTIFIER,
            "version": "1",
        },
    ],
)
def test_linux_keyring_reference_rejects_ambiguous_metadata(payload):
    with pytest.raises(ValueError, match="secret_reference_invalid"):
        parse_secret_reference(payload)


def test_linux_keyring_provider_resolves_without_secret_in_argv_or_repr():
    runner = FakeKeyctl(present=True, value=SECRET.encode())
    keyring = LinuxUserKeyring(runner=runner)
    provider = LinuxKernelKeyringSecretProvider(keyring=keyring)

    resolved = provider.resolve(_reference())

    assert resolved is not None
    assert resolved.reveal() == SECRET
    assert SECRET not in repr(keyring)
    assert SECRET not in repr(provider)
    assert all(SECRET not in " ".join(command) for command in runner.commands)
    assert runner.commands[-1] == ("/usr/bin/keyctl", "pipe", runner.key_id)


def test_linux_keyring_provider_missing_is_stable_and_non_leaking():
    provider = LinuxKernelKeyringSecretProvider(
        keyring=LinuxUserKeyring(runner=FakeKeyctl())
    )
    with pytest.raises(SecretResolutionError) as caught:
        provider.resolve(_reference())
    assert caught.value.code is SecretResolutionErrorCode.MISSING
    assert SECRET not in str(caught.value)
    assert (
        LinuxKernelKeyringSecretProvider(
            keyring=LinuxUserKeyring(runner=FakeKeyctl())
        ).resolve(_reference(required=False))
        is None
    )


def test_linux_keyring_store_updates_via_stdin_and_applies_bounded_lifetime():
    runner = FakeKeyctl()
    keyring = LinuxUserKeyring(runner=runner)

    key_id = keyring.store(IDENTIFIER, SECRET, timeout_seconds=604_800)
    keyring.store(IDENTIFIER, f"{SECRET}-rotated", timeout_seconds=86_400)

    assert key_id == runner.key_id
    assert runner.value == f"{SECRET}-rotated".encode()
    assert any(command[1] == "padd" for command in runner.commands)
    assert any(
        command[1:] == ("link", runner.key_id, "@u") for command in runner.commands
    )
    assert any(
        command[1:] == ("unlink", runner.key_id, "@s") for command in runner.commands
    )
    assert any(command[1] == "update" for command in runner.commands)
    assert any(
        command[1:] == ("setperm", runner.key_id, "0x3f3f0000")
        for command in runner.commands
    )
    assert any(
        command[1:] == ("timeout", runner.key_id, "86400")
        for command in runner.commands
    )
    assert all(SECRET not in " ".join(command) for command in runner.commands)
    assert SECRET.encode() in tuple(item for item in runner.inputs if item is not None)


def test_linux_keyring_remove_revokes_the_value():
    runner = FakeKeyctl(present=True, value=SECRET.encode())
    keyring = LinuxUserKeyring(runner=runner)
    assert keyring.remove(IDENTIFIER) is True
    assert keyring.remove(IDENTIFIER) is False
    assert runner.present is False


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("setperm", "linux_keyring_permission_failed"),
        ("timeout", "linux_keyring_timeout_failed"),
        ("link", "linux_keyring_link_failed"),
        ("unlink", "linux_keyring_unlink_failed"),
    ],
)
def test_linux_keyring_store_discards_value_when_hardening_fails(
    failure,
    message,
):
    runner = FakeKeyctl(fail_action=failure)
    keyring = LinuxUserKeyring(runner=runner)

    with pytest.raises(RuntimeError, match=message):
        keyring.store(IDENTIFIER, SECRET)

    assert runner.present is False
    assert runner.value == b""
    assert any(command[1] == "revoke" for command in runner.commands)


def test_linux_keyring_cli_hidden_input_never_prints_value(monkeypatch, capsys):
    runner = FakeKeyctl()
    keyring = LinuxUserKeyring(runner=runner)
    monkeypatch.setattr(
        "auto_researcher.secret_cli.LinuxUserKeyring",
        lambda: keyring,
    )
    monkeypatch.setattr("getpass.getpass", lambda _prompt: SECRET)

    assert main(["keyring", "store", "--identifier", IDENTIFIER]) == 0
    output = capsys.readouterr().out
    assert "LINUX_KEYRING_SECRET_STORED" in output
    assert SECRET not in output
    assert main(["keyring", "status", "--identifier", IDENTIFIER]) == 0
    assert main(["keyring", "remove", "--identifier", IDENTIFIER]) == 0
