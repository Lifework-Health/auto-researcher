"""Operator CLI for runtime-only secret lifecycle operations."""

from __future__ import annotations

import argparse
import getpass

from auto_researcher.secrets.linux_keyring import (
    DEFAULT_TIMEOUT_SECONDS,
    LinuxUserKeyring,
)
from auto_researcher.secrets.models import (
    SecretProviderKind,
    SecretReference,
)


def _reference(identifier: str, *, required: bool) -> SecretReference:
    return SecretReference(
        logical_name="anthropic_api_key",
        provider=SecretProviderKind.LINUX_KERNEL_KEYRING,
        provider_identifier=identifier,
        required=required,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="provider", required=True)
    keyring_parser = subparsers.add_parser("keyring")
    commands = keyring_parser.add_subparsers(dest="command", required=True)
    for name in ("store", "status", "remove"):
        command = commands.add_parser(name)
        command.add_argument("--identifier", required=True)
        if name == "store":
            command.add_argument(
                "--timeout-seconds",
                type=int,
                default=DEFAULT_TIMEOUT_SECONDS,
            )
    args = parser.parse_args(argv)
    reference = _reference(args.identifier, required=args.command != "status")
    keyring = LinuxUserKeyring()
    try:
        if args.command == "store":
            first = getpass.getpass("Paste secret value (input hidden): ")
            second = getpass.getpass("Confirm secret value (input hidden): ")
            if first != second:
                print("PRE-RUN BLOCKED: linux_keyring_secret_confirmation_mismatch")
                return 2
            keyring.store(
                reference.provider_identifier or "",
                first,
                timeout_seconds=args.timeout_seconds,
            )
            print(
                "LINUX_KEYRING_SECRET_STORED "
                f"identifier={reference.provider_identifier} "
                f"timeout_seconds={args.timeout_seconds}"
            )
            return 0
        if args.command == "status":
            present = keyring.find(reference.provider_identifier or "") is not None
            print(
                "LINUX_KEYRING_SECRET_PRESENT"
                if present
                else "LINUX_KEYRING_SECRET_MISSING"
            )
            return 0 if present else 2
        removed = keyring.remove(reference.provider_identifier or "")
        print(
            "LINUX_KEYRING_SECRET_REMOVED"
            if removed
            else "LINUX_KEYRING_SECRET_ALREADY_MISSING"
        )
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"PRE-RUN BLOCKED: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
