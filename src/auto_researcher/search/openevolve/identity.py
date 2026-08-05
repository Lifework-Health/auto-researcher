"""Domain-separated identities for OpenEvolve objects and source payloads."""

from __future__ import annotations

import hashlib

from auto_researcher.runtime.identity import payload_hash

CANONICAL_IDENTITY_VERSION = "canonical-json-sha256-v1"


def openevolve_hash(domain: str, value) -> str:
    return payload_hash(
        {
            "identity_version": CANONICAL_IDENTITY_VERSION,
            "domain": domain,
            "payload": value,
        }
    )


def source_hash(source: str) -> str:
    canonical = source.replace("\r\n", "\n").replace("\r", "\n")
    if not canonical.endswith("\n"):
        canonical += "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def candidate_id(
    *,
    search_request_id: str,
    component_interface_hash: str,
    source_sha256: str,
) -> str:
    digest = openevolve_hash(
        "openevolve-candidate-identity-v1",
        {
            "search_request_id": search_request_id,
            "component_interface_hash": component_interface_hash,
            "source_hash": source_sha256,
        },
    )
    return f"candidate-{digest[:24]}"
