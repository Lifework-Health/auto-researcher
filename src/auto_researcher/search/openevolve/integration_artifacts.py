"""Transactional identity-bound adapter and executor evidence bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.upstream_models import (
    ExecutorIsolationResult,
    HardenedExecutorPolicy,
    UpstreamOpenEvolveAdapterContract,
    UpstreamOpenEvolveAdapterState,
)


def publish_integration_bundle(
    root: Path,
    run_id: str,
    contract: UpstreamOpenEvolveAdapterContract,
    state: UpstreamOpenEvolveAdapterState,
    policy: HardenedExecutorPolicy,
    isolation: ExecutorIsolationResult,
) -> tuple[tuple[str, ...], str]:
    target = root / "runs" / run_id / "openevolve-integration"
    payloads = {
        "upstream_identity.json": {
            "repository": contract.upstream_repository,
            "tag": contract.upstream_tag,
            "commit": contract.upstream_commit,
            "package_version": contract.upstream_package_version,
            "wheel_sha256": contract.upstream_wheel_sha256,
        },
        "adapter_contract.json": contract.model_dump(mode="json"),
        "adapter_state.json": state.model_dump(mode="json"),
        "upstream_mapping_summary.json": {
            "authoritative_identity": "AUTO_RESEARCHER",
            "proposal_count": state.proposal_count,
            "recommendations": state.upstream_parent_recommendations,
        },
        "upstream_feature_boundary.json": {"disabled": contract.unsupported_features},
        "executor_manifest.json": {
            "policy_hash": payload_hash(policy),
            "isolation_hash": payload_hash(isolation),
        },
        "image_identity.json": {
            "image_reference": policy.image_reference,
            "image_digest": policy.image_digest,
            "base_image_digest": policy.base_image_digest,
            "entrypoint_hash": policy.entrypoint_hash,
            "build_recipe_hash": policy.build_recipe_hash,
        },
        "isolation_policy.json": policy.model_dump(mode="json"),
        "execution_request.json": {
            "executor_id": policy.executor_id,
            "network": policy.network_mode,
            "environment_inheritance": False,
        },
        "execution_result.json": {
            "isolation_verified": isolation.network_isolation_verified
            and isolation.mount_isolation_verified
        },
        "network_isolation_result.json": {
            "verified": isolation.network_isolation_verified,
            "checks": dict(isolation.safe_checks),
        },
        "mount_isolation_result.json": {"verified": isolation.mount_isolation_verified},
        "resource_summary.json": {"bounded": True, "policy": policy.executor_id},
        "sanitised_log.json": {"persisted_raw_log": False},
    }
    hashes = {
        name: hashlib.sha256(
            (
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            ).encode()
        ).hexdigest()
        for name, value in payloads.items()
    }
    manifest = {
        "schema": "openevolve-integration-bundle-v1",
        "payload_hashes": hashes,
        "bundle_hash": payload_hash(hashes),
    }
    payloads["manifest.json"] = manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".openevolve-integration-", dir=target.parent)
    )
    try:
        for name, value in payloads.items():
            (staging / name).write_text(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
                + "\n"
            )
        if target.exists():
            existing = json.loads((target / "manifest.json").read_text())
            if existing != manifest:
                raise ValueError("openevolve_integration_artefact_conflict")
            return tuple(
                str((target / name).relative_to(root)) for name in sorted(payloads)
            ), manifest["bundle_hash"]
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return tuple(
        str((target / name).relative_to(root)) for name in sorted(payloads)
    ), manifest["bundle_hash"]
