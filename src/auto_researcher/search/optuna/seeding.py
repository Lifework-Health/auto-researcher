"""Exact Optuna 4.9.0 sampler-specific distributed seed policy."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class DistributedSamplerSeedPolicy(StrEnum):
    """Runtime-reviewed meaning of a sampler's constructor seed."""

    NATIVE_DEFAULT = "NATIVE_DEFAULT"
    WORKER_DISTINCT = "WORKER_DISTINCT"
    STUDY_SHARED = "STUDY_SHARED"
    UNSEEDED_DISTRIBUTED = "UNSEEDED_DISTRIBUTED"
    DISTRIBUTED_UNSUPPORTED = "DISTRIBUTED_UNSUPPORTED"


NATIVE_DISTRIBUTED_SEED_POLICIES: Mapping[str, DistributedSamplerSeedPolicy] = (
    MappingProxyType(
        {
            "native_default": DistributedSamplerSeedPolicy.NATIVE_DEFAULT,
            "tpe": DistributedSamplerSeedPolicy.WORKER_DISTINCT,
            "random": DistributedSamplerSeedPolicy.WORKER_DISTINCT,
            "cmaes": DistributedSamplerSeedPolicy.DISTRIBUTED_UNSUPPORTED,
            "gp": DistributedSamplerSeedPolicy.WORKER_DISTINCT,
            "nsgaii": DistributedSamplerSeedPolicy.WORKER_DISTINCT,
            "nsgaiii": DistributedSamplerSeedPolicy.WORKER_DISTINCT,
            "qmc": DistributedSamplerSeedPolicy.STUDY_SHARED,
            "grid": DistributedSamplerSeedPolicy.STUDY_SHARED,
            "brute_force": DistributedSamplerSeedPolicy.UNSEEDED_DISTRIBUTED,
        }
    )
)

CUSTOM_DISTRIBUTED_SEED_POLICIES = frozenset(
    {
        DistributedSamplerSeedPolicy.WORKER_DISTINCT,
        DistributedSamplerSeedPolicy.STUDY_SHARED,
        DistributedSamplerSeedPolicy.UNSEEDED_DISTRIBUTED,
    }
)


@dataclass(frozen=True)
class NativeSamplerSeedPlan:
    policy: DistributedSamplerSeedPolicy
    sampler_seed: int | None
    independent_sampler_seed: int | None = None


def worker_distinct_seed(
    study_seed: int,
    *,
    worker_id: str,
    worker_session_id: str,
) -> int:
    """Derive an ordinary stochastic-worker seed from operational identity."""

    material = f"{study_seed}\x1f{worker_id}\x1f{worker_session_id}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def native_distributed_seed_policy(
    sampler_type: str,
) -> DistributedSamplerSeedPolicy:
    try:
        return NATIVE_DISTRIBUTED_SEED_POLICIES[sampler_type]
    except KeyError:
        raise ValueError("optuna_native_sampler_seed_policy_missing") from None


def native_sampler_seed_plan(
    sampler_type: str,
    *,
    shared_workers: bool,
    study_seed: int,
    worker_seed: int,
    qmc_scramble: bool = False,
) -> NativeSamplerSeedPlan:
    """Resolve constructor seeds without claiming durable distributed RNG state."""

    policy = native_distributed_seed_policy(sampler_type)
    if not shared_workers:
        return NativeSamplerSeedPlan(
            policy=policy,
            sampler_seed=(
                None
                if policy is DistributedSamplerSeedPolicy.NATIVE_DEFAULT
                else study_seed
            ),
            independent_sampler_seed=(study_seed if sampler_type == "qmc" else None),
        )
    if policy is DistributedSamplerSeedPolicy.NATIVE_DEFAULT:
        return NativeSamplerSeedPlan(policy=policy, sampler_seed=None)
    if policy is DistributedSamplerSeedPolicy.WORKER_DISTINCT:
        return NativeSamplerSeedPlan(policy=policy, sampler_seed=worker_seed)
    if policy is DistributedSamplerSeedPolicy.STUDY_SHARED:
        return NativeSamplerSeedPlan(
            policy=policy,
            sampler_seed=(
                study_seed if sampler_type != "qmc" or qmc_scramble else None
            ),
            independent_sampler_seed=(worker_seed if sampler_type == "qmc" else None),
        )
    if policy is DistributedSamplerSeedPolicy.UNSEEDED_DISTRIBUTED:
        return NativeSamplerSeedPlan(policy=policy, sampler_seed=None)
    raise ValueError("optuna_sampler_shared_worker_incompatible")
