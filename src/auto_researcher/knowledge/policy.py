"""Platform ceilings and deterministic trust helpers."""

from auto_researcher.knowledge.models import (
    KnowledgeGroundingPolicy,
    KnowledgeTrustTier,
)

PLATFORM_MAXIMUM_REFERENCES = 100
PLATFORM_MAXIMUM_ASSERTIONS = 10_000
PLATFORM_MAXIMUM_ENTITIES = 10_000
PLATFORM_PRIOR_CAPS = {
    KnowledgeTrustTier.CURATED: 0.9,
    KnowledgeTrustTier.CORPUS: 0.7,
    KnowledgeTrustTier.LIVE: 0.3,
    KnowledgeTrustTier.UNVERIFIED: 0.3,
}


def validate_policy_ceiling(policy: KnowledgeGroundingPolicy) -> None:
    if policy.maximum_references > PLATFORM_MAXIMUM_REFERENCES:
        raise ValueError("knowledge reference limit exceeds platform ceiling")
    if policy.maximum_assertions > PLATFORM_MAXIMUM_ASSERTIONS:
        raise ValueError("knowledge assertion limit exceeds platform ceiling")
    if policy.maximum_entities > PLATFORM_MAXIMUM_ENTITIES:
        raise ValueError("knowledge entity limit exceeds platform ceiling")
    for tier, platform_cap in PLATFORM_PRIOR_CAPS.items():
        configured = float(policy.tier_prior_weight_caps.get(tier.value, 0))
        if configured > platform_cap:
            raise ValueError(f"{tier.value} prior cap exceeds platform ceiling")


def prior_cap(
    policy: KnowledgeGroundingPolicy,
    tier: KnowledgeTrustTier,
) -> float:
    validate_policy_ceiling(policy)
    return min(
        float(policy.tier_prior_weight_caps.get(tier.value, 0.0)),
        PLATFORM_PRIOR_CAPS[tier],
    )
