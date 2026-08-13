"""Synthetic offline research fixtures; none are assertions about live sources."""

from __future__ import annotations

from datetime import UTC, date, datetime

from auto_researcher.research_intelligence.models import (
    ApplicabilityAssessment,
    ApplicabilityLevel,
    Availability,
    EvidenceCategory,
    EvidenceQuality,
    FindingCandidate,
    FindingStance,
    QuantitativeResult,
    RetrievedSourceMaterial,
    SourceCandidate,
    SourceType,
    TrustClassification,
)

RETRIEVED_AT = datetime(2026, 8, 13, 12, tzinfo=UTC)


def _source(
    *,
    key: str,
    title: str,
    source_type: SourceType,
    domains: tuple[str, ...],
    task_contexts: tuple[str, ...] = (),
    dataset_contexts: tuple[str, ...] = (),
    quality: float = 0.8,
) -> SourceCandidate:
    return SourceCandidate(
        title=title,
        authors=("Offline Fixture Author",),
        organisation="Auto Researcher test fixtures",
        source_type=source_type,
        publication_or_update_date=date(2024, 1, 15),
        retrieved_at=RETRIEVED_AT,
        reference_identity=f"fixture:{key}",
        uri=f"fixture://research-intelligence/{key}",
        source_version="fixture-v1",
        task_contexts=task_contexts,
        dataset_contexts=dataset_contexts,
        domains=domains,
        code_availability=Availability.AVAILABLE,
        data_availability=Availability.PARTIAL,
        trust_classification=TrustClassification.MODERATE,
        quality_score=quality,
        provenance_version="synthetic-offline-corpus-v1",
    )


def _finding(
    *,
    key: str,
    claim: str,
    category: EvidenceCategory,
    domain: str,
    task_id: str,
    applicability: float,
    stance: FindingStance = FindingStance.SUPPORTS,
    implication: str = "Run a controlled ablation before adopting this choice.",
    quantitative_results: tuple[QuantitativeResult, ...] = (),
) -> FindingCandidate:
    level = ApplicabilityLevel.HIGH if applicability >= 0.75 else ApplicabilityLevel.LOW
    return FindingCandidate(
        claim_key=key,
        claim=claim,
        category=category,
        stance=stance,
        quantitative_results=quantitative_results,
        method_or_intervention="Synthetic fixture method description",
        dataset_or_population="Synthetic, non-patient fixture population",
        conditions=("Offline fixture conditions apply",),
        limitations=(
            "This is deterministic test material, not a live literature claim",
        ),
        applicability_assessments=(
            ApplicabilityAssessment(
                task_id=task_id,
                domain=domain,
                level=level,
                score=applicability,
                rationale="Fixture-defined relevance for deterministic tests.",
            ),
        ),
        evidence_quality=EvidenceQuality.HIGH,
        confidence=0.82,
        hypothesis_tags=(f"{domain}.baseline",),
        experiment_design_implications=(implication,),
    )


def feta_nnunet_corpus() -> tuple[RetrievedSourceMaterial, ...]:
    domain = "fetal-mri-segmentation"
    task = "feta_seg_search@1.0"
    context = {
        "domains": (domain,),
        "task_contexts": (task,),
        "dataset_contexts": ("FeTA",),
    }
    return (
        RetrievedSourceMaterial(
            source=_source(
                key="nnunet-official-docs",
                title="Fixture nnU-Net official documentation",
                source_type=SourceType.OFFICIAL_DOCUMENTATION,
                **context,
            ),
            findings=(
                _finding(
                    key="baseline.self_configuring_unet",
                    claim="A self-configuring U-Net is a strong segmentation baseline.",
                    category=EvidenceCategory.STRONG_BASELINE,
                    domain=domain,
                    task_id=task,
                    applicability=0.94,
                    quantitative_results=(
                        QuantitativeResult(
                            metric="fixture macro score",
                            value=0.81,
                            unit="dimensionless",
                            uncertainty="Synthetic fixture value only",
                        ),
                    ),
                ),
                _finding(
                    key="planning.patch_and_batch_constraints",
                    claim="Patch size and batch constraints should be planned jointly.",
                    category=EvidenceCategory.ESTABLISHED_CHOICE,
                    domain=domain,
                    task_id=task,
                    applicability=0.9,
                ),
                _finding(
                    key="planning.spacing_resampling",
                    claim="Dataset-specific spacing should inform resampling configuration.",
                    category=EvidenceCategory.ACTIONABLE_PRIOR,
                    domain=domain,
                    task_id=task,
                    applicability=0.92,
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                key="nnunet-code",
                title="Fixture nnU-Net implementation evidence",
                source_type=SourceType.IMPLEMENTATION_CODE_EVIDENCE,
                **context,
            ),
            findings=(
                _finding(
                    key="preprocessing.foreground_crop",
                    claim="Foreground-aware cropping can reduce irrelevant background.",
                    category=EvidenceCategory.ESTABLISHED_CHOICE,
                    domain=domain,
                    task_id=task,
                    applicability=0.9,
                ),
                _finding(
                    key="architecture.depth_topology",
                    claim="Network depth should remain compatible with the planned patch topology.",
                    category=EvidenceCategory.ESTABLISHED_CHOICE,
                    domain=domain,
                    task_id=task,
                    applicability=0.88,
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                key="feta-challenge",
                title="Fixture FeTA challenge result",
                source_type=SourceType.BENCHMARK_CHALLENGE_RESULT,
                **context,
            ),
            findings=(
                _finding(
                    key="domain_shift.reconstruction",
                    claim="Reconstruction-domain shift can degrade segmentation.",
                    category=EvidenceCategory.FAILURE_MODE,
                    domain=domain,
                    task_id=task,
                    applicability=0.96,
                ),
                _finding(
                    key="failure.tissue_specific",
                    claim="Tissue classes may have materially different failure profiles.",
                    category=EvidenceCategory.FAILURE_MODE,
                    domain=domain,
                    task_id=task,
                    applicability=0.93,
                ),
                _finding(
                    key="spacing.fixed_target",
                    claim="A single fixed target spacing is preferable.",
                    category=EvidenceCategory.ACTIONABLE_PRIOR,
                    domain=domain,
                    task_id=task,
                    applicability=0.84,
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                key="feta-preprint",
                title="Fixture fetal MRI preprint",
                source_type=SourceType.PREPRINT,
                quality=0.68,
                **context,
            ),
            findings=(
                _finding(
                    key="spacing.fixed_target",
                    claim="A single fixed target spacing is not always preferable.",
                    category=EvidenceCategory.ACTIONABLE_PRIOR,
                    domain=domain,
                    task_id=task,
                    applicability=0.8,
                    stance=FindingStance.CONTRADICTS,
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                key="segmentation-paper",
                title="Fixture peer-reviewed segmentation paper",
                source_type=SourceType.PEER_REVIEWED_PAPER,
                quality=0.92,
                **context,
            ),
            findings=(
                _finding(
                    key="augmentation.bias_field",
                    claim="Bias-field augmentation is a plausible robustness prior.",
                    category=EvidenceCategory.UNDEREXPLORED_HYPOTHESIS,
                    domain=domain,
                    task_id=task,
                    applicability=0.62,
                ),
                _finding(
                    key="baseline.residual_encoder",
                    claim="A modern residual encoder is a plausible baseline family.",
                    category=EvidenceCategory.STRONG_BASELINE,
                    domain=domain,
                    task_id=task,
                    applicability=0.76,
                ),
            ),
        ),
    )


def tabular_corpus() -> tuple[RetrievedSourceMaterial, ...]:
    domain = "imbalanced-tabular-classification"
    task = "credit_risk_fixture@1.0"
    return (
        RetrievedSourceMaterial(
            source=_source(
                key="tabular-official-docs",
                title="Fixture tabular library documentation",
                source_type=SourceType.OFFICIAL_DOCUMENTATION,
                domains=(domain,),
                task_contexts=(task,),
            ),
            findings=(
                _finding(
                    key="baseline.class_weighting",
                    claim="Class weighting is a low-cost imbalanced-class baseline.",
                    category=EvidenceCategory.STRONG_BASELINE,
                    domain=domain,
                    task_id=task,
                    applicability=0.91,
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                key="tabular-paper",
                title="Fixture peer-reviewed tabular study",
                source_type=SourceType.PEER_REVIEWED_PAPER,
                domains=(domain,),
                task_contexts=(task,),
            ),
            findings=(
                _finding(
                    key="validation.target_leakage",
                    claim="Target leakage can invalidate tabular validation.",
                    category=EvidenceCategory.FAILURE_MODE,
                    domain=domain,
                    task_id=task,
                    applicability=0.97,
                ),
            ),
        ),
    )
