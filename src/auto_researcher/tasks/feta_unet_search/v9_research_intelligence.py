"""Reviewed, frozen research-intelligence inputs for the V9 campaign.

These records are deliberately conservative.  They preserve source identity,
state transfer limitations explicitly, and create advisory Director evidence.
They do not expose an experiment-construction API.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from auto_researcher.research_intelligence import (
    ApplicabilityAssessment,
    ApplicabilityLevel,
    Availability,
    DeterministicEvidenceSynthesiser,
    EvidenceCategory,
    EvidenceQuality,
    FindingCandidate,
    LiteratureEvidenceCandidate,
    LiteratureScoutMode,
    LiteratureScoutPolicy,
    LiteratureScoutRequest,
    LiteratureSourceType,
    OfflineResearchScout,
    ResearchProgrammeContext,
    RetrievedSourceMaterial,
    SourceCandidate,
    SourceType,
    TrustClassification,
    build_literature_scout_brief,
    build_research_director_knowledge_library,
    knowledge_library_as_landscape_evidence,
    literature_brief_as_landscape_evidence,
)

V9_PROGRAMME_ID = "feta_unet_search@1.0"
V9_EVIDENCE_CUTOFF = date(2026, 8, 24)
V9_RETRIEVED_AT = datetime(2026, 8, 24, 12, tzinfo=UTC)
V9_KNOWLEDGE_CONTENT_VERSION = "feta-v9-reviewed-primary-sources-v1"
V9_SCOUT_POLICY_ID = "feta-v9-primary-source-scout-v1"


def _applicability(score: float, rationale: str) -> ApplicabilityAssessment:
    level = (
        ApplicabilityLevel.HIGH
        if score >= 0.75
        else ApplicabilityLevel.MODERATE
        if score >= 0.5
        else ApplicabilityLevel.LOW
    )
    return ApplicabilityAssessment(
        task_id=V9_PROGRAMME_ID,
        domain="fetal-mri-segmentation",
        dataset_id="FeTA-development-fold-0",
        level=level,
        score=score,
        rationale=rationale,
    )


def _source(
    *,
    title: str,
    authors: tuple[str, ...],
    reference_identity: str,
    uri: str,
    publication_date: date,
    source_type: SourceType = SourceType.PEER_REVIEWED_PAPER,
    code_available: bool = True,
) -> SourceCandidate:
    return SourceCandidate(
        title=title,
        authors=authors,
        organisation=None,
        source_type=source_type,
        publication_or_update_date=publication_date,
        retrieved_at=V9_RETRIEVED_AT,
        reference_identity=reference_identity,
        uri=uri,
        source_version="reviewed-2026-08-24",
        task_contexts=(V9_PROGRAMME_ID,),
        dataset_contexts=("FeTA-development-fold-0",),
        domains=("medical-image-segmentation", "fetal-mri-segmentation"),
        code_availability=(
            Availability.AVAILABLE if code_available else Availability.UNKNOWN
        ),
        data_availability=Availability.PARTIAL,
        trust_classification=TrustClassification.HIGH,
        quality_score=0.9,
        quality_assessment_basis=(
            "Primary paper reviewed for method identity and transfer limits; "
            "its reported benchmark is not treated as FeTA evidence."
        ),
        provenance_version="operator-reviewed-primary-source-v1",
    )


def _finding(
    *,
    key: str,
    claim: str,
    category: EvidenceCategory,
    method: str,
    population: str,
    score: float,
    rationale: str,
    limitations: tuple[str, ...],
    implications: tuple[str, ...],
    tags: tuple[str, ...],
) -> FindingCandidate:
    return FindingCandidate(
        claim_key=key,
        claim=claim,
        category=category,
        method_or_intervention=method,
        dataset_or_population=population,
        conditions=(
            "Retain the frozen FeTA fold, objective, preprocessing and holdout boundary.",
            "Adopt only after bounded parameter, CUDA-memory and runtime validation.",
        ),
        limitations=limitations,
        applicability_assessments=(_applicability(score, rationale),),
        evidence_quality=EvidenceQuality.HIGH,
        confidence=0.82,
        hypothesis_tags=tags,
        experiment_design_implications=implications,
    )


def reviewed_v9_materials() -> tuple[RetrievedSourceMaterial, ...]:
    """Return the small primary-source corpus approved for V9 planning."""

    return (
        RetrievedSourceMaterial(
            source=_source(
                title="Attention U-Net: Learning Where to Look for the Pancreas",
                authors=("Ozan Oktay", "Jo Schlemper", "Loic Le Folgoc"),
                reference_identity="arxiv:1804.03999",
                uri="https://arxiv.org/abs/1804.03999",
                publication_date=date(2018, 4, 11),
                source_type=SourceType.PREPRINT,
            ),
            findings=(
                _finding(
                    key="architecture.attention_gated_skip",
                    claim=(
                        "Attention gates can condition skip information so a U-Net "
                        "suppresses irrelevant activations before decoder fusion."
                    ),
                    category=EvidenceCategory.UNDEREXPLORED_HYPOTHESIS,
                    method="Attention gates added to U-Net skip pathways.",
                    population="The source evaluates abdominal CT, not fetal MRI.",
                    score=0.72,
                    rationale=(
                        "The mechanism directly targets skip-feature selectivity, but "
                        "its transfer to reconstructed fetal MRI is unverified."
                    ),
                    limitations=(
                        "No FeTA or fetal-MRI evaluation is reported.",
                        "The mechanism may increase compute without improving small-tissue boundaries.",
                    ),
                    implications=(
                        "Screen two fixed AttentionUnet roots before allowing local HPO.",
                        "Compare external-CSF and grey-matter Dice, not macro Dice alone.",
                    ),
                    tags=("architecture.attention", "error.external_csf"),
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                title="UNETR: Transformers for 3D Medical Image Segmentation",
                authors=("Ali Hatamizadeh", "Yucheng Tang", "Vishwesh Nath"),
                reference_identity="arxiv:2103.10504",
                uri="https://arxiv.org/abs/2103.10504",
                publication_date=date(2021, 3, 18),
            ),
            findings=(
                _finding(
                    key="architecture.unetr_global_context",
                    claim=(
                        "A transformer encoder can provide global 3D context while "
                        "U-shaped skip connections recover multiscale detail."
                    ),
                    category=EvidenceCategory.UNDEREXPLORED_HYPOTHESIS,
                    method="UNETR transformer encoder with convolutional decoder.",
                    population="Multi-organ CT and brain-tumour MRI benchmarks.",
                    score=0.55,
                    rationale=(
                        "Global context is relevant, but data scale and reconstruction "
                        "shift make direct FeTA transfer uncertain."
                    ),
                    limitations=(
                        "The source does not evaluate fetal MRI.",
                        "Transformer data efficiency and runtime may be poor for this fold size.",
                    ),
                    implications=(
                        "Treat UNETR as a fixed feasibility pilot, not an evolutionary family.",
                        "Require measured CUDA memory and throughput before promotion eligibility.",
                    ),
                    tags=("architecture.transformer", "context.global"),
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                title="Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images",
                authors=("Ali Hatamizadeh", "Vishwesh Nath", "Yucheng Tang"),
                reference_identity="arxiv:2201.01266",
                uri="https://arxiv.org/abs/2201.01266",
                publication_date=date(2022, 1, 4),
            ),
            findings=(
                _finding(
                    key="architecture.swin_hierarchical_context",
                    claim=(
                        "Shifted-window attention offers hierarchical local-to-global "
                        "context for 3D MRI segmentation."
                    ),
                    category=EvidenceCategory.UNDEREXPLORED_HYPOTHESIS,
                    method="Hierarchical Swin-transformer encoder with U-shaped decoder.",
                    population="Adult brain-tumour MRI benchmarks.",
                    score=0.6,
                    rationale=(
                        "The MRI modality is closer than CT, but anatomy, labels, data "
                        "volume and pathology differ materially from FeTA."
                    ),
                    limitations=(
                        "The source is not a fetal-brain study.",
                        "Reported benefits may depend on pretraining unavailable to the campaign.",
                    ),
                    implications=(
                        "Run one fixed SwinUNETR root with no external pretraining.",
                        "Do not spend promotion budget unless the early curve is competitive.",
                    ),
                    tags=("architecture.transformer", "architecture.swin"),
                ),
            ),
        ),
        RetrievedSourceMaterial(
            source=_source(
                title="BOHB: Robust and Efficient Hyperparameter Optimization at Scale",
                authors=("Stefan Falkner", "Aaron Klein", "Frank Hutter"),
                reference_identity="pmlr:v80:falkner18a",
                uri="https://proceedings.mlr.press/v80/falkner18a.html",
                publication_date=date(2018, 7, 3),
            ),
            findings=(
                _finding(
                    key="search.multifidelity_model_based_allocation",
                    claim=(
                        "Model-based proposal and bandit-style resource allocation can "
                        "combine broad screening with selective high-fidelity investment."
                    ),
                    category=EvidenceCategory.ESTABLISHED_CHOICE,
                    method="BOHB combines Bayesian optimisation with Hyperband.",
                    population="General hyperparameter-optimisation benchmarks.",
                    score=0.78,
                    rationale=(
                        "The allocation principle matches V9, while the campaign keeps "
                        "its own lineage, resume and evidence constraints."
                    ),
                    limitations=(
                        "Generic benchmark efficiency does not guarantee reliable early Dice ranking.",
                        "V8 observed only two 150-epoch trajectories, so ranking calibration is sparse.",
                    ),
                    implications=(
                        "Use a denser 15-to-30-to-50 fidelity ladder and preserve graduation reserve.",
                        "Let the Planner compile exact allocations; the Director supplies strategy only.",
                    ),
                    tags=("search.multifidelity", "search.allocation"),
                ),
            ),
        ),
    )


def build_v9_knowledge_library():
    context = ResearchProgrammeContext(
        task_id=V9_PROGRAMME_ID,
        domains=("fetal-mri-segmentation",),
        dataset_ids=("FeTA-development-fold-0",),
        as_of_date=V9_EVIDENCE_CUTOFF,
    )
    cards = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(reviewed_v9_materials()),
        context,
        synthesised_at=V9_RETRIEVED_AT,
    ).evidence_cards
    return build_research_director_knowledge_library(
        cards,
        programme_id=V9_PROGRAMME_ID,
        content_version=V9_KNOWLEDGE_CONTENT_VERSION,
        evidence_cutoff=V9_EVIDENCE_CUTOFF,
        maximum_cards=8,
    )


class _ReviewedCandidateProvider:
    def search(self, request: LiteratureScoutRequest):
        source_types = {
            SourceType.PREPRINT: LiteratureSourceType.PREPRINT,
            SourceType.PEER_REVIEWED_PAPER: LiteratureSourceType.PEER_REVIEWED,
        }
        return tuple(
            LiteratureEvidenceCandidate(
                source_identifier=item.source.reference_identity,
                title=item.source.title,
                source_type=source_types[item.source.source_type],
                uri=str(item.source.uri),
                publication_date=item.source.publication_or_update_date,
                retrieved_at=item.source.retrieved_at,
                claim=item.findings[0].claim,
                stance=item.findings[0].stance.value,
                relevance=item.findings[0].applicability_assessments[0].score,
                applicability=item.findings[0].applicability_assessments[0].rationale,
                limitations=item.findings[0].limitations,
            )
            for item in reviewed_v9_materials()
        )


def build_v9_literature_brief(*, mode: LiteratureScoutMode):
    request = LiteratureScoutRequest(
        programme_id=V9_PROGRAMME_ID,
        trigger="v8_postmortem_and_v9_architecture_selection",
        questions=(
            "Which bounded attention mechanism could improve tissue-specific boundaries?",
            "Which transformer family merits a no-pretraining feasibility pilot?",
            "How should scarce high-fidelity epochs be allocated after weak early ranking?",
        ),
        task_context=(
            "Fold-0 3D fetal-brain MRI segmentation; macro Dice remains the objective; "
            "external CSF and grey matter are the weakest V8 tissues."
        ),
        evidence_cutoff=V9_EVIDENCE_CUTOFF,
        mode=mode,
    )
    policy = LiteratureScoutPolicy(
        policy_id=V9_SCOUT_POLICY_ID,
        maximum_questions=3,
        maximum_sources=6,
        maximum_claim_characters=1_000,
        allowed_source_types=frozenset(
            {LiteratureSourceType.PEER_REVIEWED, LiteratureSourceType.PREPRINT}
        ),
    )
    return build_literature_scout_brief(request, policy, _ReviewedCandidateProvider())


def v9_director_evidence():
    """Return the two immutable advisory inputs bound into V9."""

    return (
        knowledge_library_as_landscape_evidence(build_v9_knowledge_library()),
        literature_brief_as_landscape_evidence(
            build_v9_literature_brief(mode=LiteratureScoutMode.LIVE)
        ),
    )


__all__ = [
    "V9_EVIDENCE_CUTOFF",
    "V9_KNOWLEDGE_CONTENT_VERSION",
    "V9_PROGRAMME_ID",
    "V9_SCOUT_POLICY_ID",
    "build_v9_knowledge_library",
    "build_v9_literature_brief",
    "reviewed_v9_materials",
    "v9_director_evidence",
]
