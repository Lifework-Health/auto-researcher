from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from auto_researcher.contracts.models import EvaluationResult
from auto_researcher.research_intelligence.models import (
    EXTERNAL_EVIDENCE_BOUNDARY,
    ResearchProgrammeContext,
    SourceCandidate,
    SourceRecord,
    SourceType,
    materialise_source,
)
from tests.research_intelligence_fixtures import feta_nnunet_corpus


def test_all_required_source_types_are_explicit_and_identity_is_stable():
    corpus = feta_nnunet_corpus()
    assert {item.source.source_type for item in corpus} == set(SourceType)
    first = materialise_source(corpus[0].source)
    second = materialise_source(corpus[0].source)
    assert first == second
    assert first.evidence_boundary == EXTERNAL_EVIDENCE_BOUNDARY
    assert SourceRecord.model_validate_json(first.model_dump_json()) == first


def test_source_identity_tampering_and_future_publication_fail_closed():
    source = materialise_source(feta_nnunet_corpus()[0].source)
    with pytest.raises(ValidationError, match="identity_mismatch"):
        SourceRecord.model_validate(
            {**source.model_dump(mode="python"), "source_content_hash": "0" * 64}
        )
    with pytest.raises(ValidationError, match="publication date"):
        SourceCandidate.model_validate(
            {
                **feta_nnunet_corpus()[0].source.model_dump(mode="python"),
                "publication_or_update_date": date(2027, 1, 1),
            }
        )


def test_unverified_source_cannot_claim_arbitrarily_high_quality():
    candidate = feta_nnunet_corpus()[0].source
    with pytest.raises(ValidationError, match="trust-classification cap"):
        SourceCandidate.model_validate(
            {
                **candidate.model_dump(mode="python"),
                "trust_classification": "UNVERIFIED",
                "quality_score": 0.9,
            }
        )


def test_source_type_is_not_a_universal_quality_hierarchy():
    candidate = feta_nnunet_corpus()[0].source
    official = SourceCandidate.model_validate(
        {**candidate.model_dump(mode="python"), "source_type": "OFFICIAL_DOCUMENTATION"}
    )
    paper = SourceCandidate.model_validate(
        {**candidate.model_dump(mode="python"), "source_type": "PEER_REVIEWED_PAPER"}
    )
    assert official.quality_score == paper.quality_score
    assert official.quality_assessment_basis == paper.quality_assessment_basis


def test_external_research_cannot_be_relabelled_as_measured_evaluation_evidence():
    source = materialise_source(feta_nnunet_corpus()[0].source)
    assert "primary_score" not in SourceRecord.model_fields
    assert source.evidence_boundary == "EXTERNAL_RESEARCH_INTELLIGENCE"
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(source.model_dump(mode="python"))


def test_programme_context_rejects_duplicate_scope_values():
    with pytest.raises(ValidationError, match="unique"):
        ResearchProgrammeContext(
            task_id="task@1.0",
            domains=("domain", "domain"),
            as_of_date=date(2026, 8, 13),
        )
