from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from auto_researcher.research_intelligence import (
    LiteratureEvidenceCandidate,
    LiteratureScoutMode,
    LiteratureScoutPolicy,
    LiteratureScoutRequest,
    LiteratureSourceType,
    build_literature_scout_brief,
    evaluate_literature_scout_shadow,
    literature_brief_as_landscape_evidence,
)

NOW = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)


class FakeProvider:
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        self.calls = 0

    def search(self, _request):
        self.calls += 1
        return self.candidates


def _request(mode=LiteratureScoutMode.SHADOW):
    return LiteratureScoutRequest(
        programme_id="feta-v9",
        trigger="post_v8_postmortem",
        questions=("Which mechanisms address topology errors?",),
        task_context=(
            "Fold-0 fetal MRI segmentation with a locked macro-Dice objective."
        ),
        evidence_cutoff=date(2026, 8, 23),
        mode=mode,
    )


def _policy(maximum_sources=4):
    return LiteratureScoutPolicy(
        policy_id="v9-literature-shadow-v1",
        maximum_questions=2,
        maximum_sources=maximum_sources,
        allowed_source_types=frozenset(LiteratureSourceType),
    )


def _candidate(identifier="doi:10.0000/example"):
    return LiteratureEvidenceCandidate(
        source_identifier=identifier,
        title="A bounded fixture paper",
        source_type=LiteratureSourceType.PEER_REVIEWED,
        uri=identifier,
        publication_date=date(2025, 1, 1),
        retrieved_at=NOW,
        claim="Topology-aware objectives may reduce disconnected predictions.",
        stance="SUPPORTS",
        relevance=0.83,
        applicability=(
            "Comparable 3D multi-class segmentation, but not the locked FeTA split."
        ),
        limitations=("Different dataset", "No direct evidence on this campaign"),
    )


def test_shadow_brief_is_cited_deterministic_and_has_no_action_authority():
    provider = FakeProvider((_candidate(), _candidate()))
    first = build_literature_scout_brief(_request(), _policy(), provider)
    second = build_literature_scout_brief(_request(), _policy(), provider)

    assert first == second
    assert len(first.evidence) == 1
    assert first.evidence[0].source_identifier == "doi:10.0000/example"
    assert first.evidence[0].trust_boundary == "UNTRUSTED_EXTERNAL_EVIDENCE"
    report = evaluate_literature_scout_shadow(first)
    assert report.experiment_authority_exercised is False
    assert report.evidence_reference_ids == (first.evidence[0].evidence_id,)
    assert provider.calls == 2


def test_shadow_brief_cannot_enter_live_research_landscape():
    brief = build_literature_scout_brief(
        _request(LiteratureScoutMode.SHADOW), _policy(), FakeProvider((_candidate(),))
    )
    with pytest.raises(ValueError, match="literature_scout_live_mode_required"):
        literature_brief_as_landscape_evidence(brief)


def test_reviewed_live_brief_becomes_separately_typed_untrusted_evidence():
    brief = build_literature_scout_brief(
        _request(LiteratureScoutMode.LIVE), _policy(), FakeProvider((_candidate(),))
    )
    evidence = literature_brief_as_landscape_evidence(brief)

    assert evidence.evidence_type == "LITERATURE"
    assert evidence.evidence_hash == brief.brief_hash
    assert evidence.reference_ids == (brief.evidence[0].evidence_id,)
    assert evidence.safe_payload["items"][0]["trust_boundary"] == (
        "UNTRUSTED_EXTERNAL_EVIDENCE"
    )


def test_source_and_question_budgets_fail_closed():
    request = _request().model_copy(
        update={"questions": ("Question one?", "Question two?", "Question three?")}
    )
    with pytest.raises(ValueError, match="literature_scout_question_budget_exceeded"):
        build_literature_scout_brief(request, _policy(), FakeProvider(()))

    with pytest.raises(ValueError, match="literature_scout_source_budget_exceeded"):
        build_literature_scout_brief(
            _request(),
            _policy(maximum_sources=1),
            FakeProvider((_candidate(), _candidate("doi:10.0000/other"))),
        )


def test_untraceable_uri_is_rejected_before_director_context():
    invalid = _candidate().model_copy(update={"uri": "file:///tmp/untrusted.txt"})
    with pytest.raises(ValueError, match="literature_scout_source_uri_invalid"):
        build_literature_scout_brief(_request(), _policy(), FakeProvider((invalid,)))
