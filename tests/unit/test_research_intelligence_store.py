from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from auto_researcher.research_intelligence.brief import DeterministicBriefBuilder
from auto_researcher.research_intelligence.models import (
    BriefSection,
    EvidenceQuery,
    ResearchProgrammeContext,
    materialise_source,
)
from auto_researcher.research_intelligence.scout import OfflineResearchScout
from auto_researcher.research_intelligence.store import SQLiteEvidenceStore
from auto_researcher.research_intelligence.synthesis import (
    DeterministicEvidenceSynthesiser,
)
from tests.research_intelligence_fixtures import feta_nnunet_corpus

NOW = datetime(2026, 8, 13, 14, tzinfo=UTC)
CONTEXT = ResearchProgrammeContext(
    task_id="feta_seg_search@1.0",
    domains=("fetal-mri-segmentation",),
    dataset_ids=("FeTA",),
    hypothesis_ids=("fetal-mri-segmentation.baseline",),
    as_of_date=date(2026, 8, 13),
)


def _result():
    return DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(feta_nnunet_corpus()), CONTEXT, synthesised_at=NOW
    )


def test_source_evidence_and_relationships_survive_restart(tmp_path):
    path = tmp_path / "research.sqlite3"
    result = _result()
    store = SQLiteEvidenceStore(path)
    refresh = store.store_synthesis(result)
    assert store.store_synthesis(result) == refresh
    store.close()

    reopened = SQLiteEvidenceStore(path)
    assert reopened.latest_refresh() == refresh
    assert reopened.get_source(result.source_records[0].source_id) is not None
    assert reopened.get_evidence(result.evidence_cards[0].evidence_id) is not None
    assert (
        len(reopened.conflicting_evidence(EvidenceQuery(task_id=CONTEXT.task_id))) == 2
    )
    sources = reopened.sources_supporting_claim("spacing.fixed_target")
    assert len(sources) == 1
    reopened.close()


def test_unchanged_source_retrievals_and_no_change_scans_are_distinct_after_restart(
    tmp_path,
):
    path = tmp_path / "research.sqlite3"
    material = feta_nnunet_corpus()[0]
    t1 = NOW
    t2 = NOW + timedelta(hours=1)
    source_t1 = material.source.model_copy(update={"retrieved_at": t1})
    source_t2 = material.source.model_copy(update={"retrieved_at": t2})
    synthesiser = DeterministicEvidenceSynthesiser()
    first = synthesiser.synthesise(
        OfflineResearchScout((material.model_copy(update={"source": source_t1}),)),
        CONTEXT,
        synthesised_at=t1,
    )
    second = synthesiser.synthesise(
        OfflineResearchScout((material.model_copy(update={"source": source_t2}),)),
        CONTEXT,
        synthesised_at=t2,
    )
    assert first.source_records == second.source_records
    assert tuple(card.evidence_id for card in first.evidence_cards) == tuple(
        card.evidence_id for card in second.evidence_cards
    )

    store = SQLiteEvidenceStore(path)
    refresh_t1 = store.store_synthesis(first)
    refresh_t2 = store.store_synthesis(second)
    store.close()

    reopened = SQLiteEvidenceStore(path)
    observations = reopened.source_retrievals(first.source_records[0].source_version_id)
    assert tuple(item.retrieved_at for item in observations) == (t1, t2)
    assert refresh_t1.refresh_id != refresh_t2.refresh_id
    assert refresh_t1.evidence_snapshot_id == refresh_t2.evidence_snapshot_id
    assert refresh_t2.new_source_version_ids == ()
    assert refresh_t2.new_evidence_card_ids == ()
    assert reopened.get_refresh(refresh_t1.refresh_id) == refresh_t1
    assert reopened.get_refresh(refresh_t2.refresh_id) == refresh_t2
    assert reopened.latest_refresh() == refresh_t2
    assert (
        reopened.get_source(first.source_records[0].source_id)
        == first.source_records[0]
    )
    reopened.close()


def test_new_source_version_creates_refresh_without_overwriting_history(tmp_path):
    path = tmp_path / "research.sqlite3"
    store = SQLiteEvidenceStore(path)
    original = _result()
    first_refresh = store.store_synthesis(original)
    material = feta_nnunet_corpus()[0]
    revised_source = material.source.model_copy(
        update={
            "title": "Fixture nnU-Net documentation, revised",
            "source_version": "fixture-v2",
            "retrieved_at": NOW + timedelta(hours=1),
        }
    )
    revised = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout((material.model_copy(update={"source": revised_source}),)),
        CONTEXT,
        synthesised_at=NOW + timedelta(hours=1),
    )
    second_refresh = store.store_synthesis(revised)
    assert second_refresh.refresh_id != first_refresh.refresh_id
    stable_id = revised.source_records[0].source_id
    original_source = materialise_source(material.source)
    assert stable_id == original_source.source_id
    assert store.get_source(stable_id) == revised.source_records[0]
    assert (
        store.get_source(stable_id, original_source.source_version_id)
        == original_source
    )
    store.close()


def test_context_baseline_failure_hypothesis_and_refresh_queries(tmp_path):
    store = SQLiteEvidenceStore(tmp_path / "research.sqlite3")
    result = _result()
    store.store_synthesis(result)
    query = EvidenceQuery(
        task_id=CONTEXT.task_id,
        domain="fetal-mri-segmentation",
        dataset_id="FeTA",
    )
    assert {card.claim_key for card in store.strongest_known_baselines(query)} == {
        "baseline.residual_encoder",
        "baseline.self_configuring_unet",
    }
    assert {card.claim_key for card in store.known_failure_modes(query)} == {
        "domain_shift.reconstruction",
        "failure.tissue_specific",
    }
    assert store.query_evidence(
        query.model_copy(update={"hypothesis_id": "fetal-mri-segmentation.baseline"})
    )
    assert store.evidence_newer_than(NOW - timedelta(seconds=1))
    assert store.evidence_newer_than(NOW + timedelta(seconds=1)) == ()
    store.close()


def test_incremental_refresh_updates_derived_conflict_links_without_changing_identity(
    tmp_path,
):
    corpus = feta_nnunet_corpus()
    supporting_source = next(
        item
        for item in corpus
        if item.source.reference_identity == "fixture:feta-challenge"
    )
    supporting_finding = next(
        item
        for item in supporting_source.findings
        if item.claim_key == "spacing.fixed_target"
    )
    supporting_only = supporting_source.model_copy(
        update={"findings": (supporting_finding,)}
    )
    contradicting_source = next(
        item
        for item in corpus
        if item.source.reference_identity == "fixture:feta-preprint"
    )
    store = SQLiteEvidenceStore(tmp_path / "research.sqlite3")
    initial = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout((supporting_only,)), CONTEXT, synthesised_at=NOW
    )
    store.store_synthesis(initial)
    initial_spacing = next(
        card
        for card in initial.evidence_cards
        if card.claim_key == "spacing.fixed_target"
    )
    assert initial_spacing.conflicting_evidence_ids == ()

    incremental = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout((contradicting_source,)),
        CONTEXT,
        synthesised_at=NOW + timedelta(hours=1),
    )
    store.store_synthesis(incremental)
    conflicts = store.conflicting_evidence(EvidenceQuery(task_id=CONTEXT.task_id))
    assert {card.evidence_id for card in conflicts} == {
        initial_spacing.evidence_id,
        incremental.evidence_cards[0].evidence_id,
    }
    assert all(len(card.conflicting_evidence_ids) == 1 for card in conflicts)
    brief = DeterministicBriefBuilder().build(
        store, CONTEXT, generated_at=NOW + timedelta(hours=1)
    )
    assert len(brief.unresolved_uncertainties) == 1
    assert set(brief.unresolved_uncertainties[0].evidence_card_ids) == {
        card.evidence_id for card in conflicts
    }
    store.close()


def test_brief_is_a_traceable_view_and_preserves_unresolved_conflicts(tmp_path):
    store = SQLiteEvidenceStore(tmp_path / "research.sqlite3")
    result = _result()
    store.store_synthesis(result)
    brief = DeterministicBriefBuilder().build(store, CONTEXT, generated_at=NOW)
    known_ids = {card.evidence_id for card in result.evidence_cards}
    sections = (
        brief.actionable_priors,
        brief.known_strong_baselines,
        brief.established_choices,
        brief.likely_failure_modes,
        brief.underexplored_hypotheses,
        brief.unresolved_uncertainties,
        brief.experiment_design_implications,
    )
    entries = tuple(entry for section in sections for entry in section)
    assert entries
    assert all(set(entry.evidence_card_ids) <= known_ids for entry in entries)
    assert (
        brief.unresolved_uncertainties[0].section == BriefSection.UNRESOLVED_CONFLICTS
    )
    assert len(brief.unresolved_uncertainties[0].evidence_card_ids) == 2
    assert brief.evidence_boundary == "EXTERNAL_RESEARCH_INTELLIGENCE"
    store.close()
