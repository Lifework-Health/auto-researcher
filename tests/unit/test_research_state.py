from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from auto_researcher.research_intelligence.brief import DeterministicBriefBuilder
from auto_researcher.research_intelligence.models import ResearchProgrammeContext
from auto_researcher.research_intelligence.scout import OfflineResearchScout
from auto_researcher.research_intelligence.synthesis import (
    DeterministicEvidenceSynthesiser,
)
from auto_researcher.research_intelligence.store import SQLiteEvidenceStore
from auto_researcher.research_state import (
    CandidateNextAction,
    ConfidenceAssessment,
    ConfidenceLevel,
    DiagnosticObservation,
    EvidenceReferences,
    ExperimentIntent,
    ExperimentModality,
    ExperimentStatus,
    ExternalEvidenceReference,
    Fidelity,
    HypothesisOrigin,
    HypothesisStatus,
    InformationValueClass,
    InternalExperimentalObservation,
    ObservationRole,
    PlannerDecision,
    PlannerInference,
    ProgrammeContext,
    RecordType,
    ResearchExperiment,
    ResearchHypothesis,
    ResearchObjective,
    ResearchProgramme,
    ResearchStateQueries,
    ResearchUncertainty,
    ResourceImplication,
    SQLiteResearchStateStore,
    StructuredResultReference,
    UncertaintyStatus,
    WorkStatus,
    external_evidence_reference,
    external_evidence_references_from_brief,
    research_programme_identity,
)
from tests.research_intelligence_fixtures import feta_nnunet_corpus, tabular_corpus

NOW = datetime(2026, 8, 13, 15, tzinfo=UTC)


def _cards(*, tabular: bool = False):
    if tabular:
        context = ResearchProgrammeContext(
            task_id="credit_risk_fixture@1.0",
            domains=("imbalanced-tabular-classification",),
            as_of_date=date(2026, 8, 13),
        )
        corpus = tabular_corpus()
    else:
        context = ResearchProgrammeContext(
            task_id="feta_seg_search@1.0",
            domains=("fetal-mri-segmentation",),
            dataset_ids=("FeTA",),
            as_of_date=date(2026, 8, 13),
        )
        corpus = feta_nnunet_corpus()
    return (
        DeterministicEvidenceSynthesiser()
        .synthesise(OfflineResearchScout(corpus), context, synthesised_at=NOW)
        .evidence_cards
    )


def _programme(*, tabular: bool = False) -> ResearchProgramme:
    name = "credit robustness" if tabular else "FeTA robustness"
    task_id = "credit_risk_fixture" if tabular else "feta_seg_search"
    domains = (
        ("imbalanced-tabular-classification",)
        if tabular
        else ("fetal-mri-segmentation",)
    )
    architectures = ("xgboost-config:v1",) if tabular else ("nnunet-plan:v3",)
    return ResearchProgramme(
        programme_id=research_programme_identity(name, "1"),
        programme_version="1",
        name=name,
        objectives=(
            ResearchObjective(
                objective_id="robustness",
                statement="Improve robust validation performance.",
                metric_ids=("primary_score",),
                success_criteria=("replicated improvement",),
            ),
        ),
        context=ProgrammeContext(
            task_id=task_id,
            task_version="1.0",
            domains=domains,
            architecture_references=architectures,
            dataset_references=("dataset:v1",),
            evaluator_references=("evaluator:v2",),
        ),
        created_at=NOW,
    )


def _external(programme, card, *, offset: int = 0):
    return external_evidence_reference(
        programme.programme_id,
        card,
        evidence_store_reference="sqlite:///research-intelligence.sqlite3",
        recorded_at=NOW + timedelta(minutes=offset),
    )


def _observation(programme, *, observation_id="obs-primary", value=0.84):
    return InternalExperimentalObservation(
        programme_id=programme.programme_id,
        observation_id=observation_id,
        experiment_id="experiment-augmentation",
        candidate_id="candidate-bias-field",
        result_id=f"result-{observation_id}",
        metric_id="dice.mean",
        measured_value=value,
        fidelity=Fidelity(level="full", dataset_fraction=1.0, replicate_index=0),
        dataset_reference="FeTA:v1",
        split_reference="folds:v2",
        evaluator_reference="feta-evaluator",
        evaluator_version="2.0",
        observed_at=NOW,
        provenance_reference=f"provenance:{observation_id}",
        observation_role=ObservationRole.PRIMARY,
        recorded_at=NOW,
    )


def _hypotheses(programme, card_id, observation_id="obs-primary"):
    motivation = EvidenceReferences(external_evidence_card_ids=(card_id,))
    confidence = ConfidenceAssessment(
        level=ConfidenceLevel.LOW,
        rationale="Plausible but not yet discriminated in this task.",
    )
    left = ResearchHypothesis(
        programme_id=programme.programme_id,
        hypothesis_id="hyp-augmentation",
        proposition="Bias-field augmentation improves reconstruction robustness.",
        status=HypothesisStatus.UNRESOLVED,
        origin=HypothesisOrigin.EXTERNAL_EVIDENCE,
        motivation="External work identifies a plausible robustness prior.",
        motivating_evidence=motivation,
        competing_hypothesis_ids=("hyp-schedule",),
        confidence=confidence,
        proposed_discriminating_experiment="Ablate augmentation under fixed scheduling.",
        recorded_at=NOW,
    )
    right = ResearchHypothesis(
        programme_id=programme.programme_id,
        hypothesis_id="hyp-schedule",
        proposition="The apparent gain is caused by scheduling, not augmentation.",
        status=HypothesisStatus.UNRESOLVED,
        origin=HypothesisOrigin.HUMAN,
        motivation="The candidate changed both factors.",
        motivating_evidence=motivation,
        competing_hypothesis_ids=("hyp-augmentation",),
        confidence=confidence,
        proposed_discriminating_question="Does the gain remain under constant LR?",
        recorded_at=NOW,
    )
    supported = left.model_copy(
        update={
            "revision": 2,
            "status": HypothesisStatus.SUPPORTED,
            "supporting_evidence": EvidenceReferences(
                internal_observation_ids=(observation_id,)
            ),
            "confidence": ConfidenceAssessment(
                level=ConfidenceLevel.MODERATE,
                rationale="One full-fidelity primary observation supports it.",
            ),
            "recorded_at": NOW + timedelta(minutes=2),
        }
    )
    refuted = right.model_copy(
        update={
            "revision": 2,
            "status": HypothesisStatus.REFUTED,
            "refuting_evidence": EvidenceReferences(
                internal_observation_ids=(observation_id,)
            ),
            "recorded_at": NOW + timedelta(minutes=2),
        }
    )
    return left, right, supported, refuted


def _experiment(programme, card_id, *, revision=1, status=ExperimentStatus.PROPOSED):
    return ResearchExperiment(
        programme_id=programme.programme_id,
        revision=revision,
        experiment_id="experiment-augmentation",
        candidate_ids=("candidate-bias-field",),
        experiment_spec_reference="experiment-spec:augmentation-v1",
        intent=ExperimentIntent(
            research_question="Does augmentation improve robustness?",
            hypothesis_ids=("hyp-augmentation",),
            expected_learning="Separate augmentation effects from schedule effects.",
            modality=ExperimentModality.ABLATION,
            motivated_by_evidence=EvidenceReferences(
                external_evidence_card_ids=(card_id,)
            ),
            information_value_class=InformationValueClass.DISCRIMINATING,
            stop_criterion="Stop if both matched candidates underperform baseline.",
        ),
        status=status,
        observation_ids=("obs-primary",)
        if status == ExperimentStatus.COMPLETED
        else (),
        started_at=NOW if status != ExperimentStatus.PROPOSED else None,
        completed_at=NOW + timedelta(hours=1)
        if status == ExperimentStatus.COMPLETED
        else None,
        recorded_at=NOW + timedelta(minutes=revision),
    )


def _inference(programme, card_id):
    return PlannerInference(
        programme_id=programme.programme_id,
        inference_id="inference-augmentation",
        interpretation="The matched candidate supports augmentation over scheduling.",
        derived_from_evidence=EvidenceReferences(
            external_evidence_card_ids=(card_id,),
            internal_observation_ids=("obs-primary",),
        ),
        hypothesis_ids=("hyp-augmentation", "hyp-schedule"),
        inference_version="1",
        recorded_at=NOW + timedelta(hours=2),
    )


def _decision(programme, card_id):
    return PlannerDecision(
        programme_id=programme.programme_id,
        decision_id="decision-replicate",
        action="Run a second-seed replication of the augmentation candidate.",
        rationale="The full-fidelity gain is useful but has only one replicate.",
        supporting_evidence=EvidenceReferences(
            external_evidence_card_ids=(card_id,),
            internal_observation_ids=("obs-primary",),
        ),
        inference_ids=("inference-augmentation",),
        hypothesis_ids=("hyp-augmentation",),
        experiment_ids=("experiment-augmentation",),
        alternatives_considered=("Adopt immediately", "Discard the candidate"),
        resource_implications=(
            ResourceImplication(
                resource="GPU time",
                implication="One full-fidelity replication.",
                amount=8,
                unit="GPU-hours",
            ),
        ),
        decision_version="1",
        decided_at=NOW + timedelta(hours=2),
        recorded_at=NOW + timedelta(hours=2),
    )


def test_external_evidence_enters_by_reference_without_type_conversion():
    programme = _programme()
    card = _cards()[0]
    reference = _external(programme, card)
    assert reference.evidence_card_id == card.evidence_id
    assert reference.evidence_boundary == "EXTERNAL_RESEARCH_INTELLIGENCE"
    assert "claim" not in type(reference).model_fields
    with pytest.raises(ValidationError):
        InternalExperimentalObservation.model_validate(reference.model_dump())


def test_retrieval_and_refresh_events_cannot_enter_as_scientific_evidence(tmp_path):
    context = ResearchProgrammeContext(
        task_id="feta_seg_search@1.0",
        domains=("fetal-mri-segmentation",),
        dataset_ids=("FeTA",),
        as_of_date=date(2026, 8, 13),
    )
    result = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(feta_nnunet_corpus()), context, synthesised_at=NOW
    )
    evidence_store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    refresh = evidence_store.store_synthesis(result)

    with pytest.raises(ValidationError):
        ExternalEvidenceReference.model_validate(
            result.source_retrievals[0].model_dump()
        )
    with pytest.raises(ValidationError):
        ExternalEvidenceReference.model_validate(refresh.model_dump())
    assert "retrieval_id" not in ExternalEvidenceReference.model_fields
    assert "refresh_id" not in ExternalEvidenceReference.model_fields
    evidence_store.close()


def test_brief_is_resolved_to_cards_instead_of_persisting_brief_prose(tmp_path):
    programme = _programme()
    context = ResearchProgrammeContext(
        task_id="feta_seg_search@1.0",
        domains=("fetal-mri-segmentation",),
        dataset_ids=("FeTA",),
        as_of_date=date(2026, 8, 13),
    )
    result = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(feta_nnunet_corpus()), context, synthesised_at=NOW
    )
    evidence_store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    evidence_store.store_synthesis(result)
    brief = DeterministicBriefBuilder().build(evidence_store, context, generated_at=NOW)
    references = external_evidence_references_from_brief(
        programme.programme_id,
        brief,
        evidence_store,
        evidence_store_reference="sqlite:///evidence.sqlite3",
        recorded_at=NOW,
    )
    brief_ids = {
        evidence_id
        for section in (
            brief.actionable_priors,
            brief.known_strong_baselines,
            brief.established_choices,
            brief.likely_failure_modes,
            brief.underexplored_hypotheses,
            brief.unresolved_uncertainties,
            brief.experiment_design_implications,
        )
        for entry in section
        for evidence_id in entry.evidence_card_ids
    }
    assert {item.evidence_card_id for item in references} == brief_ids
    assert all("statement" not in type(item).model_fields for item in references)
    evidence_store.close()


def test_incremental_conflict_reconciliation_preserves_state_lineage_identity(
    tmp_path,
):
    programme = _programme()
    context = ResearchProgrammeContext(
        task_id="feta_seg_search@1.0",
        domains=("fetal-mri-segmentation",),
        dataset_ids=("FeTA",),
        as_of_date=date(2026, 8, 13),
    )
    corpus = feta_nnunet_corpus()
    initial_corpus = tuple(
        item
        for item in corpus
        if item.source.reference_identity != "fixture:feta-preprint"
    )
    initial = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(initial_corpus), context, synthesised_at=NOW
    )
    evidence_store = SQLiteEvidenceStore(tmp_path / "evidence.sqlite3")
    evidence_store.store_synthesis(initial)
    initial_card = next(
        card
        for card in initial.evidence_cards
        if card.claim_key == "spacing.fixed_target"
    )
    initial_reference = _external(programme, initial_card)
    hypothesis = ResearchHypothesis(
        programme_id=programme.programme_id,
        hypothesis_id="hyp-fixed-spacing",
        proposition="A fixed target spacing transfers to the current task.",
        status=HypothesisStatus.UNRESOLVED,
        origin=HypothesisOrigin.EXTERNAL_EVIDENCE,
        motivation="The initial external evidence supports testing fixed spacing.",
        motivating_evidence=EvidenceReferences(
            external_evidence_card_ids=(initial_card.evidence_id,)
        ),
        confidence=ConfidenceAssessment(
            level=ConfidenceLevel.LOW,
            rationale="External evidence has not been tested on this programme.",
        ),
        recorded_at=NOW,
    )
    state_store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    state_store.create_programme(programme)
    state_store.append_many((initial_reference, hypothesis))

    refreshed = DeterministicEvidenceSynthesiser().synthesise(
        OfflineResearchScout(corpus),
        context,
        synthesised_at=NOW + timedelta(hours=1),
    )
    evidence_store.store_synthesis(refreshed)
    reconciled_card = evidence_store.get_evidence(initial_card.evidence_id)
    assert reconciled_card is not None
    assert reconciled_card.conflicting_evidence_ids
    assert reconciled_card.evidence_id == initial_card.evidence_id
    assert reconciled_card.evidence_content_hash == initial_card.evidence_content_hash

    reconciled_reference = _external(programme, reconciled_card)
    assert reconciled_reference.evidence_card_id == initial_reference.evidence_card_id
    assert (
        reconciled_reference.evidence_content_hash
        == initial_reference.evidence_content_hash
    )
    assert "supporting_evidence_ids" not in ExternalEvidenceReference.model_fields
    assert "conflicting_evidence_ids" not in ExternalEvidenceReference.model_fields
    lineage = ResearchStateQueries(
        state_store.load_state(programme.programme_id)
    ).what_evidence_supports_conclusion(RecordType.HYPOTHESIS, hypothesis.hypothesis_id)
    assert lineage.evidence.external_evidence_card_ids == (initial_card.evidence_id,)
    state_store.close()
    evidence_store.close()


def test_internal_measurement_and_planner_interpretation_are_distinct_types():
    programme = _programme()
    observation = _observation(programme)
    assert observation.evidence_boundary == "INTERNAL_EXPERIMENTAL_OBSERVATION"
    with pytest.raises(ValidationError):
        PlannerInference.model_validate(observation.model_dump())
    with pytest.raises(ValidationError):
        InternalExperimentalObservation.model_validate(
            {
                **observation.model_dump(),
                "measured_value": None,
                "structured_result_reference": None,
            }
        )


def test_diagnostic_observation_has_its_own_boundary_and_summary_channel(tmp_path):
    programme = _programme()
    diagnostic = DiagnosticObservation(
        programme_id=programme.programme_id,
        diagnostic_observation_id="diagnostic-class-collapse",
        diagnostic_run_id="diagnostic-run-1",
        diagnostic_kind="class_specific_failure",
        subject_references=("candidate-bias-field",),
        result_reference=StructuredResultReference(
            reference_id="diagnostic-result-1",
            schema_version="1",
            location="artefact://diagnostics/class-collapse.json",
        ),
        finding="The smallest class accounts for most validation failures.",
        diagnostic_system_reference="diagnostic-intelligence",
        diagnostic_system_version="future-v1-contract",
        observed_at=NOW,
        provenance_reference="provenance:diagnostic-run-1",
        recorded_at=NOW,
    )
    store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    store.create_programme(programme)
    store.append(diagnostic)
    learned = ResearchStateQueries(
        store.load_state(programme.programme_id)
    ).what_have_we_learned()
    assert diagnostic.evidence_boundary == "INTERNAL_DIAGNOSTIC_OBSERVATION"
    assert learned.diagnostic_observations == (diagnostic,)
    assert learned.internal_observations == ()
    store.close()


def test_planner_decision_requires_typed_support_and_resource_implication():
    programme = _programme()
    card_id = _cards()[0].evidence_id
    with pytest.raises(ValidationError, match="typed supporting evidence"):
        _decision(programme, card_id).model_copy(
            update={"supporting_evidence": EvidenceReferences()}
        ).__class__.model_validate(
            {
                **_decision(programme, card_id).model_dump(),
                "supporting_evidence": EvidenceReferences().model_dump(),
            }
        )
    with pytest.raises(ValidationError):
        PlannerDecision.model_validate(
            {**_decision(programme, card_id).model_dump(), "rationale": ""}
        )


def test_competing_hypotheses_append_supporting_and_refuting_revisions(tmp_path):
    programme = _programme()
    card = _cards()[0]
    external = _external(programme, card)
    observation = _observation(programme)
    left, right, supported, refuted = _hypotheses(programme, card.evidence_id)
    store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    store.create_programme(programme)
    store.append_many((external, observation, left, right, supported, refuted))
    state = store.load_state(programme.programme_id)
    by_id = {item.hypothesis_id: item for item in state.hypotheses}
    assert by_id["hyp-augmentation"].status == HypothesisStatus.SUPPORTED
    assert by_id["hyp-schedule"].status == HypothesisStatus.REFUTED
    assert (
        len(
            store.record_history(
                programme.programme_id, RecordType.HYPOTHESIS, "hyp-augmentation"
            )
        )
        == 2
    )
    store.close()


def test_unresolved_uncertainty_survives_restart_and_is_queryable(tmp_path):
    programme = _programme()
    card = _cards()[0]
    external = _external(programme, card)
    left, right, *_ = _hypotheses(programme, card.evidence_id)
    uncertainty = ResearchUncertainty(
        programme_id=programme.programme_id,
        uncertainty_id="uncertainty-replication",
        question="Does the gain replicate across seeds?",
        status=UncertaintyStatus.OPEN,
        affects_hypothesis_ids=("hyp-augmentation",),
        affects_evidence=EvidenceReferences(
            external_evidence_card_ids=(card.evidence_id,)
        ),
        recorded_at=NOW,
    )
    path = tmp_path / "state.sqlite3"
    store = SQLiteResearchStateStore(path)
    store.create_programme(programme)
    store.append_many((external, left, right, uncertainty))
    stable_revision_id = (
        store.load_state(programme.programme_id).revision_history[-1].state_revision_id
    )
    store.close()

    reopened = SQLiteResearchStateStore(path)
    state = reopened.load_state(programme.programme_id)
    summary = ResearchStateQueries(state).what_remains_uncertain()
    assert (
        summary.unresolved_uncertainties[0].uncertainty_id == "uncertainty-replication"
    )
    assert state.revision_history[-1].state_revision_id == stable_revision_id
    reopened.close()


def test_evidence_update_does_not_erase_historical_inference_or_decision(tmp_path):
    programme = _programme()
    cards = _cards()
    first, update = cards[0], cards[1]
    external = _external(programme, first)
    observation = _observation(programme)
    left, right, supported, refuted = _hypotheses(programme, first.evidence_id)
    experiment = _experiment(programme, first.evidence_id)
    inference = _inference(programme, first.evidence_id)
    decision = _decision(programme, first.evidence_id)
    store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    store.create_programme(programme)
    store.append_many(
        (
            external,
            observation,
            left,
            right,
            supported,
            refuted,
            experiment,
            inference,
            decision,
        )
    )
    store.append(_external(programme, update, offset=5))
    state = store.load_state(programme.programme_id)
    assert {item.evidence_card_id for item in state.external_evidence} == {
        first.evidence_id,
        update.evidence_id,
    }
    assert state.planner_inferences == (inference,)
    assert state.planner_decisions == (decision,)
    lineage = ResearchStateQueries(state).evidence_influencing_decision(
        decision.decision_id
    )
    assert first.evidence_id in lineage.evidence.external_evidence_card_ids
    assert update.evidence_id not in lineage.evidence.external_evidence_card_ids
    store.close()


def test_learning_experiment_rationale_decision_and_exact_lineage_queries(tmp_path):
    programme = _programme()
    card = _cards()[0]
    external = _external(programme, card)
    observation = _observation(programme)
    left, right, supported, refuted = _hypotheses(programme, card.evidence_id)
    experiment = _experiment(programme, card.evidence_id)
    inference = _inference(programme, card.evidence_id)
    decision = _decision(programme, card.evidence_id)
    next_action = CandidateNextAction(
        programme_id=programme.programme_id,
        next_action_id="next-ablation",
        action="Run the fixed-schedule augmentation ablation.",
        rationale="This directly discriminates the alternatives.",
        status=WorkStatus.CANDIDATE,
        hypothesis_ids=("hyp-augmentation", "hyp-schedule"),
        motivated_by_evidence=EvidenceReferences(
            internal_observation_ids=(observation.observation_id,)
        ),
        expected_information_value=InformationValueClass.DISCRIMINATING,
        recorded_at=NOW,
    )
    store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    store.create_programme(programme)
    store.append_many(
        (
            external,
            observation,
            left,
            right,
            supported,
            refuted,
            experiment,
            inference,
            decision,
            next_action,
        )
    )
    queries = ResearchStateQueries(store.load_state(programme.programme_id))
    learned = queries.what_have_we_learned()
    assert learned.internal_observations == (observation,)
    assert learned.supported_hypotheses == (supported,)
    rationale = queries.why_was_experiment_run(experiment.experiment_id)
    assert rationale.hypothesis_ids == ("hyp-augmentation",)
    assert rationale.informed_decision_ids == (decision.decision_id,)
    assert queries.decision_informed_by_experiment(experiment.experiment_id) == (
        decision,
    )
    lineage = queries.what_evidence_supports_conclusion(
        RecordType.PLANNER_DECISION, decision.decision_id
    )
    assert lineage.inference_ids == (inference.inference_id,)
    assert lineage.hypothesis_ids == ("hyp-augmentation", "hyp-schedule")
    assert lineage.evidence.external_evidence_card_ids == (card.evidence_id,)
    assert lineage.evidence.internal_observation_ids == (observation.observation_id,)
    discriminators = queries.most_direct_discriminating_actions()
    assert {item.source_id for item in discriminators} == {
        "hyp-augmentation",
        "hyp-schedule",
        "next-ablation",
    }
    store.close()


def test_active_and_completed_experiment_work_is_reconstructed(tmp_path):
    programme = _programme()
    card = _cards()[0]
    external = _external(programme, card)
    observation = _observation(programme)
    left, right, *_ = _hypotheses(programme, card.evidence_id)
    proposed = _experiment(programme, card.evidence_id)
    active = _experiment(
        programme,
        card.evidence_id,
        revision=2,
        status=ExperimentStatus.ACTIVE,
    )
    completed = _experiment(
        programme,
        card.evidence_id,
        revision=3,
        status=ExperimentStatus.COMPLETED,
    )
    store = SQLiteResearchStateStore(tmp_path / "state.sqlite3")
    store.create_programme(programme)
    store.append_many((external, observation, left, right, proposed, active))
    assert store.active_work(programme.programme_id) == (active,)
    store.append(completed)
    assert store.active_work(programme.programme_id) == ()
    assert store.completed_work(programme.programme_id) == (completed,)
    store.close()


@pytest.mark.parametrize("tabular", [False, True], ids=["imaging", "non-imaging"])
def test_same_state_contract_works_across_task_domains(tmp_path, tabular):
    programme = _programme(tabular=tabular)
    card = _cards(tabular=tabular)[0]
    store = SQLiteResearchStateStore(tmp_path / f"state-{tabular}.sqlite3")
    store.create_programme(programme)
    store.append(_external(programme, card))
    reconstructed = store.load_state(programme.programme_id)
    assert reconstructed.programme.context.domains == programme.context.domains
    assert reconstructed.external_evidence[0].evidence_card_id == card.evidence_id
    store.close()


def test_restart_reconstruction_preserves_record_and_revision_identities(tmp_path):
    programme = _programme()
    card = _cards()[0]
    path = tmp_path / "state.sqlite3"
    store = SQLiteResearchStateStore(path)
    store.create_programme(programme)
    revision = store.append(_external(programme, card))
    original = store.load_state(programme.programme_id)
    store.close()

    reopened = SQLiteResearchStateStore(path)
    reconstructed = reopened.load_state(programme.programme_id)
    assert reconstructed.programme.programme_id == programme.programme_id
    assert reconstructed.external_evidence == original.external_evidence
    assert reconstructed.revision_history == (revision,)
    assert reopened.append(original.external_evidence[0]) == revision
    reopened.close()
