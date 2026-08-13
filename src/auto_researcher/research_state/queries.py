"""Deterministic query semantics over a reconstructed Research State."""

from __future__ import annotations

from pydantic import Field

from auto_researcher.research_state.models import (
    CandidateNextAction,
    DiagnosticObservation,
    EvidenceReferences,
    ExperimentIntent,
    HypothesisStatus,
    InformationValueClass,
    InternalExperimentalObservation,
    PlannerInference,
    ProgrammeDecisionBasis,
    RecordType,
    ResearchHypothesis,
    ResearchState,
    ResearchStateModel,
    ResearchUncertainty,
    UncertaintyStatus,
)


class LearningSummary(ResearchStateModel):
    internal_observations: tuple[InternalExperimentalObservation, ...]
    diagnostic_observations: tuple[DiagnosticObservation, ...]
    planner_inferences: tuple[PlannerInference, ...]
    supported_hypotheses: tuple[ResearchHypothesis, ...]
    refuted_hypotheses: tuple[ResearchHypothesis, ...]


class UncertaintySummary(ResearchStateModel):
    unresolved_uncertainties: tuple[ResearchUncertainty, ...]
    unresolved_hypotheses: tuple[ResearchHypothesis, ...]


class EvidenceLineage(ResearchStateModel):
    conclusion_type: RecordType
    conclusion_id: str = Field(min_length=1)
    motivating_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    direct_supporting_evidence: EvidenceReferences = Field(
        default_factory=EvidenceReferences
    )
    hypothesis_supporting_evidence: EvidenceReferences = Field(
        default_factory=EvidenceReferences
    )
    refuting_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    derived_evidence: EvidenceReferences = Field(default_factory=EvidenceReferences)
    inference_ids: tuple[str, ...] = ()
    hypothesis_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    programme_bases: tuple[ProgrammeDecisionBasis, ...] = ()

    @property
    def all_related_evidence(self) -> EvidenceReferences:
        return (
            self.motivating_evidence.merged(self.direct_supporting_evidence)
            .merged(self.hypothesis_supporting_evidence)
            .merged(self.refuting_evidence)
            .merged(self.derived_evidence)
        )

    @property
    def evidence(self) -> EvidenceReferences:
        """Compatibility aggregate; role-specific fields remain authoritative."""

        return self.all_related_evidence


class ExperimentRationale(ResearchStateModel):
    experiment_id: str
    intent: ExperimentIntent
    direct_motivating_evidence: EvidenceReferences
    hypothesis_motivating_evidence: EvidenceReferences
    inference_motivating_evidence: EvidenceReferences
    hypothesis_ids: tuple[str, ...]
    motivating_inference_ids: tuple[str, ...]
    informed_decision_ids: tuple[str, ...]

    @property
    def evidence(self) -> EvidenceReferences:
        return self.direct_motivating_evidence.merged(
            self.hypothesis_motivating_evidence
        ).merged(self.inference_motivating_evidence)


class DiscriminatingAction(ResearchStateModel):
    source_id: str
    action: str
    competing_hypothesis_ids: tuple[str, ...] = ()
    uncertainty_ids: tuple[str, ...] = ()


class ResearchStateQueries:
    def __init__(self, state: ResearchState) -> None:
        self.state = state
        self._hypotheses = {item.hypothesis_id: item for item in state.hypotheses}
        self._inferences = {
            item.inference_id: item for item in state.planner_inferences
        }
        self._decisions = {item.decision_id: item for item in state.planner_decisions}
        self._experiments = {item.experiment_id: item for item in state.experiments}

    def what_have_we_learned(self) -> LearningSummary:
        return LearningSummary(
            internal_observations=self.state.internal_observations,
            diagnostic_observations=self.state.diagnostic_observations,
            planner_inferences=self.state.planner_inferences,
            supported_hypotheses=tuple(
                item
                for item in self.state.hypotheses
                if item.status == HypothesisStatus.SUPPORTED
            ),
            refuted_hypotheses=tuple(
                item
                for item in self.state.hypotheses
                if item.status == HypothesisStatus.REFUTED
            ),
        )

    def what_remains_uncertain(self) -> UncertaintySummary:
        return UncertaintySummary(
            unresolved_uncertainties=tuple(
                item
                for item in self.state.uncertainties
                if item.status in {UncertaintyStatus.OPEN, UncertaintyStatus.REDUCED}
            ),
            unresolved_hypotheses=tuple(
                item
                for item in self.state.hypotheses
                if item.status
                in {HypothesisStatus.PROPOSED, HypothesisStatus.UNRESOLVED}
            ),
        )

    def hypotheses(
        self, status: HypothesisStatus | None = None
    ) -> tuple[ResearchHypothesis, ...]:
        if status is None:
            return self.state.hypotheses
        return tuple(item for item in self.state.hypotheses if item.status == status)

    def what_evidence_supports_conclusion(
        self, conclusion_type: RecordType, conclusion_id: str
    ) -> EvidenceLineage:
        motivating_evidence = EvidenceReferences()
        direct_supporting_evidence = EvidenceReferences()
        hypothesis_supporting_evidence = EvidenceReferences()
        refuting_evidence = EvidenceReferences()
        derived_evidence = EvidenceReferences()
        inference_ids: set[str] = set()
        hypothesis_ids: set[str] = set()
        uncertainty_ids: set[str] = set()
        experiment_ids: set[str] = set()
        programme_bases: tuple[ProgrammeDecisionBasis, ...] = ()

        def add_hypothesis(hypothesis_id: str, *, direct: bool = False) -> None:
            nonlocal motivating_evidence, direct_supporting_evidence
            nonlocal hypothesis_supporting_evidence, refuting_evidence
            hypothesis = self._hypotheses.get(hypothesis_id)
            if hypothesis is None or hypothesis_id in hypothesis_ids:
                return
            hypothesis_ids.add(hypothesis_id)
            motivating_evidence = motivating_evidence.merged(
                hypothesis.motivating_evidence
            )
            if direct:
                direct_supporting_evidence = direct_supporting_evidence.merged(
                    hypothesis.supporting_evidence
                )
            else:
                hypothesis_supporting_evidence = hypothesis_supporting_evidence.merged(
                    hypothesis.supporting_evidence
                )
            refuting_evidence = refuting_evidence.merged(hypothesis.refuting_evidence)
            for inference_id in hypothesis.origin_inference_ids:
                add_inference(inference_id)

        def add_inference(inference_id: str) -> None:
            nonlocal derived_evidence
            inference = self._inferences.get(inference_id)
            if inference is None or inference_id in inference_ids:
                return
            inference_ids.add(inference_id)
            uncertainty_ids.update(inference.uncertainty_ids)
            derived_evidence = derived_evidence.merged(inference.derived_from_evidence)
            for hypothesis_id in inference.hypothesis_ids:
                add_hypothesis(hypothesis_id)

        if conclusion_type == RecordType.HYPOTHESIS:
            if conclusion_id not in self._hypotheses:
                raise KeyError(conclusion_id)
            add_hypothesis(conclusion_id, direct=True)
        elif conclusion_type == RecordType.PLANNER_INFERENCE:
            if conclusion_id not in self._inferences:
                raise KeyError(conclusion_id)
            add_inference(conclusion_id)
        elif conclusion_type == RecordType.PLANNER_DECISION:
            decision = self._decisions.get(conclusion_id)
            if decision is None:
                raise KeyError(conclusion_id)
            direct_supporting_evidence = direct_supporting_evidence.merged(
                decision.supporting_evidence
            )
            uncertainty_ids.update(decision.uncertainty_ids)
            experiment_ids.update(decision.experiment_ids)
            programme_bases = decision.programme_bases
            for inference_id in decision.inference_ids:
                add_inference(inference_id)
            for hypothesis_id in decision.hypothesis_ids:
                add_hypothesis(hypothesis_id)
        else:
            raise ValueError(
                "only hypotheses, inferences, and decisions are conclusions"
            )

        return EvidenceLineage(
            conclusion_type=conclusion_type,
            conclusion_id=conclusion_id,
            motivating_evidence=motivating_evidence,
            direct_supporting_evidence=direct_supporting_evidence,
            hypothesis_supporting_evidence=hypothesis_supporting_evidence,
            refuting_evidence=refuting_evidence,
            derived_evidence=derived_evidence,
            inference_ids=tuple(sorted(inference_ids)),
            hypothesis_ids=tuple(sorted(hypothesis_ids)),
            uncertainty_ids=tuple(sorted(uncertainty_ids)),
            experiment_ids=tuple(sorted(experiment_ids)),
            programme_bases=programme_bases,
        )

    def why_was_experiment_run(self, experiment_id: str) -> ExperimentRationale:
        experiment = self._experiments.get(experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        hypothesis_evidence = EvidenceReferences()
        inference_evidence = EvidenceReferences()
        inference_ids: set[str] = set()
        for hypothesis_id in experiment.intent.hypothesis_ids:
            hypothesis = self._hypotheses.get(hypothesis_id)
            if hypothesis is not None:
                hypothesis_evidence = hypothesis_evidence.merged(
                    hypothesis.motivating_evidence
                )
                for inference_id in hypothesis.origin_inference_ids:
                    inference = self._inferences.get(inference_id)
                    if inference is not None:
                        inference_ids.add(inference_id)
                        inference_evidence = inference_evidence.merged(
                            inference.derived_from_evidence
                        )
        return ExperimentRationale(
            experiment_id=experiment_id,
            intent=experiment.intent,
            direct_motivating_evidence=experiment.intent.motivated_by_evidence,
            hypothesis_motivating_evidence=hypothesis_evidence,
            inference_motivating_evidence=inference_evidence,
            hypothesis_ids=experiment.intent.hypothesis_ids,
            motivating_inference_ids=tuple(sorted(inference_ids)),
            informed_decision_ids=tuple(
                sorted(
                    item.decision_id
                    for item in self.state.planner_decisions
                    if experiment_id in item.experiment_ids
                )
            ),
        )

    def decision_informed_by_experiment(self, experiment_id: str):
        return tuple(
            item
            for item in self.state.planner_decisions
            if experiment_id in item.experiment_ids
        )

    def evidence_influencing_decision(self, decision_id: str) -> EvidenceLineage:
        return self.what_evidence_supports_conclusion(
            RecordType.PLANNER_DECISION, decision_id
        )

    def most_direct_discriminating_actions(self) -> tuple[DiscriminatingAction, ...]:
        actions: list[DiscriminatingAction] = []
        for hypothesis in self.state.hypotheses:
            if not hypothesis.competing_hypothesis_ids:
                continue
            action = (
                hypothesis.proposed_discriminating_experiment
                or hypothesis.proposed_discriminating_question
            )
            if action:
                actions.append(
                    DiscriminatingAction(
                        source_id=hypothesis.hypothesis_id,
                        action=action,
                        competing_hypothesis_ids=hypothesis.competing_hypothesis_ids,
                    )
                )
        actions.extend(
            self._next_action_as_discriminator(item)
            for item in self.state.candidate_next_actions
            if item.expected_information_value == InformationValueClass.DISCRIMINATING
        )
        return tuple(sorted(actions, key=lambda item: (item.source_id, item.action)))

    @staticmethod
    def _next_action_as_discriminator(
        item: CandidateNextAction,
    ) -> DiscriminatingAction:
        return DiscriminatingAction(
            source_id=item.next_action_id,
            action=item.action,
            competing_hypothesis_ids=item.competing_hypothesis_ids,
            uncertainty_ids=item.uncertainty_ids,
        )
