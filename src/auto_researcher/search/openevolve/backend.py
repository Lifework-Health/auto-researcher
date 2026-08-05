"""Generic deterministic control for bounded OpenEvolve candidate lifecycles."""

from __future__ import annotations

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.identity import (
    candidate_id,
    openevolve_hash,
    source_hash,
)
from auto_researcher.search.openevolve.models import (
    CandidateOutcome,
    CandidateStatus,
    LineageRecord,
    MutationOperatorPolicy,
    MutationReservation,
    ObjectiveDirection,
    OpenEvolveCandidate,
    OpenEvolvePopulationState,
    OpenEvolveSearchContract,
    OpenEvolveSearchResult,
    ReplacementPolicy,
    SandboxPolicy,
    SelectionPolicy,
)
from auto_researcher.search.openevolve.protocols import (
    EvolvableComponent,
    MutationOperator,
)
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.validation import validate_candidate
from auto_researcher.tasks.models import ExperimentMetadata


class OpenEvolveBackend:
    backend_id = "generic-openevolve-backend-v1"

    def __init__(
        self,
        component: EvolvableComponent,
        metadata: ExperimentMetadata,
        verifier_identity: str,
        mutation_operator: MutationOperator,
        sandbox_runner: LocalSandboxRunner,
    ) -> None:
        self.component = component
        self.component_spec = component.component_spec()
        self.metadata = metadata
        self.verifier_identity = verifier_identity
        self.mutation_operator = mutation_operator
        self.sandbox_runner = sandbox_runner

    @property
    def interface_hash(self) -> str:
        spec = self.component_spec
        return openevolve_hash(
            "openevolve-component-interface-v1",
            {
                "component_id": spec.component_id,
                "component_version": spec.component_version,
                "contract": spec.immutable_interface_contract,
                "entry_point": spec.entry_point,
                "parameter_schema": spec.parameter_schema,
                "output_schema": spec.output_schema,
                "allowed_files": spec.allowed_files,
            },
        )

    @property
    def dependency_hash(self) -> str:
        return openevolve_hash(
            "openevolve-dependency-manifest-v1",
            {
                "imports": self.component_spec.allowed_imports,
                "dependencies": self.component_spec.allowed_dependencies,
            },
        )

    @property
    def evaluator_identity(self) -> str:
        return f"{self.metadata.evaluator_id}@{self.metadata.code_version}"

    def create_search_contract(
        self,
        request: SearchRequest,
        contract: ResearchContract,
    ) -> OpenEvolveSearchContract:
        if request.search_type != SearchType.OPENEVOLVE:
            raise ValueError("OpenEvolve requires an OPENEVOLVE SearchRequest")
        if request.search_type not in contract.allowed_search_types:
            raise ValueError("OpenEvolve is not permitted by the research contract")
        raw = request.search_space.get("openevolve")
        if not isinstance(raw, dict):
            raise ValueError("openevolve_finite_configuration_required")
        required = {
            "population_size",
            "maximum_generations",
            "maximum_wall_time_seconds",
            "maximum_model_calls",
            "maximum_failed_candidates",
            "maximum_consecutive_failures",
            "maximum_artefact_bytes",
            "random_seed",
            "objective_direction",
            "sandbox_policy_id",
            "evaluator_identity",
            "verifier_identity",
        }
        if not required.issubset(raw):
            raise ValueError("openevolve_finite_configuration_required")
        runner_id = getattr(self.sandbox_runner, "runner_id", "")
        expected_sandbox = (
            "openevolve-hardened-executor-v1"
            if runner_id == "openevolve-hardened-executor-v1"
            else "openevolve-sandbox-v1"
        )
        if raw["sandbox_policy_id"] != expected_sandbox:
            raise ValueError("openevolve_sandbox_policy_unavailable")
        if raw["evaluator_identity"] != self.evaluator_identity:
            raise ValueError("openevolve_evaluator_identity_mismatch")
        if raw["verifier_identity"] != self.verifier_identity:
            raise ValueError("openevolve_verifier_identity_mismatch")
        maximum_evaluations = request.experiment_budget
        if maximum_evaluations > contract.maximum_experiments:
            raise ValueError("openevolve_candidate_budget_exceeds_contract")
        source_sha = source_hash(self.component_spec.seed_source)
        seed_id = candidate_id(
            search_request_id=request.request_id,
            component_interface_hash=self.interface_hash,
            source_sha256=source_sha,
        )
        sandbox = SandboxPolicy(
            policy_id=str(raw["sandbox_policy_id"]),
            cpu_time_seconds=int(raw.get("candidate_cpu_time_seconds", 2)),
            wall_time_seconds=float(raw.get("candidate_wall_time_seconds", 3)),
            memory_bytes=int(raw.get("candidate_memory_bytes", 256 * 1024 * 1024)),
            process_limit=int(raw.get("candidate_process_limit", 1)),
            output_bytes=int(raw.get("candidate_output_bytes", 64_000)),
            log_bytes=int(raw.get("candidate_log_bytes", 8_000)),
            file_count_limit=int(raw.get("candidate_file_count_limit", 8)),
            dependency_allowlist=self.component_spec.allowed_dependencies,
        )
        search_contract = OpenEvolveSearchContract(
            search_request_id=request.request_id,
            task_id=contract.task_id,
            task_version=contract.task_version,
            evolvable_component_id=self.component_spec.component_id,
            evolvable_component_version=self.component_spec.component_version,
            seed_candidate_id=seed_id,
            population_size=int(raw["population_size"]),
            maximum_generations=int(raw["maximum_generations"]),
            maximum_candidate_evaluations=maximum_evaluations,
            maximum_wall_time_seconds=float(raw["maximum_wall_time_seconds"]),
            maximum_model_calls=int(raw["maximum_model_calls"]),
            maximum_failed_candidates=int(raw["maximum_failed_candidates"]),
            maximum_consecutive_failures=int(raw["maximum_consecutive_failures"]),
            maximum_artefact_bytes=int(raw["maximum_artefact_bytes"]),
            mutation_operator_policy=MutationOperatorPolicy(
                policy_id="structured-full-file-replacement-v1",
                allowed_operator_ids=(self.mutation_operator.operator_id,),
                maximum_patch_bytes=self.component_spec.maximum_source_bytes,
                maximum_resulting_source_bytes=self.component_spec.maximum_source_bytes,
            ),
            selection_policy=SelectionPolicy(
                direction=ObjectiveDirection(str(raw["objective_direction"])),
                objective_metric=contract.primary_metric,
            ),
            replacement_policy=ReplacementPolicy(),
            sandbox_policy=sandbox,
            evaluator_identity=self.evaluator_identity,
            verifier_identity=self.verifier_identity,
            random_seed=int(raw["random_seed"]),
            objective_threshold=(
                float(raw["objective_threshold"])
                if raw.get("objective_threshold") is not None
                else None
            ),
        )
        if (
            search_contract.maximum_model_calls
            < self.mutation_operator.model_calls_per_mutation
        ):
            raise ValueError("openevolve_model_call_budget_too_small")
        return search_contract

    def seed_candidate(
        self, search_contract: OpenEvolveSearchContract
    ) -> OpenEvolveCandidate:
        spec = self.component_spec
        source_sha = source_hash(spec.seed_source)
        return OpenEvolveCandidate(
            candidate_id=search_contract.seed_candidate_id,
            search_request_id=search_contract.search_request_id,
            parent_candidate_ids=(),
            generation=0,
            birth_index=0,
            mutation_operator="seed",
            mutation_description="Task-owned immutable seed candidate.",
            mutable_file=spec.mutable_file,
            source_payload=spec.seed_source,
            source_hash=source_sha,
            component_interface_hash=self.interface_hash,
            dependency_manifest_hash=self.dependency_hash,
            sandbox_policy_id=search_contract.sandbox_policy.policy_id,
            status=CandidateStatus.PROPOSED,
            creation_provenance="SEED",
        )

    def initialise_population(
        self, search_contract: OpenEvolveSearchContract
    ) -> OpenEvolvePopulationState:
        seed = self.seed_candidate(search_contract)
        return OpenEvolvePopulationState(
            search_request_id=search_contract.search_request_id,
            search_contract_hash=payload_hash(search_contract),
            generation=0,
            source_hashes=(),
            random_seed_state=search_contract.random_seed,
            current_candidate_id=seed.candidate_id,
        )

    def reserve_mutation(
        self,
        search_contract: OpenEvolveSearchContract,
        population: OpenEvolvePopulationState,
        parent: OpenEvolveCandidate,
    ) -> MutationReservation:
        generation = population.generation + 1
        birth_index = population.budget.candidate_proposals + 1
        request_hash = openevolve_hash(
            "openevolve-mutation-request-v1",
            {
                "search_contract_hash": population.search_contract_hash,
                "parent_candidate_id": parent.candidate_id,
                "parent_source_hash": parent.source_hash,
                "generation": generation,
                "birth_index": birth_index,
                "operator": self.mutation_operator.operator_id,
            },
        )
        return MutationReservation(
            reservation_id=f"mutation-{request_hash[:24]}",
            search_request_id=search_contract.search_request_id,
            parent_candidate_ids=(parent.candidate_id,),
            generation=generation,
            birth_index=birth_index,
            mutation_operator=self.mutation_operator.operator_id,
            input_source_hash=parent.source_hash,
            mutation_request_hash=request_hash,
        )

    def mutate_candidate(
        self,
        reservation: MutationReservation,
        parent: OpenEvolveCandidate,
        search_contract: OpenEvolveSearchContract,
    ) -> OpenEvolveCandidate:
        source, description, call_id = self.mutation_operator.mutate(
            reservation, parent, self.component_spec
        )
        encoded = source.encode("utf-8", errors="strict")
        if (
            len(encoded)
            > search_contract.mutation_operator_policy.maximum_resulting_source_bytes
        ):
            raise ValueError("candidate_output_limit")
        source_sha = source_hash(source)
        return OpenEvolveCandidate(
            candidate_id=candidate_id(
                search_request_id=search_contract.search_request_id,
                component_interface_hash=self.interface_hash,
                source_sha256=source_sha,
            ),
            search_request_id=search_contract.search_request_id,
            parent_candidate_ids=reservation.parent_candidate_ids,
            generation=reservation.generation,
            birth_index=reservation.birth_index,
            mutation_operator=reservation.mutation_operator,
            mutation_description=description,
            mutable_file=self.component_spec.mutable_file,
            source_payload=source,
            source_hash=source_sha,
            parent_source_hash=parent.source_hash,
            component_interface_hash=self.interface_hash,
            dependency_manifest_hash=self.dependency_hash,
            sandbox_policy_id=search_contract.sandbox_policy.policy_id,
            status=CandidateStatus.PROPOSED,
            model_call_id=call_id,
            creation_provenance=self.mutation_operator.provenance,
        )

    def validate(self, candidate: OpenEvolveCandidate):
        return validate_candidate(candidate, self.component_spec)

    def prepare(
        self, candidate: OpenEvolveCandidate, search_contract: OpenEvolveSearchContract
    ):
        return self.sandbox_runner.prepare(
            candidate,
            self.component_spec,
            search_contract.sandbox_policy,
            self.component.seed_configuration(),
        )

    @staticmethod
    def _outcome_key(outcome: CandidateOutcome, direction: ObjectiveDirection):
        objective = outcome.objective_value
        score_key = (
            float("inf")
            if objective is None
            else (-objective if direction == ObjectiveDirection.MAXIMIZE else objective)
        )
        return (
            -int(outcome.constraint_compliant),
            -int(outcome.verified),
            score_key,
            outcome.candidate_id,
        )

    def update_population(
        self,
        population: OpenEvolvePopulationState,
        search_contract: OpenEvolveSearchContract,
        candidate: OpenEvolveCandidate,
        outcome: CandidateOutcome,
    ) -> OpenEvolvePopulationState:
        if any(
            item.candidate_id == candidate.candidate_id
            and item.generation == candidate.generation
            and item.source_hash_after == candidate.source_hash
            for item in population.lineage
        ):
            return population
        existing = {item.candidate_id: item for item in population.outcomes}
        if (
            candidate.candidate_id in existing
            and outcome.rejection_reason == "candidate_duplicate"
        ):
            budget = population.budget.model_copy(
                update={
                    "generations_used": max(
                        population.budget.generations_used,
                        candidate.generation,
                    ),
                    "candidate_proposals": population.budget.candidate_proposals + 1,
                    "model_calls": population.budget.model_calls
                    + self.mutation_operator.model_calls_per_mutation,
                    "failed_candidates": population.budget.failed_candidates + 1,
                    "consecutive_failures": population.budget.consecutive_failures + 1,
                }
            )
            lineage = LineageRecord(
                candidate_id=candidate.candidate_id,
                parent_candidate_ids=candidate.parent_candidate_ids,
                generation=candidate.generation,
                mutation_operator=candidate.mutation_operator,
                model_call_id=candidate.model_call_id,
                source_hash_before=candidate.parent_source_hash,
                source_hash_after=candidate.source_hash,
                validation_code="candidate_duplicate",
                selection_outcome="rejected",
                rejection_reason="candidate_duplicate",
                replacement_outcome="no_change",
            )
            diversity = dict(population.diversity_metadata)
            diversity.update(
                {
                    "mechanism": "source_hash_uniqueness_v1",
                    "unique_source_count": len(set(population.source_hashes)),
                    "duplicate_rejections": int(
                        diversity.get("duplicate_rejections", 0)
                    )
                    + 1,
                }
            )
            return population.model_copy(
                update={
                    "generation": max(population.generation, candidate.generation),
                    "lineage": (*population.lineage, lineage),
                    "diversity_metadata": diversity,
                    "budget": budget,
                    "current_candidate_id": None,
                    "current_reservation_id": None,
                }
            )
        if candidate.candidate_id in existing:
            if existing[candidate.candidate_id] != outcome:
                raise ValueError("conflicting_completed_candidate_identity")
            return population
        outcomes = (*population.outcomes, outcome)
        valid = [item for item in outcomes if item.status == CandidateStatus.VERIFIED]
        ranked = sorted(
            valid,
            key=lambda item: self._outcome_key(
                item, search_contract.selection_policy.direction
            ),
        )
        active = tuple(
            item.candidate_id for item in ranked[: search_contract.population_size]
        )
        evaluated = (
            (*population.evaluated_candidate_ids, candidate.candidate_id)
            if outcome.evaluation is not None
            else population.evaluated_candidate_ids
        )
        failed = (
            (*population.failed_candidate_ids, candidate.candidate_id)
            if outcome.status in {CandidateStatus.FAILED, CandidateStatus.REJECTED}
            else population.failed_candidate_ids
        )
        failure = outcome.status in {CandidateStatus.FAILED, CandidateStatus.REJECTED}
        preparation = candidate.preparation_result
        budget = population.budget.model_copy(
            update={
                "generations_used": max(
                    population.budget.generations_used, candidate.generation
                ),
                "candidate_proposals": population.budget.candidate_proposals + 1,
                "successful_preparations": population.budget.successful_preparations
                + int(
                    preparation is not None
                    and preparation.execution_status.value == "COMPLETED"
                ),
                "failed_preparations": population.budget.failed_preparations
                + int(
                    preparation is not None
                    and preparation.execution_status.value != "COMPLETED"
                ),
                "candidate_evaluations": population.budget.candidate_evaluations
                + int(outcome.evaluation is not None),
                "verifier_calls": population.budget.verifier_calls
                + int(outcome.verification is not None),
                "model_calls": population.budget.model_calls
                + (
                    self.mutation_operator.model_calls_per_mutation
                    if candidate.generation > 0
                    else 0
                ),
                "candidate_runtime": population.budget.candidate_runtime
                + (preparation.runtime_seconds if preparation is not None else 0),
                "wall_time_elapsed": population.budget.wall_time_elapsed
                + (preparation.runtime_seconds if preparation is not None else 0),
                "failed_candidates": population.budget.failed_candidates + int(failure),
                "consecutive_failures": population.budget.consecutive_failures + 1
                if failure
                else 0,
            }
        )
        lineage = LineageRecord(
            candidate_id=candidate.candidate_id,
            parent_candidate_ids=candidate.parent_candidate_ids,
            generation=candidate.generation,
            mutation_operator=candidate.mutation_operator,
            model_call_id=candidate.model_call_id,
            source_hash_before=candidate.parent_source_hash,
            source_hash_after=candidate.source_hash,
            validation_code=(candidate.validation_result.safe_error_code or "valid")
            if candidate.validation_result
            else "not_validated",
            evaluation_identity=candidate.evaluation_identity,
            selection_outcome=outcome.selection_outcome,
            rejection_reason=outcome.rejection_reason,
            replacement_outcome=outcome.replacement_outcome,
        )
        return population.model_copy(
            update={
                "generation": max(population.generation, candidate.generation),
                "active_population_candidate_ids": active,
                "archive_candidate_ids": (
                    *population.archive_candidate_ids,
                    candidate.candidate_id,
                ),
                "evaluated_candidate_ids": evaluated,
                "failed_candidate_ids": failed,
                "best_known_candidate_ids": tuple(
                    item.candidate_id for item in ranked[:1]
                ),
                "source_hashes": (*population.source_hashes, candidate.source_hash),
                "outcomes": outcomes,
                "lineage": (*population.lineage, lineage),
                "diversity_metadata": {
                    "mechanism": "source_hash_uniqueness_v1",
                    "unique_source_count": len(
                        set((*population.source_hashes, candidate.source_hash))
                    ),
                },
                "budget": budget,
                "current_candidate_id": None,
                "current_reservation_id": None,
            }
        )

    def stop_reason(
        self,
        population: OpenEvolvePopulationState,
        search_contract: OpenEvolveSearchContract,
    ) -> str | None:
        budget = population.budget
        if (
            budget.candidate_evaluations
            >= search_contract.maximum_candidate_evaluations
        ):
            return "maximum_candidate_evaluations_reached"
        if budget.generations_used >= search_contract.maximum_generations:
            return "maximum_generations_reached"
        if (
            budget.model_calls >= search_contract.maximum_model_calls
            and self.mutation_operator.model_calls_per_mutation
        ):
            return "maximum_model_calls_reached"
        if budget.wall_time_elapsed >= search_contract.maximum_wall_time_seconds:
            return "maximum_wall_time_reached"
        if budget.failed_candidates >= search_contract.maximum_failed_candidates:
            return "maximum_failed_candidates_reached"
        if budget.artefact_bytes >= search_contract.maximum_artefact_bytes:
            return "maximum_artefact_bytes_reached"
        if budget.consecutive_failures >= search_contract.maximum_consecutive_failures:
            return "maximum_failed_candidates_reached"
        if (
            search_contract.objective_threshold is not None
            and population.best_known_candidate_ids
        ):
            best = next(
                item
                for item in population.outcomes
                if item.candidate_id == population.best_known_candidate_ids[0]
            )
            if (
                best.constraint_compliant
                and best.verified
                and best.objective_value is not None
            ):
                reached = (
                    best.objective_value >= search_contract.objective_threshold
                    if search_contract.selection_policy.direction
                    == ObjectiveDirection.MAXIMIZE
                    else best.objective_value <= search_contract.objective_threshold
                )
                if reached:
                    return "objective_reached"
        if (
            population.outcomes
            and not population.active_population_candidate_ids
            and budget.consecutive_failures
        ):
            return "no_valid_candidates"
        return None

    def final_result(
        self, population: OpenEvolvePopulationState
    ) -> OpenEvolveSearchResult:
        return OpenEvolveSearchResult(
            search_request_id=population.search_request_id,
            search_contract_hash=population.search_contract_hash,
            candidates_proposed=population.budget.candidate_proposals,
            candidates_evaluated=population.budget.candidate_evaluations,
            candidates_failed=population.budget.failed_candidates,
            generations_completed=population.budget.generations_used,
            best_candidate_ids=population.best_known_candidate_ids,
            feasible_candidate_found=bool(population.best_known_candidate_ids),
            stop_reason=population.stop_reason or "operator_stop",
        )
