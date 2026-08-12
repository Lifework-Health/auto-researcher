from __future__ import annotations

import json

from auto_researcher.agents.models import StructuredModelResponse
from auto_researcher.contracts.enums import ProvenanceKind, SearchType
from auto_researcher.contracts.models import SearchRequest
from auto_researcher.providers.fake_production import (
    FakeProductionStructuredModelClient,
)
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.live_boundary import (
    assert_no_prohibited_dynamic_content,
)
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import (
    build_approved_live_upstream_runtime,
    mutation_constraints,
)
from auto_researcher.tasks.feta_seg.manifests import (
    DATASET_RELEASE,
    EXPECTED_MANIFEST_HASH,
)
from auto_researcher.tasks.feta_seg_evolve import (
    default_feta_evolve_openevolve_configuration,
)
from auto_researcher.tasks.feta_seg_evolve.configuration import (
    FeTASegEvolveConfiguration,
)
from auto_researcher.tasks.feta_seg_evolve.evaluator import (
    EVALUATOR_ID,
    evaluator_code_version,
)
from auto_researcher.tasks.feta_seg_evolve.openevolve import COSINE_SOURCE
from auto_researcher.tasks.models import ExperimentMetadata
from tests.unit.test_openevolve_metadata_only_boundary import (
    PROMPT,
    make_bridge,
    metadata_only_evidence,
)


class CapturingFakeProvider(FakeProductionStructuredModelClient):
    def __init__(self, response: dict) -> None:
        super().__init__(
            provider="fake-production",
            model_id="fake-model-20260101",
            response=response,
        )
        self.model_requests: list[dict] = []

    def generate_structured(self, **kwargs) -> StructuredModelResponse:
        self.model_requests.append(json.loads(kwargs["user_prompt"]))
        return super().generate_structured(**kwargs)


def test_feta_fake_production_metadata_only_mutation_lifecycle(tmp_path):
    evidence = metadata_only_evidence()
    response = {
        "protocol_version": "upstream-mutation-envelope-v1",
        "mutable_file": "candidate.py",
        "source": COSINE_SOURCE,
        "description": "Bounded schedule mutation using the attested schema.",
    }
    provider = CapturingFakeProvider(response)
    bridge = make_bridge(evidence, provider_factory=lambda: provider)
    adapter, hardened_executor = build_approved_live_upstream_runtime(
        evidence["adapter"],
        bridge,
        evidence["policy"],
        evidence["isolation"],
        task=evidence["task"],
        component_spec=evidence["spec"],
        workspace_root=tmp_path / "hardened-workspace",
    )

    dataset_version = f"{DATASET_RELEASE}+{EXPECTED_MANIFEST_HASH}"
    metadata = ExperimentMetadata(
        evaluator_id=EVALUATOR_ID,
        code_version=evaluator_code_version(dataset_version),
        dataset_version=dataset_version,
        provenance=ProvenanceKind.REAL,
    )
    backend = OpenEvolveBackend(
        evidence["component"],
        metadata,
        "deterministic-verifier-v1@feta-seg-evolve-evidence-policy-v2",
        adapter,
        hardened_executor,
    )
    configuration = default_feta_evolve_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": 1,
            "maximum_candidate_evaluations": 2,
            "maximum_model_calls": 1,
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    )
    request = SearchRequest(
        request_id=evidence["context"].search_request_id,
        hypothesis_id="feta-metadata-only-hypothesis",
        search_type=SearchType.OPENEVOLVE,
        target="mean_subject_macro_dice",
        search_space=configuration,
        experiment_budget=2,
        rationale="Offline fake-production metadata-only lifecycle.",
    )
    search = backend.create_search_contract(request, evidence["research_contract"])
    seed = backend.seed_candidate(search)
    population = backend.initialise_population(search)
    reservation = backend.reserve_mutation(search, population, seed)
    candidate = backend.mutate_candidate(reservation, seed, search)
    validation = backend.validate(candidate)
    assert validation.status.value == "VALID"
    candidate = candidate.model_copy(update={"validation_result": validation})

    # CPU-only preparation fixture. The operational backend above retains the
    # attested HardenedDockerExecutor and this test never invokes Docker/CUDA.
    preparation = LocalSandboxRunner(tmp_path / "local-preparation").prepare(
        candidate,
        evidence["spec"],
        search.sandbox_policy,
        evidence["component"].seed_configuration(),
    )
    assert preparation.execution_status.value == "COMPLETED"
    experiment = evidence["component"].candidate_to_experiment(
        candidate,
        preparation,
        request,
        evidence["research_contract"],
        metadata,
        run_id=evidence["context"].run_id,
    )
    evolved = FeTASegEvolveConfiguration.model_validate(experiment.configuration)

    assert provider.invocation_count == 1
    assert candidate.generation == 1
    assert candidate.parent_candidate_ids == (seed.candidate_id,)
    assert candidate.model_call_id is not None
    assert evolved.candidate_provenance.candidate_id == candidate.candidate_id
    assert evolved.training_policy.learning_rate.family == "cosine"
    assert evolved.base_configuration.seed == 20260807
    assert evolved.base_configuration.spacing_mm == (0.5, 0.5, 0.5)
    assert evolved.base_configuration.patch_size == (128, 128, 128)

    model_request = provider.model_requests[0]
    assert set(model_request) == {
        "protocol",
        "parent",
        "mutable_file",
        "interface_contract",
        "maximum_source_bytes",
        "mutation_constraints",
    }
    assert model_request["parent"]["code"] == seed.source_payload
    assert model_request["mutation_constraints"] == mutation_constraints(
        evidence["spec"]
    ).model_dump(mode="json")
    assert_no_prohibited_dynamic_content(model_request)
    encoded = json.dumps(model_request, sort_keys=True).casefold()
    assert "/must-not-cross-boundary/feta" not in encoded
    assert "data_dir" not in encoded
    assert "workspace_dir" not in encoded
    assert "output_dir" not in encoded
    assert payload_hash(PROMPT) == evidence["approval"].prompt_hash
