"""Mandatory automatic verifier invocation."""

from auto_researcher.graph.state import ResearchState
from auto_researcher.provenance.reuse import VerificationReuseRecord
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.runtime.identity import payload_hash


def verify_evidence(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    experiment = state["experiment_spec"]
    evaluation = state["evaluation_result"]
    assert experiment is not None and evaluation is not None
    evaluation_hash = payload_hash(evaluation)
    verifier_version = getattr(
        dependencies.verifier,
        "version",
        dependencies.verifier.verifier_id,
    )
    policy_version = dependencies.verification_policy.policy_id
    identity_hash = payload_hash(
        {
            "run_id": state["run_id"],
            "experiment_id": experiment.experiment_id,
            "evaluation_payload_hash": evaluation_hash,
            "verifier_version": verifier_version,
            "verification_policy_version": policy_version,
        }
    )
    existing = dependencies.provenance_store.get_verification_reuse(
        state["run_id"],
        experiment.experiment_id,
    )
    reused = existing is not None
    if existing is not None:
        if existing.scientific_identity_hash != identity_hash:
            raise RuntimeError("conflicting_completed_verification_identity")
        if payload_hash(existing.result) != existing.result_payload_hash:
            raise RuntimeError("stored_verification_payload_hash_mismatch")
        verification = existing.result
    else:
        verification = dependencies.verifier.verify(
            experiment,
            evaluation,
            state["contract"],
            claimed_score=evaluation.primary_score,
        )
        dependencies.provenance_store.append_verification_reuse(
            VerificationReuseRecord(
                run_id=state["run_id"],
                experiment_id=experiment.experiment_id,
                scientific_identity_hash=identity_hash,
                evaluation_payload_hash=evaluation_hash,
                result_payload_hash=payload_hash(verification),
                verifier_version=verifier_version,
                verification_policy_version=policy_version,
                result=verification,
            )
        )
    return {
        "verification_result": verification,
        "executed_nodes": ["verify_evidence_reused" if reused else "verify_evidence"],
    }
