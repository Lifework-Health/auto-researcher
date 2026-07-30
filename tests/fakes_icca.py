from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

from auto_researcher.tasks.icca_nbs.bindings import ICCABindings


class _CodeEnum(Enum):
    def __init__(self, doc_name: str, code_name: str, slug: str) -> None:
        self.doc_name = doc_name
        self.code_name = code_name
        self.slug = slug


class FakeNetwork(_CodeEnum):
    IDEKER = ("Ideker", "Ideker", "ideker")
    OMNI = ("Omni", "Omni", "omni")


class FakeAlignment(_CodeEnum):
    INTERSECT = ("Intersect", "intersect", "intersect")
    NETWORK_ZERO_PAD = ("NetworkZeroPad", "full", "full")


class FakeStatus(str, Enum):
    COMPLETE = "complete"


class NumpyLikeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def make_fake_icca_bindings(
    *,
    eligible: bool = True,
    fail_evaluation: bool = False,
) -> tuple[ICCABindings, dict]:
    calls = {
        "load_cohort": 0,
        "cache_get": 0,
        "evaluate": 0,
        "objective": 0,
        "k_values": None,
    }

    def load_cohort(data_dir, *, verbose=False):
        calls["load_cohort"] += 1
        return SimpleNamespace(mutations="fake-mutations")

    def paths_factory(workspace_dir):
        return SimpleNamespace(workspace_dir=workspace_dir)

    class Cache:
        def __init__(self, paths):
            self.paths = paths

        def get(self, mutations, network, alignment, alpha):
            calls["cache_get"] += 1
            return SimpleNamespace(
                matrix=[[0.1, 0.2], [0.2, 0.1]],
                patient_ids=["internal-patient-a", "internal-patient-b"],
            )

    def evaluate(matrix, patient_ids, cohort, *, k_values, r, config):
        calls["evaluate"] += 1
        calls["k_values"] = list(k_values)
        if fail_evaluation:
            raise RuntimeError("fake scientific failure")
        k = k_values[0]
        gates = {
            "logrank_pass": eligible,
            "clinical_pass": eligible,
            "floors_pass": eligible,
            "eligible": eligible,
            "diagnostic_only": False,
        }
        result = SimpleNamespace(
            selected_k=k,
            eligible=eligible,
            eligibility=gates,
            metrics={
                "pac": NumpyLikeScalar(0.2),
                "c_index": {"cv": NumpyLikeScalar(0.71)},
                "status": FakeStatus.COMPLETE,
            },
            selection_inputs={
                "pac": NumpyLikeScalar(0.2),
                "promising": eligible,
            },
            per_cluster={
                1: {"size": NumpyLikeScalar(55), "events": NumpyLikeScalar(20)}
            },
            provenance="REAL",
        )
        return SimpleNamespace(per_k={k: result}, selected_k=k)

    def objective(result):
        calls["objective"] += 1
        pac = result.selection_inputs["pac"].item()
        return (1.0 - pac) if result.eligible else -pac

    return (
        ICCABindings(
            load_cohort=load_cohort,
            harness_paths_factory=paths_factory,
            network_type=FakeNetwork,
            alignment_type=FakeAlignment,
            propagation_cache_factory=Cache,
            evaluate=evaluate,
            stability_objective=objective,
            alpha_bounds=(0.3, 0.9),
            k_bounds=(4, 8),
            package_version="fake-0.1",
            code_version="fake-v2-commit",
        ),
        calls,
    )
