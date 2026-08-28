from __future__ import annotations

from types import SimpleNamespace

import pytest

from auto_researcher.tasks.feta_unet_ensemble import final_holdout
from auto_researcher.tasks.feta_unet_ensemble.final_holdout import (
    FINAL_HOLDOUT_MANIFEST_SCHEMA,
    _fold_cache_paths,
    _validate_release_manifest,
    subject_bootstrap_interval,
)
from auto_researcher.tasks.feta_unet_ensemble.models import EnsembleMember
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
)


def _rows():
    return tuple({"macro_dice": 0.75 + index / 100} for index in range(12))


def _manifest():
    return {
        "schema_version": FINAL_HOLDOUT_MANIFEST_SCHEMA,
        "evaluation_id": "v11-v8-dynunet-final-test",
        "decision": {
            "scope": "one-time-final-sealed-holdout",
            "candidate_family": "v8-dynunet",
            "inference_rule": "equal-probability-mean-five-fold",
            "post_processing": "none",
            "selection_frozen_before_holdout": True,
            "result_feedback_prohibited": True,
        },
        "member": {"source": "bound-by-loader"},
    }


def _source():
    configuration = FeTAUNetSearchConfiguration(
        profile="five_fold_confirmation",
        fold_count=5,
        maximum_epochs=150,
        model_variant="dynunet",
        feature_width="v8_dyn_balanced_5",
        architecture_budget="dynunet-15m-150m-v1",
        features=(64, 128, 256, 512, 768),
        kernel_profile="large_front",
        residual_blocks=True,
        deep_supervision_heads=1,
    )
    return SimpleNamespace(
        configuration=configuration,
        member=SimpleNamespace(experiment_id="experiment-v8"),
        checkpoint_sha256s=tuple(f"{index:064x}" for index in range(5)),
    )


def test_subject_bootstrap_interval_is_deterministic_and_subject_level():
    first = subject_bootstrap_interval(_rows(), samples=2_000)
    second = subject_bootstrap_interval(_rows(), samples=2_000)
    assert first == second
    assert first["lower"] < 0.805 < first["upper"]
    assert first["confidence_level"] == 0.95


def test_release_manifest_accepts_only_frozen_v8_dynunet(monkeypatch):
    source = _source()
    monkeypatch.setattr(
        final_holdout, "load_cross_validation_member_source", lambda _: source
    )
    assert _validate_release_manifest(_manifest()) is source

    unsafe = _manifest()
    unsafe["decision"]["result_feedback_prohibited"] = False
    with pytest.raises(ValueError, match="manifest_invalid"):
        _validate_release_manifest(unsafe)

    source.configuration = source.configuration.model_copy(
        update={"model_variant": "basic_unet"}
    )
    with pytest.raises(ValueError, match="candidate_not_frozen_v8_dynunet"):
        _validate_release_manifest(_manifest())


def test_fold_cache_identity_separates_all_five_models(tmp_path):
    source = _source()
    source.member = EnsembleMember(
        experiment_id="experiment-v8",
        checkpoint_sha256="f" * 64,
        configuration_identity="1" * 64,
        architecture_identity="architecture",
        dataset_manifest_hash="dataset",
        split_hash="split",
        fold_hash="fold",
        preprocessing_identity="preprocess",
        label_mapping_identity="labels",
        inference_identity="inference",
    )
    subject = SimpleNamespace(subject_id="sub-001")
    paths = tuple(_fold_cache_paths(tmp_path, source, fold, subject) for fold in range(5))
    assert len({item[0] for item in paths}) == 5
    assert len({item[2] for item in paths}) == 5
