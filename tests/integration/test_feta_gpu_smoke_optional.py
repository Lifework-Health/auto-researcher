import os
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.feta_data
def test_real_feta_cuda_engineering_smoke_is_non_scientific():
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real FeTA engineering smoke")
    value = os.getenv("AUTO_RESEARCHER_FETA_DATA_DIR")
    if not value:
        pytest.skip("set AUTO_RESEARCHER_FETA_DATA_DIR for the real FeTA smoke")
    from auto_researcher.tasks.feta_seg.runner import run_engineering_smoke

    result = run_engineering_smoke(Path(value))
    assert result["scientific_baseline"] is False
    assert result["reusable_as_baseline_evidence"] is False
    assert result["dataset_identity_exact"] is True
    assert result["holdout_subjects_evaluated"] == 0
    assert result["metric_panel_complete"] is True
    assert result["all_labels_valid"] is True
    assert result["peak_gpu_memory_bytes"] > 0
