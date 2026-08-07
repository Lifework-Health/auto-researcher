from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration  # noqa: E402
from auto_researcher.tasks.feta_seg.model import (  # noqa: E402
    create_segresnet,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_seg.metrics import (  # noqa: E402
    cubical_betti_numbers,
    cubical_euler_characteristic,
    evaluate_subject_segmentation,
    physical_fov_diagonal,
    physical_hd95,
    topology_metrics,
    volume_similarity_counts,
)
from auto_researcher.tasks.feta_seg.runner import (  # noqa: E402
    restore_prediction_to_native,
)
from auto_researcher.tasks.feta_seg.trainer import (  # noqa: E402
    checkpoint_reference,
    create_loss,
    seed_everything,
    sliding_window_predict,
)
from auto_researcher.tasks.feta_seg.transforms import create_transforms  # noqa: E402


def test_segresnet_shape_finite_loss_and_backward():
    seed_everything(0)
    model = create_segresnet(FeTASegConfiguration())
    inputs = torch.randn(1, 1, 32, 32, 32)
    labels = torch.randint(0, 8, (1, 1, 32, 32, 32))
    output = model(inputs)
    loss = create_loss()(output, labels)
    loss.backward()
    assert output.shape == (1, 8, 32, 32, 32)
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert trainable_parameter_count(model) == 18_968_456


def test_checkpoint_reference_is_relative_and_hashed(tmp_path: Path):
    checkpoint = tmp_path / "fold-0" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"generated-checkpoint-fixture")
    result = checkpoint_reference(
        checkpoint, fold=0, best_epoch=5, score=0.5, output_root=tmp_path
    )
    assert result["relative_path"] == "fold-0/best.pt"
    assert len(result["sha256"]) == 64
    assert result["size_bytes"] == len(b"generated-checkpoint-fixture")


def test_deterministic_transforms_preserve_integer_labels_and_normalise(tmp_path: Path):
    import nibabel as nib

    image = np.zeros((8, 8, 8), dtype=np.float32)
    image[2:6, 2:6, 2:6] = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 1
    label = np.zeros((8, 8, 8), dtype=np.int16)
    label[2:6, 2:6, 2:6] = 1
    image_path, label_path = tmp_path / "image.nii.gz", tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(label, np.eye(4)), label_path)
    result = create_transforms(training=False)(
        {"image": image_path, "label": label_path}
    )
    assert set(torch.unique(result["label"]).tolist()) <= {0.0, 1.0}
    foreground = result["image"][result["image"] != 0]
    assert abs(float(foreground.mean())) < 1e-5


def test_augmentation_is_training_only_and_sliding_window_is_whole_volume():
    training = create_transforms(training=True)
    validation = create_transforms(training=False)
    training_names = {type(item).__name__ for item in training.transforms}
    validation_names = {type(item).__name__ for item in validation.transforms}
    assert "RandFlipd" in training_names and "RandFlipd" not in validation_names
    inputs = torch.randn(1, 1, 16, 16, 16)
    result = sliding_window_predict(inputs, torch.nn.Identity(), FeTASegConfiguration())
    assert result.shape == inputs.shape


def test_hd95_uses_physical_mm_and_empty_prediction_penalty():
    actual = np.zeros((5, 5, 5), dtype=bool)
    predicted = np.zeros_like(actual)
    actual[1, 1, 1] = True
    predicted[2, 1, 1] = True
    value, empty = physical_hd95(actual, predicted, (2.0, 1.0, 1.0))
    assert value == pytest.approx(2.0)
    assert empty is False
    penalty, empty = physical_hd95(actual, np.zeros_like(actual), (2.0, 1.0, 1.0))
    assert penalty == pytest.approx(
        physical_fov_diagonal(actual.shape, (2.0, 1.0, 1.0))
    )
    assert empty is True and np.isfinite(penalty)


def test_volume_similarity_arithmetic_and_empty_prediction():
    assert volume_similarity_counts(10, 10) == pytest.approx(1.0)
    assert volume_similarity_counts(10, 5) == pytest.approx(2 / 3)
    assert volume_similarity_counts(10, 0) == 0.0
    with pytest.raises(ValueError, match="feta_subject_tissue_absent"):
        volume_similarity_counts(0, 1)


def test_known_cubical_topology_fixtures():
    one = np.zeros((5, 5, 5), dtype=bool)
    one[2, 2, 2] = True
    two = one.copy()
    two[0, 0, 0] = True
    cavity = np.ones((3, 3, 3), dtype=bool)
    cavity[1, 1, 1] = False
    loop = np.zeros((3, 3, 3), dtype=bool)
    loop[:, :, 1] = True
    loop[1, 1, 1] = False
    assert cubical_euler_characteristic(one) == 1
    assert cubical_betti_numbers(one) == (1, 0, 0)
    assert cubical_betti_numbers(two) == (2, 0, 0)
    assert cubical_betti_numbers(cavity) == (1, 0, 1)
    assert cubical_betti_numbers(loop) == (1, 1, 0)
    assert topology_metrics(two, 2)["euler_distance"] == 0
    assert topology_metrics(one, 2)["euler_distance"] == 1


def test_complete_panel_empty_prediction_is_finite_and_flagged():
    actual = np.zeros((7, 7, 7), dtype=np.uint8)
    for label in range(1, 8):
        actual[label - 1, 0, 0] = label
    predicted = np.zeros_like(actual)
    metrics = evaluate_subject_segmentation(actual, predicted, (0.5, 0.5, 0.5))
    assert metrics["macro_dice"] == 0.0
    assert metrics["macro_volume_similarity"] == 0.0
    assert metrics["empty_prediction_count"] == 7
    assert np.isfinite(metrics["macro_hd95_mm"])
    assert all(row["empty_prediction"] for row in metrics["per_class"].values())


def test_native_geometry_restore_uses_nearest_labels(tmp_path: Path):
    import nibabel as nib

    reference_path = tmp_path / "reference.nii.gz"
    native = np.zeros((4, 4, 4), dtype=np.uint8)
    nib.save(nib.Nifti1Image(native, np.eye(4)), reference_path)
    prediction = np.zeros((2, 2, 2), dtype=np.uint8)
    prediction[1, 1, 1] = 7
    restored = restore_prediction_to_native(
        prediction, np.diag([2.0, 2.0, 2.0, 1.0]), reference_path
    )
    assert restored.shape == native.shape
    assert set(np.unique(restored)).issubset({0, 7})
