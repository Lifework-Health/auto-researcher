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
