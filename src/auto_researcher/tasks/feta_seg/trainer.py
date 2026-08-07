"""Pinned FeTA training primitives; full five-fold orchestration requires CUDA."""

import hashlib
import random
from pathlib import Path

from auto_researcher.tasks.feta_seg.configuration import FeTASegConfiguration


def checkpoint_reference(
    path: Path, *, fold: int, best_epoch: int, score: float, output_root: Path
) -> dict:
    resolved = path.resolve()
    relative = resolved.relative_to(output_root.resolve())
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "fold": fold,
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest,
        "best_epoch": best_epoch,
        "validation_score": score,
    }


def create_loss():
    try:
        from monai.losses import DiceCELoss
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


def create_optimizer(model, configuration: FeTASegConfiguration):
    try:
        from torch.optim import AdamW
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )


def require_full_baseline_environment() -> dict:
    try:
        import torch
        import monai
        import nibabel
        import numpy
        import scipy  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("feta_cuda_unavailable_for_full_baseline")
    return {
        "torch": torch.__version__,
        "monai": monai.__version__,
        "nibabel": nibabel.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "amp": "cuda-float16",
        "deterministic_seed_base": 20260807,
    }


def seed_everything(fold: int) -> int:
    seed = 20260807 + fold
    random.seed(seed)
    try:
        import numpy as np
        import torch
        from monai.utils import set_determinism
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    return seed


def sliding_window_predict(inputs, model, configuration: FeTASegConfiguration):
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return sliding_window_inference(
        inputs,
        roi_size=configuration.patch_size,
        sw_batch_size=configuration.sliding_window_batch_size,
        predictor=model,
        overlap=configuration.inference_overlap,
        mode=configuration.inference_blending,
    )
