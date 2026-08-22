"""Protected atomic probability caches for sequential ensemble inference."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

from auto_researcher.tasks.feta_unet_ensemble.aggregation import (
    _validated_probability_tensor,
)
from auto_researcher.tasks.feta_unet_ensemble.models import ProbabilityCacheRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_probability_cache(
    path: Path,
    probabilities: Any,
    *,
    subject_id: str,
    member_identity: str,
) -> ProbabilityCacheRecord:
    """Write one protected float32 tensor without overwriting prior evidence."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional FeTA dependency
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    target = path.expanduser().resolve()
    if target.exists():
        raise ValueError("feta_unet_probability_cache_exists")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tensor = _validated_probability_tensor(probabilities)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, tensor, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        os.chmod(target, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return ProbabilityCacheRecord(
        subject_id=subject_id,
        member_identity=member_identity,
        probability_sha256=_sha256(target),
        shape=tuple(int(item) for item in tensor.shape),
        size_bytes=target.stat().st_size,
    )


def load_probability_cache(path: Path, record: ProbabilityCacheRecord):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional FeTA dependency
        raise RuntimeError("feta_metric_dependencies_unavailable") from exc
    source = path.expanduser().resolve()
    if (
        not source.is_file()
        or source.stat().st_size != record.size_bytes
        or _sha256(source) != record.probability_sha256
    ):
        raise ValueError("feta_unet_probability_cache_identity_mismatch")
    with source.open("rb") as handle:
        tensor = np.load(handle, allow_pickle=False)
    validated = _validated_probability_tensor(tensor)
    if tuple(validated.shape) != record.shape:
        raise ValueError("feta_unet_probability_cache_identity_mismatch")
    return validated


__all__ = ["load_probability_cache", "write_probability_cache"]
