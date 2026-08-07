"""Local-only FeTA inventory and canonical, path-free dataset identity."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg.metrics import LABEL_NAMES
from auto_researcher.tasks.models import DatasetManifest, TaskRuntimeContext

DATASET_RELEASE = "feta-2.1-export-80"
EXPECTED_MANIFEST_HASH = (
    "6d6f375fda99512a93bbaaa715d6edb5031c4d4f2356584b578f2ebd9631eacf"
)
MANIFEST_VERSION = "feta-dataset-manifest-v1"
LOADER_VERSION = "feta-flat-nifti-loader-v1"
FILE_PATTERN = re.compile(r"^(sub-\d{3})_rec-(mial|irtk)_(T2w|dseg)\.nii(?:\.gz)?$")


@dataclass(frozen=True)
class FeTASubject:
    subject_id: str
    reconstruction_method: str
    image_path: Path
    segmentation_path: Path
    image_sha256: str
    segmentation_sha256: str
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    labels: tuple[int, ...]

    def canonical(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "reconstruction_method": self.reconstruction_method,
            "image_identity": self.image_path.name,
            "segmentation_identity": self.segmentation_path.name,
            "image_sha256": self.image_sha256,
            "segmentation_sha256": self.segmentation_sha256,
            "shape": list(self.shape),
            "spacing": list(self.spacing),
            "labels": list(self.labels),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_pairs(
    data_dir: Path,
) -> tuple[dict[str, tuple[Path, Path, str]], tuple[str, ...]]:
    root = data_dir.expanduser().resolve()
    if (root / "mri_gz").is_dir():
        root = root / "mri_gz"
    grouped: dict[str, dict[str, list[Path] | str]] = {}
    for path in sorted(root.glob("*.nii*")):
        match = FILE_PATTERN.match(path.name)
        if not match:
            continue
        subject_id, method, kind = match.groups()
        entry = grouped.setdefault(
            subject_id, {"method": method, "T2w": [], "dseg": []}
        )
        if entry["method"] != method:
            raise ValueError("feta_duplicate_subject_reconstruction")
        values = entry[kind]
        assert isinstance(values, list)
        values.append(path)
    pairs: dict[str, tuple[Path, Path, str]] = {}
    warnings: list[str] = []
    for subject_id, entry in sorted(grouped.items()):
        images = entry["T2w"]
        labels = entry["dseg"]
        assert isinstance(images, list) and isinstance(labels, list)
        if not images:
            raise ValueError("feta_image_missing")
        if not labels:
            raise ValueError("feta_label_missing")
        if len(labels) != 1:
            raise ValueError("feta_duplicate_segmentation")
        compressed = [path for path in images if path.name.endswith(".nii.gz")]
        selected = compressed[0] if len(compressed) == 1 else images[0]
        if len(images) > 1:
            if any(_sha256(path) != _sha256(selected) for path in images):
                # Container hashes can differ; voxel equality is verified during audit.
                warnings.append(f"{subject_id}:duplicate_image_container_selected_gzip")
            else:
                warnings.append(f"{subject_id}:duplicate_image_container")
        pairs[subject_id] = (selected, labels[0], str(entry["method"]))
    return pairs, tuple(warnings)


def inspect_subjects(
    data_dir: Path, *, inspect_labels: bool = True
) -> tuple[FeTASubject, ...]:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("feta_nifti_dependencies_unavailable") from exc
    pairs, _ = discover_pairs(data_dir)
    subjects: list[FeTASubject] = []
    for subject_id, (image_path, label_path, method) in pairs.items():
        image: Any = nib.load(str(image_path))
        label: Any = nib.load(str(label_path))
        if image.shape != label.shape or len(image.shape) != 3:
            raise ValueError("feta_image_label_geometry_mismatch")
        labels = (
            tuple(int(item) for item in np.unique(label.dataobj))
            if inspect_labels
            else tuple(range(8))
        )
        if labels != tuple(range(8)):
            raise ValueError("feta_invalid_labels")
        subjects.append(
            FeTASubject(
                subject_id,
                method,
                image_path,
                label_path,
                _sha256(image_path),
                _sha256(label_path),
                cast(tuple[int, int, int], tuple(int(item) for item in image.shape)),
                cast(
                    tuple[float, float, float],
                    tuple(
                        round(float(item), 7) for item in image.header.get_zooms()[:3]
                    ),
                ),
                labels,
            )
        )
    return tuple(subjects)


def canonical_manifest_payload(subjects: tuple[FeTASubject, ...]) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_release": DATASET_RELEASE,
        "loader_version": LOADER_VERSION,
        "subject_count": len(subjects),
        "reconstruction_counts": dict(
            sorted(Counter(item.reconstruction_method for item in subjects).items())
        ),
        "label_schema": {str(key): value for key, value in LABEL_NAMES.items()},
        "labels": list(range(8)),
        "subjects": [item.canonical() for item in subjects],
    }


def manifest_hash(subjects: tuple[FeTASubject, ...]) -> str:
    return payload_hash(canonical_manifest_payload(subjects))


def build_dataset_manifest(context: TaskRuntimeContext) -> DatasetManifest:
    created_at = context.manifest_created_at or datetime.now(UTC)
    if context.task_options.get("mode") == "smoke" and context.data_dir is None:
        return DatasetManifest(
            task_id="feta_seg",
            dataset_version="feta-generated-smoke-v1",
            files=(),
            hashes={},
            loader_version=LOADER_VERSION,
            created_at=created_at,
            metadata={"scientific_baseline": False, "holdout_subjects_evaluated": 0},
        )
    if context.data_dir is None:
        raise RuntimeError("feta_data_unavailable")
    # The registered byte identities prove the audited label content. Building a
    # runtime manifest reads hold-out headers/hashes but never decodes hold-out
    # label voxels; only development labels enter training or metric code.
    subjects = inspect_subjects(context.data_dir, inspect_labels=False)
    payload = canonical_manifest_payload(subjects)
    identity = manifest_hash(subjects)
    if identity != EXPECTED_MANIFEST_HASH:
        raise ValueError("feta_dataset_identity_mismatch")
    return DatasetManifest(
        task_id="feta_seg",
        dataset_version=f"{DATASET_RELEASE}+{identity}",
        files=tuple(item.image_path.name for item in subjects)
        + tuple(item.segmentation_path.name for item in subjects),
        hashes={item.image_path.name: item.image_sha256 for item in subjects}
        | {item.segmentation_path.name: item.segmentation_sha256 for item in subjects},
        loader_version=LOADER_VERSION,
        created_at=created_at,
        metadata={
            **payload,
            "manifest_hash": identity,
            "absolute_paths_in_identity": False,
        },
    )
