import os
from pathlib import Path

import pytest

from auto_researcher.tasks.feta_seg.manifests import inspect_subjects, manifest_hash


@pytest.mark.feta_data
def test_local_feta_manifest_matches_locked_identity():
    value = os.getenv("AUTO_RESEARCHER_FETA_DATA_DIR")
    if not value:
        pytest.skip(
            "set AUTO_RESEARCHER_FETA_DATA_DIR to run the local FeTA identity gate"
        )
    subjects = inspect_subjects(Path(value))
    assert len(subjects) == 80
    assert all(subject.labels == tuple(range(8)) for subject in subjects)
    assert (
        manifest_hash(subjects)
        == "6d6f375fda99512a93bbaaa715d6edb5031c4d4f2356584b578f2ebd9631eacf"
    )
