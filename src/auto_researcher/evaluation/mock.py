"""Backward-compatible construction of the synthetic task evaluator."""

from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic.evaluator import SyntheticEvaluator
from auto_researcher.tasks.synthetic.task import SyntheticTask


class MockEvaluator(SyntheticEvaluator):
    """Compatibility name retained for PR 1 callers."""

    def __init__(self) -> None:
        task = SyntheticTask()
        context = TaskRuntimeContext()
        super().__init__(
            context,
            task.experiment_metadata(context),
            task.dataset_manifest(context),
        )
