"""Safe atomic knowledge artefacts outside checkpoint state."""

from pathlib import Path

from auto_researcher.knowledge.models import (
    KnowledgeBundle,
    KnowledgeRetrievalRequest,
)
from auto_researcher.tasks.artifacts import atomic_json_write, safe_segment
from auto_researcher.tasks.models import TaskRuntimeContext

KNOWLEDGE_ARTEFACT_FILENAMES = (
    "retrieval_request.json",
    "query_plan.json",
    "graph_snapshot.json",
    "knowledge_bundle.json",
    "validation_summary.json",
)


def knowledge_artefact_references(
    context: TaskRuntimeContext,
    retrieval_id: str,
) -> tuple[str, ...]:
    if context.output_dir is None or not context.run_id:
        return ()
    run_id = safe_segment(context.run_id, "run_id")
    retrieval = safe_segment(retrieval_id, "retrieval_id")
    prefix = Path("runs") / run_id / "knowledge" / retrieval
    return tuple(
        (prefix / filename).as_posix() for filename in KNOWLEDGE_ARTEFACT_FILENAMES
    )


def write_knowledge_artefacts(
    context: TaskRuntimeContext,
    request: KnowledgeRetrievalRequest,
    bundle: KnowledgeBundle,
) -> None:
    references = knowledge_artefact_references(context, request.retrieval_id)
    if not references:
        return
    assert context.output_dir is not None
    values = (
        request,
        request.query_plan,
        bundle.graph_snapshot,
        bundle,
        bundle.validation_result,
    )
    for relative, value in zip(references, values, strict=True):
        atomic_json_write(context.output_dir / relative, value)
