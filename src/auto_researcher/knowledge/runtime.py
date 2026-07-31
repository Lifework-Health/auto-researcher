"""Replay-safe execution of one bounded knowledge retrieval."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from auto_researcher.contracts.enums import KnowledgeRetrievalStatus
from auto_researcher.knowledge.artifacts import (
    knowledge_artefact_references,
    write_knowledge_artefacts,
)
from auto_researcher.knowledge.identity import bundle_content_hash
from auto_researcher.knowledge.identity import content_hash
from auto_researcher.knowledge.models import (
    KnowledgeBundle,
    KnowledgeErrorCode,
    KnowledgeGroundingPolicy,
    KnowledgeProviderConfiguration,
    KnowledgeRetrievalRecord,
    KnowledgeRetrievalRequest,
)
from auto_researcher.knowledge.protocols import (
    KnowledgeProvider,
    KnowledgeProviderError,
)
from auto_researcher.knowledge.store import (
    KnowledgeRetrievalStore,
    retrieval_record_id,
)
from auto_researcher.knowledge.validation import KnowledgeBundleValidator
from auto_researcher.tasks.models import TaskRuntimeContext


class KnowledgeRetrievalExecutionError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class KnowledgeRetrievalCoordinator:
    def __init__(
        self,
        *,
        store: KnowledgeRetrievalStore,
        validator: KnowledgeBundleValidator,
        runtime_context: TaskRuntimeContext,
        clock: Callable[[], datetime],
    ) -> None:
        self.store = store
        self.validator = validator
        self.runtime_context = runtime_context
        self.clock = clock

    def replay(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeBundle | None:
        """Return an exact completed bundle without touching the provider."""
        selected = self._select_retrieval(request.retrieval_id)
        completed = self._completed(selected)
        if completed is None:
            return None
        assert completed.bundle is not None
        return completed.bundle

    def run(
        self,
        request: KnowledgeRetrievalRequest,
        provider: KnowledgeProvider,
        configuration: KnowledgeProviderConfiguration,
        policy: KnowledgeGroundingPolicy,
    ) -> tuple[KnowledgeBundle, bool]:
        replayed = self.replay(request)
        if replayed is not None:
            return replayed, True
        selected = self._select_retrieval(request.retrieval_id)
        selected_request = (
            request
            if selected == request.retrieval_id
            else self.store.latest(selected).request
        )
        latest = self.store.latest(selected)
        retry_of = latest.retry_of_retrieval_id if latest else None
        reservation = self._record(
            selected_request,
            status=KnowledgeRetrievalStatus.RESERVED,
            retry_of=retry_of,
            provider_request_started=True,
        )
        self.store.append(reservation)
        try:
            if content_hash(policy) != selected_request.grounding_policy_hash:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.BUNDLE_VALIDATION_FAILED.value
                )
            draft = provider.retrieve(selected_request)
            references = knowledge_artefact_references(
                self.runtime_context,
                selected_request.retrieval_id,
            )
            draft = draft.model_copy(
                update={
                    "artefact_references": references,
                    "bundle_hash": "0" * 64,
                }
            )
            draft = draft.model_copy(update={"bundle_hash": bundle_content_hash(draft)})
            bundle = self.validator.validate(
                draft,
                policy,
                provider_id=provider.provider_id,
                schema_version=configuration.schema_version,
                content_version=configuration.content_version,
                maximum_records=configuration.maximum_records,
                query_plan_hash=selected_request.query_plan_hash,
            )
            if not bundle.validation_result.passed:
                raise KnowledgeProviderError(
                    KnowledgeErrorCode.BUNDLE_VALIDATION_FAILED.value
                )
            write_knowledge_artefacts(
                self.runtime_context,
                selected_request,
                bundle,
            )
        except KnowledgeProviderError as exc:
            try:
                code = KnowledgeErrorCode(exc.code)
            except ValueError:
                code = KnowledgeErrorCode.BUNDLE_VALIDATION_FAILED
            self.store.append(
                self._record(
                    selected_request,
                    status=KnowledgeRetrievalStatus.FAILED,
                    retry_of=retry_of,
                    errors=(code,),
                    provider_request_started=True,
                )
            )
            raise KnowledgeRetrievalExecutionError(code.value) from None
        except (ValueError, OSError):
            code = KnowledgeErrorCode.BUNDLE_VALIDATION_FAILED
            self.store.append(
                self._record(
                    selected_request,
                    status=KnowledgeRetrievalStatus.FAILED,
                    retry_of=retry_of,
                    errors=(code,),
                    provider_request_started=True,
                )
            )
            raise KnowledgeRetrievalExecutionError(code.value) from None
        except Exception:
            code = KnowledgeErrorCode.RETRIEVAL_INDETERMINATE
            self.store.append(
                self._record(
                    selected_request,
                    status=KnowledgeRetrievalStatus.INDETERMINATE,
                    retry_of=retry_of,
                    errors=(code,),
                    provider_request_started=True,
                )
            )
            raise KnowledgeRetrievalExecutionError(code.value) from None
        completed_record = self._record(
            selected_request,
            status=KnowledgeRetrievalStatus.COMPLETED,
            retry_of=retry_of,
            bundle=bundle,
            provider_request_started=True,
        )
        self.store.append(completed_record)
        return bundle, False

    def _select_retrieval(self, base_id: str) -> str:
        latest = self.store.latest(base_id)
        if latest is None:
            return base_id
        if latest.status == KnowledgeRetrievalStatus.COMPLETED:
            return base_id
        if (
            latest.status == KnowledgeRetrievalStatus.RESERVED
            and latest.provider_request_started
        ):
            self._mark_indeterminate(latest)
            raise KnowledgeRetrievalExecutionError(
                KnowledgeErrorCode.RETRIEVAL_INDETERMINATE.value
            )
        if latest.status == KnowledgeRetrievalStatus.FAILED:
            raise KnowledgeRetrievalExecutionError(latest.errors[0].value)
        if latest.status == KnowledgeRetrievalStatus.INDETERMINATE:
            all_records = self.store.list_records(latest.run_id)
            descendants = {base_id}
            changed = True
            while changed:
                changed = False
                for item in all_records:
                    if (
                        item.retry_of_retrieval_id in descendants
                        and item.retrieval_id not in descendants
                    ):
                        descendants.add(item.retrieval_id)
                        changed = True
            latest_children = [
                item
                for child_id in sorted(descendants - {base_id})
                if (item := self.store.latest(child_id)) is not None
            ]
            completed = [
                item
                for item in latest_children
                if item.status == KnowledgeRetrievalStatus.COMPLETED
            ]
            if len(completed) > 1:
                raise KnowledgeRetrievalExecutionError(
                    "conflicting_completed_knowledge_retrievals"
                )
            if completed:
                return completed[0].retrieval_id
            for item in latest_children:
                if (
                    item.status == KnowledgeRetrievalStatus.RESERVED
                    and item.provider_request_started
                ):
                    self._mark_indeterminate(item)
            unused = [
                item
                for item in latest_children
                if item.status == KnowledgeRetrievalStatus.RESERVED
                and not item.provider_request_started
            ]
            if len(unused) == 1:
                return unused[0].retrieval_id
            raise KnowledgeRetrievalExecutionError(
                KnowledgeErrorCode.RETRIEVAL_INDETERMINATE.value
            )
        return base_id

    def _mark_indeterminate(
        self,
        record: KnowledgeRetrievalRecord,
    ) -> None:
        self.store.append(
            record.model_copy(
                update={
                    "record_id": retrieval_record_id(
                        record.retrieval_id,
                        KnowledgeRetrievalStatus.INDETERMINATE,
                        len(self.store.records_for_retrieval(record.retrieval_id)) + 1,
                    ),
                    "status": KnowledgeRetrievalStatus.INDETERMINATE,
                    "errors": (KnowledgeErrorCode.RETRIEVAL_INDETERMINATE,),
                    "created_at": self.clock(),
                }
            )
        )

    def _completed(
        self,
        retrieval_id: str,
    ) -> KnowledgeRetrievalRecord | None:
        completed = [
            item
            for item in self.store.records_for_retrieval(retrieval_id)
            if item.status == KnowledgeRetrievalStatus.COMPLETED
        ]
        if not completed:
            return None
        if len({item.bundle.bundle_hash for item in completed if item.bundle}) != 1:
            raise KnowledgeRetrievalExecutionError(
                "conflicting_completed_knowledge_retrievals"
            )
        return completed[0]

    def _record(
        self,
        request: KnowledgeRetrievalRequest,
        *,
        status: KnowledgeRetrievalStatus,
        retry_of: str | None = None,
        bundle: KnowledgeBundle | None = None,
        errors: tuple[KnowledgeErrorCode, ...] = (),
        provider_request_started: bool,
    ) -> KnowledgeRetrievalRecord:
        ordinal = len(self.store.records_for_retrieval(request.retrieval_id)) + 1
        return KnowledgeRetrievalRecord(
            record_id=retrieval_record_id(
                request.retrieval_id,
                status,
                ordinal,
            ),
            retrieval_id=request.retrieval_id,
            run_id=request.run_id,
            cycle=request.cycle,
            status=status,
            request=request,
            bundle=bundle,
            errors=errors,
            retry_of_retrieval_id=retry_of,
            provider_request_started=provider_request_started,
            created_at=self.clock(),
        )
