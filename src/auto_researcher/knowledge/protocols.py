"""Runtime-checkable provider-neutral knowledge boundaries."""

from typing import Protocol, runtime_checkable

from auto_researcher.knowledge.models import (
    KnowledgeBundle,
    KnowledgeProviderConfiguration,
    KnowledgeReadinessResult,
    KnowledgeRetrievalRequest,
)


@runtime_checkable
class KnowledgeProvider(Protocol):
    provider_id: str
    provider_version: str

    def execution_template_hashes(self) -> dict[str, str]: ...

    def readiness(
        self,
        configuration: KnowledgeProviderConfiguration,
    ) -> KnowledgeReadinessResult: ...

    def retrieve(self, request: KnowledgeRetrievalRequest) -> KnowledgeBundle: ...

    def close(self) -> None: ...


class KnowledgeProviderError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)
