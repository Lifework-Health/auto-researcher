"""Instance-scoped knowledge provider registry."""

from collections.abc import Callable

from auto_researcher.knowledge.protocols import KnowledgeProvider

KnowledgeProviderFactory = Callable[[], KnowledgeProvider]


class KnowledgeProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, KnowledgeProviderFactory] = {}

    def register(self, provider_id: str, factory: KnowledgeProviderFactory) -> None:
        if provider_id in self._factories:
            raise ValueError(f"knowledge provider {provider_id!r} already registered")
        self._factories[provider_id] = factory

    def get(self, provider_id: str) -> KnowledgeProvider:
        try:
            provider = self._factories[provider_id]()
        except KeyError as exc:
            raise KeyError(f"unknown knowledge provider {provider_id!r}") from exc
        if not isinstance(provider, KnowledgeProvider):
            raise TypeError(
                f"knowledge provider {provider_id!r} does not implement the protocol"
            )
        return provider

    def contains(self, provider_id: str) -> bool:
        return provider_id in self._factories

    def list_providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
