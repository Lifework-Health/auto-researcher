"""One task-neutral registry for installed search backend capabilities."""

from __future__ import annotations

from typing import Any

from auto_researcher.contracts.enums import SearchType
from auto_researcher.search.protocols import SearchCapability


class SearchBackendRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[SearchType, SearchCapability] = {}
        self._backends: dict[SearchType, Any] = {}

    def register(
        self,
        capability: SearchCapability,
        backend: Any | None = None,
    ) -> None:
        if capability.search_type in self._capabilities:
            raise ValueError(
                f"search backend {capability.search_type.value} already registered"
            )
        if capability.available and backend is None:
            raise ValueError("an available search capability requires a backend")
        self._capabilities[capability.search_type] = capability
        if backend is not None:
            self._backends[capability.search_type] = backend

    def capability(self, search_type: SearchType) -> SearchCapability:
        try:
            return self._capabilities[search_type]
        except KeyError as exc:
            raise ValueError(
                f"search backend {search_type.value} is not registered"
            ) from exc

    def backend(self, search_type: SearchType):
        if not self.capability(search_type).available:
            raise ValueError(f"search backend {search_type.value} is unavailable")
        return self._backends[search_type]

    def capabilities(self) -> dict[SearchType, SearchCapability]:
        return dict(self._capabilities)
