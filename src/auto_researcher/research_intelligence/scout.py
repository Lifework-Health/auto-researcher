"""Deterministic scout for already-retrieved fixture or operator material."""

from auto_researcher.research_intelligence.models import RetrievedSourceMaterial


class OfflineResearchScout:
    scout_id = "already-retrieved-material-scout"
    scout_version = "offline-research-scout-v1"

    def __init__(self, materials: tuple[RetrievedSourceMaterial, ...]) -> None:
        self._materials = materials

    def collect(self) -> tuple[RetrievedSourceMaterial, ...]:
        return tuple(self._materials)
