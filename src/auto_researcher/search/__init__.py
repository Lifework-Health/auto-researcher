"""Search backend interfaces."""

from auto_researcher.search.direct import DirectSearchBackend
from auto_researcher.search.protocols import SearchBackend

__all__ = ["DirectSearchBackend", "SearchBackend"]
