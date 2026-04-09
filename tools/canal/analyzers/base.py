"""Base class for analyses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Analysis(ABC):
    """Base class for all analyses.

    Subclasses declare which collectors they need via `requires`,
    and implement `run()` to produce a typed result from collected data.
    """

    name: str
    # Collector names this analysis needs
    requires: list[str]
    # How many models this analysis expects (1 for most, 2 for fusion)
    model_count: int = 1

    @abstractmethod
    def run(self, collected: dict[tuple[int, str], Any]) -> Any:
        """Run the analysis on collected data.

        Args:
            collected: dict keyed by (model_index, collector_name) → collector output
        """
        ...
