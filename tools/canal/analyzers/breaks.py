"""Graph break analysis."""

from __future__ import annotations

from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import Break, BreaksResult, ExplainData


class BreaksAnalysis(Analysis):
    name = "breaks"
    requires = ["explain"]

    def run(self, collected: dict[tuple[int, str], Any]) -> BreaksResult:
        data: ExplainData = collected[(0, "explain")]

        breaks = [
            Break(
                type=br.reason.split("\n")[0],
                reason=br.reason,
                user_stack=br.user_stack,
            )
            for br in data.break_reasons
        ]

        return BreaksResult(
            graph_count=data.graph_count,
            graph_break_count=data.graph_break_count,
            op_count=data.op_count,
            breaks=breaks,
            graphs=data.graphs,
        )
