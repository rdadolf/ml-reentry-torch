"""FX graph extraction."""

from __future__ import annotations

from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import CompileDebugData, FXResult, Subgraph


class FXAnalysis(Analysis):
    name = "fx"
    requires = ["compile_debug"]

    def run(self, collected: dict[tuple[int, str], Any]) -> FXResult:
        data: CompileDebugData = collected[(0, "compile_debug")]
        # Extract only FX-relevant fields per subgraph
        return FXResult(
            subgraphs=[
                Subgraph(
                    name=sg.name,
                    fx_graph=sg.fx_graph,
                    fx_graph_transformed=sg.fx_graph_transformed,
                    fx_graph_runnable=sg.fx_graph_runnable,
                )
                for sg in data.subgraphs
            ]
        )
