"""Inductor IR extraction."""

from __future__ import annotations

from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import CompileDebugData, IRResult, Subgraph


class IRAnalysis(Analysis):
    name = "ir"
    requires = ["compile_debug"]

    def run(self, collected: dict[tuple[int, str], Any]) -> IRResult:
        data: CompileDebugData = collected[(0, "compile_debug")]
        return IRResult(
            subgraphs=[
                Subgraph(
                    name=sg.name,
                    ir_pre_fusion=sg.ir_pre_fusion,
                    ir_post_fusion=sg.ir_post_fusion,
                )
                for sg in data.subgraphs
            ]
        )
