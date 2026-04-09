"""Generated kernel code extraction."""

from __future__ import annotations

from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import CodegenResult, CompileDebugData, Subgraph


class CodegenAnalysis(Analysis):
    name = "codegen"
    requires = ["compile_debug"]

    def run(self, collected: dict[tuple[int, str], Any]) -> CodegenResult:
        data: CompileDebugData = collected[(0, "compile_debug")]
        return CodegenResult(
            subgraphs=[
                Subgraph(
                    name=sg.name,
                    output_code=sg.output_code,
                    provenance=sg.provenance,
                )
                for sg in data.subgraphs
            ]
        )
