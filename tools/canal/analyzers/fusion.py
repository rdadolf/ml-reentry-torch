"""Fusion analysis — A/B comparison of two model variants."""

from __future__ import annotations

import re
from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import CompileDebugData, FusionReport, FusionResult


def _count_ir_nodes(ir_text: str | None, pattern: str) -> int:
    """Count scheduler nodes in IR text."""
    if not ir_text:
        return 0
    return len(re.findall(pattern, ir_text, re.MULTILINE))


def _fusion_stats(data: CompileDebugData, model_name: str) -> FusionReport:
    """Compute fusion stats from compile debug output."""
    total_pre = 0
    total_post = 0
    total_kernels = 0

    for sg in data.subgraphs:
        total_pre += _count_ir_nodes(sg.ir_pre_fusion, r"^\s*\w+\s*:")
        total_post += _count_ir_nodes(sg.ir_post_fusion, r"^\s*\w+\s*:")
        # Count kernel functions in generated code
        if sg.output_code:
            total_kernels += len(re.findall(r"^def ", sg.output_code, re.MULTILINE))

    fusion_ratio = 1.0 - (total_post / total_pre) if total_pre > 0 else 0.0

    return FusionReport(
        model_name=model_name,
        kernel_count=total_kernels,
        pre_fusion_nodes=total_pre,
        post_fusion_nodes=total_post,
        fusion_ratio=fusion_ratio,
    )


class FusionAnalysis(Analysis):
    name = "fusion"
    requires = ["compile_debug"]
    model_count = 2

    def run(self, collected: dict[tuple[int, str], Any]) -> FusionResult:
        reports = []
        for i in range(2):
            data: CompileDebugData = collected[(i, "compile_debug")]
            # Use base dir name as fallback model name
            reports.append(_fusion_stats(data, model_name=data.base))
        return FusionResult(reports=reports)
