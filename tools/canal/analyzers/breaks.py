"""Categorized graph break analysis."""

from __future__ import annotations

import re
from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import BreaksResult, CategorizedBreak, ExplainData

_CATEGORIES = [
    ("control_flow", re.compile(r"control flow|if|while|for|branch", re.I)),
    ("side_effect", re.compile(r"side.?effect|print|log|write", re.I)),
    ("unsupported_op", re.compile(r"unsupported|not supported|unimplemented", re.I)),
    ("dynamic_shape", re.compile(r"dynamic|symbolic|shape", re.I)),
    ("data_dependent", re.compile(r"data.?dependent|item\(\)|\.item", re.I)),
]


def _categorize(reason: str) -> str:
    for category, pattern in _CATEGORIES:
        if pattern.search(reason):
            return category
    return "other"


class BreaksAnalysis(Analysis):
    name = "breaks"
    requires = ["explain"]

    def run(self, collected: dict[tuple[int, str], Any]) -> BreaksResult:
        data: ExplainData = collected[(0, "explain")]

        breaks = [
            CategorizedBreak(
                reason=br.reason,
                category=_categorize(br.reason),
                user_stack=br.user_stack,
            )
            for br in data.break_reasons
        ]

        categories: dict[str, int] = {}
        for b in breaks:
            categories[b.category] = categories.get(b.category, 0) + 1

        return BreaksResult(
            graph_count=data.graph_count,
            graph_break_count=data.graph_break_count,
            op_count=data.op_count,
            breaks=breaks,
            categories=categories,
        )
