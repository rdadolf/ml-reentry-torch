"""Inductor pass observation — which passes ran, what changed."""

from __future__ import annotations

import difflib
from collections import defaultdict
from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import (
    PassEntry,
    PassesResult,
    PassLogData,
    PatternInfo,
)

# Passes known to use apply_gm_pass (receive GraphModule, not Graph)
_MODULE_PASSES = {
    "constant_fold_uniform_value",
    "decompose_scan_to_while_loop",
    "decompose_map_to_while_loop",
}


class PassesAnalysis(Analysis):
    name = "passes"
    requires = ["pass_observer"]

    def run(self, collected: dict[tuple[int, str], Any]) -> PassesResult:
        data: PassLogData = collected[(0, "pass_observer")]

        # Group children by parent_order
        children: dict[int, list] = defaultdict(list)
        top_level = []
        for e in data.entries:
            if e.parent_order is not None:
                children[e.parent_order].append(e)
            else:
                top_level.append(e)

        results = []
        idx = 0

        for e in top_level:
            kids = children.get(e.order, [])

            # Aggregate match counts from children
            match_count = e.match_count
            for kid in kids:
                if kid.match_count is not None:
                    match_count = (match_count or 0) + kid.match_count

            # Collect patterns from all children (captured at apply time)
            patterns: list[PatternInfo] = []
            for kid in kids:
                patterns.extend(kid.patterns)
            # If this entry itself has patterns (e.g. PM pass with no
            # wrapping GTO), use those
            if not patterns and e.patterns:
                patterns = e.patterns

            pattern_count = sum(p.count for p in patterns) if patterns else None

            # Determine category
            if patterns or any(k.patterns for k in kids):
                category = "Pattern Matcher Pass"
            elif e.name in _MODULE_PASSES:
                category = "Module Pass"
            else:
                category = "Graph Pass"

            # Diff for changed passes
            diff_text = None
            if e.changed:
                diff_lines = difflib.unified_diff(
                    e.before_graph.splitlines(keepends=True),
                    e.after_graph.splitlines(keepends=True),
                    fromfile=f"before ({e.name})",
                    tofile=f"after ({e.name})",
                    n=3,
                )
                diff_text = "".join(diff_lines)

            results.append(
                PassEntry(
                    name=e.name,
                    subsystem=e.subsystem,
                    order=idx,
                    changed=e.changed,
                    category=category,
                    match_count=match_count,
                    pattern_count=pattern_count,
                    patterns=patterns,
                    diff=diff_text,
                )
            )
            idx += 1

        changed_count = sum(1 for e in results if e.changed)
        total_matches = sum(e.match_count for e in results if e.match_count is not None)
        return PassesResult(
            entries=results,
            total=len(results),
            changed_count=changed_count,
            total_matches=total_matches,
        )
