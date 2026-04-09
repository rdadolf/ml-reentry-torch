"""Inductor pass enumeration."""

from __future__ import annotations

from typing import Any

from tools.canal.analyzers.base import Analysis
from tools.canal.types import CompileDebugData, InductorPass, PassesResult


def _enumerate_passes() -> list[InductorPass]:
    """Discover Inductor passes from torch._inductor source."""
    passes = []
    try:
        from torch._inductor.fx_passes import (  # type: ignore[attr-defined]
            post_grad,
            pre_grad,
        )

        for name in dir(pre_grad):
            obj = getattr(pre_grad, name)
            if callable(obj) and not name.startswith("_"):
                passes.append(
                    InductorPass(name=name, category="pre_grad", active=False)
                )
        for name in dir(post_grad):
            obj = getattr(post_grad, name)
            if callable(obj) and not name.startswith("_"):
                passes.append(
                    InductorPass(name=name, category="post_grad", active=False)
                )
    except ImportError:
        pass
    return passes


class PassesAnalysis(Analysis):
    name = "passes"
    requires = ["compile_debug"]

    def run(self, collected: dict[tuple[int, str], Any]) -> PassesResult:
        data: CompileDebugData = collected[(0, "compile_debug")]
        passes = _enumerate_passes()

        # Cross-reference with debug log to mark active passes
        log_text = data.inductor_log or ""
        for p in passes:
            if p.name in log_text:
                p.active = True

        active = sum(1 for p in passes if p.active)
        return PassesResult(passes=passes, total=len(passes), active_count=active)
