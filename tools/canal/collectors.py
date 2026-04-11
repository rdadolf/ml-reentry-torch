"""Collectors — run expensive operations once, return typed data."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from glob import glob
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tools.canal.types import (
    BreakReason,
    CompileDebugData,
    ExplainData,
    PassLogData,
    PassLogEntry,
    PatternInfo,
    Subgraph,
)

# Files we look for inside each model__*_inference_*/ subdir
_ARTIFACT_FILES = {
    "fx_graph": "fx_graph_readable.py",
    "fx_graph_transformed": "fx_graph_transformed.py",
    "fx_graph_runnable": "fx_graph_runnable.py",
    "ir_pre_fusion": "ir_pre_fusion.txt",
    "ir_post_fusion": "ir_post_fusion.txt",
    "output_code": "output_code.py",
    "provenance": "inductor_provenance_tracking_node_mappings.json",
}


class CompileDebugCollector:
    """Runs torch.compile with debug tracing, parses output."""

    name = "compile_debug"

    def collect(
        self,
        model: nn.Module,
        input_fn: Callable[..., tuple],
        compile_options: dict[str, Any],
        device: str,
    ) -> CompileDebugData:
        torch._dynamo.reset()
        debug_dir = tempfile.mkdtemp(prefix="canal_debug_")

        inductor_config = torch._inductor.config  # type: ignore[attr-defined]
        with inductor_config.patch(
            {
                "trace.enabled": True,
                "trace.debug_dir": debug_dir,
                "fx_graph_cache": False,
                "fx_graph_remote_cache": False,
            }
        ):
            compiled = torch.compile(model, **compile_options)
            with torch.no_grad(), torch.device(device):
                compiled(*input_fn())

        return _parse_debug_dir(debug_dir)


class ExplainCollector:
    """Runs torch._dynamo.explain()."""

    name = "explain"

    def collect(
        self,
        model: nn.Module,
        input_fn: Callable[..., tuple],
        compile_options: dict[str, Any],
        device: str,
    ) -> ExplainData:
        torch._dynamo.reset()
        with torch.no_grad(), torch.device(device):
            explanation = torch._dynamo.explain(model)(*input_fn())

        return ExplainData(
            graph_count=explanation.graph_count,
            graph_break_count=explanation.graph_break_count,
            op_count=explanation.op_count,
            break_reasons=[
                BreakReason(
                    reason=str(br.reason),
                    user_stack=[str(f) for f in (br.user_stack or [])],
                )
                for br in explanation.break_reasons
            ],
            graphs=[g.print_readable(False) for g in explanation.graphs],
        )


class PassObserverCollector:
    """Instruments torch.compile to observe every Inductor pass."""

    name = "pass_observer"

    def collect(
        self,
        model: nn.Module,
        input_fn: Callable[..., tuple],
        compile_options: dict[str, Any],
        device: str,
    ) -> PassLogData:
        from torch._inductor.pattern_matcher import PatternMatcherPass
        from torch.fx.passes.graph_transform_observer import (
            GraphTransformObserver,
        )

        torch._dynamo.reset()
        entries: list[PassLogEntry] = []
        counter = [0]
        # Stack of order IDs for tracking parent-child nesting
        order_stack: list[int] = []
        gto_state: dict[int, dict] = {}

        orig_init = GraphTransformObserver.__init__
        orig_enter = GraphTransformObserver.__enter__
        orig_exit = GraphTransformObserver.__exit__
        orig_pm_apply = PatternMatcherPass.apply

        def patched_init(self_gto, gm, passname, subsystem=None, log_url=None):
            orig_init(self_gto, gm, passname, subsystem, log_url)
            self_gto._obs_passname = passname
            self_gto._obs_subsystem = subsystem

        def patched_enter(self_gto):
            before = _snapshot_graph(self_gto.gm)
            order = counter[0]
            counter[0] += 1
            parent = order_stack[-1] if order_stack else None
            order_stack.append(order)
            gto_state[id(self_gto)] = {
                "before": before,
                "order": order,
                "parent_order": parent,
                "name": getattr(self_gto, "_obs_passname", self_gto.passname),
                "subsystem": getattr(self_gto, "_obs_subsystem", self_gto.subsystem),
            }
            return orig_enter(self_gto)

        def patched_exit(self_gto, exc_type, exc_val, exc_tb):
            state = gto_state.pop(id(self_gto), None)
            result = orig_exit(self_gto, exc_type, exc_val, exc_tb)
            if state is not None:
                order_stack.pop()
                after = _snapshot_graph(self_gto.gm)
                entries.append(
                    PassLogEntry(
                        name=state["name"],
                        subsystem=state["subsystem"],
                        order=state["order"],
                        parent_order=state["parent_order"],
                        before_graph=state["before"],
                        after_graph=after,
                        changed=state["before"] != after,
                    )
                )
            return result

        def patched_pm_apply(self_pm, gm):
            # Extract pattern info before running
            pats = _extract_pattern_info(self_pm)
            count = orig_pm_apply(self_pm, gm)
            # Annotate the most recent entry with this pass_name
            pass_name = (
                self_pm.pass_name
                if self_pm.pass_name is not None
                else "pattern_matcher"
            )
            for entry in reversed(entries):
                if entry.name == pass_name and entry.match_count is None:
                    entry.match_count = count
                    entry.patterns = pats
                    break
            return count

        GraphTransformObserver.__init__ = patched_init  # type: ignore[assignment]
        GraphTransformObserver.__enter__ = patched_enter  # type: ignore[assignment]
        GraphTransformObserver.__exit__ = patched_exit  # type: ignore[assignment]
        PatternMatcherPass.apply = patched_pm_apply  # type: ignore[assignment]

        try:
            inductor_config = torch._inductor.config  # type: ignore[attr-defined]
            with inductor_config.patch(
                {
                    "fx_graph_cache": False,
                    "fx_graph_remote_cache": False,
                }
            ):
                compiled = torch.compile(model, **compile_options)
                with torch.no_grad(), torch.device(device):
                    compiled(*input_fn())
        finally:
            GraphTransformObserver.__init__ = orig_init  # type: ignore[assignment]
            GraphTransformObserver.__enter__ = orig_enter  # type: ignore[assignment]
            GraphTransformObserver.__exit__ = orig_exit  # type: ignore[assignment]
            PatternMatcherPass.apply = orig_pm_apply  # type: ignore[assignment]

        return PassLogData(entries=entries)


_ANON_HANDLER_NAMES = {"fn", "_", None}

_CATEGORY_NAME_MAP = {
    "LoweringPatternEntry": "Lowering",
    "GraphPatternEntry": "Graph",
    "ReplacementPatternEntry": "Replacement",
}


def _extract_pattern_info(pm: Any) -> list[PatternInfo]:
    """Extract pattern names and categories from a live PatternMatcherPass."""
    counts: dict[tuple[str, str], int] = {}
    for entries in pm.patterns.values():
        for entry in entries:
            handler = getattr(entry, "handler", None)
            name = getattr(handler, "__name__", None) if handler else None
            if not name or name in _ANON_HANDLER_NAMES:
                name = "<anonymous>"
            cls_name = type(entry).__name__
            cat = _CATEGORY_NAME_MAP.get(cls_name, "Unknown")
            key = (name, cat)
            counts[key] = counts.get(key, 0) + 1
    return [
        PatternInfo(name=n, category=c, count=cnt)
        for (n, c), cnt in sorted(counts.items())
    ]


def _snapshot_graph(gm: Any) -> str:
    """Get text representation of the FX graph."""
    try:
        if isinstance(gm, torch.fx.GraphModule):
            return gm.print_readable(False)
        elif isinstance(gm, torch.fx.Graph):
            return str(gm)
        return "(unknown graph type)"
    except Exception:
        return "(unavailable)"


# ── Collector registry ──────────────────────────────────────────────

COLLECTORS = {
    "compile_debug": CompileDebugCollector(),
    "explain": ExplainCollector(),
    "pass_observer": PassObserverCollector(),
}


# ── Debug dir parsing ───────────────────────────────────────────────


def _find_inductor_dir(base: Path) -> Path | None:
    """Locate torchinductor/ under either config API or env var layout."""
    # Config API layout: base/torchinductor/model__*/
    direct = base / "torchinductor"
    if direct.is_dir() and glob(str(direct / "model__*")):
        return direct
    # Env var layout: base/torch_compile_debug/run_*/torchinductor/
    run_dirs = sorted(glob(str(base / "torch_compile_debug" / "run_*")))
    if run_dirs:
        candidate = Path(run_dirs[-1]) / "torchinductor"
        if candidate.is_dir():
            return candidate
    return None


def _parse_debug_dir(base: str | Path) -> CompileDebugData:
    """Walk a TORCH_COMPILE_DEBUG output dir into a CompileDebugData."""
    base = Path(base)
    inductor_dir = _find_inductor_dir(base)
    if inductor_dir is None:
        return CompileDebugData(base=str(base))

    model_dirs = sorted(glob(str(inductor_dir / "model__*")))
    subgraphs = []
    for md in model_dirs:
        md = Path(md)
        fields: dict[str, Any] = {"name": md.name}
        for key, filename in _ARTIFACT_FILES.items():
            fpath = md / filename
            if fpath.exists():
                text = fpath.read_text()
                if filename.endswith(".json"):
                    fields[key] = json.loads(text)
                else:
                    fields[key] = text
        subgraphs.append(Subgraph(**fields))

    # Top-level logs
    log_root = inductor_dir.parent
    dynamo_log = log_root / "torchdynamo" / "debug.log"
    inductor_logs = sorted(glob(str(inductor_dir / "*_debug.log")))

    return CompileDebugData(
        base=str(base),
        subgraphs=subgraphs,
        dynamo_log=dynamo_log.read_text() if dynamo_log.exists() else None,
        inductor_log=Path(inductor_logs[0]).read_text() if inductor_logs else None,
    )
