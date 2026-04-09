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
        )


# ── Collector registry ──────────────────────────────────────────────

COLLECTORS = {
    "compile_debug": CompileDebugCollector(),
    "explain": ExplainCollector(),
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
