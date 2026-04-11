"""Shared dataclasses for canal collectors and analyzers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Collector output types ──────────────────────────────────────────


@dataclass
class Subgraph:
    """One compiled subgraph from TORCH_COMPILE_DEBUG output."""

    name: str
    fx_graph: str | None = None
    fx_graph_transformed: str | None = None
    fx_graph_runnable: str | None = None
    ir_pre_fusion: str | None = None
    ir_post_fusion: str | None = None
    output_code: str | None = None
    provenance: dict | None = None


@dataclass
class CompileDebugData:
    """Parsed output from a TORCH_COMPILE_DEBUG run."""

    base: str
    subgraphs: list[Subgraph] = field(default_factory=list)
    dynamo_log: str | None = None
    inductor_log: str | None = None


@dataclass
class BreakReason:
    """A single graph break reason from explain()."""

    reason: str
    user_stack: list[str] = field(default_factory=list)


@dataclass
class ExplainData:
    """Output from torch._dynamo.explain()."""

    graph_count: int
    graph_break_count: int
    op_count: int
    break_reasons: list[BreakReason] = field(default_factory=list)
    graphs: list[str] = field(default_factory=list)  # printed FX graphs


# ── Analysis result types ───────────────────────────────────────────


@dataclass
class FXResult:
    """FX graph extraction per subgraph."""

    subgraphs: list[Subgraph]


@dataclass
class IRResult:
    """Inductor IR extraction per subgraph."""

    subgraphs: list[Subgraph]


@dataclass
class CodegenResult:
    """Generated kernel code per subgraph."""

    subgraphs: list[Subgraph]


@dataclass
class Break:
    """A single graph break."""

    type: str  # first line of reason (the gb_type label)
    reason: str  # full reason string
    user_stack: list[str] = field(default_factory=list)


@dataclass
class BreaksResult:
    """Graph break report."""

    graph_count: int
    graph_break_count: int
    op_count: int
    breaks: list[Break] = field(default_factory=list)
    graphs: list[str] = field(default_factory=list)


@dataclass
class FusionReport:
    """Fusion stats for a single model variant."""

    model_name: str
    kernel_count: int
    pre_fusion_nodes: int
    post_fusion_nodes: int
    fusion_ratio: float


@dataclass
class FusionResult:
    """A/B fusion comparison."""

    reports: list[FusionReport]


@dataclass
class PassLogEntry:
    """Raw pass observation from the collector (includes full graph text)."""

    name: str
    subsystem: str | None
    order: int
    parent_order: int | None  # order of enclosing GTO, None if top-level
    before_graph: str
    after_graph: str
    changed: bool
    match_count: int | None = None
    patterns: list[PatternInfo] = field(default_factory=list)


@dataclass
class PassLogData:
    """Raw output from the pass observer collector."""

    entries: list[PassLogEntry] = field(default_factory=list)


@dataclass
class PatternInfo:
    """A named pattern within a PatternMatcherPass."""

    name: str  # handler name or "<anonymous>"
    category: str  # Graph, Replacement, Lowering
    count: int  # how many entries with this name


@dataclass
class PassEntry:
    """Record of a single pass execution."""

    name: str
    subsystem: str | None
    order: int
    changed: bool
    category: str  # "Graph Pass", "Module Pass", "Pattern Matcher Pass"
    match_count: int | None = None
    pattern_count: int | None = None  # total patterns in a PatternMatcherPass
    patterns: list[PatternInfo] = field(default_factory=list)
    # Only stored for changed passes to keep JSON reasonable
    diff: str | None = None


@dataclass
class PassesResult:
    """Inductor pass observation report."""

    entries: list[PassEntry] = field(default_factory=list)
    total: int = 0
    changed_count: int = 0
    total_matches: int = 0


# ── Top-level experiment result ─────────────────────────────────────


@dataclass
class ExperimentResult:
    """Top-level result from a single experiment."""

    name: str
    analysis: str
    models: list[str]
    device: str
    compile_options: dict[str, Any]
    timestamp: str
    torch_version: str
    result: Any  # analysis-specific dataclass

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentResult:
        """Load from a JSON file. Result field is a plain dict (untyped)."""
        data = json.loads(Path(path).read_text())
        return cls(**data)
