"""Renderers for canal-view — pretty-print experiment results."""

from __future__ import annotations

import re
from typing import Any


def render_summary(data: dict, verbose: bool = False) -> str:
    """Dispatch to the appropriate summary renderer based on analysis type."""
    analysis = data.get("analysis", "unknown")
    renderer = _RENDERERS.get(analysis)
    if renderer is None:
        return f"No renderer for analysis type: {analysis!r}"

    header = _render_header(data)
    body = renderer(data["result"])
    parts = [header, body]

    if verbose:
        verbose_fn = _VERBOSE_RENDERERS.get(analysis)
        if verbose_fn:
            parts.append("--- verbose ---")
            parts.append(verbose_fn(data["result"]))

    return "\n".join(parts)


def _render_header(data: dict) -> str:
    models = ", ".join(data.get("models", []))
    return f"=== {data['name']} [{models} => {data['analysis']}] on: {data['device']}"


# ── FX ──────────────────────────────────────────────────────────────


def _render_fx(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        lines.append(f"subgraph: {sg['name']}")
        fx = sg.get("fx_graph") or ""
        ops = [ln.strip() for ln in fx.split("\n") if "torch.ops." in ln]
        op_names = [_extract_op_name(op) for op in ops]
        lines.append(f"  ops: {len(ops)}")
        if op_names:
            counts: dict[str, int] = {}
            for name in op_names:
                counts[name] = counts.get(name, 0) + 1
            summary = ", ".join(
                f"{name} ({n})" if n > 1 else name for name, n in counts.items()
            )
            lines.append(f"  op list: {summary}")

        transformed = sg.get("fx_graph_transformed")
        if transformed and transformed != fx:
            lines.append("  (FX graph was modified by optimization passes)")
        elif transformed:
            lines.append("  (no FX transforms performed)")

    if not result.get("subgraphs"):
        lines.append("(no subgraphs captured)")
    return "\n".join(lines)


def _extract_op_name(line: str) -> str:
    """Extract op name from a line like 'addmm = torch.ops.aten.addmm.default(...)'"""
    m = re.search(r"torch\.ops\.(\w+\.\w+)", line)
    return m.group(1) if m else line.strip().split("=")[0].strip()


# ── IR ──────────────────────────────────────────────────────────────


def _render_ir(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        lines.append(f"subgraph: {sg['name']}")
        pre = sg.get("ir_pre_fusion") or ""
        post = sg.get("ir_post_fusion") or ""

        # Parse node types from pre-fusion IR
        pre_types = _parse_ir_node_types(pre)
        post_types = _parse_ir_node_types(post)

        lines.append(f"  pre-fusion:  {sum(pre_types.values())} nodes")
        for ntype, count in sorted(pre_types.items()):
            lines.append(f"    {ntype}: {count}")

        lines.append(f"  post-fusion: {sum(post_types.values())} nodes")
        for ntype, count in sorted(post_types.items()):
            lines.append(f"    {ntype}: {count}")

        # Show what ops map to which nodes
        kernels = _parse_ir_kernels(pre)
        if kernels:
            lines.append("  ops:")
            for node_name, kernel in kernels:
                lines.append(f"    {node_name}: {kernel}")

    if not result.get("subgraphs"):
        lines.append("(no subgraphs captured)")
    return "\n".join(lines)


def _parse_ir_node_types(ir_text: str) -> dict[str, int]:
    """Count node types in IR dump (e.g. SchedulerNode, ExternKernelSchedulerNode)."""
    counts: dict[str, int] = {}
    for m in re.finditer(r"^(op\d+): (\w+)", ir_text, re.MULTILINE):
        ntype = m.group(2)
        counts[ntype] = counts.get(ntype, 0) + 1
    return counts


def _parse_ir_kernels(ir_text: str) -> list[tuple[str, str]]:
    """Extract (node_name, kernel) pairs from IR."""
    results = []
    for m in re.finditer(r"^(op\d+)\.node\.kernel\s*=\s*(.+)$", ir_text, re.MULTILINE):
        results.append((m.group(1), m.group(2).strip()))
    return results


# ── Codegen ─────────────────────────────────────────────────────────


def _render_codegen(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        lines.append(f"subgraph: {sg['name']}")
        code = sg.get("output_code") or ""
        total_lines = code.count("\n")
        # Compiled kernels: cpp_fused_*, triton_*, or async_compile calls
        kernels = re.findall(
            r"^(\w*(?:cpp_fused|triton)\w*)\s*=\s*async_compile",
            code,
            re.MULTILINE,
        )
        # Extern kernel calls (MKL, cuBLAS, etc.)
        externs = re.findall(r"extern_kernels\.(\w+)\(", code)
        extern_counts: dict[str, int] = {}
        for e in externs:
            extern_counts[e] = extern_counts.get(e, 0) + 1

        lines.append(f"  generated code: {total_lines} lines")
        lines.append(f"  compiled kernels: {len(kernels)}")
        if kernels:
            lines.append(f"  kernel names: {', '.join(kernels)}")
        if extern_counts:
            ext_str = ", ".join(
                f"{name} ({n})" if n > 1 else name for name, n in extern_counts.items()
            )
            lines.append(f"  extern calls: {ext_str}")

    if not result.get("subgraphs"):
        lines.append("(no subgraphs captured)")
    return "\n".join(lines)


# ── Breaks ──────────────────────────────────────────────────────────


def _render_breaks(result: dict) -> str:
    lines = []
    lines.append(f"graphs: {result['graph_count']}")
    lines.append(f"graph breaks: {result['graph_break_count']}")
    lines.append(f"ops: {result['op_count']}")

    breaks = result.get("breaks", [])
    if breaks:
        lines.append("breaks:")
        for b in breaks:
            lines.append(f"  [{b['type']}]")
            # Include Explanation line if present
            for rl in b["reason"].split("\n")[1:]:
                stripped = rl.strip()
                if stripped.startswith("Explanation:"):
                    lines.append(f"    {stripped}")
                    break
    elif result["graph_break_count"] == 0:
        lines.append("(clean compilation — no graph breaks)")

    return "\n".join(lines)


# ── Fusion ──────────────────────────────────────────────────────────


def _render_fusion(result: dict) -> str:
    lines = []
    reports = result.get("reports", [])
    if not reports:
        return "(no fusion data)"

    hdr = f"{'variant':<30s} {'kernels':>8s} {'pre':>6s} {'post':>6s} {'ratio':>8s}"
    lines.append(hdr)
    lines.append(f"{'-' * 30} {'-' * 8} {'-' * 6} {'-' * 6} {'-' * 8}")
    for r in reports:
        name = r["model_name"]
        if len(name) > 30:
            name = "..." + name[-27:]
        lines.append(
            f"{name:<30s} {r['kernel_count']:>8d} "
            f"{r['pre_fusion_nodes']:>6d} "
            f"{r['post_fusion_nodes']:>6d} "
            f"{r['fusion_ratio']:>7.1%}"
        )
    return "\n".join(lines)


# ── Passes ──────────────────────────────────────────────────────────


def _render_passes(result: dict) -> str:
    entries = result.get("entries", [])
    total = result.get("total", 0)
    changed = result.get("changed_count", 0)
    total_matches = result.get("total_matches", 0)

    # Column widths — CHILD_W chosen so child categories
    # align with parent categories (2 + NAME_W = 4 + CHILD_W)
    NAME_W = 52
    CHILD_W = NAME_W - 2
    CAT_W = 22

    lines: list[str] = []
    current_subsystem = None

    for e in entries:
        sub = e.get("subsystem") or ""
        if sub != current_subsystem:
            current_subsystem = sub
            lines.append(f"  {sub}")

        delta = "Δ" if e.get("changed") else " "
        cat = e.get("category", "")
        name = e.get("name", "")
        idx = e.get("order", "")
        mc = e.get("match_count")
        pc = e.get("pattern_count")

        suffix = ""
        if mc is not None and pc is not None:
            suffix = f"{mc} matches ({pc} patterns)"
        elif mc is not None:
            suffix = f"{mc} matches"
        elif pc is not None:
            suffix = f"({pc} patterns)"

        label = f"[{idx}] {name}"
        lines.append(f"{delta} {label:<{NAME_W}s} {cat:<{CAT_W}s} {suffix}")

        for p in e.get("patterns", []):
            pname = p["name"]
            pcat = p["category"]
            pcount = p["count"]
            cnt = f" x{pcount}" if pcount > 1 else ""
            plabel = f"{pname}{cnt}"
            lines.append(f"    {plabel:<{CHILD_W}s} {pcat}")

    lines.append(
        f"  Total: {total} passes, {changed} changed, {total_matches} pattern matches"
    )
    return "\n".join(lines)


# ── Verbose renderers ───────────────────────────────────────────────


def _verbose_fx(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        # Show post-transform graph if available, otherwise pre-transform
        code = sg.get("fx_graph_transformed") or sg.get("fx_graph")
        if code:
            lines.append(f"[{sg['name']}] FX graph (post-transform):")
            lines.append(code.rstrip())
    return "\n".join(lines) if lines else "(no FX graph)"


def _verbose_ir(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        pre = sg.get("ir_pre_fusion")
        post = sg.get("ir_post_fusion")
        if pre:
            lines.append(f"[{sg['name']}] IR pre-fusion:")
            lines.append(pre.rstrip())
        if post and post != pre:
            lines.append(f"\n[{sg['name']}] IR post-fusion:")
            lines.append(post.rstrip())
        elif post:
            lines.append("(post-fusion IR identical to pre-fusion)")
    return "\n".join(lines) if lines else "(no IR)"


def _verbose_codegen(result: dict) -> str:
    lines = []
    for sg in result.get("subgraphs", []):
        code = sg.get("output_code")
        if code:
            lines.append(f"[{sg['name']}] generated code:")
            lines.append(code.rstrip())
    return "\n".join(lines) if lines else "(no generated code)"


def _verbose_breaks(result: dict) -> str:
    lines = []
    graphs = result.get("graphs", [])
    breaks = result.get("breaks", [])

    for i, graph in enumerate(graphs):
        lines.append(f"[graph {i}]")
        lines.append(graph.rstrip())
        # Show break after this graph if there is one
        if i < len(breaks):
            br = breaks[i]
            lines.append(f"  ^^^ BREAK: [{br['type']}]")
            # Show explanation if present
            for rl in br["reason"].split("\n")[1:]:
                stripped = rl.strip()
                if stripped.startswith("Explanation:"):
                    lines.append(f"      {stripped}")
                    break
            if br.get("user_stack"):
                for frame in br["user_stack"]:
                    lines.append(f"      at {frame}")
        lines.append("")

    return "\n".join(lines) if lines else "(no graphs captured)"


def _verbose_fusion(result: dict) -> str:
    # Fusion doesn't have raw artifacts beyond the stats
    return "(use fx/ir/codegen for full artifacts on each variant)"


def _verbose_passes(result: dict) -> str:
    lines = []
    entries = result.get("entries", [])
    changed = [e for e in entries if e.get("changed")]
    if not changed:
        return "(no passes changed the graph)"
    for e in changed:
        sub = e.get("subsystem") or ""
        lines.append(f"[{e['order']}] {e['name']} ({sub})")
        diff = e.get("diff")
        if diff:
            lines.append(diff.rstrip())
        else:
            lines.append("  (changed but no diff available)")
        lines.append("")
    return "\n".join(lines)


# ── Registry ────────────────────────────────────────────────────────

_RENDERERS: dict[str, Any] = {
    "fx": _render_fx,
    "ir": _render_ir,
    "codegen": _render_codegen,
    "breaks": _render_breaks,
    "fusion": _render_fusion,
    "passes": _render_passes,
}

_VERBOSE_RENDERERS: dict[str, Any] = {
    "fx": _verbose_fx,
    "ir": _verbose_ir,
    "codegen": _verbose_codegen,
    "breaks": _verbose_breaks,
    "fusion": _verbose_fusion,
    "passes": _verbose_passes,
}
