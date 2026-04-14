"""DEV-128: Dispatch tracing with custom ops.

Traces custom_silu_ffn and its native counterpart (silu_ffn) in two modes:
  1. Eager — full ATen dispatch sequence via DispatchTracer
  2. torch.compile (default) — only unfused ops via CompiledDispatchTracer

Saves JSON traces to traces/ and prints a side-by-side comparison showing:
  - Which dispatch keys fire in each mode
  - Where the custom op appears vs native ops
  - What the compiler fused away
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from shared.models import ALL
from tools.tracer import CompiledDispatchTracer, DispatchTracer

TRACE_DIR = Path(__file__).parent / "traces"
MODELS = ["custom_silu_ffn", "silu_ffn"]


def trace_eager(model_name: str) -> DispatchTracer:
    case = ALL[model_name]()
    case.model.eval()
    tracer = DispatchTracer()
    with torch.no_grad(), tracer:
        case.model(*case.make_input())
    return tracer


def trace_compiled(model_name: str) -> CompiledDispatchTracer:
    case = ALL[model_name]()
    case.model.eval()
    compiled = torch.compile(case.model)
    with torch.no_grad():
        compiled(*case.make_input())  # warmup
    tracer = CompiledDispatchTracer()
    with torch.no_grad(), tracer:
        compiled(*case.make_input())
    return tracer


def save_trace(tracer: DispatchTracer, name: str) -> Path:
    path = TRACE_DIR / f"{name}.json"
    path.write_text(tracer.trace.to_json())
    return path


def op_summary(tracer: DispatchTracer) -> dict:
    """Extract summary stats from a trace."""
    events = tracer.trace.events
    ops = Counter(e.op for e in events)
    namespaces = sorted({e.namespace for e in events})
    custom_ops = [e for e in events if e.namespace != "aten"]
    return {
        "total_ops": len(events),
        "unique_ops": len(ops),
        "namespaces": namespaces,
        "op_counts": ops.most_common(),
        "custom_ops": [(e.op, e.index) for e in custom_ops],
    }


def print_comparison(results: dict[str, dict[str, dict]]) -> None:
    sep = "=" * 72
    for model_name, modes in results.items():
        print(f"\n{sep}")
        print(f"  {model_name}")
        print(sep)

        for mode_name, summary in modes.items():
            print(f"\n  [{mode_name}]")
            print(f"    Total ops:   {summary['total_ops']}")
            print(f"    Unique ops:  {summary['unique_ops']}")
            print(f"    Namespaces:  {', '.join(summary['namespaces'])}")
            if summary["custom_ops"]:
                print(f"    Custom ops:  {summary['custom_ops']}")
            print("    Top ops:")
            for op, count in summary["op_counts"][:8]:
                print(f"      {op}: {count}")

        # Diff: what did compile fuse away?
        if "eager" in modes and "compile" in modes:
            eager_ops = set(dict(modes["eager"]["op_counts"]).keys())
            compiled_ops = set(dict(modes["compile"]["op_counts"]).keys())
            fused = eager_ops - compiled_ops
            new = compiled_ops - eager_ops
            print("\n  [compile delta]")
            print(f"    Ops fused away:  {len(fused)}")
            if fused:
                for op in sorted(fused):
                    print(f"      - {op}")
            if new:
                print("    New ops in compiled:")
                for op in sorted(new):
                    print(f"      + {op}")
            e_count = modes["eager"]["total_ops"]
            c_count = modes["compile"]["total_ops"]
            print(f"    Op count: {e_count} (eager) -> {c_count} (compiled)")


def main() -> None:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict]] = {}

    for model_name in MODELS:
        results[model_name] = {}
        print(f"Tracing {model_name} (eager)...", flush=True)
        eager = trace_eager(model_name)
        save_trace(eager, f"{model_name}_eager")
        results[model_name]["eager"] = op_summary(eager)

        print(f"Tracing {model_name} (compile)...", flush=True)
        compiled = trace_compiled(model_name)
        save_trace(compiled, f"{model_name}_compile")
        results[model_name]["compile"] = op_summary(compiled)

    print_comparison(results)

    # Save combined summary
    summary_path = TRACE_DIR / "comparison_summary.json"
    # Convert Counters to serializable form
    serializable = {}
    for model, modes in results.items():
        serializable[model] = {}
        for mode, summary in modes.items():
            serializable[model][mode] = {
                **summary,
                "op_counts": dict(summary["op_counts"]),
            }
    summary_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nTraces saved to {TRACE_DIR}/")


if __name__ == "__main__":
    main()
