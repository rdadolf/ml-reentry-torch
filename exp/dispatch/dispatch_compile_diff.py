#!/usr/bin/env python3
"""Compare eager vs compiled dispatch traces for all model cases.

Outputs a markdown table showing op counts, unique ops, and which ops
were fused away by torch.compile (Inductor backend, CUDA).

Usage: python exp/dispatch/dispatch_compile_diff.py
"""

import warnings

import torch

from shared.models import ALL_CASES
from tools.tracer import CompiledDispatchTracer, DispatchTracer

warnings.filterwarnings("ignore")


def main() -> None:
    rows: list[tuple[str, int, int, int, int, set[str]]] = []

    for name, make_case in ALL_CASES.items():
        torch._dynamo.reset()
        with torch.device("cuda"):
            case = make_case()
            case.model.eval()

            # Eager trace
            t1 = DispatchTracer()
            with torch.no_grad(), t1:
                case.model(*case.make_input())
            eager_count = len(t1.trace.events)
            eager_ops = {e.op for e in t1.trace.events}

            # Compiled trace
            compiled_model = torch.compile(case.model)
            with torch.no_grad():
                compiled_model(*case.make_input())  # warmup
            t2 = CompiledDispatchTracer()
            with torch.no_grad(), t2:
                compiled_model(*case.make_input())
            compiled_count = len(t2.trace.events)
            compiled_ops = {e.op for e in t2.trace.events}

            fused = eager_ops - compiled_ops

            rows.append(
                (
                    name,
                    eager_count,
                    len(eager_ops),
                    compiled_count,
                    len(compiled_ops),
                    fused,
                )
            )

    # Print markdown table
    print(
        f"| {'Model':<20} | {'Eager ops':>10} | {'Unique':>6} "
        f"| {'Compiled ops':>12} | {'Unique':>6} | Fused away |"
    )
    print(f"|{'':-<22}|{'':-<12}|{'':-<8}|{'':-<14}|{'':-<8}|{'':-<40}|")
    for name, ec, eu, cc, cu, fused in rows:
        fused_str = ", ".join(sorted(fused)) if fused else "(none)"
        print(f"| {name:<20} | {ec:>10} | {eu:>6} | {cc:>12} | {cu:>6} | {fused_str} |")


if __name__ == "__main__":
    main()
