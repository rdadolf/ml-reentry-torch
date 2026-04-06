#!/usr/bin/env python3
"""Trace dispatch events for a named model.

Usage: tracer mlp                  # eager, JSON to stdout
       tracer mlp --compile        # compiled (shows only unfused ops)
       tracer mlp --gpu            # on CUDA
       tracer mlp -H               # short human-readable summary
       tracer --list               # available models
"""

import argparse
from collections import Counter

import torch

from shared.models import ALL_CASES, ModelCase
from tools.tracer import CompiledDispatchTracer, DispatchTracer


def _trace(case: ModelCase, compile: bool = False) -> DispatchTracer:
    case.model.eval()
    model = case.model
    if compile:
        model = torch.compile(model)
        with torch.no_grad():
            model(*case.make_input())  # warmup — triggers compilation
        tracer = CompiledDispatchTracer()
    else:
        tracer = DispatchTracer()
    with torch.no_grad(), tracer:
        model(*case.make_input())
    return tracer


def _human_summary(
    tracer: DispatchTracer, model: str, device: str, compiled: bool
) -> str:
    events = tracer.trace.events
    ops = Counter(e.op for e in events)
    top = ops.most_common(5)
    top_str = ", ".join(f"{name} ({n})" for name, n in top)
    namespaces = sorted({e.namespace for e in events})
    mode = "compiled" if compiled else "eager"
    return (
        f"{model} [{device}, {mode}]: {len(events)} ops, "
        f"{len(ops)} unique, "
        f"namespaces: {', '.join(namespaces)}\n"
        f"  top ops: {top_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace dispatch events for a model.")
    parser.add_argument(
        "model",
        nargs="?",
        choices=ALL_CASES.keys(),
        help="model name from ALL_CASES",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="place model and inputs on CUDA"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model; trace shows only unfused ops",
    )
    parser.add_argument(
        "-H",
        "--human",
        action="store_true",
        help="short human-readable summary instead of JSON",
    )
    parser.add_argument(
        "--list", action="store_true", help="list available models and exit"
    )
    args = parser.parse_args()

    if args.list or args.model is None:
        print("Available models:")
        for name in ALL_CASES:
            print(f"  {name}")
        raise SystemExit(0)

    device = "cuda" if args.gpu else "cpu"
    with torch.device(device):
        case = ALL_CASES[args.model]()
        tracer = _trace(case, compile=args.compile)

    if args.human:
        print(_human_summary(tracer, args.model, device, args.compile))
    else:
        print(tracer.trace.to_json())


if __name__ == "__main__":
    main()
