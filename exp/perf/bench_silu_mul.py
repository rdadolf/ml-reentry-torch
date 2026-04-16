"""DEV-132: Wall-clock benchmarks for silu_and_mul custom ops.

Compares 4 implementations across 3 execution modes and 3 tensor sizes.

Implementations:
  native     — F.silu(gate) * up (standard PyTorch, no custom op)
  cpp_cuda   — reentry::silu_and_mul (C++ define/impl registration)
  custom_op  — reentry::silu_and_mul_py (@custom_op decorator)
  triton     — reentry::silu_and_mul_triton (@triton_op)

Modes:
  eager            — no compilation
  compiled         — torch.compile(fullgraph=True)
  compiled+graphs  — torch.compile(fullgraph=True, mode="reduce-overhead")

Tensor sizes chosen to be compute-relevant (bandwidth-bound regime):
  (128, 4096), (512, 4096), (1024, 8192)

Output: Compare table with median latency per cell.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.benchmark import Compare, Timer

import shared.custom_ops  # noqa: F401 — registers reentry:: ops

SIZES = [
    (128, 4096),
    (512, 4096),
    (1024, 8192),
]

MIN_RUN_TIME = 2.0
WARMUP_ITERS = 5
WARMUP_ITERS_GRAPHS = 10


def native_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


IMPLS = {
    "native": native_silu_mul,
    "cpp_cuda": torch.ops.reentry.silu_and_mul,
    "custom_op": torch.ops.reentry.silu_and_mul_py,
    "triton": torch.ops.reentry.silu_and_mul_triton,
}

COMPILE_MODES = {
    "eager": None,
    "compiled": {"fullgraph": True},
    "compiled+graphs": {"fullgraph": True, "mode": "reduce-overhead"},
}

# cpp_cuda uses STABLE_TORCH_LIBRARY_IMPL to dispatch directly to a C++
# CUDA kernel.  This dispatch path is invisible to CUDA graph capture,
# producing an empty graph (no-op replay).  Skip that combination.
SKIP = {("cpp_cuda", "compiled+graphs")}


def warmup(fn, gate, up, n=WARMUP_ITERS):
    for _ in range(n):
        fn(gate, up)
    torch.cuda.synchronize()


def build_fn(impl_fn, compile_kwargs):
    if compile_kwargs is None:
        return impl_fn
    return torch.compile(impl_fn, **compile_kwargs)


def main():
    # Match Timer's default num_threads=1 so warmup and measurement
    # agree on global state.  Without this, Timer's thread-count change
    # triggers dynamo guard invalidation → recompile (and, for
    # compiled+graphs, expensive CUDA graph re-capture) inside the
    # timed region.
    torch.set_num_threads(1)

    results = []
    cells = [
        (size, impl_name, mode_name)
        for size in SIZES
        for impl_name in IMPLS
        for mode_name in COMPILE_MODES
        if (impl_name, mode_name) not in SKIP
    ]
    n_cells = len(cells)

    prev_size = None
    gate = up = torch.empty(0, device="cuda")
    for cell, (size, impl_name, mode_name) in enumerate(cells, 1):
        if size != prev_size:
            gate = torch.randn(size, device="cuda", dtype=torch.float32)
            up = torch.randn(size, device="cuda", dtype=torch.float32)
            prev_size = size

        compile_kwargs = COMPILE_MODES[mode_name]
        impl_fn = IMPLS[impl_name]

        # Reset dynamo cache before each compiled measurement.
        # Compiled wrappers sharing the same underlying function
        # pollute each other's code-object cache.
        if compile_kwargs is not None:
            torch._dynamo.reset()

        fn = build_fn(impl_fn, compile_kwargs)
        n_warmup = (
            WARMUP_ITERS_GRAPHS
            if compile_kwargs and "mode" in compile_kwargs
            else WARMUP_ITERS
        )
        warmup(fn, gate, up, n=n_warmup)

        t = Timer(
            stmt="fn(gate, up)",
            globals={"fn": fn, "gate": gate, "up": up},
            label=str(size),
            sub_label=impl_name,
            description=mode_name,
        )
        m = t.blocked_autorange(min_run_time=MIN_RUN_TIME)
        results.append(m)
        warn = " (!)" if m.has_warnings else ""
        print(
            f"  [{cell:2d}/{n_cells}]  {impl_name:12s} | "
            f"{mode_name:16s} | {size} | "
            f"{m.median * 1e6:>8.1f} ± {m.iqr * 1e6:>6.1f} us "
            f"({len(m.times)} blocks){warn}"
        )

    print("\n")
    compare = Compare(results)
    compare.trim_significant_figures()
    compare.highlight_warnings()
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    main()
