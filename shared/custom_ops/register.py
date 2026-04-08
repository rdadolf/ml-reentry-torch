"""Register custom ops in the `reentry` namespace.

Usage:
    import shared.custom_ops  # side-effect: loads C++ and registers ops
    torch.ops.reentry.identity(x)
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# ── Build the C++ extension via JIT ─────────────────────────────────
os.environ.setdefault("CXX", "clang++")
_EXT_DIR = Path(__file__).resolve().parent
load(
    name="reentry_ops",
    sources=[str(_EXT_DIR / "ops.cpp")],
    extra_cflags=["-std=c++20"],
    verbose=True,
)

# ── identity ────────────────────────────────────────────────────────


def identity(x: torch.Tensor) -> torch.Tensor:
    """Returns a copy of the input tensor."""
    return torch.ops.reentry.identity(x)


@torch.library.register_fake("reentry::identity")
def _identity_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


# ── silu_and_mul ────────────────────────────────────────────────────


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU and multiply operation."""
    return torch.ops.reentry.silu_and_mul(gate, up)


@torch.library.register_fake("reentry::silu_and_mul")
def _silu_and_mul_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.dtype == torch.double:
        return torch.empty_like(gate)
    else:
        return torch.empty_like(up)
