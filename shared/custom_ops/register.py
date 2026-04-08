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


def _identity_backward(ctx, grad_output):
    return grad_output.clone()


def _identity_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


@torch.library.register_fake("reentry::identity")
def _identity_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


torch.library.register_autograd(
    "reentry::identity", _identity_backward, setup_context=_identity_context
)

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


def _silu_and_mul_backward(ctx, grad):
    gate, up = ctx.saved_tensors
    sigmoid_gate = torch.sigmoid(gate)
    grad_gate = grad * up * sigmoid_gate * (1 + gate * (1 - sigmoid_gate))
    grad_up = grad * gate * sigmoid_gate
    return grad_gate, grad_up


def _setup_silu_and_mul_context(ctx, inputs, output):
    gate, up = inputs
    saved_gate, saved_up = None, None
    if ctx.needs_input_grad[0]:
        saved_gate = gate
    if ctx.needs_input_grad[1]:
        saved_up = up
    ctx.save_for_backward(saved_gate, saved_up)


torch.library.register_autograd(
    "reentry::silu_and_mul",
    _silu_and_mul_backward,
    setup_context=_setup_silu_and_mul_context,
)
