"""Register custom ops in the `reentry` namespace.

Usage:
    import shared.custom_ops  # side-effect: loads C++ and registers ops
    torch.ops.reentry.identity(x)
"""

import os
from pathlib import Path

import torch
import triton
from torch.library import triton_op, wrap_triton
from torch.utils.cpp_extension import load
from triton import language as tl

# ── Build the C++ extension via JIT ─────────────────────────────────
os.environ.setdefault("CXX", "clang++")
_EXT_DIR = Path(__file__).resolve().parent
load(
    name="reentry_ops",
    sources=[str(_EXT_DIR / "ops.cpp"), str(_EXT_DIR / "ops.cu")],
    extra_cflags=["-std=c++20"],
    extra_cuda_cflags=["-std=c++20"],
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


# ── silu_and_mul_py ─────────────────────────────────────────────────


@torch.library.custom_op("reentry::silu_and_mul_py", mutates_args=())
def silu_and_mul_py(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Pure Python implementation of the silu_and_mul op, for testing."""
    return gate * torch.sigmoid(gate) * up


# ── silu_and_mul_triton ─────────────────────────────────────────────


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_and_mul_triton_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask)
    up = tl.load(up_ptr + offsets, mask=mask)
    output = gate * tl.sigmoid(gate) * up
    tl.store(output_ptr + offsets, output, mask=mask)


@triton_op("reentry::silu_and_mul_triton", mutates_args=())
def silu_and_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Triton implementation of the silu_and_mul op."""
    assert gate.shape == up.shape, "Input tensors must have the same shape"
    assert gate.is_cuda and up.is_cuda, "Inputs must be CUDA tensors"
    output = torch.empty_like(gate)
    n_elements = gate.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    wrap_triton(silu_and_mul_triton_kernel)[grid](gate, up, output, n_elements)  # type: ignore[arg-type]
    return output


# ── silu_and_mul ────────────────────────────────────────────────────


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU and multiply operation."""
    return torch.ops.reentry.silu_and_mul(gate, up)


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

# ── silu_and_mul joint fake tensor registration ─────────────────────
# Since all the version of silu_and_mul have the same signature, we
# can register a single fake implementation for all of them.


@torch.library.register_fake("reentry::silu_and_mul_triton")
@torch.library.register_fake("reentry::silu_and_mul_py")
@torch.library.register_fake("reentry::silu_and_mul")
def _silu_and_mul_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.dtype == torch.double:
        return torch.empty_like(gate)
    else:
        return torch.empty_like(up)
