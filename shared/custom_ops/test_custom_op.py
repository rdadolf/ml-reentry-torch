"""Tests for reentry custom ops."""

import pytest
import torch
import torch.nn.functional as F

import shared.custom_ops  # noqa: F401 — side-effect: registers ops

# ── Op table ────────────────────────────────────────────────────────
# Each entry: (op, inputs, reference_fn)
# reference_fn(*inputs) produces the expected output.


OP_CASES = {
    "identity": {
        "op": lambda: torch.ops.reentry.identity,
        "inputs": lambda: (torch.randn(4, 8),),
        "reference": lambda x: x.clone(),
        "dtypes": [torch.float32, torch.float64],
    },
    "silu_and_mul": {
        "op": lambda: torch.ops.reentry.silu_and_mul,
        "inputs": lambda: (torch.randn(4, 8), torch.randn(4, 8)),
        "reference": lambda gate, up: F.silu(gate) * up,
        "dtypes": [torch.float32, torch.float64],
    },
    "silu_and_mul_py": {
        "op": lambda: torch.ops.reentry.silu_and_mul_py,
        "inputs": lambda: (torch.randn(4, 8), torch.randn(4, 8)),
        "reference": lambda gate, up: F.silu(gate) * up,
        "dtypes": [torch.float32, torch.float64],
    },
    "silu_and_mul_triton": {
        "op": lambda: torch.ops.reentry.silu_and_mul_triton,
        "inputs": lambda: (
            torch.randn(4, 8, device="cuda"),
            torch.randn(4, 8, device="cuda"),
        ),
        "reference": lambda gate, up: F.silu(gate) * up,
        "dtypes": [torch.float32],
        "cuda": True,
    },
}

_NO_AUTOGRAD = {"silu_and_mul_py", "silu_and_mul_triton"}
_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.fixture(
    params=[
        pytest.param(name, marks=[_requires_cuda] if OP_CASES[name].get("cuda") else [])
        for name in OP_CASES
    ],
)
def op_case(request):
    return OP_CASES[request.param]


class TestOps:
    def test_correctness(self, op_case):
        inputs = op_case["inputs"]()
        result = op_case["op"]()(*inputs)
        expected = op_case["reference"](*inputs)
        assert torch.allclose(result, expected, atol=1e-6), (
            f"max diff: {(result - expected).abs().max()}"
        )

    def test_not_alias(self, op_case):
        """Output does not alias any input."""
        inputs = op_case["inputs"]()
        result = op_case["op"]()(*inputs)
        for inp in inputs:
            assert result.data_ptr() != inp.data_ptr()

    def test_preserves_dtype(self, op_case):
        for dtype in op_case["dtypes"]:
            inputs = tuple(t.to(dtype=dtype) for t in op_case["inputs"]())
            result = op_case["op"]()(*inputs)
            assert result.dtype == dtype

    def test_empty(self, op_case):
        inputs = op_case["inputs"]()
        device = inputs[0].device
        empty_inputs = tuple(torch.empty(0, device=device) for _ in inputs)
        result = op_case["op"]()(*empty_inputs)
        assert result.shape == (0,)

    def test_opcheck(self, op_case):
        inputs = op_case["inputs"]()
        torch.library.opcheck(op_case["op"](), inputs)

    def test_gradcheck(self, op_case, request):
        """Verify gradient math via finite differences."""
        if request.node.callspec.id in _NO_AUTOGRAD:
            pytest.xfail("no autograd formula registered")
        inputs = tuple(t.double().requires_grad_(True) for t in op_case["inputs"]())
        torch.autograd.gradcheck(op_case["op"](), inputs)

    def test_compile(self, op_case):
        op = op_case["op"]()

        @torch.compile(fullgraph=True)
        def f(*args):
            return op(*args)

        inputs = op_case["inputs"]()
        result = f(*inputs)
        expected = op_case["reference"](*inputs)
        assert torch.allclose(result, expected, atol=1e-6)
