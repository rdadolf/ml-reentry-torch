"""Test models for torch.compile analysis (DEV-122).

Models designed to exercise specific compiler behaviors:
graph breaks, fusion boundaries, custom op interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.models.catalog import RMSNorm, SwiGLUFFN, ToyLlama
from shared.models.common import ModelCase, deterministic

# ── Clean compilation ───────────────────────────────────────────────


class PointwiseChain(nn.Module):
    """Pure pointwise chain. Baseline for fusion."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x + 1
        x = x * 2
        x = F.relu(x)
        x = x - 0.5
        return x


def pointwise_chain_case() -> ModelCase:
    return ModelCase(
        model=PointwiseChain(),
        make_input=deterministic(lambda: (torch.randn(1, 64),)),
    )


class MatmulChain(nn.Module):
    """Sequential matmuls, no activations."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(dim, dim, bias=False)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.l3 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l3(self.l2(self.l1(x)))


def matmul_chain_case() -> ModelCase:
    return ModelCase(
        model=MatmulChain(),
        make_input=deterministic(lambda: (torch.randn(1, 64),)),
    )


class LayerNormChain(nn.Module):
    """LayerNorm decomposes into pointwise ops — tests fusion with linears."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.linear(self.norm1(x)))


def layernorm_chain_case() -> ModelCase:
    return ModelCase(
        model=LayerNormChain(),
        make_input=deterministic(lambda: (torch.randn(1, 8, 64),)),
    )


# ── Graph break triggers ───────────────────────────────────────────


class DataDependentBranch(nn.Module):
    """Data-dependent control flow — should cause a graph break."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2
        if x.sum() > 0:
            x = x + 1
        x = x * 3
        return x


def data_dependent_branch_case() -> ModelCase:
    return ModelCase(
        model=DataDependentBranch(),
        # Use ones so the branch condition is always True and the break fires
        make_input=deterministic(lambda: (torch.ones(2, 32),)),
    )


class DynamicShape(nn.Module):
    """Dynamic shapes from data-dependent masking."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        filtered = x[mask]  # dynamic shape
        return filtered.sum().unsqueeze(0)


def dynamic_shape_case() -> ModelCase:
    return ModelCase(
        model=DynamicShape(),
        make_input=deterministic(lambda: (torch.randn(4, 32),)),
    )


# ── Custom ops ──────────────────────────────────────────────────────


class CustomIdentity(nn.Module):
    """Linear → custom identity → linear. Identity should be transparent."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from shared.custom_ops.register import identity

        x = F.relu(self.l1(x))
        x = identity(x)
        return self.l2(x)


def custom_identity_case() -> ModelCase:
    return ModelCase(
        model=CustomIdentity(),
        make_input=deterministic(lambda: (torch.randn(1, 64),)),
    )


class CustomSiluFFN(nn.Module):
    """SwiGLU with custom silu_and_mul op, surrounded by norms and linears."""

    def __init__(self, dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from shared.custom_ops.register import silu_and_mul

        h = self.norm(x)
        h = self.w2(silu_and_mul(self.w1(h), self.w3(h)))
        return self.out(h + x)  # residual


def custom_silu_ffn_case() -> ModelCase:
    return ModelCase(
        model=CustomSiluFFN(),
        make_input=deterministic(lambda: (torch.randn(1, 8, 64),)),
    )


class _CustomSwiGLUFFN(nn.Module):
    """Drop-in SwiGLUFFN replacement using silu_and_mul custom op."""

    def __init__(self, original: SwiGLUFFN):
        super().__init__()
        self.w1 = original.w1
        self.w2 = original.w2
        self.w3 = original.w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from shared.custom_ops.register import silu_and_mul

        return self.w2(silu_and_mul(self.w1(x), self.w3(x)))


class CustomLlamaFFN(ToyLlama):
    """ToyLlama with SwiGLUFFN replaced by custom silu_and_mul op."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for block in self.layers:
            assert isinstance(block.mlp, SwiGLUFFN)
            block.mlp = _CustomSwiGLUFFN(block.mlp)


def custom_llama_ffn_case() -> ModelCase:
    return ModelCase(
        model=CustomLlamaFFN(num_layers=1),
        make_input=deterministic(lambda: (torch.randint(0, 256, (1, 16)),)),
    )


# ── Fusion pairs (native counterparts) ─────────────────────────────


class CustomPointwiseChain(nn.Module):
    """PointwiseChain with custom identity splitting the chain."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from shared.custom_ops.register import identity

        x = F.relu(x)
        x = x + 1
        x = identity(x)  # fusion barrier
        x = x * 2
        x = F.relu(x)
        x = x - 0.5
        return x


def custom_pointwise_chain_case() -> ModelCase:
    return ModelCase(
        model=CustomPointwiseChain(),
        make_input=deterministic(lambda: (torch.randn(1, 64),)),
    )


class SiluFFN(nn.Module):
    """Native SwiGLU — same structure as CustomSiluFFN but using F.silu."""

    def __init__(self, dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.w2(F.silu(self.w1(h)) * self.w3(h))
        return self.out(h + x)


def silu_ffn_case() -> ModelCase:
    return ModelCase(
        model=SiluFFN(),
        make_input=deterministic(lambda: (torch.randn(1, 8, 64),)),
    )
