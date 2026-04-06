"""Toy model configurations for experiments.

Each model is packaged as a ModelCase: a named tuple of the model and a
deterministic input generator. Use the catalog at the bottom for discovery.
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torchtune.models.llama3 import llama3
from torchtune.modules.transformer import TransformerDecoder

type InputGenerator = Callable[..., tuple[Any, ...]]
"""callable(seed=0) -> args tuple for model.forward()"""


class ModelCase(NamedTuple):
    """A model paired with a deterministic input generator."""

    model: nn.Module
    make_input: InputGenerator


def _seeded(fn: Callable[..., tuple[Any, ...]]) -> Callable[..., tuple[Any, ...]]:
    """Wrap a function so it sets the torch manual seed from a seed= kwarg."""

    def wrapper(*, seed: int = 0) -> tuple[Any, ...]:
        torch.manual_seed(seed)
        return fn()

    return wrapper


# ── MLP ──────────────────────────────────────────────────────────────
# Dispatch: straightforward linear chain. Every op is a single ATen call
# (aten.addmm for Linear, aten.relu for ReLU). No decomposition surprises.
# Compile: compiles cleanly with zero graph breaks. Inductor fuses the
# ReLU into the preceding matmul (pointwise fusion).
# Optimization: baseline — if this doesn't compile/fuse, nothing will.


class MLP(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def mlp_case() -> ModelCase:
    return ModelCase(
        model=MLP(),
        make_input=_seeded(lambda: (torch.randn(1, 64),)),
    )


# ── Residual MLP ────────────────────────────────────────────────────
# Dispatch: same ops as MLP plus aten.add for the skip connection.
# Compile: the skip connection introduces a diamond in the dataflow graph.
# Inductor must schedule the residual add after the block, which tests
# whether fusion can cross the skip boundary.
# Optimization: the add is a pointwise op adjacent to other pointwise ops
# (ReLU) — a natural fusion target. Interesting to see if Inductor fuses
# the block output + residual add + subsequent ops into one kernel.


class ResidualMLP(nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 128, num_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, dim),
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


def residual_mlp_case() -> ModelCase:
    return ModelCase(
        model=ResidualMLP(),
        make_input=_seeded(lambda: (torch.randn(1, 64),)),
    )


# ── CNN ─────────────────────────────────────────────────────────────
# Dispatch: Conv2d has a backend selection step (dispatch key BackendSelect
# chooses between multiple convolution algorithms). BatchNorm dispatches
# differently in train vs eval mode (running stats update vs. frozen).
# Compile: convolutions are opaque to Inductor on CPU (calls into MKL/oneDNN),
# but the surrounding pointwise ops (ReLU, BatchNorm in eval) can fuse.
# Optimization: the Conv→BN→ReLU pattern is a classic fusion target in
# inference frameworks. Interesting to see what torch.compile does vs.
# what a dedicated inference compiler would fuse.


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def cnn_case() -> ModelCase:
    return ModelCase(
        model=TinyCNN(),
        make_input=_seeded(lambda: (torch.randn(1, 1, 16, 16),)),
    )


# ── GRU (RNN) ───────────────────────────────────────────────────────
# Dispatch: nn.GRU has a special-cased CuDNN dispatch path on CUDA that
# collapses the entire recurrence into a single kernel. On CPU, it
# decomposes into per-timestep ops (linear + sigmoid + tanh + elementwise).
# The dispatch traces will look radically different per device.
# Compile: RNNs are historically hard for torch.compile — the variable-length
# loop over timesteps can cause graph breaks with dynamic shapes. The CuDNN
# path is opaque to Inductor. The decomposed CPU path is traceable but
# creates a long sequential graph.
# Optimization: a stress test for the compiler. The gap between the CuDNN
# fused kernel and the decomposed path is exactly the kind of thing
# torch.compile is supposed to close — does it?


class TinyGRU(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1])


def gru_case() -> ModelCase:
    return ModelCase(
        model=TinyGRU(),
        make_input=_seeded(lambda: (torch.randn(1, 8, 32),)),
    )


# ── Embedding + Pooling ────────────────────────────────────────────
# Dispatch: nn.EmbeddingBag fuses the lookup + reduction into a single
# ATen op (aten.embedding_bag). This is a sparse→dense boundary:
# the input is integer indices, the output is dense vectors. Contrast
# with nn.Embedding → manual mean which dispatches as two separate ops.
# Compile: EmbeddingBag is a single opaque op — nothing for Inductor to
# fuse into. But the layers after it (Linear, activation) are standard
# fusion targets. Interesting boundary between "unfusable sparse lookup"
# and "fusable dense compute".
# Optimization: tests how the compiler handles mixed sparse/dense graphs
# and whether it can optimize the dense tail independently.


class EmbeddingClassifier(nn.Module):
    def __init__(
        self, vocab_size: int = 256, embed_dim: int = 64, num_classes: int = 10
    ):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        return self.head(x)


def embedding_case() -> ModelCase:
    return ModelCase(
        model=EmbeddingClassifier(),
        make_input=_seeded(lambda: (torch.randint(0, 256, (1, 10)),)),
    )


# ── Sparse GNN Layer ────────────────────────────────────────────────
# Dispatch: sparse matmul (torch.sparse.mm) dispatches through SparseCPU /
# SparseCUDA keys — a completely different code path from dense ops.
# The adjacency matrix is sparse CSR, node features are dense. This is
# how PyG and DGL implement message passing under the hood.
# Compile: torch.compile support for sparse tensors is incomplete as of
# 2025. This model tests the boundary: does it graph-break on the sparse
# matmul? Can Inductor fuse the dense ops around it?
# Optimization: this is the gap. Sparse ops are effectively opaque to the
# compiler, similar to custom ops (M2/M4). If you later work on improving
# sparse support in the compiler stack, this model is your baseline.


class SparseGNNLayer(nn.Module):
    """Single GCN-style message passing layer: A @ X @ W + b.

    Uses a sparse adjacency matrix for the graph structure and dense
    linear transforms for node features. This is the core pattern in
    every GNN framework (PyG, DGL, etc.), stripped to its essentials.
    """

    def __init__(self, in_features: int = 32, out_features: int = 32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: sparse [num_nodes, num_nodes], x: dense [num_nodes, in_features]
        x = torch.sparse.mm(adj, x)  # message passing (sparse @ dense)
        x = self.linear(x)  # node feature transform (dense)
        return torch.relu(x)


class TinyGNN(nn.Module):
    """2-layer GCN with sparse adjacency. No PyG dependency."""

    def __init__(
        self,
        num_features: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 8,
    ):
        super().__init__()
        self.layer1 = SparseGNNLayer(num_features, hidden_dim)
        self.layer2 = SparseGNNLayer(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x, adj)
        x = self.layer2(x, adj)
        return x


def _random_sparse_adj(num_nodes: int, density: float = 0.1) -> torch.Tensor:
    """Generate a random sparse adjacency matrix (COO format)."""
    nnz = int(num_nodes * num_nodes * density)
    row = torch.randint(0, num_nodes, (nnz,))
    col = torch.randint(0, num_nodes, (nnz,))
    values = torch.ones(nnz)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (num_nodes, num_nodes)
    ).coalesce()
    return adj


def sparse_gnn_case() -> ModelCase:
    return ModelCase(
        model=TinyGNN(),
        make_input=_seeded(lambda: (torch.randn(16, 32), _random_sparse_adj(16))),
    )


# ── Transformer Encoder Layer ───────────────────────────────────────
# Dispatch: multi-head attention dispatches to scaled_dot_product_attention,
# which has multiple backend paths (FlashAttention, memory-efficient,
# math fallback) selected at dispatch time based on input properties.
# LayerNorm decomposes into mean/var/normalize ops.
# Compile: the attention backend selection is interesting — torch.compile
# can pick a different SDPA backend than eager. The feedforward block
# (Linear→activation→Linear) fuses well.
# Optimization: a single encoder layer is the building block of the full
# transformer. Understanding its dispatch/compile behavior maps directly
# to understanding larger models.


def transformer_encoder_layer(
    d_model: int = 64, nhead: int = 4, dim_feedforward: int = 128
) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        batch_first=True,
    )


def transformer_case() -> ModelCase:
    return ModelCase(
        model=transformer_encoder_layer(),
        make_input=_seeded(lambda: (torch.randn(1, 8, 64),)),
    )


# ── Toy Llama ───────────────────────────────────────────────────────
# Dispatch: modern transformer with RoPE (rotary positional embeddings),
# GQA (grouped-query attention), RMSNorm, and SwiGLU feedforward. Each
# of these has different dispatch characteristics than their older
# counterparts (RoPE: trig ops + complex indexing; RMSNorm: no mean
# subtraction unlike LayerNorm; SwiGLU: gated activation with silu).
# Compile: the most complex model in the set. RoPE's precomputed
# sin/cos table and the GQA head expansion are potential graph break
# or dynamic shape triggers.
# Optimization: the target architecture. If the dispatch tracer and
# compile analysis work on this, they work on real models.


def toy_llama(num_layers: int = 2) -> TransformerDecoder:
    """~107K param llama3 model."""
    return llama3(
        vocab_size=256,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        max_seq_len=128,
        intermediate_dim=128,
    )


def toy_llama_case() -> ModelCase:
    return ModelCase(
        model=toy_llama(num_layers=1),
        make_input=_seeded(lambda: (torch.randint(0, 256, (1, 16)),)),
    )


# ── Catalog ─────────────────────────────────────────────────────────

ALL_CASES: dict[str, Callable[[], ModelCase]] = {
    "mlp": mlp_case,
    "residual_mlp": residual_mlp_case,
    "cnn": cnn_case,
    "gru": gru_case,
    "embedding": embedding_case,
    "sparse_gnn": sparse_gnn_case,
    "transformer": transformer_case,
    "toy_llama": toy_llama_case,
}
