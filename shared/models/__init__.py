"""Shared model definitions and catalog.

Usage:
    from shared.models import TEST_MODELS, ModelCase
    from shared.models.catalog import MLP, ToyLlama
"""

from collections.abc import Callable

from shared.models.catalog import CATALOG
from shared.models.common import InputGenerator, ModelCase, deterministic
from shared.models.compiler_models import (
    custom_identity_case,
    custom_llama_ffn_case,
    custom_pointwise_chain_case,
    custom_py_silu_ffn_case,
    custom_silu_ffn_case,
    custom_triton_silu_ffn_case,
    data_dependent_branch_case,
    dynamic_shape_case,
    layernorm_chain_case,
    matmul_chain_case,
    pointwise_chain_case,
    silu_ffn_case,
)

TEST_MODELS: dict[str, Callable[[], ModelCase]] = {
    **CATALOG,
    # Compiler test models
    "pointwise_chain": pointwise_chain_case,
    "matmul_chain": matmul_chain_case,
    "layernorm_chain": layernorm_chain_case,
    "data_dependent_branch": data_dependent_branch_case,
    "dynamic_shape": dynamic_shape_case,
    "custom_identity": custom_identity_case,
    "custom_silu_ffn": custom_silu_ffn_case,
    "custom_py_silu_ffn": custom_py_silu_ffn_case,
    "custom_llama_ffn": custom_llama_ffn_case,
    "custom_pointwise_chain": custom_pointwise_chain_case,
    "silu_ffn": silu_ffn_case,
}

# Models requiring CUDA — not in TEST_MODELS to avoid breaking CPU-only test sweeps.
CUDA_MODELS: dict[str, Callable[[], ModelCase]] = {
    "custom_triton_silu_ffn": custom_triton_silu_ffn_case,
}

# Complete registry for name-based lookup (canal, experiments).
ALL_MODELS: dict[str, Callable[[], ModelCase]] = {**TEST_MODELS, **CUDA_MODELS}

__all__ = [
    "TEST_MODELS",
    "ALL_MODELS",
    "CUDA_MODELS",
    "ModelCase",
    "InputGenerator",
    "deterministic",
]
