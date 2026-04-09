"""Shared model definitions and catalog.

Usage:
    from shared.models import ALL, ModelCase
    from shared.models.catalog import MLP, ToyLlama
"""

from collections.abc import Callable

from shared.models.catalog import CATALOG
from shared.models.common import InputGenerator, ModelCase, deterministic
from shared.models.compiler_models import (
    custom_identity_case,
    custom_llama_ffn_case,
    custom_pointwise_chain_case,
    custom_silu_ffn_case,
    data_dependent_branch_case,
    dynamic_shape_cat_case,
    inplace_on_input_case,
    layernorm_chain_case,
    matmul_chain_case,
    pointwise_chain_case,
    silu_ffn_case,
)

ALL: dict[str, Callable[[], ModelCase]] = {
    **CATALOG,
    # Compiler test models
    "pointwise_chain": pointwise_chain_case,
    "matmul_chain": matmul_chain_case,
    "layernorm_chain": layernorm_chain_case,
    "data_dependent_branch": data_dependent_branch_case,
    "dynamic_shape_cat": dynamic_shape_cat_case,
    "inplace_on_input": inplace_on_input_case,
    "custom_identity": custom_identity_case,
    "custom_silu_ffn": custom_silu_ffn_case,
    "custom_llama_ffn": custom_llama_ffn_case,
    "custom_pointwise_chain": custom_pointwise_chain_case,
    "silu_ffn": silu_ffn_case,
}

__all__ = ["ALL", "ModelCase", "InputGenerator", "deterministic"]
