"""Toy model configurations for experiments."""

from torchtune.models.llama3 import llama3
from torchtune.modules.transformer import TransformerDecoder


def toy_llama(num_layers: int = 2) -> TransformerDecoder:
    """~107K param llama3 model. RoPE, GQA, RMSNorm, SwiGLU — all the
    architecturally interesting bits, small enough to trace/compile instantly."""
    return llama3(
        vocab_size=256,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=2,
        embed_dim=64,
        max_seq_len=128,
        intermediate_dim=128,
    )
