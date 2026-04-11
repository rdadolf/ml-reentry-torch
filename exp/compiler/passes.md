# FX Pass Analysis (DEV-126)

## CPU

| Model | Passes | Changed | Matches | What changed |
|-------|--------|---------|---------|--------------|
| cnn | 25 | 2 | 1 | `remove_noop_ops`, `reorder_for_locality` |
| custom_identity | 25 | 0 | 0 | — |
| custom_llama_ffn | 25 | 1 | 34 | `reorder_for_locality` |
| custom_pointwise_chain | 25 | 0 | 0 | — |
| custom_silu_ffn | 25 | 1 | 8 | `reorder_for_locality` |
| data_dependent_branch | 50 | 0 | 0 | — (2 subgraphs) |
| dynamic_shape | 50 | 0 | 0 | — (2 subgraphs) |
| embedding | 25 | 1 | 2 | `reorder_for_locality` |
| gru | 0 | 0 | 0 | not traced |
| layernorm_chain | 25 | 1 | 2 | `reorder_for_locality` |
| matmul_chain | 25 | 0 | 0 | — |
| mlp | 25 | 0 | 0 | — |
| pointwise_chain | 25 | 0 | 0 | — |
| residual_mlp | 25 | 0 | 0 | — |
| silu_ffn | 25 | 1 | 8 | `reorder_for_locality` |
| sparse_gnn | 0 | 0 | 0 | not traced |
| toy_llama | 25 | 1 | 34 | `reorder_for_locality` |
| transformer | 25 | 0 | 0 | — |

On CPU, only two FX passes ever change the graph:
- **`reorder_for_locality`** — reorders nodes to place producers closer to consumers. Fires on any model with enough complexity (attention, SwiGLU, layernorm).
- **`remove_noop_ops`** — removes identity alias nodes. Only fires on CNN (BatchNorm alias nodes in eval mode).

Pattern matches without graph changes (e.g., 34 for toy_llama) are from `early_patterns` handlers (`pointless_view`, `pointless_view_pair`, `pointless_permute_pair`) that match but decide the view/permute isn't actually redundant.

Graph break models (`data_dependent_branch`, `dynamic_shape`) run the full pipeline twice — once per subgraph.

## CUDA

| Model | Passes | Changed | Matches | What changed |
|-------|--------|---------|---------|--------------|
| cnn | 25 | 2 | 1 | `remove_noop_ops`, `reorder_for_locality` |
| custom_identity | — | — | — | ERROR: no CUDA kernel for `reentry::identity` |
| custom_llama_ffn | — | — | — | ERROR: no CUDA kernel for `reentry::silu_and_mul` |
| custom_pointwise_chain | — | — | — | ERROR: no CUDA kernel for `reentry::identity` |
| custom_silu_ffn | — | — | — | ERROR: no CUDA kernel for `reentry::silu_and_mul` |
| data_dependent_branch | 50 | 0 | 0 | — (2 subgraphs) |
| dynamic_shape | 50 | 0 | 0 | — (2 subgraphs) |
| embedding | 25 | 2 | 3 | `reorder_for_locality`, `pass_pattern_2`(1 match) |
| gru | 0 | 0 | 0 | not traced |
| layernorm_chain | 25 | 1 | 2 | `reorder_for_locality` |
| matmul_chain | 25 | 0 | 0 | — |
| mlp | 25 | 1 | 2 | `pass_pattern_2`(2 matches) |
| pointwise_chain | 25 | 0 | 0 | — |
| residual_mlp | 25 | 1 | 4 | `pass_pattern_2`(4 matches) |
| silu_ffn | 25 | 1 | 8 | `reorder_for_locality` |
| sparse_gnn | 0 | 0 | 0 | not traced |
| toy_llama | 25 | 1 | 34 | `reorder_for_locality` |
| transformer | 25 | 0 | 0 | — |

CUDA adds one significant pass:
- **`pass_pattern_2`** (post-grad pattern matcher, round 3) — decomposes `addmm` (fused bias+matmul) into `mm` + `add` so Triton can fuse the add into the kernel. Fires on mlp (2 matches), residual_mlp (4), and embedding (1). Does not fire on toy_llama because the llama architecture uses `mm` directly (no bias terms in attention/FFN projections).

Custom op models fail on CUDA because our C++ custom ops (`reentry::identity`, `reentry::silu_and_mul`) have no CUDA kernels registered.

## Observations

1. The FX pass pipeline is largely idle for these models. Most of the 25 passes run, check for patterns, find nothing, and return. The real compilation work (kernel fusion, codegen) happens downstream in the Inductor scheduler, which operates on a different IR.

2. `reorder_for_locality` is the most broadly active pass. It doesn't add or remove ops — it reorders nodes in the graph so that data producers are closer to their consumers, improving cache locality during codegen.

3. The `addmm` decomposition (`pass_pattern_2`) is the only CUDA-specific FX transformation observed. It splits fused bias+matmul into separate ops to enable Triton epilog fusion. This doesn't apply to models that already use bias-free `mm` (toy_llama, silu_ffn).

4. The 34 pattern matches on toy_llama (from `early_patterns`: `pointless_view`, `pointless_permute_pair`) are false positives — the handlers fire but determine the matches aren't worth transforming. The match count reflects "how many times a handler was invoked," not "how many transformations were applied."

5. Models that aren't traced by Dynamo (gru, sparse_gnn) produce 0 passes. Graph-broken models produce 2x passes (one full pipeline per subgraph).
