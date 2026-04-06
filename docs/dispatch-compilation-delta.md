# Dispatch Compilation Delta

Comparison of ATen dispatch traces between eager execution and torch.compile
(Inductor backend) across all model cases. Each model runs on CUDA (RTX 4070).

**How it works:** A `DispatchTracer` (`ignore_compile_internals=False`) captures
the full eager ATen op sequence — torch.compile falls back to eager when this
mode is active. A `CompiledDispatchTracer` (`ignore_compile_internals=True`)
allows torch.compile to run normally; fused ops (replaced by Triton kernels)
vanish from the trace, while unfused ops (e.g., cuBLAS matmul) remain visible.

The diff shows exactly which ops the compiler fused away.

## Results

| Model                |  Eager ops | Unique | Compiled ops | Unique | Fused away |
|----------------------|------------|--------|--------------|--------|----------------------------------------|
| mlp                  |          9 |      4 |            4 |      3 | aten::addmm, aten::relu, aten::t |
| residual_mlp         |         13 |      5 |            5 |      2 | aten::add.Tensor, aten::addmm, aten::relu, aten::t |
| cnn                  |         12 |      9 |            4 |      3 | aten::addmm, aten::cudnn_batch_norm, aten::max_pool2d_with_indices, aten::mean.dim, aten::relu, aten::t, aten::view |
| gru                  |          8 |      8 |            8 |      8 | (none) |
| embedding            |          9 |      7 |            4 |      4 | aten::addmm, aten::arange.start_step, aten::relu, aten::t, aten::view |
| sparse_gnn           |         18 |     12 |           18 |     12 | (none) |
| transformer          |         43 |     14 |            2 |      2 | aten::_scaled_dot_product_efficient_attention, aten::add.Tensor, aten::addmm, aten::clone, aten::native_layer_norm, aten::permute, aten::relu, aten::select.int, aten::squeeze.dim, aten::t, aten::transpose.int, aten::unsqueeze, aten::view |
| toy_llama            |         96 |     19 |           10 |      3 | aten::_fused_rms_norm, aten::_unsafe_view, aten::add.Tensor, aten::clone, aten::embedding, aten::expand, aten::mm, aten::mul.Tensor, aten::select.int, aten::silu, aten::slice.Tensor, aten::stack, aten::sub.Tensor, aten::t, aten::transpose.int, aten::unsqueeze, aten::view |

## Observations

- **GRU**: zero fusion — RNN control flow prevents Inductor from fusing anything.
- **Sparse GNN**: zero fusion — sparse ops are opaque to Inductor, same as custom ops.
- **Transformer / Llama**: 90–95% fusion — this is the architecture torch.compile was designed for. Nearly all ops collapse into fused Triton kernels.
- **CNN**: cuDNN batch norm, convolutions, and pooling all fuse away; only a few ops survive.
- **MLP / Residual MLP**: moderate fusion — the linear + activation pattern fuses well.
