# Graph Breaks

## Data-dependent branching

Model:
```
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2
        if x.sum() > 0:
            x = x + 1
        return x
```

`breaks` canal analysis and `canal-view` verbose output:
```
=== data_dependent_branch_breaks [data_dependent_branch => breaks] on: cpu
graphs: 2
graph breaks: 1
ops: 0
breaks:
  [generic_jump TensorVariable()]
--- verbose ---
[graph 0]
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 32]"):
        l_x_ = L_x_

        # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
        x: "f32[2, 32]" = l_x_.mul(2);  l_x_ = None
        sum_1: "f32[]" = x.sum()
        gt: "b8[]" = sum_1.gt(0);  sum_1 = None
        return (gt, x)
  ^^^ BREAK: [generic_jump TensorVariable()]
      at <FrameSummary file /x/workspace/shared/models/compiler_models.py, line 87 in forward>

[graph 1]
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 32]"):
        l_x_ = L_x_

        # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
        x: "f32[2, 32]" = l_x_.add(1);  l_x_ = None
        x_1: "f32[2, 32]" = x.mul(3);  x = None
        return (x_1,)
```

Pretty straightforward. `x.sum() > 0` is data-dependent and can't be resovled until runtime, so the graph can't be monolithic. In this example, our input which forced the lazy compilation to execute was `torch.ones(2, 32)` (which sums to more than zero), so we get the `True` path compiled. If we negate the example input (`torch.ones(2, 32)*-1`), we predictably get the opposite path generated:

```
[graph 1]
...
        x: "f32[2, 32]" = l_x_.mul(3);  l_x_ = None
        return (x,)
```

## Data-dependent shapes

Model:
```
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x > 0
        filtered = x[mask]  # dynamic shape
        return filtered.sum().unsqueeze(0)
```

`breaks` canal analysis and `canal-view` verbose output:
```
=== dynamic_shape_breaks [dynamic_shape => breaks] on: cpu
graphs: 2
graph breaks: 1
ops: 0
breaks:
  [Dynamic shape operator]
    Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
--- verbose ---
[graph 0]
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 32]"):
        l_x_ = L_x_

        # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
        mask: "b8[4, 32]" = l_x_.gt(0);  l_x_ = None
        return (mask,)
  ^^^ BREAK: [Dynamic shape operator]
      Explanation: Operator `aten.nonzero.default`'s output shape depends on input Tensor data.
      at <FrameSummary file /x/workspace/shared/models/compiler_models.py, line 106 in forward>
      at <FrameSummary file /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py, line 116 in __torch_function__>

[graph 1]
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "f32[67]"):
        l_stack0_ = L_stack0_

        # File: /opt/venv/lib/python3.12/site-packages/torch/utils/_device.py:116 in __torch_function__, code: return func(*args, **kwargs)
        sum_1: "f32[]" = l_stack0_.sum();  l_stack0_ = None
        unsqueeze: "f32[1]" = sum_1.unsqueeze(0);  sum_1 = None
        return (unsqueeze,)
```

Reason is captured better here: the shape of `x[mask]` is data-dependent and listed clearly. The break itself is clear in the FX code just after the mask. The subsequent `graph 1` shows `L_stack0_` as `f32[67]`, an irregularly-sized type which matches the number of positive values in the sample data we used.

## Break Sweep

I went ahead and swept over all the test models I made, just to see, then fed the results of `canal-view` to Claude for markdown formatting. As expected, intentional graph breaks were breaks, and nothing else had a break (these are very simple and abstract models). 

| Model | Graphs | Breaks | Ops | Break type |
|-------|--------|--------|-----|------------|
| cnn | 1 | 0 | 9 | clean |
| custom_identity | 1 | 0 | 4 | clean |
| custom_llama_ffn | 1 | 0 | 34 | clean |
| custom_pointwise_chain | 1 | 0 | 6 | clean |
| custom_silu_ffn | 1 | 0 | 6 | clean |
| data_dependent_branch | 2 | 1 | 0 | generic_jump TensorVariable() |
| dynamic_shape | 2 | 1 | 0 | Dynamic shape operator |
| embedding | 1 | 0 | 4 | clean |
| gru | 0 | -1 | 0 | not traced |
| layernorm_chain | 1 | 0 | 3 | clean |
| matmul_chain | 1 | 0 | 3 | clean |
| mlp | 1 | 0 | 5 | clean |
| pointwise_chain | 1 | 0 | 5 | clean |
| residual_mlp | 1 | 0 | 6 | clean |
| silu_ffn | 1 | 0 | 6 | clean |
| sparse_gnn | 0 | -1 | 0 | not traced |
| toy_llama | 1 | 0 | 34 | clean |
| transformer | 1 | 0 | 1 | clean |

Notes:
- `gru` and `sparse_gnn` report 0 graphs / -1 breaks — Dynamo does not trace these models (opaque MKL kernel for GRU, incomplete sparse tensor support for GNN).
- All custom op models (`custom_identity`, `custom_silu_ffn`, `custom_llama_ffn`, `custom_pointwise_chain`) compile cleanly with no graph breaks.
- `alias_mutation` compiles cleanly — modern Dynamo handles aliased in-place mutation natively.

