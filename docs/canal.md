# Canal — torch.compile Analysis Tool

Canal is a config-driven tool for capturing and analyzing torch.compile artifacts:
FX graphs, Inductor IR, generated Triton/C++ code, and compiler diagnostics.

## CLI

```bash
canal <config.py>                    # run all experiments
canal <config.py> --list             # list experiments without running
canal <config.py> --only <name>      # run a single experiment
canal <config.py> -o <dir>           # set output directory
```

Default output goes to `canal_output/<timestamp>/`.

## Config Files

A config file is a Python file that defines an `EXPERIMENTS` list.
Each entry is an `ExperimentConfig`, built using the `experiment()` helper.

### Minimal example

```python
from tools.canal.dsl import experiment

EXPERIMENTS = [
    experiment("mlp", analysis="fx"),
]
```

This auto-names the experiment `"mlp_fx"` and captures FX graphs for the MLP model.

### Full example

```python
import torch
import torch.nn as nn

from shared.models import ModelCase
from tools.canal.dsl import experiment, sweep, all_models

# Inline model definition
class MyModel(nn.Module):
    def forward(self, x):
        return x * 2 + 1

EXPERIMENTS = [
    # Catalog model, different analyses
    experiment("mlp", analysis="fx"),
    experiment("mlp", analysis="ir"),
    experiment("mlp", analysis="codegen"),
    experiment("mlp", analysis="explain"),

    # Custom model — name required for callables
    experiment(
        lambda: ModelCase(MyModel(), lambda seed=0: (torch.randn(2, 4),)),
        name="custom_model",
        analysis="explain",
    ),

    # Fusion: two models (A/B comparison)
    experiment("mlp", "transformer", analysis="fusion"),

    # Sweep: one experiment per catalog model
    *sweep(all_models(), analysis="explain"),

    # Multiple analyses via comprehension
    *[experiment("mlp", analysis=a) for a in ["fx", "ir", "codegen"]],

    # Override compile options
    experiment("mlp", analysis="fx", compile_options={"fullgraph": True}),
]
```

## DSL Reference

### `experiment(*models, name=None, analysis="fx", compile_options=None, device="cpu", **extra)`

Create an experiment config.

Positional args are model sources:
- **str** — key into `shared.models.ALL_CASES` (e.g. `"mlp"`, `"toy_llama"`)
- **callable** — function returning a `ModelCase` or `(model, input_fn)` tuple

Most analyses need one model. Fusion needs two.

Keyword args:
- **name** — experiment name, used as output filename. Auto-generated from model names + analysis if omitted. Required when using callables with no `__name__`.
- **analysis** — which analysis to run (see below)
- **compile_options** — dict of kwargs passed to `torch.compile()`
- **device** — `"cpu"` or `"cuda"`

Auto-naming: `experiment("mlp", analysis="ir")` → name `"mlp_ir"`.
Fusion: `experiment("mlp", "transformer", analysis="fusion")` → name `"mlp_transformer_fusion"`.
Name collisions are detected at run time and resolved with `_1`, `_2` suffixes.

### `sweep(models, analysis="fx", **kw)`

Creates one experiment per model key. Returns a list — use `*sweep(...)` to unpack.

### `all_models()`

Returns all available model keys from `shared.models.ALL_CASES`:
`mlp`, `residual_mlp`, `cnn`, `gru`, `embedding`, `sparse_gnn`, `transformer`, `toy_llama`.

## Analyses

Each analysis declares which collectors it needs. Collectors are run once
even if multiple analyses need them.

| Analysis | Collectors | Models | Description |
|----------|-----------|--------|-------------|
| `fx` | compile_debug | 1 | FX graph pre/post optimization |
| `ir` | compile_debug | 1 | Inductor IR pre/post fusion |
| `codegen` | compile_debug | 1 | Generated Triton/C++ kernel code |
| `breaks` | explain | 1 | Categorized graph break report |
| `fusion` | compile_debug | 2 | A/B kernel and fusion comparison |
| `passes` | compile_debug | 1 | Inductor pass enumeration |

### Collectors

- **compile_debug** — runs `torch.compile()` with `TORCH_COMPILE_DEBUG` tracing enabled.
  Produces FX graphs, Inductor IR, generated code, and provenance mappings.
- **explain** — runs `torch._dynamo.explain()`.
  Produces graph count, break count, op count, and break reasons.

## Output Structure

Each run produces a directory:

```
canal_output/<timestamp>/
  manifest.json        # experiment names, run timestamp
  mlp_fx.json          # one file per experiment
  mlp_ir.json
  ...
```

Output filenames always match the experiment name.

Each experiment JSON is a serialized `ExperimentResult`:

```json
{
  "name": "mlp_fx",
  "analysis": "fx",
  "models": ["mlp"],
  "device": "cpu",
  "compile_options": {},
  "timestamp": "...",
  "torch_version": "...",
  "result": {
    "subgraphs": [
      {
        "name": "model__0_inference_0.0",
        "fx_graph": "class <lambda>(torch.nn.Module):\n  ...",
        "fx_graph_transformed": "...",
        "fx_graph_runnable": "..."
      }
    ]
  }
}
```

The `result` field contains the analysis-specific dataclass (see `tools/canal/types.py`).

## ExperimentConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | auto | Experiment name (auto-generated if omitted) |
| `models` | `tuple[str \| Callable, ...]` | required | Model source(s) |
| `analysis` | `str` | `"fx"` | Analysis to run |
| `compile_options` | `dict` | `{}` | Kwargs for `torch.compile()` |
| `device` | `str` | `"cpu"` | Target device |
| `extra` | `dict` | `{}` | Analysis-specific config |

## Available Catalog Models

| Key | Architecture | Notes |
|-----|-------------|-------|
| `mlp` | 3-layer MLP | Baseline — compiles cleanly, fuses well |
| `residual_mlp` | MLP with skip connections | Tests fusion across residual boundaries |
| `cnn` | Conv2d + BatchNorm + pooling | Conv ops are opaque to Inductor on CPU |
| `gru` | GRU + linear head | RNN uses opaque MKL kernel, may not produce Inductor output |
| `embedding` | EmbeddingBag + MLP head | Sparse/dense boundary |
| `sparse_gnn` | 2-layer GCN with sparse adjacency | Sparse matmul not fully supported by compiler |
| `transformer` | Single TransformerEncoderLayer | SDPA backend selection, feedforward fusion |
| `toy_llama` | Llama3-style decoder (RoPE, GQA, SwiGLU) | Target architecture for M4 custom op work |
