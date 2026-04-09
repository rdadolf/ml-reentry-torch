"""Demo canal config — exercises various analyses."""

import torch
import torch.nn as nn

from shared.models import ModelCase, deterministic
from tools.canal.dsl import experiment


# A model that deliberately causes graph breaks
class GraphBreakModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2
        if x.sum() > 0:  # data-dependent control flow
            x = x + 1
        return x


EXPERIMENTS = [
    # FX graph capture
    experiment("mlp", analysis="fx"),
    experiment("transformer", analysis="fx"),
    # Inductor IR
    experiment("mlp", analysis="ir"),
    # Generated code
    experiment("mlp", analysis="codegen"),
    # Graph break analysis
    experiment("mlp", analysis="breaks"),
    # Breaks on a model with graph breaks
    experiment(
        lambda: ModelCase(
            GraphBreakModel(), deterministic(lambda: (torch.randn(2, 4),))
        ),
        name="control_flow",
        analysis="breaks",
    ),
]
