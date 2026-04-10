import torch

from shared.models import ModelCase, deterministic
from shared.models.compiler_models import DataDependentBranch
from tools.canal.dsl import experiment

data_dependent_branch_neg = ModelCase(
    model=DataDependentBranch(),
    make_input=deterministic(lambda: (torch.ones(2, 32) * -1,)),
)

EXPERIMENTS = [
    # Graph break analysis on a model with graph breaks
    experiment("data_dependent_branch", analysis="breaks"),
    experiment(
        lambda: data_dependent_branch_neg,
        analysis="breaks",
        name="data_dependent_neg_branch_breaks",
    ),
    experiment("dynamic_shape", analysis="breaks"),
]
