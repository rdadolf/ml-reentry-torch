"""Shared types and utilities for model definitions."""

from collections.abc import Callable
from typing import Any, NamedTuple

import torch
import torch.nn as nn

type InputGenerator = Callable[..., tuple[Any, ...]]
"""callable(seed=0) -> args tuple for model.forward()"""


class ModelCase(NamedTuple):
    """A model paired with a deterministic input generator."""

    model: nn.Module
    make_input: InputGenerator


def deterministic(fn: InputGenerator) -> InputGenerator:
    """Wrap a function so it sets the torch manual seed from a seed= kwarg."""

    def wrapper(*, seed: int = 0) -> tuple[Any, ...]:
        torch.manual_seed(seed)
        return fn()

    return wrapper
