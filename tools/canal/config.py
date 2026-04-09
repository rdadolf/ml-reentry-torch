"""Experiment configuration."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """One experiment to run."""

    name: str
    models: tuple[str | Callable, ...]
    analysis: str = "fx"
    compile_options: dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> list[ExperimentConfig]:
    """Import a Python config file and return its EXPERIMENTS list."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_canal_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path} as a Python module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_canal_config"] = mod
    spec.loader.exec_module(mod)
    experiments = getattr(mod, "EXPERIMENTS", None)
    if experiments is None:
        raise ValueError(f"{path} must define EXPERIMENTS list")
    # Flatten any nested lists (from sweep())
    flat = []
    for item in experiments:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat
