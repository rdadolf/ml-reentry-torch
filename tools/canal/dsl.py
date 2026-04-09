"""DSL helpers for canal config files."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.canal.config import ExperimentConfig


def _get_catalog() -> dict:
    from shared.models import ALL

    return ALL


def _auto_name(models: tuple[str | Callable, ...], analysis: str) -> str:
    """Generate a default experiment name from models and analysis."""
    parts = []
    for m in models:
        if isinstance(m, str):
            parts.append(m)
        else:
            parts.append(getattr(m, "__name__", "custom"))
    return "_".join(parts) + "_" + analysis


def experiment(
    *models: str | Callable,
    name: str | None = None,
    analysis: str = "fx",
    compile_options: dict[str, Any] | None = None,
    device: str = "cpu",
    **extra: Any,
) -> ExperimentConfig:
    """Create an experiment config.

    Positional args are model sources — catalog keys (str) or callables
    returning ModelCase or (model, input_fn). Most analyses need one model;
    fusion needs two.

    Name is auto-generated from model names + analysis if not provided.
    Required when using callable models with no __name__.
    """
    if not models:
        raise ValueError("At least one model is required")

    if name is None:
        name = _auto_name(models, analysis)

    return ExperimentConfig(
        name=name,
        models=models,
        analysis=analysis,
        compile_options=compile_options or {},
        device=device,
        extra=extra,
    )


def sweep(
    models: list[str],
    analysis: str = "fx",
    **kw: Any,
) -> list[ExperimentConfig]:
    """One experiment per model."""
    return [experiment(m, analysis=analysis, **kw) for m in models]


def all_models() -> list[str]:
    """All available model keys from the catalog."""
    return list(_get_catalog().keys())
