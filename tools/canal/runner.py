"""Experiment runner — collector dedup, lazy-analyze, JSON output."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from shared.models import ALL, ModelCase
from tools.canal.analyzers import ANALYZERS
from tools.canal.collectors import COLLECTORS
from tools.canal.config import ExperimentConfig
from tools.canal.types import ExperimentResult


def resolve_model(source: str | Callable) -> ModelCase:
    """Resolve a model source to a ModelCase."""
    if isinstance(source, str):
        return ALL[source]()
    result = source()
    if isinstance(result, ModelCase):
        return result
    model, input_fn = result
    return ModelCase(model=model, make_input=input_fn)


def _model_label(source: str | Callable) -> str:
    """Human-readable label for a model source."""
    if isinstance(source, str):
        return source
    return getattr(source, "__name__", "<custom>")


def run_one(exp: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment: collect needed data, then analyze."""
    analysis = ANALYZERS.get(exp.analysis)
    if analysis is None:
        raise ValueError(
            f"Unknown analysis {exp.analysis!r}. Available: {list(ANALYZERS)}"
        )

    # Validate model count
    if len(exp.models) < analysis.model_count:
        raise ValueError(
            f"Analysis {exp.analysis!r} needs {analysis.model_count} model(s), "
            f"got {len(exp.models)}"
        )

    # Resolve models
    cases = [resolve_model(m) for m in exp.models[: analysis.model_count]]
    for case in cases:
        case.model.eval()

    # Determine which collectors are needed (keyed by (model_idx, collector_name))
    needed: set[tuple[int, str]] = set()
    for model_idx in range(analysis.model_count):
        for collector_name in analysis.requires:
            needed.add((model_idx, collector_name))

    # Run collectors, deduped, with dynamo reset between each
    collected: dict[tuple[int, str], Any] = {}
    for model_idx, collector_name in sorted(needed):
        torch._dynamo.reset()
        collector = COLLECTORS[collector_name]
        case = cases[model_idx]
        collected[(model_idx, collector_name)] = collector.collect(
            model=case.model,
            input_fn=case.make_input,
            compile_options=exp.compile_options,
            device=exp.device,
        )

    # Run analysis
    result = analysis.run(collected)

    return ExperimentResult(
        name=exp.name,
        analysis=exp.analysis,
        models=[_model_label(m) for m in exp.models],
        device=exp.device,
        compile_options=exp.compile_options,
        timestamp=datetime.now(UTC).isoformat(),
        torch_version=torch.__version__,
        result=result,
    )


def run_all(
    experiments: list[ExperimentConfig],
    output_dir: str | Path | None = None,
) -> Path:
    """Run all experiments, write JSON results to output_dir."""
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"canal_output/{ts}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect and fix name collisions
    seen: dict[str, int] = {}
    for exp in experiments:
        if exp.name in seen:
            seen[exp.name] += 1
            new_name = f"{exp.name}_{seen[exp.name]}"
            print(
                f"  warning: duplicate name {exp.name!r}, renaming to {new_name!r}",
                file=sys.stderr,
            )
            exp.name = new_name
        else:
            seen[exp.name] = 0

    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "experiments": [],
    }

    for exp in experiments:
        print(f"  [{exp.name}] running...", end=" ", flush=True)
        result = run_one(exp)
        out_file = output_dir / f"{exp.name}.json"
        out_file.write_text(result.to_json())
        manifest["experiments"].append(exp.name)
        print("done")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return output_dir
