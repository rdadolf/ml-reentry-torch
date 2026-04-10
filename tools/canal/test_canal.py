"""Smoke tests for canal — "is this still working?" after changes."""

import json

import pytest
import torch
import torch.nn as nn

from shared.models import ALL, ModelCase, deterministic
from tools.canal.config import ExperimentConfig, load_config
from tools.canal.runner import resolve_model, run_one
from tools.canal.types import ExperimentResult

# ── Config loading ──────────────────────────────────────────────────


def test_load_config(tmp_path):
    config_file = tmp_path / "test_config.py"
    config_file.write_text(
        "from tools.canal.dsl import experiment\n"
        "EXPERIMENTS = [experiment('mlp', analysis='fx')]\n"
    )
    exps = load_config(config_file)
    assert len(exps) == 1
    assert exps[0].name == "mlp_fx"
    assert exps[0].models == ("mlp",)


def test_experiment_config_defaults():
    exp = ExperimentConfig(name="test", models=("mlp",))
    assert exp.analysis == "fx"
    assert exp.device == "cpu"
    assert exp.compile_options == {}


def test_auto_naming():
    from tools.canal.dsl import experiment

    auto = experiment("mlp", analysis="ir")
    assert auto.name == "mlp_ir"


def test_auto_naming_fusion():
    from tools.canal.dsl import experiment

    auto = experiment("mlp", "transformer", analysis="fusion")
    assert auto.name == "mlp_transformer_fusion"


# ── Model resolution ────────────────────────────────────────────────


def test_resolve_catalog_model():
    case = resolve_model("mlp")
    assert isinstance(case, ModelCase)
    assert callable(case.make_input)


def test_resolve_custom_model():
    def make_case():
        return ModelCase(nn.Linear(4, 4), deterministic(lambda: (torch.randn(1, 4),)))

    case = resolve_model(make_case)
    out = case.model(*case.make_input())
    assert out.shape == (1, 4)


# ── FX analysis ────────────────────────────────────────────────────


def test_fx_mlp():
    exp = ExperimentConfig(name="test_mlp", models=("mlp",), analysis="fx")
    result = run_one(exp)
    assert isinstance(result, ExperimentResult)
    assert result.analysis == "fx"
    assert len(result.result.subgraphs) >= 1
    assert result.result.subgraphs[0].fx_graph is not None


# ── IR analysis ────────────────────────────────────────────────────


def test_ir_mlp():
    exp = ExperimentConfig(name="test_ir", models=("mlp",), analysis="ir")
    result = run_one(exp)
    assert len(result.result.subgraphs) >= 1
    assert result.result.subgraphs[0].ir_pre_fusion is not None


# ── Codegen analysis ───────────────────────────────────────────────


def test_codegen_mlp():
    exp = ExperimentConfig(name="test_cg", models=("mlp",), analysis="codegen")
    result = run_one(exp)
    assert len(result.result.subgraphs) >= 1
    assert result.result.subgraphs[0].output_code is not None


# ── Breaks analysis ────────────────────────────────────────────────


def test_breaks_clean_model():
    exp = ExperimentConfig(name="test_breaks", models=("mlp",), analysis="breaks")
    result = run_one(exp)
    assert result.result.graph_break_count == 0
    assert result.result.breaks == []


def test_breaks_with_control_flow():
    class BranchModel(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x + 1
            return x - 1

    def make_case():
        return ModelCase(BranchModel(), deterministic(lambda: (torch.ones(2, 4),)))

    exp = ExperimentConfig(
        name="test_breaks_cf", models=(make_case,), analysis="breaks"
    )
    result = run_one(exp)
    assert len(result.result.breaks) > 0 or result.result.graph_count > 1


# ── JSON roundtrip ──────────────────────────────────────────────────


def test_result_json_roundtrip():
    exp = ExperimentConfig(name="test_rt", models=("mlp",), analysis="breaks")
    result = run_one(exp)
    text = result.to_json()
    loaded = json.loads(text)
    assert loaded["name"] == "test_rt"
    assert loaded["result"]["graph_count"] >= 1


# ── All catalog models ──────────────────────────────────────────────


@pytest.mark.parametrize("model_key", list(ALL.keys()))
def test_fx_all_models(model_key):
    """Every catalog model should run without crashing."""
    exp = ExperimentConfig(name=f"test_{model_key}", models=(model_key,), analysis="fx")
    result = run_one(exp)
    assert isinstance(result, ExperimentResult)
