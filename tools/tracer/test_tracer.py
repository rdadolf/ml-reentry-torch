"""Smoke tests for the dispatch tracer across all shared models.

Run: pytest tools/tracer/test_tracer.py -v
"""

import torch

from shared.models import ALL, ModelCase, deterministic
from shared.models.catalog import MLP
from tools.tracer import DispatchTracer


def _trace(case: ModelCase) -> DispatchTracer:
    case.model.eval()
    tracer = DispatchTracer()
    with torch.no_grad(), tracer:
        case.model(*case.make_input())
    return tracer


class TestDispatchSmoke:
    """Verify that the tracer captures ops for every model without crashing."""

    def test_all_models_produce_events(self) -> None:
        for name, make_case in ALL.items():
            tracer = _trace(make_case())
            assert len(tracer.trace.events) > 0, f"{name}: no dispatch events"

    def test_events_have_op_names(self) -> None:
        tracer = _trace(ModelCase(MLP(), deterministic(lambda: (torch.randn(1, 64),))))
        for event in tracer.trace.events:
            assert event.op, "event missing op"
            assert "::" in event.op, f"unexpected op format: {event.op}"

    def test_json_roundtrips(self) -> None:
        import json

        tracer = _trace(ModelCase(MLP(), deterministic(lambda: (torch.randn(1, 64),))))
        data = json.loads(tracer.trace.to_json())
        assert len(data) == len(tracer.trace.events)
        assert all("op" in e for e in data)
        assert all("inputs" in e for e in data)
        assert all("outputs" in e for e in data)

    def test_summary_is_nonempty(self) -> None:
        tracer = _trace(ModelCase(MLP(), deterministic(lambda: (torch.randn(1, 64),))))
        summary = tracer.trace.summary()
        assert len(summary) > 0
        assert "aten::" in summary

    def test_deterministic_input(self) -> None:
        for name, make_case in ALL.items():
            case = make_case()
            a = case.make_input(seed=42)
            b = case.make_input(seed=42)
            for t_a, t_b in zip(a, b):
                if isinstance(t_a, torch.Tensor) and isinstance(t_b, torch.Tensor):
                    if t_a.is_sparse:
                        t_a, t_b = t_a.to_dense(), t_b.to_dense()
                    assert torch.equal(t_a, t_b), f"{name}: inputs not deterministic"
