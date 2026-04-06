"""Dispatch tracer using TorchDispatchMode.

Intercepts ATen operator dispatch and records a structured JSON trace:
op name, named inputs (tensor metadata + scalar values), outputs, and
the backend kernel that handled the op.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# ── Dispatch key inference ───────────────────────────────────────────

_LAYOUT_TO_KEY_PREFIX: dict[torch.layout, str] = {
    torch.strided: "",
    torch.sparse_coo: "Sparse",
    torch.sparse_csr: "SparseCsr",
    torch.sparse_csc: "SparseCsr",  # same dispatch key family
    torch.sparse_bsr: "SparseCsr",
    torch.sparse_bsc: "SparseCsr",
}


def _infer_dispatch_key(tensors: list[torch.Tensor]) -> str | None:
    """Best-effort dispatch key from input tensors (device + layout)."""
    if not tensors:
        return None
    t = tensors[0]
    prefix = _LAYOUT_TO_KEY_PREFIX.get(t.layout, "")
    device = t.device.type.upper()
    if device == "CPU":
        return f"{prefix}CPU" if prefix else "CPU"
    if device == "CUDA":
        return f"{prefix}CUDA" if prefix else "CUDA"
    return device


def _lookup_backend(
    op_name: str,
    dispatch_key: str | None,
    vtable: dict[str, list[VTableEntry]] | None,
) -> str | None:
    """Look up the backend method from the pre-built vtable."""
    if vtable is None or dispatch_key is None or op_name not in vtable:
        return None
    for entry in vtable[op_name]:
        if entry.key == dispatch_key and entry.registered:
            return entry.method
    return None


# ── Tensor metadata ─────────────────────────────────────────────────


def _tensor_meta(t: torch.Tensor) -> dict[str, Any]:
    """Extract JSON-serializable metadata from a tensor."""
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "device": str(t.device),
        "layout": str(t.layout).removeprefix("torch."),
    }


# ── Event / Trace data ──────────────────────────────────────────────


@dataclass
class DispatchEvent:
    """A single dispatched operator call."""

    index: int
    op: str
    namespace: str
    backend: str | None
    inputs: dict[str, Any]
    outputs: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "op": self.op,
            "namespace": self.namespace,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.backend is not None:
            d["backend"] = self.backend
        return d


@dataclass
class DispatchTrace:
    """Structured trace of a model's dispatch events."""

    events: list[DispatchEvent] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps([e.to_dict() for e in self.events], indent=indent)

    def summary(self) -> str:
        lines = []
        for e in self.events:
            tensor_inputs = {k: v for k, v in e.inputs.items() if isinstance(v, dict)}
            in_str = ", ".join(str(v["shape"]) for v in tensor_inputs.values())
            out_str = ", ".join(str(o["shape"]) for o in e.outputs)
            lines.append(f"{e.op}  [{in_str}] -> [{out_str}]")
        return "\n".join(lines)


# ── Tracer ───────────────────────────────────────────────────────────

if TYPE_CHECKING:
    from exp.dispatch.dispatch_vtable import VTableEntry


class DispatchTracer(TorchDispatchMode):
    """Context manager that records a dispatch trace (eager mode).

    With ignore_compile_internals=False (default), any active
    TorchDispatchMode causes torch.compile to fall back to eager.
    This tracer captures the full eager ATen op sequence.

    Usage:
        tracer = DispatchTracer()
        with tracer:
            output = model(input)
        print(tracer.trace.summary())
        print(tracer.trace.to_json())
    """

    def __init__(
        self,
        vtable: dict[str, list[Any]] | None = None,
    ) -> None:
        super().__init__()
        self.trace = DispatchTrace()
        self._vtable = vtable
        self._index = 0

    def __torch_dispatch__(
        self,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}

        # ── Build named inputs from schema ───────────────────────
        schema_args = func._schema.arguments
        inputs: dict[str, Any] = {}
        input_tensors: list[torch.Tensor] = []

        for i, schema_arg in enumerate(schema_args):
            name = schema_arg.name
            # Positional args first, then kwargs
            if i < len(args):
                val = args[i]
            elif name in kwargs:
                val = kwargs[name]
            else:
                continue  # default value, not provided

            is_tensor = schema_arg.type.isSubtypeOf(torch._C.TensorType.get())
            if is_tensor and isinstance(val, torch.Tensor):
                inputs[name] = _tensor_meta(val)
                input_tensors.append(val)
            elif isinstance(val, torch.Tensor):
                # Tensor passed for a non-tensor schema slot (e.g., Optional[Tensor])
                inputs[name] = _tensor_meta(val)
                input_tensors.append(val)
            elif isinstance(val, (int, float, bool, str, type(None))):
                inputs[name] = val
            elif isinstance(val, (list, tuple)):
                # Could be size/shape args like [1, 64] or tensor lists like stack()
                serialized = []
                for v in val:
                    if isinstance(v, torch.Tensor):
                        serialized.append(_tensor_meta(v))
                        input_tensors.append(v)
                    else:
                        serialized.append(v)
                inputs[name] = serialized
            else:
                inputs[name] = str(val)

        # ── Execute ──────────────────────────────────────────────
        result = func(*args, **kwargs)

        # ── Build outputs ────────────────────────────────────────
        outputs: list[dict[str, Any]] = []
        if isinstance(result, torch.Tensor):
            outputs.append(_tensor_meta(result))
        elif isinstance(result, (tuple, list)):
            for v in result:
                if isinstance(v, torch.Tensor):
                    outputs.append(_tensor_meta(v))

        # ── Backend lookup ───────────────────────────────────────
        dispatch_key = _infer_dispatch_key(input_tensors)
        backend = _lookup_backend(func.name(), dispatch_key, self._vtable)

        self.trace.events.append(
            DispatchEvent(
                index=self._index,
                op=func.name(),
                namespace=func.namespace,
                backend=backend,
                inputs=inputs,
                outputs=outputs,
            )
        )
        self._index += 1

        return result


class CompiledDispatchTracer(DispatchTracer):
    """Tracer that observes compiled execution.

    With ignore_compile_internals=True, torch.compile runs normally.
    Fused ops (replaced by Triton kernels) vanish from the trace.
    Unfused ops (e.g., cuBLAS matmul via extern_kernels) remain visible.
    The diff between DispatchTracer and CompiledDispatchTracer traces
    shows exactly what the compiler fused away.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True
