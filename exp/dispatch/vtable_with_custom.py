"""Static vtable comparison: custom ops vs native counterparts.

Imports custom ops first so the dispatcher sees them, then builds the
vtable and prints a side-by-side of actual kernel registrations (filtering
out the generic backend fallbacks that every op inherits).
"""

from __future__ import annotations

import shared.custom_ops  # noqa: F401 — must load before building vtable
from exp.dispatch.dispatch_vtable import VTableEntry, build_dispatch_table

# Ops to compare: (display name, dispatcher name)
OPS = [
    ("reentry::silu_and_mul", "reentry::silu_and_mul"),
    ("reentry::identity", "reentry::identity"),
    ("aten::silu", "aten::silu"),
    ("aten::mul.Tensor", "aten::mul.Tensor"),
    ("aten::mm", "aten::mm"),
]

# Generic fallbacks inherited by every op — not interesting for comparison.
_GENERIC_FALLBACKS = {
    "Python",
    "FuncTorchDynamicLayerBackMode",
    "Functionalize",
    "Named",
    "Conjugate",
    "Negative",
    "ZeroTensor",
    "Tracer",
    "FuncTorchBatched",
    "BatchedNestedTensor",
    "Batched",
    "FuncTorchGradWrapper",
    "PythonTLSSnapshot",
    "FuncTorchDynamicLayerFrontMode",
    "PreDispatch",
    "PythonDispatcher",
    "FuncTorchVmapMode",
    "VmapMode",
}


def is_interesting(entry: VTableEntry) -> bool:
    """Keep entries that reflect actual registrations, not generic fallbacks."""
    if entry.key in _GENERIC_FALLBACKS:
        return False
    if not entry.registered:
        return False
    # "fallthrough" registrations are structural, not interesting
    if "fallthrough" in entry.loc:
        return False
    return True


def classify(entry: VTableEntry) -> str:
    """Short label for how the entry was registered."""
    if "autograd kernel" in entry.method:
        return "autograd"
    if "backend fallback" in entry.method:
        return "fallback"
    if "default backend kernel" in entry.method:
        return "default"
    return "kernel"


def format_comparison(
    vtable: dict[str, list[VTableEntry]],
    ops: list[tuple[str, str]],
) -> str:
    """Format a markdown-style table comparing dispatch registrations."""
    # Collect interesting entries per op
    op_entries: dict[str, dict[str, tuple[str, str]]] = {}
    all_keys: set[str] = set()

    for display, name in ops:
        entries = vtable.get(name, [])
        interesting = {
            e.key: (classify(e), e.loc.split("/")[-1].rstrip(" )"))
            for e in entries
            if is_interesting(e)
        }
        op_entries[display] = interesting
        all_keys |= interesting.keys()

    # Group dispatch keys for readability
    key_groups = [
        ("Backend", ["CPU", "CUDA", "Undefined", "Meta", "MkldnnCPU"]),
        ("Autograd", [k for k in sorted(all_keys) if k.startswith("Autograd")]),
        ("Sparse", [k for k in sorted(all_keys) if "Sparse" in k]),
        ("Quantized", [k for k in sorted(all_keys) if "Quantized" in k]),
        ("NestedTensor", [k for k in sorted(all_keys) if "Nested" in k]),
        ("Autocast", [k for k in sorted(all_keys) if "Autocast" in k]),
    ]
    # Catch anything not in a group
    grouped = {k for _, keys in key_groups for k in keys}
    ungrouped = sorted(all_keys - grouped)
    if ungrouped:
        key_groups.append(("Other", ungrouped))

    op_names = [display for display, _ in ops]
    col_width = max(len(n) for n in op_names) + 2

    lines = []
    # Header
    header = f"{'Key':<30}" + "".join(f"{n:<{col_width}}" for n in op_names)
    lines.append(header)
    lines.append("-" * len(header))

    for group_name, keys in key_groups:
        # Only show keys that at least one op has registered
        active_keys = [k for k in keys if k in all_keys]
        if not active_keys:
            continue

        # Collapse autograd keys if all ops have the same pattern
        if group_name == "Autograd" and len(active_keys) > 3:
            # Summarize: show AutogradCPU as representative
            rep = "AutogradCPU"
            row = f"{'Autograd* (x' + str(len(active_keys)) + ')':<30}"
            for display in op_names:
                val = op_entries[display].get(rep, ("—", ""))
                row += f"{val[0]:<{col_width}}"
            lines.append(row)
            continue

        for key in active_keys:
            # Skip rows where every op is either "—" or "default"
            vals = [op_entries[display].get(key, ("—", ""))[0] for display in op_names]
            if all(v in ("—", "default") for v in vals):
                continue
            row = f"{key:<30}"
            for v in vals:
                row += f"{v:<{col_width}}"
            lines.append(row)

    return "\n".join(lines)


def main() -> None:
    vtable = build_dispatch_table()
    print(format_comparison(vtable, OPS))


if __name__ == "__main__":
    main()
