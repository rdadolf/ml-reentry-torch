"""Dump and visualize the PyTorch dispatch virtual table."""

import re
from collections import Counter
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import torch

sns.set_palette("deep")

_ENTRY_RE = re.compile(
    r"(\w+):\s*(.*registered|unregistered)\s+(?:at|by)\s+(.+)\s*\[(.*)\]"
)


@dataclass(frozen=True)
class VTableEntry:
    """A single row in an operator's dispatch table."""

    key: str
    registered: bool
    loc: str
    method: str

    @classmethod
    def from_string(cls, line: str) -> "VTableEntry":
        """Parse one line of _dispatch_dump_table output."""
        m = _ENTRY_RE.match(line)
        if not m:
            raise ValueError(f"line does not match expected format: {line}")
        key, registered, loc, method = m.groups()
        return cls(
            key=key,
            registered=registered == "registered",
            loc=loc,
            method=method,
        )


def build_dispatch_table() -> dict[str, list[VTableEntry]]:
    """Parse the full dispatch table for every registered op."""
    all_ops = torch._C._dispatch_get_all_op_names()
    return {
        op: [
            VTableEntry.from_string(line)
            for line in torch._C._dispatch_dump_table(op).split("\n")
            if line
        ]
        for op in all_ops
        if op
    }


def get_all_keys(vtable: dict[str, list[VTableEntry]]) -> set[str]:
    """Return the set of unique dispatch keys across all ops."""
    return {entry.key for entries in vtable.values() for entry in entries}


def plot_dispatch_table_sizes(vtable: dict[str, list[VTableEntry]]) -> None:
    """Histogram of dispatch table sizes (entries per op)."""
    sizes = [len(entries) for entries in vtable.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(sizes, bins=30, ax=ax)
    ax.set_xlabel("Dispatch Table Size")
    ax.set_ylabel("Number of Ops")
    ax.set_title("Dispatch Table Sizes")
    fig.tight_layout()
    fig.savefig("dispatch_table_sizes.png")


def plot_key_frequency(vtable: dict[str, list[VTableEntry]]) -> None:
    """Horizontal bar chart of how often each dispatch key appears."""
    counts = Counter(
        entry.key
        for entries in vtable.values()
        for entry in entries
        if entry.registered
    )
    total_ops = len(vtable)
    keys, freqs = zip(*counts.most_common())
    keys = ("ALL OPS", *keys)
    freqs = (total_ops, *freqs)

    fig, ax = plt.subplots(figsize=(14, 18))
    colors = sns.color_palette()
    hue_groups = ["hi"] + ["default"] * (len(keys) - 1)
    sns.barplot(
        x=list(freqs),
        y=list(keys),
        hue=hue_groups,
        palette={"default": colors[0], "hi": colors[1]},
        legend=False,
    )

    ax.axvline(x=total_ops, color=colors[1], linestyle="--", linewidth=0.8)
    ax.set_xlabel("Number of Ops")
    ax.set_ylabel("Dispatch Key")
    ax.set_title("Dispatch Key Frequency (registered entries)")
    fig.tight_layout()
    fig.savefig("dispatch_key_frequency.png")


if __name__ == "__main__":
    vtable = build_dispatch_table()
    print(f"Total ops: {len(vtable)}")
    print(f"Total entries: {sum(len(entries) for entries in vtable.values())}")
    print(f"Unique keys: {len(get_all_keys(vtable))}")

    methods = Counter(e.method for entries in vtable.values() for e in entries)
    reg_types = Counter(
        "fallthrough" if not e.registered else "registered"
        for entries in vtable.values()
        for e in entries
    )

    print("\nBy registration type:")
    for kind, count in reg_types.most_common():
        print(f"  {kind}: {count}")
    print("\nBy method:")
    for method, count in methods.most_common():
        print(f"  {method}: {count}")

    plot_dispatch_table_sizes(vtable)
    plot_key_frequency(vtable)
