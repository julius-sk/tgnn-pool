"""How many bytes a snapshot graph occupies, by layout.

One definition, used by both B2 and B3.  If these two derivations ever
disagreed about what a dataset costs, both would be worthless, so neither owns
the formula.

Three layouts, differing only in how the structure grows with the number of
snapshots T:

  expanded        Column indices are copied into every later snapshot:
                  sum_t |E_<=t| ~ E(T+1)/2 entries.  Grows linearly in T with a
                  large constant.  This is the naive reading of a snapshot
                  model, and it is the baseline, not a proposal.
  delta           One CSR per snapshot holding that snapshot's new edges.
                  Column indices stored once in total; row pointers stored
                  once PER SNAPSHOT, at full node length.  Grows linearly in T
                  with a constant of (N+1) * index_bytes.  This is what the
                  project runs today.
  shared_rowptr   One time-sorted CSR plus a timestamp per edge; snapshot t is
                  a prefix cut inside each row.  Constant in T.  Needs a
                  sampler that can cut inside a row, which the sampler in use
                  cannot do.

The gap between `delta` and `shared_rowptr` is therefore the price of a missing
sampler feature, in bytes.
"""
from __future__ import annotations

LAYOUTS = ("expanded", "delta", "shared_rowptr")


def feature_bytes(nodes: int, dim: int, dtype_bytes: int) -> int:
    return nodes * dim * dtype_bytes


def structure_bytes(layout: str, nodes: int, edges: int, snapshots: int,
                    idx_bytes: int, time_bytes: int = 1) -> dict[str, int]:
    """Structure bytes, split into the parts that behave differently in T."""
    rowptr_unit = (nodes + 1) * idx_bytes
    if layout == "expanded":
        # Assumes a uniform edge arrival rate.  Measured arrival grows
        # monotonically, so this UNDER-states the later snapshots.
        colind = int(edges * (snapshots + 1) / 2) * idx_bytes
        return {"rowptr": snapshots * rowptr_unit, "colind": colind, "edge_time": 0}
    if layout == "delta":
        return {"rowptr": snapshots * rowptr_unit,
                "colind": edges * idx_bytes, "edge_time": 0}
    if layout == "shared_rowptr":
        return {"rowptr": rowptr_unit, "colind": edges * idx_bytes,
                "edge_time": edges * time_bytes}
    raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")


def dataset_bytes(layout: str, nodes: int, edges: int, snapshots: int,
                  dim: int, dtype_bytes: int, idx_bytes: int,
                  time_bytes: int = 1) -> int:
    return (feature_bytes(nodes, dim, dtype_bytes)
            + sum(structure_bytes(layout, nodes, edges, snapshots,
                                  idx_bytes, time_bytes).values()))


def max_snapshots(layout: str, nodes: int, edges: int, capacity: int,
                  feat: int, idx_bytes: int, time_bytes: int = 1,
                  ceiling: int = 1000) -> int:
    """Largest T whose feature table plus structure still fits in `capacity`."""
    budget = capacity - feat
    if budget <= 0:
        return 0
    lo, hi = 0, ceiling
    while lo < hi:
        mid = (lo + hi + 1) // 2
        total = sum(structure_bytes(layout, nodes, edges, mid,
                                    idx_bytes, time_bytes).values())
        if total <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo
