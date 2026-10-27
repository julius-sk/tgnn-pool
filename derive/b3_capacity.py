#!/usr/bin/env python3
"""B3 - how many snapshots fit in the pool, and which layout decides it.

The pool's selling point is capacity.  A snapshot model spends that capacity on
two things: one feature table, which does not grow with the number of snapshots,
and a structure, which does.  How fast the structure grows is a layout choice,
and the three layouts below differ by more than an order of magnitude.

  expanded        Materialise the time-expanded graph: snapshot t holds every
                  edge with time <= t, so edges are copied into every later
                  snapshot.  Column indices cost sum_t |E_<=t| ~ E(T+1)/2.
                  Nobody builds this on purpose; it is the naive reading of the
                  model and it is the baseline the other two improve on.

  delta           One CSR per snapshot holding only that snapshot's new edges.
                  Column indices are stored once in total.  Each snapshot still
                  carries a full-length row pointer array, and that duplication
                  is the whole cost.  THIS IS WHAT THE PROJECT RUNS TODAY.

  shared_rowptr   One time-sorted CSR: row pointers once, column indices once,
                  plus one timestamp per edge.  Snapshot t is a prefix cut
                  inside each row.  Structure stops growing with T altogether.
                  Requires a sampler that can cut inside a row, which the
                  current sampler cannot do -- which is precisely why the
                  project runs `delta`.

So this derivation prices a library limitation.  Run it to find out what that
limitation costs at full scale.

Runs on a laptop.  No pool, no GPU, no dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool import config
from tgnnpool.config import gb
from tgnnpool.result import Record, banner

from tgnnpool.sizes import (LAYOUTS, feature_bytes, structure_bytes,
                            max_snapshots)


def validate(wl: config.Config, idx_bytes: int) -> list[str]:
    """Check the model against the two structure sizes that were measured."""
    lines = []
    n = wl.require("window", "nodes")
    predicted = (n + 1) * idx_bytes
    measured = wl.require("layout", "rowptr_bytes_per_snapshot")
    err = (predicted - measured) / measured * 100
    lines.append(f"  row pointers per snapshot, window scale: "
                 f"model {gb(predicted)}  measured {gb(measured)}  ({err:+.1f}%)")
    t = wl.require("window", "snapshots")
    predicted_total = t * predicted
    measured_total = wl.require("layout", "rowptr_total_bytes")
    err = (predicted_total - measured_total) / measured_total * 100
    lines.append(f"  row pointers, all {t} snapshots:          "
                 f"model {gb(predicted_total)}  measured {gb(measured_total)}  ({err:+.1f}%)")
    return lines


def report(scale: str, wl: config.Config, capacity: int, dtype_bytes: int,
           idx_bytes: int) -> dict:
    nodes = wl.require(scale, "nodes")
    edges = wl.require(scale, "edges")
    want_t = wl.require(scale, "snapshots")
    dim = wl.require("full", "feature_dim")
    feat = feature_bytes(nodes, dim, dtype_bytes)

    print(f"\n{scale} graph: {nodes:,} nodes, {edges:,} edges, "
          f"{dim}-dim features, {want_t} snapshots wanted")
    print(f"  feature table ({dtype_bytes} B/value): {gb(feat)} "
          f"of {gb(capacity)} pool  -> {gb(capacity - feat)} left for structure")

    rows = {}
    print(f"\n  {'layout':<16}{'structure at T='+str(want_t):>22}"
          f"{'total':>14}{'fits':>7}{'max T':>8}")
    for layout in LAYOUTS:
        parts = structure_bytes(layout, nodes, edges, want_t, idx_bytes)
        struct = sum(parts.values())
        total = feat + struct
        tmax = max_snapshots(layout, nodes, edges, capacity, feat,
                             idx_bytes, ceiling=max(1000, want_t * 10))
        ceiling = max(1000, want_t * 10)
        fits = "yes" if total <= capacity else "no"
        shown = f">{ceiling}" if tmax >= ceiling else str(tmax)
        print(f"  {layout:<16}{gb(struct):>22}{gb(total):>14}{fits:>7}{shown:>8}")
        rows[layout] = {"structure_bytes": struct, "total_bytes": total,
                        "fits": total <= capacity, "max_snapshots": tmax,
                        "parts": parts}

    delta, shared = rows["delta"], rows["shared_rowptr"]
    cost = delta["structure_bytes"] - shared["structure_bytes"]
    print(f"\n  cost of the row-pointer duplication at T={want_t}: {gb(cost)}"
          f"  ({cost / capacity * 100:.1f}% of the pool)")
    if not delta["fits"] and shared["fits"]:
        print(f"  -> the layout in use misses this scale by "
              f"{want_t - delta['max_snapshots']} snapshot(s); the shared-row-pointer "
              f"layout clears it with {gb(capacity - shared['total_bytes'])} to spare.")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-bytes", type=float, default=None,
                    help="override the pool capacity, in bytes")
    ap.add_argument("--dtype-bytes", type=int, default=4,
                    help="feature element width: 4 for fp32, 2 for fp16 (default 4)")
    ap.add_argument("--idx-bytes", type=int, default=None,
                    help="index width; default comes from the workload config")
    ap.add_argument("--also-current", action="store_true",
                    help="repeat the table at the size the pool is formatted at today")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    mc, wl = config.machine(), config.workload()
    capacity = int(args.pool_bytes) if args.pool_bytes else mc.require("pool", "usable_bytes")
    idx_bytes = args.idx_bytes or wl.require("layout", "index_bytes")

    banner("B3", "snapshot capacity, and what the layout costs")
    print(f"pool capacity {gb(capacity)}; features {args.dtype_bytes} B per value; "
          f"indices {idx_bytes} B")

    print("\nmodel check against measured structure sizes:")
    for line in validate(wl, idx_bytes):
        print(line)

    outputs = {"capacity_bytes": capacity, "dtype_bytes": args.dtype_bytes,
               "index_bytes": idx_bytes, "scales": {}}
    for scale in ("window", "full"):
        outputs["scales"][scale] = report(scale, wl, capacity,
                                          args.dtype_bytes, idx_bytes)

    if args.also_current:
        current = mc.require("pool", "currently_formatted_bytes")
        print("\n" + "-" * 72)
        print(f"same question at the size the pool is formatted at today, {gb(current)}:")
        outputs["scales_at_current_format"] = {
            s: report(s, wl, current, args.dtype_bytes, idx_bytes)
            for s in ("window", "full")}

    if not args.no_write:
        rec = Record(
            experiment="B3",
            evidence="derived",
            summary=("Snapshot capacity in a pool of the stated size, for three "
                     "structure layouts; the per-snapshot row-pointer duplication "
                     "the current sampler forces is what sets the limit."),
            inputs={"capacity_bytes": capacity, "dtype_bytes": args.dtype_bytes,
                    "index_bytes": idx_bytes, "workload": wl.get_or("mag", "name")},
            outputs=outputs,
            caveats=[
                "The expanded layout assumes a uniform edge arrival rate. Measured "
                "arrival grows monotonically (38.5M to 97.5M per snapshot over the "
                "window), so the real expanded cost is HIGHER than reported and its "
                "max T is correspondingly lower.",
                "Full-graph node and edge counts come from the dataset paper, not "
                "from a build on this cluster. Only the window numbers are measured "
                "here.",
                "Capacity accounting ignores the allocator's own overhead: the heap "
                "expands in fixed steps, so bytes taken from the device exceed bytes "
                "live. Treat every max-T as an upper bound.",
                "The shared-row-pointer row is a layout that cannot be run today: it "
                "needs a sampler that cuts inside a row. It is priced here to size "
                "that missing feature, not reported as a measurement.",
            ],
        )
        print(f"\nwrote {rec.write().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
