#!/usr/bin/env python3
"""B2 - where pooled memory wins, as a function of host count.

The pool here is dual-headed, so no experiment on this cluster can go past two
hosts.  Any claim about more hosts has to come from a model calibrated on the
points that can be measured.  This is that model.  It is load-bearing, not
decorative: it is the only route to a statement about N > 2.

Three architectures, three different things that run out:

  pool        One copy of the dataset, mapped by every host.  Per-host
              residency is the working set, not the dataset.  What runs out is
              pool CAPACITY, which does not improve with N, and pool
              BANDWIDTH, which is shared and so divides by N.
  distdgl     The dataset is partitioned and the partitions overlap, because
              sampling needs halo nodes.  Per-host residency is S*r(N)/N, which
              falls with N but slower than 1/N.  What runs out is per-host
              DRAM -- and not at the steady-state size, because loading a
              partition transiently needs far more than holding it.
  wholegraph  The dataset is spread across GPU memory, S/N per GPU.  What runs
              out is aggregate VRAM, which is the smallest pool of the three.

Two crossovers come out:

  N_cap   smallest host count at which an architecture holds the dataset at
          all.  For the pool this is 1 or never, because adding hosts does not
          add pool capacity -- which makes the LAYOUT (see --layout) decide it.
  N_bw    host count at which the pool's shared bandwidth stops meeting
          demand.  This turns on a quantity nobody has measured: whether two
          readers each get the single-reader bandwidth or split it.  The
          script reports both ends rather than a number.  X2b closes it.

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
from tgnnpool.sizes import LAYOUTS, dataset_bytes


def replication(n_hosts: int, alpha: float) -> float:
    """Halo overlap r(N): bytes resident across all hosts, divided by S.

    r(1) = 1 by definition.  r grows with N because cutting a graph into more
    parts leaves more boundary, and boundary nodes' features are replicated so
    the sampler can reach them without a round trip.  The linear form is a
    stand-in: alpha is NOT measured here, so every distdgl number is inferred.
    """
    return 1.0 + alpha * (n_hosts - 1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=("window", "full"), default="full")
    ap.add_argument("--layout", choices=LAYOUTS, default="delta",
                    help="structure layout; decides the dataset size (default delta, "
                         "which is what the project runs today)")
    ap.add_argument("--max-hosts", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.20,
                    help="halo growth in r(N)=1+alpha(N-1); NOT measured (default 0.20)")
    ap.add_argument("--dtype-bytes", type=int, default=4)
    ap.add_argument("--pool-bytes", type=float, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    mc, wl = config.machine(), config.workload()
    idx_bytes = wl.require("layout", "index_bytes")
    total = dataset_bytes(
        args.layout,
        nodes=wl.require(args.scale, "nodes"),
        edges=wl.require(args.scale, "edges"),
        snapshots=wl.require(args.scale, "snapshots"),
        dim=wl.require("full", "feature_dim"),
        dtype_bytes=args.dtype_bytes,
        idx_bytes=idx_bytes,
    )
    capacity = int(args.pool_bytes) if args.pool_bytes else mc.require("pool", "usable_bytes")
    dram = mc.require("host", "dram_bytes")
    vram = mc.require("gpu", "memory_bytes")

    # Holding a partition is not the peak. Loading it is. Measured on the window
    # workload: 112 GB resident, 207 GB at the moment of load.
    peak_factor = (mc.require("distdgl", "shm_load_peak_bytes")
                   / mc.require("distdgl", "shm_steady_bytes"))

    # A pool host holds only what it is working on. Measured gather volume per
    # batch, doubled for double buffering; nothing here turns on the factor.
    working_set = 2 * wl.require("training", "gather_bytes_per_batch")

    banner("B2", "placement crossover")
    print(f"{args.scale} graph, {args.layout} layout: {gb(total)} "
          f"(features + structure, {args.dtype_bytes} B per value)")
    print(f"limits: pool {gb(capacity)} shared | DRAM {gb(dram)} per host | "
          f"VRAM {gb(vram)} per host")
    print(f"distdgl load peak {peak_factor:.2f}x steady [measured]; "
          f"halo r(N)=1+{args.alpha}(N-1) [inferred]")

    print("\nper-host residency, and whether that host can actually hold it")
    print(f"  {'N':>3}  {'pool':>20}  {'distdgl':>20}  {'wholegraph':>20}")
    feasible: dict[str, int | None] = {"pool": None, "distdgl": None, "wholegraph": None}
    table = []
    for n in range(1, args.max_hosts + 1):
        row: dict = {"hosts": n}
        cells = []
        for arch in ("pool", "distdgl", "wholegraph"):
            if arch == "pool":
                need, ok = working_set, total <= capacity
            elif arch == "distdgl":
                need = total * replication(n, args.alpha) / n
                ok = need * peak_factor <= dram
            else:
                need = total / n
                ok = need <= vram
            if ok and feasible[arch] is None:
                feasible[arch] = n
            row[arch] = {"bytes": need, "fits": ok}
            cells.append(f"{gb(need) + (' ok' if ok else ' --'):>20}")
        table.append(row)
        if n <= 8 or n == args.max_hosts:
            print(f"  {n:>3}  " + "  ".join(cells))

    print("\nsmallest host count at which the dataset is held at all")
    for arch in ("pool", "distdgl", "wholegraph"):
        n = feasible[arch]
        where = f"N = {n}" if n else f"never within {args.max_hosts} hosts"
        print(f"  {arch:<12}{where}")

    if feasible["pool"] is None:
        print(f"  -> the pool does NOT hold the {args.scale} graph in the {args.layout} "
              f"layout: {gb(total)} against {gb(capacity)}.\n"
              f"     Adding hosts does not help, because pool capacity is not per-host. "
              f"The\n     layout is the only lever. Re-run with --layout shared_rowptr "
              f"to price it.")
    elif feasible["pool"] == 1 and (feasible["wholegraph"] or 10 ** 9) > 1:
        print(f"  -> the pool holds it on one host; GPU memory needs "
              f"{feasible['wholegraph']}, host memory needs {feasible['distdgl']}.\n"
              f"     That is the capacity argument as a host count rather than a ratio.")

    # ------------------------------------------------------------ bandwidth
    print("\nbandwidth")
    per_batch = wl.require("training", "gather_bytes_per_batch")
    train_s = wl.require("training", "seconds_per_batch")
    demand = per_batch / train_s
    single = mc.require("pool", "gather_bytes_per_s_single_reader")
    print(f"  one reader sustains {single / 1e9:.1f} GB/s [measured]")
    print(f"  one training host demands {demand / 1e9:.2f} GB/s "
          f"({gb(per_batch)} per batch at {train_s * 1e3:.0f} ms)")
    try:
        agg = mc.require("pool", "gather_bytes_per_s_aggregate")
        bw = {"aggregate_bytes_per_s": agg, "n_bw": agg / demand}
        print(f"  aggregate {agg / 1e9:.1f} GB/s [measured] -> "
              f"saturates at N = {agg / demand:.1f}")
    except config.Missing:
        shared_n = single / demand
        bw = {"aggregate_bytes_per_s": None, "n_bw_if_shared": shared_n,
              "n_bw_if_independent": None}
        print("  aggregate bandwidth is NOT MEASURED, so N_bw is a range:")
        print(f"    readers split one device limit  -> N_bw = {shared_n:.1f}")
        print(f"    readers scale independently     -> N_bw > {args.max_hosts}")
        print("  X2b (one reader against two, same pool) decides which. It is the "
              "cheapest\n  unrun experiment in this repository and it gates every "
              "N > 2 claim.")

    if not args.no_write:
        rec = Record(
            experiment="B2",
            evidence="derived",
            summary=("Per-host residency and bandwidth demand for pooled memory, "
                     "partitioned host memory and GPU memory against host count, "
                     "identifying which resource binds for each."),
            inputs={"scale": args.scale, "layout": args.layout,
                    "dataset_bytes": total, "alpha": args.alpha,
                    "pool_bytes": capacity, "dram_bytes": dram, "vram_bytes": vram,
                    "distdgl_load_peak_factor": peak_factor},
            outputs={"table": table, "first_feasible_hosts": feasible,
                     "bandwidth": bw, "working_set_bytes": working_set},
            caveats=[
                "alpha, the halo growth rate, is not measured on this cluster, and "
                "every distdgl figure depends on it. Measure it by partitioning the "
                "same graph into 1, 2 and 4 parts and recording resident bytes.",
                "The load-peak factor is measured on the window workload at one "
                "partition and assumed to carry to other scales and part counts.",
                "The pool column assumes residency equals the working set. True "
                "while hosts only read; false as soon as one writes embeddings back, "
                "which no experiment here does yet.",
                "Demand uses the training-level batch time. The micro-benchmark "
                "batch is about six times shorter and implies demand six times "
                "higher; the honest reading is that N_bw lies between the two.",
                "The pool is dual-headed, so N > 2 cannot be run here at all. "
                "Every row past N=2 is model output, not measurement.",
            ],
        )
        print(f"\nwrote {rec.write().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
