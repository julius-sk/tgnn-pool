#!/usr/bin/env python3
"""X2 - what does it cost to admit one more snapshot?

This is the fourth row of the three-way comparison table, and the one that
does not reduce to throughput.  A snapshot model grows: the question is what
each architecture has to do when it does.

  pool        Publish the new snapshot's arrays and flip their state. Cost is
              proportional to the NEW edges. Training does not stop, and B1
              says what readers see while it happens.
  distdgl     Repartition and reload. Cost is proportional to the WHOLE graph,
              and training stops for all of it.
  wholegraph  Does not arise at this scale: the dataset does not fit in GPU
              memory to begin with (see B2).

The claim the table supports is O(delta) against O(whole graph), so the
interesting number is the ratio, and the ratio is dominated by the DistDGL
side, which is the one nobody has measured yet.

  python exp/x2_admission.py --dry-run           # plan and prediction, laptop
  python exp/x2_admission.py --backend pool      # cluster
  python exp/x2_admission.py --backend distdgl --partition-cmd '...'   # cluster
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool import config
from tgnnpool.config import gb
from tgnnpool.result import Record, banner
from tgnnpool.sizes import structure_bytes


def predict(mc: config.Config, wl: config.Config, scale: str,
            idx_bytes: int) -> dict:
    """What each architecture should cost, from measured rates."""
    nodes = wl.require(scale, "nodes")
    edges = wl.require(scale, "edges")
    snaps = wl.require(scale, "snapshots")

    # One snapshot's share of the edges, plus the row pointer array it carries.
    per_snapshot_edges = edges / snaps
    delta_bytes = per_snapshot_edges * idx_bytes + (nodes + 1) * idx_bytes
    ingest_rate = mc.require("pool", "ingest_bytes_per_s")
    pool_seconds = delta_bytes / ingest_rate

    whole = sum(structure_bytes("delta", nodes, edges, snaps, idx_bytes).values())
    out = {"delta_bytes": delta_bytes, "pool_seconds": pool_seconds,
           "whole_structure_bytes": whole}
    try:
        out["distdgl_seconds"] = mc.require("distdgl", "repartition_seconds")
        out["ratio"] = out["distdgl_seconds"] / pool_seconds
    except config.Missing:
        out["distdgl_seconds"] = None
        out["ratio"] = None
    return out


def run_pool(args, mc, wl, idx_bytes) -> dict:
    from tgnnpool.store import PoolStore
    store = PoolStore()          # raises with instructions until implemented
    payload = b"\0" * args.chunk_bytes
    started = time.perf_counter()
    for i in range(args.chunks):
        store.publish(f"x2.snapshot{args.snapshot}.obj{i}", payload,
                      version=args.snapshot + 1)
    elapsed = time.perf_counter() - started
    written = args.chunk_bytes * args.chunks
    return {"seconds": elapsed, "bytes": written,
            "bytes_per_s": written / elapsed}


def run_distdgl(args) -> dict:
    if not args.partition_cmd:
        raise SystemExit(
            "--backend distdgl needs --partition-cmd: the exact command this "
            "cluster uses to repartition and reload the graph with one more "
            "snapshot included. It is recorded in the result so the number can "
            "be reproduced.")
    started = time.perf_counter()
    proc = subprocess.run(args.partition_cmd, shell=True)
    elapsed = time.perf_counter() - started
    return {"seconds": elapsed, "returncode": proc.returncode,
            "command": args.partition_cmd}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("pool", "distdgl"), default=None)
    ap.add_argument("--scale", choices=("window", "full"), default="window")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the prediction; touches nothing")
    ap.add_argument("--snapshot", type=int, default=10,
                    help="index of the snapshot being admitted")
    ap.add_argument("--chunks", type=int, default=64)
    ap.add_argument("--chunk-bytes", type=int, default=16 << 20)
    ap.add_argument("--partition-cmd", default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    mc, wl = config.machine(), config.workload()
    idx_bytes = wl.require("layout", "index_bytes")
    pred = predict(mc, wl, args.scale, idx_bytes)

    banner("X2", "cost of admitting one more snapshot")
    print(f"{args.scale} graph, snapshot {args.snapshot}")
    print(f"  new edges in one snapshot cost {gb(pred['delta_bytes'])} to publish")
    print(f"  pool, at the measured {mc.require('pool','ingest_bytes_per_s')/1e9:.2f} "
          f"GB/s ingest rate: {pred['pool_seconds']:.1f} s  [derived]")
    if pred["distdgl_seconds"] is None:
        print("  distdgl: NOT MEASURED. It has to reread and repartition "
              f"{gb(pred['whole_structure_bytes'])} of structure plus the whole\n"
              "           feature table, and training is stopped throughout. "
              "Until that is timed\n           once, the headline ratio of this "
              "experiment does not exist.")
    else:
        print(f"  distdgl: {pred['distdgl_seconds']:.0f} s [measured]  "
              f"-> ratio {pred['ratio']:.0f}x")

    if args.dry_run or args.backend is None:
        print("\nto fill in the missing half:")
        print("  1. time one repartition-and-reload on this cluster:")
        print("       python exp/x2_admission.py --backend distdgl "
              "--partition-cmd '<the command>'")
        print("  2. put the seconds into configs/machine.yaml under "
              "distdgl.repartition_seconds")
        print("  3. rerun this script; B2 picks the number up as well")
        print("\nnothing was run.")
        return 0

    measured = run_pool(args, mc, wl, idx_bytes) if args.backend == "pool" \
        else run_distdgl(args)
    print(f"\nmeasured: {measured}")

    if not args.no_write:
        rec = Record(
            experiment="X2",
            evidence="measured",
            summary=f"Cost of admitting one snapshot on the {args.backend} backend.",
            inputs=vars(args),
            outputs={"measured": measured, "predicted": pred},
            caveats=[
                "The pool figure is an ingest rate, not an end-to-end admission: "
                "it excludes building the delta CSR from source data, which both "
                "architectures must do and which is therefore not part of the "
                "comparison.",
                "The DistDGL figure depends entirely on the command passed in. "
                "Record that command with the number or the number means nothing.",
                "Training is left running during the pool measurement and stopped "
                "during the DistDGL one, because that asymmetry is the finding, "
                "not an experimental flaw.",
            ],
        )
        print(f"wrote {rec.write().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
