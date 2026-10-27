#!/usr/bin/env python3
"""X2b - does a second reader halve the pool's bandwidth, or not?

This is the cheapest unrun experiment here and it gates the most.  Every claim
about more hosts than this cluster has runs through B2, and B2 cannot produce a
number until it knows whether the measured 5.1 GB/s of a single reader is a
per-reader rate or a device limit that readers split.  The gap between the two
answers is the difference between the pool saturating at two hosts and at
fourteen.

Procedure: one reader gathers rows for a fixed interval and reports its own
achieved bandwidth.  Then two readers do the same thing simultaneously, one per
host, and both report.  Aggregate is the sum.

  independent  aggregate ~ 2x single   -> N_bw is large, pooling scales
  shared       aggregate ~ 1x single   -> N_bw ~ 2, and that is the headline
                                           limit of this architecture

Either answer is publishable.  The second one is more interesting and is the
reason this must be run before anything is written down.

  python exp/x2b_contention.py --dry-run                 # plan, laptop
  python exp/x2b_contention.py --readers 1               # on host A
  python exp/x2b_contention.py --readers 2 --rank 0      # host A, with
  python exp/x2b_contention.py --readers 2 --rank 1      # host B, together
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool import config
from tgnnpool.result import Record, banner


def gather_loop(seconds: float, rows: int, row_bytes: int, rank: int) -> dict:
    """Gather random rows from the pool for a fixed interval.

    Deliberately uses the same access shape as training: random row indices,
    one row per lookup, the measured row width. A sequential scan would report
    a bandwidth this workload never sees.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("numpy is required on the cluster side") from exc
    try:
        import cxl_shm
    except ImportError as exc:
        raise SystemExit(
            "cxl_shm not importable. This half of X2b only runs on a host "
            "attached to the pool; use --dry-run elsewhere.") from exc

    raise NotImplementedError(
        "Fill in with the project's existing gather path rather than a new one: "
        "the bandwidth being measured must be the bandwidth training actually "
        "gets. Reuse the index_select path from the micro-benchmark, attach "
        "read-only, and never call cxl_shm_init or cxl_shm_finalize here -- both "
        "wipe the pool."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readers", type=int, default=1, choices=(1, 2))
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    mc, wl = config.machine(), config.workload()
    single = mc.require("pool", "gather_bytes_per_s_single_reader")
    row_bytes = wl.require("full", "feature_dim") * 4
    demand = (wl.require("training", "gather_bytes_per_batch")
              / wl.require("training", "seconds_per_batch"))

    banner("X2b", "pool bandwidth under concurrent readers")
    print(f"single reader, measured earlier: {single / 1e9:.1f} GB/s")
    print(f"one training host demands:       {demand / 1e9:.2f} GB/s")
    print(f"row width {row_bytes} B; random row access, {args.seconds:.0f} s per arm")
    print("\nthe two hypotheses, and what each implies")
    print(f"  independent: aggregate ~ {2 * single / 1e9:.1f} GB/s "
          f"-> N_bw ~ {2 * single / demand:.0f} hosts")
    print(f"  shared:      aggregate ~ {single / 1e9:.1f} GB/s "
          f"-> N_bw ~ {single / demand:.0f} hosts")

    if args.dry_run:
        print("\nrun it as:")
        print("  host A:  python exp/x2b_contention.py --readers 1")
        print("  then together, started within a second of each other:")
        print("    host A:  python exp/x2b_contention.py --readers 2 --rank 0")
        print("    host B:  python exp/x2b_contention.py --readers 2 --rank 1")
        print("  then put the SUM of the two-reader arms into configs/machine.yaml")
        print("  under pool.gather_bytes_per_s_aggregate, and rerun B2.")
        print("\nnothing was run.")
        return 0

    started = time.perf_counter()
    out = gather_loop(args.seconds, args.rows, row_bytes, args.rank)
    out["wall_seconds"] = time.perf_counter() - started
    print(f"\nreader {args.rank} of {args.readers}: {out}")

    if not args.no_write:
        rec = Record(
            experiment="X2b",
            evidence="measured",
            summary=(f"Pool gather bandwidth achieved by reader {args.rank} with "
                     f"{args.readers} reader(s) active."),
            inputs=vars(args),
            outputs=out,
            caveats=[
                "Each reader reports only its own rate. The aggregate is the sum "
                "across hosts and has to be assembled by hand, because the two "
                "runs are separate processes on separate machines.",
                "The arms must overlap in time or the comparison is meaningless. "
                "Record the start timestamps and discard a pair that does not "
                "overlap for most of its duration.",
                "Bind to the NUMA node the pool is near on both hosts, or the "
                "result measures the interconnect topology instead of the pool.",
            ],
        )
        print(f"wrote {rec.write().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
