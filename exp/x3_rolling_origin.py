#!/usr/bin/env python3
"""X3 - what does the usual evaluation protocol overstate?

A temporal model should be scored the way it will be used: fit on everything
up to t, predict t+1, advance t.  That is rolling-origin evaluation.  Most
snapshot-GNN work does not do it, and the reason is cost, not ignorance --
each origin needs its own partition of the graph, and partitioning is hours.

In a versioned pool an origin is not a new partition.  It is a read filter:
train on rows with version <= t.  Setting up a fold costs nothing, so the
protocol that was unaffordable becomes the default.  That makes this both a
methodological point and a direct consequence of the storage design, which is
why it belongs in the same paper as B1 and X1 rather than in a different one.

Two arms, same model, same budget, same seeds:

  random    the protocol in common use: split labelled nodes at random,
            train on all snapshots at once, report test accuracy.
  rolling   for each origin t: train on snapshots <= t, test on the labels
            that first appear in t+1. Report the mean over origins.

The number to report is the difference.  It can come out small -- that is a
publishable answer too, and a more useful one than not knowing.

  python exp/x3_rolling_origin.py --dry-run     # fold plan and machine time
  python exp/x3_rolling_origin.py --run         # cluster
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool import config
from tgnnpool.result import Record, banner


def folds(first_origin: int, last_origin: int, step: int) -> list[dict]:
    out = []
    for t in range(first_origin, last_origin + 1, step):
        out.append({"origin": t, "train_snapshots": list(range(0, t + 1)),
                    "test_snapshot": t + 1})
    return out


def estimate_hours(n_folds: int, epochs: int, seconds_per_epoch: float,
                   arms: int = 1) -> float:
    return arms * n_folds * epochs * seconds_per_epoch / 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first-origin", type=int, default=5)
    ap.add_argument("--last-origin", type=int, default=8)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=12,
                    help="epochs per fold. Must be identical across arms; a "
                         "shortened budget is fine, an unequal one is not "
                         "(default 12)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--trainer", default="python train_mag_temporal.py",
                    help="command that trains one fold; receives "
                         "--max-version and --test-snapshot")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    wl = config.workload()
    per_epoch = wl.require("training", "seconds_per_epoch")
    plan = folds(args.first_origin, args.last_origin, args.step)
    n = len(plan) * args.seeds

    banner("X3", "rolling-origin evaluation against the protocol in common use")
    print(f"{len(plan)} origins x {args.seeds} seed(s) = {n} fold(s), "
          f"{args.epochs} epochs each")
    print(f"measured {per_epoch:.0f} s per epoch -> rolling arm "
          f"{estimate_hours(n, args.epochs, per_epoch):.1f} h, "
          f"random arm {estimate_hours(args.seeds, args.epochs, per_epoch):.1f} h")
    print(f"\n  {'origin':>7}  {'train on snapshots':<24}{'test on':>9}")
    for f in plan:
        rng = f"0..{f['origin']}"
        print(f"  {f['origin']:>7}  {rng:<24}{f['test_snapshot']:>9}")

    print("\nwhat makes this cheap here, and the thing to check first:")
    print("  each fold is a read filter (version <= t), not a repartition.")
    print("  Fold setup must therefore be seconds. If it is not, the mechanism")
    print("  is not being used and the protocol argument has no support.")

    if args.dry_run or not args.run:
        print("\nnothing was run. Add --run on a host attached to the pool.")
        return 0

    results = []
    for f in plan:
        for seed in range(args.seeds):
            cmd = (f"{args.trainer} --max-version {f['origin'] + 1} "
                   f"--test-snapshot {f['test_snapshot']} "
                   f"--num-epochs {args.epochs} --seed {seed}")
            print(f"\n$ {cmd}")
            started = time.perf_counter()
            proc = subprocess.run(cmd, shell=True)
            results.append({**f, "seed": seed, "returncode": proc.returncode,
                            "seconds": time.perf_counter() - started,
                            "command": cmd})

    if not args.no_write:
        rec = Record(
            experiment="X3",
            evidence="measured",
            summary=("Rolling-origin folds over a versioned pool, for comparison "
                     "against a single random split at the same budget."),
            inputs=vars(args),
            outputs={"folds": results},
            caveats=[
                "Accuracy is not parsed here; the trainer writes its own metrics. "
                "This driver records what was run and how long it took, so that "
                "the two arms can be shown to have had an equal budget.",
                "Equal budget means equal epochs per fold, not equal total epochs. "
                "The rolling arm necessarily costs more in total; that is the cost "
                "the protocol has always had and the reason it is skipped.",
                "Early origins train on less data and will score worse for that "
                "reason alone. Report per-origin numbers, not only the mean, or "
                "the comparison confounds protocol with training-set size.",
            ],
        )
        print(f"\nwrote {rec.write().name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
