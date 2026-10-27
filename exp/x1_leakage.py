#!/usr/bin/env python3
"""X1 - does a sampler read the future while a new snapshot is being published?

B1 proves what happens under every interleaving of one writer and one reader.
X1 is the empirical counterpart: real threads at real speeds, counting the
edges that leak, through one code path that runs against either the mock store
or the pool.  Both are needed, because B1 can be right about a protocol the
hardware does not implement, and X1 on its own cannot tell a safe design from a
lucky schedule.

How a snapshot is modelled.  The pool does not hold one object per graph row;
it holds a handful of large arrays per snapshot -- row pointers, then column
indices in blocks.  Object count matters here, because a reader's chance of
landing inside a publication in flight is roughly one over the number of
objects.  Modelling per-row objects would make tearing look rare for a reason
that has nothing to do with the design.

What is counted, per reader strength:

  leaked edges  edges returned whose snapshot is later than the reader's
                logical time t.  In a temporal graph those are future
                information, so this is a correctness count.
  torn reads    payloads that failed their checksum.
  faults        use-after-retire, when reclamation is enabled.

Expected shape, from B1:   NAIVE tears and leaks | COMMITTED leaks | SNAPSHOT neither.

  python exp/x1_leakage.py                   # mock backend, laptop
  python exp/x1_leakage.py --backend pool    # cluster, needs cxl_shm
"""
from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool.result import Record, banner
from tgnnpool.store import MockStore, Reader, TornRead, UseAfterRetire

NAME = "s{snap}.obj{obj}"


def spin(seconds: float) -> None:
    """Busy-wait.  time.sleep cannot resolve tens of microseconds on every
    platform, and rounding it up to the scheduler tick would silently change
    the overlap this experiment is measuring."""
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def make_payload(snap: int, obj: int, edges: int, rng: random.Random) -> bytes:
    body = ",".join(str(rng.randrange(1 << 20)) for _ in range(edges))
    return f"{snap}|{obj}|{body}".encode()


def decode_snapshot(payload: bytes) -> int:
    return int(payload.split(b"|", 1)[0])


class Writer(threading.Thread):
    """Publishes one snapshot's objects while the sampler is running."""

    def __init__(self, store: MockStore, snap: int, objects: int, edges: int,
                 pass_pause_s: float, obj_pause_s: float, seed: int) -> None:
        super().__init__(daemon=True)
        self.store, self.snap, self.objects, self.edges = store, snap, objects, edges
        self.pass_pause_s, self.obj_pause_s = pass_pause_s, obj_pause_s
        self.rng = random.Random(seed)
        self.published = 0
        self.done = threading.Event()

    def run(self) -> None:
        # The gap between passes is real work: a payload write followed by a
        # cache write-back. It is also the only window a torn read can land in.
        self.store.hook = lambda p, n: spin(self.pass_pause_s) if p < 4 else None
        for obj in range(self.objects):
            self.store.publish(NAME.format(snap=self.snap, obj=obj),
                               make_payload(self.snap, obj, self.edges, self.rng),
                               version=self.snap + 1)
            self.published += 1
            spin(self.obj_pause_s)
        self.store.hook = lambda p, n: None
        self.done.set()


def sample(store: MockStore, mode: Reader, t: int, objects: int, seeds: int,
           hops: int, rng: random.Random, edges_per_object: int) -> dict:
    """One sampling pass at logical time t."""
    read_edges = leaked = torn = faults = 0
    for _ in range(seeds * hops):
        # A sampler asks every snapshot it knows of. It knows of t+1 as well:
        # the name is derivable and an unfiltered reader has no reason to skip
        # it. That is the whole mechanism of the leak.
        snap = rng.randrange(t + 2)
        obj = rng.randrange(objects)
        try:
            data = store.read(NAME.format(snap=snap, obj=obj), mode, at_version=t + 1)
        except TornRead:
            torn += 1
            continue
        except UseAfterRetire:
            faults += 1
            continue
        if data is None:
            continue
        read_edges += edges_per_object
        if decode_snapshot(data) > t:
            leaked += edges_per_object
    return {"edges_read": read_edges, "leaked_edges": leaked,
            "torn_reads": torn, "faults": faults}


def trial(mode: Reader, t: int, objects: int, edges: int, seeds: int, hops: int,
          pass_pause_us: float, obj_pause_us: float, head_start_frac: float,
          seed: int) -> dict:
    rng = random.Random(seed)
    store = MockStore()
    for snap in range(t + 1):                       # history, already published
        for obj in range(objects):
            store.publish(NAME.format(snap=snap, obj=obj),
                          make_payload(snap, obj, edges, rng), version=snap + 1)

    writer = Writer(store, t + 1, objects, edges,
                    pass_pause_us / 1e6, obj_pause_us / 1e6, seed + 1)
    writer.start()
    # A trainer does not wait for an ingest to begin. Let the writer get
    # partway in, which is the common case and the one read committed handles
    # worst.
    spin(head_start_frac * objects * (3 * pass_pause_us + obj_pause_us) / 1e6)
    out = sample(store, mode, t, objects, seeds, hops, rng, edges)
    writer.join(timeout=10)
    out["objects_published_concurrently"] = writer.published
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("mock", "pool"), default="mock")
    ap.add_argument("--snapshots", type=int, default=6,
                    help="reader's logical time t (default 6)")
    ap.add_argument("--objects", type=int, default=4,
                    help="arrays per snapshot: row pointers plus column-index "
                         "blocks (default 4)")
    ap.add_argument("--edges-per-object", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--pass-pause-us", type=float, default=300.0,
                    help="gap between the passes of one publish; the torn-read "
                         "window (default 300 us)")
    ap.add_argument("--obj-pause-us", type=float, default=300.0,
                    help="gap between objects (default 300 us)")
    ap.add_argument("--head-start", type=float, default=0.3,
                    help="fraction of the publication done before the sampler "
                         "starts (default 0.3)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    if args.backend == "pool":
        from tgnnpool.store import PoolStore
        PoolStore()          # raises with instructions; backend not finished yet

    publish_ms = args.objects * (3 * args.pass_pause_us + args.obj_pause_us) / 1e3
    banner("X1", "future-snapshot leakage during concurrent publication")
    print(f"backend {args.backend}; reader at t={args.snapshots}; "
          f"{args.objects} objects per snapshot; {args.seeds}x{args.hops} reads; "
          f"{args.trials} trials")
    print(f"publishing snapshot {args.snapshots + 1} takes ~{publish_ms:.0f} ms; "
          f"the sampler starts {args.head_start:.0%} of the way in")

    results: dict = {}
    print(f"\n  {'reader':<11}{'edges read':>12}{'leaked':>10}{'leak rate':>11}"
          f"{'torn':>7}{'faults':>8}")
    for mode in (Reader.NAIVE, Reader.COMMITTED, Reader.SNAPSHOT):
        agg = {"edges_read": 0, "leaked_edges": 0, "torn_reads": 0, "faults": 0,
               "objects_published_concurrently": 0}
        for i in range(args.trials):
            out = trial(mode, args.snapshots, args.objects, args.edges_per_object,
                        args.seeds, args.hops, args.pass_pause_us,
                        args.obj_pause_us, args.head_start, seed=1000 + i)
            for k in agg:
                agg[k] += out[k]
        agg["leak_rate"] = agg["leaked_edges"] / max(1, agg["edges_read"])
        results[mode.name.lower()] = agg
        print(f"  {mode.name:<11}{agg['edges_read']:>12,}{agg['leaked_edges']:>10,}"
              f"{agg['leak_rate']:>10.1%}{agg['torn_reads']:>7}{agg['faults']:>8}")

    shape_ok = (results["snapshot"]["leaked_edges"] == 0
                and results["snapshot"]["torn_reads"] == 0
                and results["committed"]["leaked_edges"] > 0
                and results["naive"]["torn_reads"] > 0)
    print("\n" + "-" * 72)
    if shape_ok:
        print("matches B1: read committed leaks the future, version filtering does not.")
        print(f"  a sampler that checks only readiness took "
              f"{results['committed']['leak_rate']:.1%} of its edges from a snapshot "
              f"it was not entitled to")
    else:
        print("DOES NOT match B1. Before this reaches the paper, find out whether "
              "the backend\npublishes in four ordered passes and whether the reader "
              "filters by version.")
    if args.backend == "mock":
        print("Mock backend: this exercises the protocol and the reader, not the "
              "hardware.\nThe same run with --backend pool is what licenses a claim "
              "about the pool.")

    if not args.no_write:
        rec = Record(
            experiment="X1",
            evidence="measured" if args.backend == "pool" else "derived",
            summary=("Edges taken from a snapshot later than the reader's logical "
                     "time by a sampler running concurrently with an ingest, per "
                     "reader strength."),
            inputs=vars(args),
            outputs=results,
            caveats=[
                "Thread timing decides how often the sampler lands inside the "
                "publication window, so the absolute leak rate is a property of "
                "this pacing. Only zero against non-zero is a claim; the magnitude "
                "is not.",
                "The mock backend is single-process and cannot exercise write-back "
                "and fence ordering between hosts. Only the pool backend can.",
                "Retirement is off here. Switching it on breaks the SNAPSHOT reader "
                "too (B1 case 2), and the pool API does nothing today to prevent it.",
                "Edges are counted per object read, not per distinct edge, so "
                "edges_read is a volume, not a set size.",
            ],
        )
        print(f"\nwrote {rec.write().name}")
    return 0 if shape_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
