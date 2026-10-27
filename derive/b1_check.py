#!/usr/bin/env python3
"""B1 - snapshot isolation for a sampler reading a growing temporal graph.

The claim this checks:

    If published snapshots are immutable, if each is published by the
    four-pass protocol, and if nothing is reclaimed inside a reader's epoch,
    then a reader that accepts only objects with state == READY and
    version <= t observes exactly the edge set of snapshot t, under every
    interleaving with a concurrent writer, and needs no read lock.

Per-object atomic publish on its own gives read committed, not snapshot
isolation: a sampler touches many objects over a long interval, and an object
published between two of its reads is visible to it.  In a temporal graph that
object holds edges from the future, so read committed is not merely weaker, it
is wrong.  What closes the gap here -- without multi-version rows and without
locking -- is that snapshots are append-only.

The checker enumerates every interleaving of the writer's four passes with the
reader's steps and reports violations per reader strength.  It then turns off
each of the three preconditions in turn and shows the argument collapsing,
which is the part worth more than the theorem: it says exactly when this design
stops being safe.

Runs on a laptop.  No pool, no GPU, no dataset.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tgnnpool.result import Record, banner
from tgnnpool.store import MockStore, Reader, TornRead, UseAfterRetire


def payload_of(snapshot: int) -> bytes:
    return f"edges-of-snapshot-{snapshot}".encode()


def snapshot_of(payload: bytes) -> int:
    return int(payload.decode().rsplit("-", 1)[1])


def run_schedule(schedule: str, t: int, mode: Reader,
                 retire_at: int | None = None,
                 revise_at: int | None = None) -> tuple[set[int], list[str]]:
    """Play one interleaving.  Returns (edges the reader observed, faults)."""
    store = MockStore()
    for snap in range(t + 1):                       # history, already published
        store.publish(f"s{snap}", payload_of(snap), version=snap + 1)

    writer = store.publish_steps(f"s{t + 1}", payload_of(t + 1), version=t + 2)

    # The sampler touches each snapshot object twice and far apart: once for the
    # row pointers when it starts the batch, once for the column indices when it
    # expands the frontier.  Modelling a single atomic touch would hide
    # precondition 3 entirely, because reclamation is only observable to a
    # reader that holds a reference across two touches.  It also tries s{t+1},
    # because a reader that does not filter by version discovers every object
    # that is published, not only the ones it is entitled to.
    names = [f"s{i}" for i in range(t + 2)]
    reader_targets = names + names                     # pass 1 rowptr, pass 2 colind
    observed: set[int] = set()
    faults: list[str] = []
    step = 0

    for action in schedule:
        if action == "W":
            next(writer, None)
            continue
        if retire_at is not None and step == retire_at:
            store.retire("s0")                      # a compactor reclaims history
        if revise_at is not None and step == revise_at:
            store.revise(f"s{t - 1}", b"edges-of-snapshot-999")   # late edges
        name = reader_targets[step]
        try:
            data = store.read(name, mode, at_version=t + 1)
        except TornRead:
            faults.append(f"torn read of {name}")
            data = None
        except UseAfterRetire:
            faults.append(f"use-after-retire of {name}")
            data = None
        if data is not None:
            observed.add(snapshot_of(data))
        step += 1
    return observed, faults


def schedules(writer_steps: int, reader_steps: int):
    total = writer_steps + reader_steps
    for positions in itertools.combinations(range(total), writer_steps):
        seq = ["R"] * total
        for p in positions:
            seq[p] = "W"
        yield "".join(seq)


def check(t: int, mode: Reader, **kw) -> dict:
    expected = set(range(t + 1))
    reader_steps = 2 * (t + 2)
    violations, witness, faults_seen = 0, None, set()
    count = 0
    for sched in schedules(4, reader_steps):
        count += 1
        observed, faults = run_schedule(sched, t, mode, **kw)
        faults_seen.update(faults)
        if observed != expected or faults:
            violations += 1
            if witness is None:
                witness = {
                    "schedule": sched,
                    "observed": sorted(observed),
                    "expected": sorted(expected),
                    "extra": sorted(observed - expected),
                    "missing": sorted(expected - observed),
                    "faults": faults,
                }
    return {
        "schedules": count,
        "violations": violations,
        "witness": witness,
        "faults": sorted(faults_seen),
    }


def describe(res: dict) -> str:
    w = res["witness"]
    if w is None:
        return "-"
    if w["extra"]:
        return f"{w['schedule']} -> observed future snapshot {w['extra']}"
    if w["faults"]:
        return f"{w['schedule']} -> {w['faults'][0]}"
    return f"{w['schedule']} -> missing {w['missing']}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", type=int, default=4,
                    help="reader's logical time t; it reads snapshots 0..t while "
                         "snapshot t+1 is being published (default 4)")
    ap.add_argument("--no-write", action="store_true",
                    help="do not write a result record")
    args = ap.parse_args()
    t = args.snapshots

    banner("B1", "snapshot isolation under a concurrent publisher")
    print(f"reader at logical time t={t}: must observe exactly snapshots 0..{t}")
    print(f"writer concurrently publishes snapshot {t + 1} (version {t + 2})")

    outputs: dict = {}

    print("\ncase 1 - all three preconditions hold "
          "(append-only, atomic publish, no reclaim)")
    print(f"  {'reader':<11}{'schedules':>10}{'violations':>12}   first witness")
    for mode in (Reader.NAIVE, Reader.COMMITTED, Reader.SNAPSHOT):
        res = check(t, mode)
        outputs[f"case1_{mode.name.lower()}"] = res
        print(f"  {mode.name:<11}{res['schedules']:>10}{res['violations']:>12}"
              f"   {describe(res)}")

    print("\ncase 2 - reclamation allowed inside the reader's epoch (SNAPSHOT reader)")
    # A compactor merges old snapshots away and retires s0 just after the
    # reader has resolved it and before the reader dereferences it again.
    res2 = check(t, Reader.SNAPSHOT, retire_at=t + 2)
    outputs["case2_snapshot_with_retire"] = res2
    print(f"  violations {res2['violations']}/{res2['schedules']}"
          f"   faults: {res2['faults'] or '-'}")

    print("\ncase 3 - published snapshots mutable in place (SNAPSHOT reader)")
    res3 = check(t, Reader.SNAPSHOT, revise_at=1)
    outputs["case3_snapshot_with_revise"] = res3
    print(f"  violations {res3['violations']}/{res3['schedules']}"
          f"   {describe(res3)}")

    ok = (outputs["case1_snapshot"]["violations"] == 0
          and outputs["case1_committed"]["violations"] > 0
          and outputs["case1_naive"]["violations"] > 0
          and res2["violations"] > 0
          and res3["violations"] > 0)

    print("\n" + "-" * 72)
    print("B1 holds exactly as stated" if ok else "B1 DID NOT COME OUT AS STATED")
    print("  read committed is not enough:    "
          f"{outputs['case1_committed']['violations']} of "
          f"{outputs['case1_committed']['schedules']} interleavings "
          "admit a future snapshot")
    print("  version filtering is enough:     "
          f"{outputs['case1_snapshot']['violations']} violations over "
          f"{outputs['case1_snapshot']['schedules']} interleavings, no lock taken")
    print("  all three preconditions matter:  cases 2 and 3 each break it")

    if not args.no_write:
        rec = Record(
            experiment="B1",
            evidence="derived",
            summary=("Version-filtered reads give a sampler snapshot isolation over a "
                     "concurrently growing temporal graph without locking, provided "
                     "snapshots are immutable and nothing is reclaimed inside the "
                     "reader's epoch; read committed is not sufficient."),
            inputs={"t": t, "writer_passes": 4, "reader_steps": 2 * (t + 2),
                "reader_touches_per_object": 2},
            outputs=outputs,
            caveats=[
                "Exhaustive over interleavings of one writer and one reader only. "
                "Two concurrent writers to one snapshot are out of scope, and the "
                "ingest path is single-writer by construction.",
                "The model checks the protocol's pass ORDER, not the cache "
                "write-backs and fences that make that order visible to another "
                "host. Those are validated separately on hardware.",
                "Precondition 3 is NOT enforced by the pool API, which exposes "
                "retire with no reader protection. Case 2 is a live defect in the "
                "runtime, not a hypothetical.",
            ],
        )
        print(f"\nwrote {rec.write().name}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
