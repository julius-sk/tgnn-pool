# B1 — snapshot isolation for a sampler on a growing temporal graph

Run it: `python derive/b1_check.py`

## The problem

A temporal graph neural network trains on snapshot `t`: the graph as it stood
at time `t`, and nothing later. Keeping that true is normally a matter of
procedure — partition the data first, then train, and change nothing while
training runs. The procedure is why nobody states it as a property.

Pooled memory makes it possible to write a new snapshot while training is in
progress, which is useful and which breaks the procedure. The question becomes
a storage question: what does a sampler see if a writer is publishing snapshot
`t+1` underneath it?

This matters more than the equivalent question in a non-temporal system. An
ordinary stale read gives you an out-of-date answer. Here, a read of the wrong
version gives the model an edge from the future of its own labels. The failure
is silent, it inflates accuracy, and it looks like a good result.

## The publish protocol

The runtime publishes an object in four ordered passes, each followed by a
write-back and a fence:

1. state → `WRITING`
2. payload
3. version and checksum
4. state → `READY`

A reader that waits for `READY` therefore never sees a partly written payload.
That is per-object atomicity, and it is where most designs stop.

## Why per-object atomicity is not enough

A sampler is not one read. It resolves row pointers, walks a frontier, reads
column indices, gathers features — many objects, over milliseconds. Per-object
atomicity gives it **read committed**: every object it sees is whole. It says
nothing about *which* objects it sees. An object published between two of the
sampler's reads is whole, is `READY`, and belongs to snapshot `t+1`.

The checker measures this directly. At `t=4`, over all 1820 interleavings of
the writer's four passes with the reader's twelve steps:

| reader | violating interleavings |
|---|---|
| accepts any bytes | 1819 / 1820 |
| accepts `READY` objects (read committed) | 1365 / 1820 |
| accepts `READY` and `version <= t` | **0 / 1820** |

X1 shows the same thing with real threads: 10.7% of edges taken from the wrong
snapshot under read committed, none under version filtering.

## The claim

> Let every published snapshot be immutable, let each be published by the
> four-pass protocol, and let nothing be reclaimed inside a reader's epoch.
> Then a reader that accepts only objects with state `READY` and
> `version <= t` observes exactly the edge set of snapshot `t`, under every
> interleaving with a concurrent writer, and takes no read lock.

Why it works, informally: an object that is `READY` and within the version
window never changes again, because snapshots are append-only. So the time at
which the reader touches it does not affect what it reads, and ordering between
reader and writer stops mattering. Append-only is what upgrades read committed
to snapshot isolation for free — without multi-version rows, and without the
read lock that a general-purpose store would need.

## The three preconditions, and what breaks each

The preconditions are the useful part. Each one is a design constraint that is
easy to violate by accident.

### 1. Immutability

Breaks if a snapshot can be revised in place: late-arriving edges backdated
into a past snapshot, or features that vary with time. The checker's case 3
turns revision on and the version-filtered reader fails **every** interleaving,
because the bytes under a version it already accepted changed.

This is also the technical reason the continuous-time branch of temporal GNNs
cannot adopt this design unchanged. Models that carry per-node memory rewrite
node state on every edge, so nothing is append-only and the argument gives
nothing.

### 2. Atomic publication

Breaks if a writer dies mid-publish, which is what the four passes exist to
survive, and which is verified separately against hardware rather than here.
This model checks the pass *order*; it does not model the cache write-backs
and fences that make that order visible to another host.

### 3. No reclamation inside a reader's epoch

**This one is not enforced by the current API.** Retirement is exposed with no
reader protection: a compactor that merges old snapshots and frees them can
free an object a sampler is holding a reference to. The checker's case 2 does
exactly that and the version-filtered reader faults in every interleaving.

The fix is standard — reference counting, or epoch-based reclamation keyed to
the version the reader opened at — and the point of stating the precondition is
that nobody notices the hole until the first compactor exists.

## Scope

- One writer and one reader. Two writers to the same object are out of scope;
  the ingest path is single-writer by construction.
- The pass order, not the memory ordering that makes it visible across hosts.
- Edge sets, not feature values. A static feature table is trivially isolated;
  a mutable one falls under precondition 1.

## Relation to prior work

Snapshot isolation is a database idea and nothing here improves on it. What is
worth writing down is the mapping: which of its preconditions hold for free in
snapshot-based temporal graph training (append-only growth does the work),
which fail in the continuous-time branch (per-node memory), and which the
runtime leaves open (reclamation). A literature check on whether this has been
stated for GNN training is listed as a prerequisite in `docs/conventions.md`
and has not been done.
