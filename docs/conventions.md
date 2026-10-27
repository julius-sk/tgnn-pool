# Conventions

## Evidence

Every number in this repository is one of three things, and which one is
recorded in the result file rather than left to the reader:

| tag | meaning |
|---|---|
| `measured` | observed on hardware, with the observation named |
| `derived` | computed from measured inputs by a formula given in the code |
| `inferred` | an argument; no number behind it |

Two rules follow:

- **A quantity nobody has measured is `null` in the config, not a default.**
  Reading one raises `Missing`, naming the experiment that would fill it in.
  A derivation cannot quietly substitute a plausible value and report the
  result as a number. Two entries are `null` today: pool aggregate bandwidth
  and partition-and-reload time.
- **Quote conditions with numbers.** "5.1 GB/s" is not a fact about the pool;
  "5.1 GB/s for one reader, random 3072-byte rows, 225 MB per iteration" is.

## Units

- Capacities and bandwidths are **decimal** (GB = 10⁹ bytes), because the
  device and dataset sizes this work is calibrated against are quoted that way.
  `gib()` is available where a binary figure is wanted; it is always labelled.
- Snapshots are counted, not dated. Where a unit is needed the dataset's
  timestamps are **years**. Material elsewhere in this project says "day"; that
  came from a framework README and is wrong.

## Validating a model before using it

`B3` reproduces a measured structure size to within 0.3% before it reports
anything. A capacity model that cannot reproduce a size someone has already
observed should not be used to make a claim about a size nobody has.

## Safety when attaching to a shared pool

The pool may hold other work. These rules are not optional:

- **Never call the runtime's initialise or finalise entry points.** Both zero
  the metadata region. Nothing in this repository calls either, and nothing
  added to it should.
- **Attach read-only** unless writing is the entire point of the experiment.
- **Look before writing.** Record what the pool already holds, and check it is
  unchanged afterwards.
- **One writer.** The isolation argument in B1 assumes a single writer, and the
  ingest path provides one. Two concurrent writers to one object are outside
  everything claimed here.

## Literature checks still outstanding

These are prerequisites for claiming novelty, not for running the code. None
has been done.

1. Has snapshot isolation, or any equivalent version-consistent read, been
   stated for GNN *training* rather than for graph analytics?
2. How far do streaming and incremental graph systems already cover
   "ingest while training, without leaking the future"?
3. Has rolling-origin evaluation been raised as a problem in snapshot temporal
   GNN work, and is the usual protocol known to be optimistic?
4. Does the distributed graph framework being compared against have an
   incremental snapshot-admission path? If it does, X2's comparison changes.

Until these are done, the arguments stand on their own evidence and claim no
priority.
