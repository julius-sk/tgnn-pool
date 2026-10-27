# tgnn-pool

Experiments and derivations for training a temporal graph neural network on
pooled, byte-addressable memory shared by several hosts.

A snapshot temporal graph grows. Every architecture that trains on one has to
answer the same three questions, and the answers differ by more than constants:

1. **Correctness.** While a new snapshot is being written, what does a sampler
   that is mid-batch see? If it can see edges from a snapshot later than the
   one it is training on, the model is reading the future, and no amount of
   throughput makes that acceptable.
2. **Capacity.** How many snapshots fit? Not how many edges — how many
   *snapshots*, because the structure, not the feature table, is what grows.
3. **Growth cost.** What does admitting one more snapshot cost, and does
   training stop while it happens?

This repository holds six numbered pieces of work that answer those questions:
three derivations that run on a laptop in seconds, and three experiments that
need a pooled-memory host.

Nothing here trains a model. The model, the dataset pipeline and the
measurements this work is calibrated against live elsewhere; this repository is
the part that can be checked independently of them.

## Status

| id | question it settles | where it runs | cost | state |
|----|--------------------|---------------|------|-------|
| **B1** | Is version-filtered reading enough for a sampler to be isolated from a concurrent writer, and what does it need? | laptop | seconds | **done** |
| **B2** | At how many hosts does pooled memory stop winning, and on which resource? | laptop | seconds | **done**, two inputs unmeasured |
| **B3** | How many snapshots fit, and which layout decides it? | laptop | seconds | **done** |
| **X1** | Does a sampler actually read the future during an ingest? | laptop (mock) / pool | minutes | **done on mock**, pool backend open |
| **X2** | What does admitting one snapshot cost, per architecture? | cluster | hours | open |
| **X2b** | Do two readers split the pool's bandwidth or not? | cluster | minutes | open — **cheapest, and gates B2** |
| **X3** | How much does the evaluation protocol in common use overstate accuracy? | cluster | ~7 h | open |

## What has come out so far

**Read committed is not enough, and the gap is measurable.** A sampler that
accepts any fully-published object — the obvious implementation — took **10.7%
of its edges from a snapshot it was not entitled to** while an ingest was in
flight (X1, mock backend). Filtering by version took none. B1 shows this holds
under all 1820 interleavings of one writer and one reader, with no read lock,
and that it needs exactly three preconditions: snapshots immutable once
published, publication atomic, and nothing reclaimed inside a reader's epoch.

**The third precondition is not enforced.** The pool API exposes retirement
with no reader protection, so a compactor can free an object a sampler is
holding. B1 case 2 reproduces it. This is a defect, not a hypothetical.

**At 500 GB, the layout decides whether the full graph fits at all.** For the
120-snapshot graph, features are 374 GB and the three structure layouts cost
745 GB, 127 GB and 12.7 GB. The layout in use today — one row-pointer array per
snapshot — lands at 501.3 GB against a 500 GB pool: it **misses by two
snapshots**. A shared row-pointer layout clears the same graph with 113 GB to
spare, but needs a sampler that can cut inside a row, which the sampler in use
cannot do. The capacity question is therefore a sampler question (B3).

**Pool bandwidth under two readers is the gating unknown.** Every statement
about more hosts than this cluster has goes through B2, and B2 cannot produce a
number until someone runs two readers at once. The answer is either "saturates
around 14 hosts" or "saturates around 2", and those support opposite papers
(X2b).

## Quick start

```bash
pip install -r requirements.txt

python derive/b1_check.py          # snapshot isolation, exhaustive
python derive/b2_crossover.py      # placement crossover vs host count
python derive/b3_capacity.py       # snapshot capacity vs layout
python exp/x1_leakage.py           # leakage, mock backend

bash scripts/run_local.sh          # all of the above, writing results/
```

Everything above runs on a laptop with no pool, no GPU and no dataset. The
cluster half:

```bash
python exp/x2b_contention.py --dry-run      # plan only
python exp/x2_admission.py   --dry-run
python exp/x3_rolling_origin.py --dry-run
```

## Layout

```
configs/      machine and workload descriptions. Every capacity, bandwidth and
              dataset size lives here; no script hardcodes one.
tgnnpool/     shared code: config loading, result records, the byte-size model,
              and the versioned store with its mock and pool backends.
derive/       B1-B3. Pure computation, no hardware.
exp/          X1-X3. X1 has a mock backend that needs no hardware.
docs/         the written arguments, and the conventions the numbers follow.
results/      one JSON record per run (gitignored).
```

## Changing the pool size

All three capacity answers come from `configs/machine.yaml`. Change
`pool.usable_bytes` and rerun; nothing else needs editing.

```bash
python derive/b3_capacity.py --pool-bytes 137438953472   # 128 GiB
python derive/b2_crossover.py --layout shared_rowptr
```

## Conventions

Every number carries its provenance: **measured** on hardware, **derived** from
measured inputs by a stated formula, or **inferred** as an argument with no
number behind it. Configuration entries for quantities nobody has measured are
`null`, and reading one raises rather than returning a default, so a derivation
cannot quietly substitute a guess. See `docs/conventions.md`.

## Safety

The experiments attach to a pool that may be shared with other work.

- Never call the runtime's initialise or finalise entry points from anything
  here. Both wipe the metadata region.
- Attach read-only unless the experiment's whole purpose is to write.
- Check what a pool already holds before writing to it.

## License

Apache-2.0.
